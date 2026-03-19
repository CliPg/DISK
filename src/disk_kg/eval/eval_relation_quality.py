"""
关系质量评估模块
使用DuIE数据集评估关系提取的准确率、召回率、方向准确率
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# max samples
MAX_SAMPLES = 2

from disk import DISK
from config.llm import llm, embeddings
from utils import TokenTracker


@dataclass
class Relation:
    """关系类"""
    subject: str      # 主体实体名称
    object: str       # 客体实体名称
    predicate: str    # 关系类型

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.object == other.object and
                self.predicate == other.predicate)

    def __hash__(self):
        return hash((self.subject, self.object, self.predicate))


@dataclass
class RelationEvaluationMetrics:
    """关系评估指标"""
    precision: float       # 准确率
    recall: float          # 召回率
    f1: float              # F1值
    direction_accuracy: float  # 方向准确率
    total_ground_truth: int    # 真实关系总数
    total_extracted: int      # 提取关系总数
    true_positive: int        # 正确关系数
    direction_correct: int    # 方向正确数
    details: List[Dict]      # 每条数据的详细信息


class RelationEvaluator:
    """关系提取评估器"""

    def __init__(self, llm, embeddings, dataset_path: str, max_samples: int = MAX_SAMPLES):
        """
        初始化评估器

        Args:
            llm: 语言模型
            embeddings: 嵌入模型
            dataset_path: 数据集文件路径
            max_samples: 最大评估样本数
        """
        self.llm = llm
        self.embeddings = embeddings
        self.dataset_path = dataset_path
        self.max_samples = max_samples

        # 创建DISK实例，启用token跟踪
        self.disk = DISK(llm=llm, embeddings=embeddings, enable_token_tracking=True)

        # 加载数据集
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """加载数据集"""
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # 限制样本数量
        return data[:self.max_samples]

    def _extract_ground_truth_relations(self, data_item: Dict) -> Set[Relation]:
        """
        从数据项中提取真实关系

        Args:
            data_item: 单条数据项

        Returns:
            真实关系集合
        """
        relations = set()

        for spo in data_item.get('spo_list', []):
            subject = spo.get('subject')
            obj = spo.get('object')
            if isinstance(obj, dict):
                obj = obj.get('@value')
            predicate = spo.get('predicate')

            if subject and obj and predicate:
                relations.add(Relation(subject=subject, object=obj, predicate=predicate))

        return relations

    def _extract_predicted_relations(self, text: str) -> List[Relation]:
        """
        从文本中提取预测关系

        Args:
            text: 输入文本

        Returns:
            预测关系列表
        """
        # 调用DISK的关系提取器
        from extractor import RelationsExtractor

        extractor = RelationsExtractor(llm=self.llm, embeddings=self.embeddings, language='zh')
        result = extractor.extract_relations(text)

        if result is None:
            return []

        # 转换为Relation格式
        relations = []
        for rel in result:
            # 获取subject和object的名称
            subject = rel.start_entity
            obj = rel.end_entity

            subject_name = subject.name if hasattr(subject, 'name') else subject
            object_name = obj.name if hasattr(obj, 'name') else obj

            relations.append(Relation(
                subject=subject_name,
                object=object_name,
                predicate=rel.label
            ))

        return relations

    def _normalize_name(self, name: str) -> str:
        """标准化实体名称（去除空格、标点等）"""
        return name.strip().replace(' ', '').replace('\t', '').replace('\n', '')

    def _match_relation(self, predicted: Relation, ground_truth: Set[Relation]) -> Tuple[bool, bool]:
        """
        匹配预测关系与真实关系

        Args:
            predicted: 预测关系
            ground_truth: 真实关系集合

        Returns:
            (是否匹配, 方向是否正确)
        """
        pred_subject_norm = self._normalize_name(predicted.subject)
        pred_object_norm = self._normalize_name(predicted.object)
        pred_predicate = predicted.predicate

        for gt_rel in ground_truth:
            gt_subject_norm = self._normalize_name(gt_rel.subject)
            gt_object_norm = self._normalize_name(gt_rel.object)
            gt_predicate = gt_rel.predicate

            # 检查实体和关系类型是否匹配
            if (pred_subject_norm == gt_subject_norm or pred_subject_norm in gt_subject_norm or gt_subject_norm in pred_subject_norm) and \
               (pred_object_norm == gt_object_norm or pred_object_norm in gt_object_norm or gt_object_norm in pred_object_norm) and \
               (pred_predicate == gt_predicate or pred_predicate in gt_predicate or gt_predicate in pred_predicate):

                # 检查方向是否正确
                direction_correct = (pred_subject_norm == gt_subject_norm and pred_object_norm == gt_object_norm)
                return (True, direction_correct)

        return (False, False)

    def evaluate(self) -> RelationEvaluationMetrics:
        """
        执行评估

        Returns:
            评估指标
        """
        true_positive = 0
        total_extracted = 0
        total_ground_truth = 0
        direction_correct = 0
        details = []

        print(f"开始评估，共 {len(self.dataset)} 条数据...")

        for idx, item in enumerate(tqdm(self.dataset, desc="Evaluating")):
            text = item.get('text', '')
            if not text:
                continue

            # 获取真实关系
            gt_relations = self._extract_ground_truth_relations(item)

            # 提取预测关系
            pred_relations = self._extract_predicted_relations(text)

            # 对预测关系按（subject, object, predicate）去重
            pred_relations_dict = {}  # (subject, object, predicate) -> relation
            for rel in pred_relations:
                key = (rel.subject, rel.object, rel.predicate)
                if key not in pred_relations_dict:
                    pred_relations_dict[key] = rel
            unique_pred_relations = list(pred_relations_dict.values())

            # 计算匹配 - 每个真实关系只能被匹配一次
            matched_gt_keys = set()  # 已被匹配的真实关系 (subject, object, predicate)
            correct_relations = []
            wrong_relations = []
            missed_relations = []

            for pred_rel in unique_pred_relations:
                matched = False
                direction_correct = False

                for gt_rel in gt_relations:
                    gt_key = (gt_rel.subject, gt_rel.object, gt_rel.predicate)
                    if gt_key in matched_gt_keys:
                        continue  # 已经被匹配过了

                    is_match, is_dir_correct = self._match_relation(pred_rel, {gt_rel})
                    if is_match:
                        matched = True
                        matched_gt_keys.add(gt_key)
                        true_positive += 1
                        if is_dir_correct:
                            direction_correct += 1
                            correct_relations.append({
                                "subject": pred_rel.subject,
                                "object": pred_rel.object,
                                "predicate": pred_rel.predicate,
                                "direction_correct": True
                            })
                        else:
                            correct_relations.append({
                                "subject": pred_rel.subject,
                                "object": pred_rel.object,
                                "predicate": pred_rel.predicate,
                                "direction_correct": False
                            })
                        break

                if not matched:
                    wrong_relations.append({
                        "subject": pred_rel.subject,
                        "object": pred_rel.object,
                        "predicate": pred_rel.predicate
                    })

            # 找出未被提取的真实关系
            for gt_rel in gt_relations:
                gt_key = (gt_rel.subject, gt_rel.object, gt_rel.predicate)
                if gt_key not in matched_gt_keys:
                    missed_relations.append({
                        "subject": gt_rel.subject,
                        "object": gt_rel.object,
                        "predicate": gt_rel.predicate
                    })

            total_ground_truth += len(gt_relations)
            total_extracted += len(unique_pred_relations)

            # 保存每条数据的详细信息
            details.append({
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "ground_truth_relations": [
                    {"subject": r.subject, "object": r.object, "predicate": r.predicate}
                    for r in gt_relations
                ],
                "extracted_relations": [
                    {"subject": r.subject, "object": r.object, "predicate": r.predicate}
                    for r in pred_relations
                ],
                "extracted_relations_unique": [
                    {"subject": r.subject, "object": r.object, "predicate": r.predicate}
                    for r in unique_pred_relations
                ],
                "correct_relations": correct_relations,
                "wrong_relations": wrong_relations,
                "missed_relations": missed_relations,
                "stats": {
                    "ground_truth_count": len(gt_relations),
                    "extracted_count": len(pred_relations),
                    "extracted_unique_count": len(unique_pred_relations),
                    "correct_count": len(correct_relations),
                    "wrong_count": len(wrong_relations),
                    "missed_count": len(missed_relations),
                    "direction_correct_count": direction_correct
                }
            })

        # 计算指标
        # 准确率 = 正确关系数 / 总提取关系数（去重后）
        precision = true_positive / total_extracted if total_extracted > 0 else 0.0
        # 召回率 = 正确关系数 / 真实关系总数
        recall = true_positive / total_ground_truth if total_ground_truth > 0 else 0.0
        # F1值
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # 方向准确率 = 方向正确数 / 正确关系数
        direction_accuracy = direction_correct / true_positive if true_positive > 0 else 0.0

        return RelationEvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            direction_accuracy=direction_accuracy,
            total_ground_truth=total_ground_truth,
            total_extracted=total_extracted,
            true_positive=true_positive,
            direction_correct=direction_correct,
            details=details
        )

    def print_results(self, metrics: RelationEvaluationMetrics):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("关系质量评估结果")
        print("=" * 60)
        print(f"{'指标':<15} {'数值':>15} {'说明':<25}")
        print("-" * 60)
        print(f"{'关系准确率':<15} {metrics.precision:>15.2%} 正确关系数 / 总关系数")
        print(f"{'关系召回率':<15} {metrics.recall:>15.2%} 正确关系数 / 真实关系总数")
        print(f"{'关系 F1 值':<15} {metrics.f1:>15.2%} 准确率和召回率的调和平均")
        print(f"{'方向准确率':<15} {metrics.direction_accuracy:>15.2%} 方向正确数 / 正确关系数")
        print("-" * 60)
        print(f"真实关系总数: {metrics.total_ground_truth}")
        print(f"提取关系总数: {metrics.total_extracted}")
        print(f"正确提取数量: {metrics.true_positive}")
        print(f"方向正确数量: {metrics.direction_correct}")
        print("=" * 60 + "\n")

    def save_results(self, metrics: RelationEvaluationMetrics, output_path: str = None):
        """保存评估结果到文件"""
        if output_path is None:
            output_path = Path(__file__).parent / "results" / "relation_quality_eval.json"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 获取token使用统计
        token_summary = self.disk.get_token_summary() if self.disk.token_tracker else None

        results = {
            "metrics": {
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "direction_accuracy": metrics.direction_accuracy,
                "total_ground_truth": metrics.total_ground_truth,
                "total_extracted": metrics.total_extracted,
                "true_positive": metrics.true_positive,
                "direction_correct": metrics.direction_correct
            },
            "token_usage": token_summary,
            "dataset_size": len(self.dataset),
            "details": metrics.details
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 同时保存一个可读的文本格式结果
        text_output_path = output_path.parent / "relation_quality_eval.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("关系质量评估详细结果\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"总指标:\n")
            f.write(f"  关系准确率: {metrics.precision:.4f}\n")
            f.write(f"  关系召回率: {metrics.recall:.4f}\n")
            f.write(f"  关系 F1 值: {metrics.f1:.4f}\n")
            f.write(f"  方向准确率: {metrics.direction_accuracy:.4f}\n\n")

            f.write(f"统计数据:\n")
            f.write(f"  真实关系总数: {metrics.total_ground_truth}\n")
            f.write(f"  提取关系总数: {metrics.total_extracted}\n")
            f.write(f"  正确提取数量: {metrics.true_positive}\n")
            f.write(f"  方向正确数量: {metrics.direction_correct}\n\n")

            # Token使用统计
            if token_summary:
                f.write("Token使用统计:\n")
                f.write(f"  总调用次数: {token_summary['total_calls']}\n")
                f.write(f"  输入Token: {token_summary['total_input_tokens']}\n")
                f.write(f"  输出Token: {token_summary['total_output_tokens']}\n")
                f.write(f"  总成本: ${token_summary['total_cost_usd']:.4f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("详细数据\n")
            f.write("=" * 60 + "\n\n")

            for detail in metrics.details:
                f.write(f"[{detail['index']}] {detail['text']}\n")
                f.write(f"  统计: 真实={detail['stats']['ground_truth_count']}, "
                       f"提取={detail['stats']['extracted_count']}, "
                       f"去重提取={detail['stats']['extracted_unique_count']}, "
                       f"正确={detail['stats']['correct_count']}, "
                       f"错误={detail['stats']['wrong_count']}, "
                       f"遗漏={detail['stats']['missed_count']}\n")

                # 正确提取的关系
                if detail['correct_relations']:
                    f.write(f"  ✓ 正确提取:\n")
                    for r in detail['correct_relations']:
                        dir_str = "→" if r['direction_correct'] else "←"
                        f.write(f"      {dir_str} {r['subject']} | {r['predicate']} | {r['object']}\n")

                # 错误提取的关系
                if detail['wrong_relations']:
                    f.write(f"  ✗ 错误提取:\n")
                    for r in detail['wrong_relations']:
                        f.write(f"      ✗ {r['subject']} | {r['predicate']} | {r['object']}\n")

                # 遗漏的关系
                if detail['missed_relations']:
                    f.write(f"  ⊗ 遗漏关系:\n")
                    for r in detail['missed_relations']:
                        f.write(f"      ⊗ {r['subject']} | {r['predicate']} | {r['object']}\n")

                f.write("\n")

        print(f"评估结果已保存到: {output_path}")
        print(f"详细结果已保存到: {text_output_path}")


def main():
    """主函数"""
    # 数据集路径（相对路径）
    dataset_path = ROOT / "datasets/DuIE2.0/duie_dev.json/duie_dev.json"

    # 创建评估器，使用项目配置的 llm 和 embeddings
    evaluator = RelationEvaluator(
        llm=llm,
        embeddings=embeddings,
        dataset_path=str(dataset_path),
        max_samples=MAX_SAMPLES
    )

    # 执行评估
    metrics = evaluator.evaluate()

    # 打印结果
    evaluator.print_results(metrics)

    # 保存结果
    evaluator.save_results(metrics)

    # 打印token使用统计
    evaluator.disk.print_token_summary()


if __name__ == "__main__":
    main()
