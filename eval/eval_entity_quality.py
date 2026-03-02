"""
实体质量评估模块
使用DuIE数据集评估实体提取的准确率、召回率、F1值和类型准确率
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# max samples
MAX_SAMPLES = 1

from disk import DISK
from config.llm import llm, embeddings
from utils import TokenTracker


@dataclass
class Entity:
    """实体类"""
    name: str
    label: str

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class EvaluationMetrics:
    """评估指标"""
    precision: float  # 准确率
    recall: float     # 召回率
    f1: float         # F1值
    type_accuracy: float  # 类型准确率
    total_ground_truth: int  # 真实实体总数
    total_extracted: int    # 提取实体总数
    true_positive: int      # 正确提取的实体数
    correct_type: int       # 类型正确的实体数
    details: List[Dict]     # 每条数据的详细信息


class EntityEvaluator:
    """实体提取评估器"""

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

    def _extract_ground_truth_entities(self, data_item: Dict) -> Set[Entity]:
        """
        从数据项中提取真实实体

        Args:
            data_item: 单条数据项

        Returns:
            真实实体集合
        """
        entities = set()

        # 从 spo_list 中提取实体
        for spo in data_item.get('spo_list', []):
            # 提取主体实体
            subject = spo.get('subject')
            subject_type = spo.get('subject_type')
            if subject and subject_type:
                # subject_type 可能是类似 {"@value": "人物"} 的格式
                if isinstance(subject_type, dict):
                    subject_type = subject_type.get('@value', subject_type)
                entities.add(Entity(name=subject, label=subject_type))

            # 提取客体实体
            obj = spo.get('object')
            if isinstance(obj, dict):
                obj = obj.get('@value')
            object_type = spo.get('object_type')
            if obj and object_type:
                if isinstance(object_type, dict):
                    object_type = object_type.get('@value', object_type)
                entities.add(Entity(name=obj, label=object_type))

        return entities

    def _extract_predicted_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取预测实体

        Args:
            text: 输入文本

        Returns:
            预测实体列表
        """
        # 调用DISK的实体提取器
        # 注意：这里需要直接调用EntitiesExtractor而不是整个流程
        from extractor import EntitiesExtractor

        extractor = EntitiesExtractor(llm=self.llm, embeddings=self.embeddings, language='zh')
        result = extractor.extract_entities(text)

        if result is None:
            return []

        # 转换为Entity格式
        entities = []
        for entity in result:
            entities.append(Entity(name=entity.name, label=entity.label))

        return entities

    def _normalize_entity_name(self, name: str) -> str:
        """标准化实体名称（去除空格、标点等）"""
        return name.strip().replace(' ', '').replace('\t', '').replace('\n', '')

    def _match_entity(self, predicted: Entity, ground_truth: Set[Entity]) -> Tuple[bool, bool]:
        """
        匹配预测实体与真实实体

        Args:
            predicted: 预测实体
            ground_truth: 真实实体集合

        Returns:
            (是否匹配, 类型是否正确)
        """
        pred_name_norm = self._normalize_entity_name(predicted.name)

        for gt_entity in ground_truth:
            gt_name_norm = self._normalize_entity_name(gt_entity.name)

            # 名称匹配（精确匹配或包含关系）
            if pred_name_norm == gt_name_norm or pred_name_norm in gt_name_norm or gt_name_norm in pred_name_norm:
                # 检查类型是否正确
                type_correct = (predicted.label == gt_entity.label or
                               predicted.label in gt_entity.label or
                               gt_entity.label in predicted.label)
                return (True, type_correct)

        return (False, False)

    def evaluate(self) -> EvaluationMetrics:
        """
        执行评估

        Returns:
            评估指标
        """
        true_positive = 0
        total_extracted = 0
        total_ground_truth = 0
        correct_type = 0
        details = []  # 每条数据的详细信息

        print(f"开始评估，共 {len(self.dataset)} 条数据...")

        for idx, item in enumerate(tqdm(self.dataset, desc="Evaluating")):
            text = item.get('text', '')
            if not text:
                continue

            # 获取真实实体
            gt_entities = self._extract_ground_truth_entities(item)

            # 提取预测实体
            pred_entities = self._extract_predicted_entities(text)

            # 计算匹配
            matched_gt = set()
            matched_pred = set()
            correct_entities = []
            missed_entities = []
            wrong_entities = []

            for pred_entity in pred_entities:
                is_match, is_type_correct = self._match_entity(pred_entity, gt_entities)
                if is_match:
                    true_positive += 1
                    matched_pred.add(pred_entity)
                    if is_type_correct:
                        correct_type += 1
                        correct_entities.append({
                            "name": pred_entity.name,
                            "label": pred_entity.label,
                            "type_correct": True
                        })
                    else:
                        correct_entities.append({
                            "name": pred_entity.name,
                            "label": pred_entity.label,
                            "type_correct": False
                        })
                    # 标记已匹配的真实实体
                    for gt_entity in gt_entities:
                        if self._normalize_entity_name(pred_entity.name) in self._normalize_entity_name(gt_entity.name):
                            matched_gt.add(gt_entity)
                            break
                else:
                    wrong_entities.append({
                        "name": pred_entity.name,
                        "label": pred_entity.label
                    })

            # 找出未被提取的真实实体
            for gt_entity in gt_entities:
                if gt_entity not in matched_gt:
                    missed_entities.append({
                        "name": gt_entity.name,
                        "label": gt_entity.label
                    })

            total_ground_truth += len(gt_entities)
            total_extracted += len(pred_entities)

            # 保存每条数据的详细信息
            details.append({
                "index": idx,
                "text": text[:100] + "..." if len(text) > 100 else text,  # 文本预览
                "ground_truth_entities": [{"name": e.name, "label": e.label} for e in gt_entities],
                "extracted_entities": [{"name": e.name, "label": e.label} for e in pred_entities],
                "correct_entities": correct_entities,
                "wrong_entities": wrong_entities,
                "missed_entities": missed_entities,
                "stats": {
                    "ground_truth_count": len(gt_entities),
                    "extracted_count": len(pred_entities),
                    "correct_count": len(correct_entities),
                    "wrong_count": len(wrong_entities),
                    "missed_count": len(missed_entities)
                }
            })

        # 计算指标
        precision = true_positive / total_extracted if total_extracted > 0 else 0.0
        recall = true_positive / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        type_accuracy = correct_type / true_positive if true_positive > 0 else 0.0

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            type_accuracy=type_accuracy,
            total_ground_truth=total_ground_truth,
            total_extracted=total_extracted,
            true_positive=true_positive,
            correct_type=correct_type,
            details=details
        )

    def print_results(self, metrics: EvaluationMetrics):
        """打印评估结果"""
        print("\n" + "=" * 60)
        print("实体质量评估结果")
        print("=" * 60)
        print(f"{'指标':<15} {'数值':>15} {'说明':<25}")
        print("-" * 60)
        print(f"{'实体准确率':<15} {metrics.precision:>15.2%} 正确实体数 / 总实体数")
        print(f"{'实体召回率':<15} {metrics.recall:>15.2%} 正确实体数 / 真实实体总数")
        print(f"{'实体 F1 值':<15} {metrics.f1:>15.2%} 准确率和召回率的调和平均")
        print(f"{'实体类型准确率':<15} {metrics.type_accuracy:>15.2%} 类型正确数 / 总实体数")
        print("-" * 60)
        print(f"真实实体总数: {metrics.total_ground_truth}")
        print(f"提取实体总数: {metrics.total_extracted}")
        print(f"正确提取数量: {metrics.true_positive}")
        print(f"类型正确数量: {metrics.correct_type}")
        print("=" * 60 + "\n")

    def save_results(self, metrics: EvaluationMetrics, output_path: str = None):
        """保存评估结果到文件"""
        if output_path is None:
            output_path = Path(__file__).parent / "results" / "entity_quality_eval.json"
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
                "type_accuracy": metrics.type_accuracy,
                "total_ground_truth": metrics.total_ground_truth,
                "total_extracted": metrics.total_extracted,
                "true_positive": metrics.true_positive,
                "correct_type": metrics.correct_type
            },
            "token_usage": token_summary,
            "dataset_size": len(self.dataset),
            "details": metrics.details
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 同时保存一个可读的文本格式结果
        text_output_path = output_path.parent / "entity_quality_eval.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("实体质量评估详细结果\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"总指标:\n")
            f.write(f"  实体准确率: {metrics.precision:.4f}\n")
            f.write(f"  实体召回率: {metrics.recall:.4f}\n")
            f.write(f"  实体 F1 值: {metrics.f1:.4f}\n")
            f.write(f"  实体类型准确率: {metrics.type_accuracy:.4f}\n\n")

            f.write(f"统计数据:\n")
            f.write(f"  真实实体总数: {metrics.total_ground_truth}\n")
            f.write(f"  提取实体总数: {metrics.total_extracted}\n")
            f.write(f"  正确提取数量: {metrics.true_positive}\n")
            f.write(f"  类型正确数量: {metrics.correct_type}\n\n")

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
                       f"正确={detail['stats']['correct_count']}, "
                       f"错误={detail['stats']['wrong_count']}, "
                       f"遗漏={detail['stats']['missed_count']}\n")

                # 正确提取的实体
                if detail['correct_entities']:
                    f.write(f"  ✓ 正确提取:\n")
                    for e in detail['correct_entities']:
                        type_str = "✓" if e['type_correct'] else "✗"
                        f.write(f"      {type_str} {e['name']} ({e['label']})\n")

                # 错误提取的实体
                if detail['wrong_entities']:
                    f.write(f"  ✗ 错误提取:\n")
                    for e in detail['wrong_entities']:
                        f.write(f"      ✗ {e['name']} ({e['label']})\n")

                # 遗漏的实体
                if detail['missed_entities']:
                    f.write(f"  ⊗ 遗漏实体:\n")
                    for e in detail['missed_entities']:
                        f.write(f"      ⊗ {e['name']} ({e['label']})\n")

                f.write("\n")

        print(f"评估结果已保存到: {output_path}")
        print(f"详细结果已保存到: {text_output_path}")


def main():
    """主函数"""
    # 数据集路径
    dataset_path = ROOT / "datasets/DuIE2.0/duie_dev.json/duie_dev.json"

    # 创建评估器，使用项目配置的 llm 和 embeddings
    evaluator = EntityEvaluator(
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
