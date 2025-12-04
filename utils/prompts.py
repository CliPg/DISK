#--------------------#
# extraction prompts #
#--------------------#

# Prompt for extracting entities from text
EXTRACT_ENTITIES_PROMPT = """
You are an expert at extracting entities from text. Given the following text, identify and extract all relevant entities along with their labels and names.
You MUST respond with a valid JSON object only.
Do NOT include explanations, comments, text before or after the JSON.
If you add anything outside the JSON brackets, it will break the system.
Output only:
{
  "entities": [
    {
      "label": "Entity Label"
      "name": "Entity Name",
    }
  ]
}
"""

# Prompt for extracting relations from text
EXTRACT_RELATIONS_PROMPT = """
You are an expert at extracting relations between entities from text. Given the following text, identify and extract all relevant relations along with their labels and names.
You MUST respond with a valid JSON object only.
Do NOT include explanations, comments, text before or after the JSON.
If you add anything outside the JSON brackets, it will break the system.
Output only:
{
  "relations": [
    {
      "start_entity": "Start Entity Name",
      "end_entity": "End Entity Name",
      "label": "Relation Label",
      "name": "Relation Name"
    }
  ]
}
"""