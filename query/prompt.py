
## Prompt construction for LLM-based sufficiency judgment

judge_sufficiency_header = (
    "You are an expert research assistant."
    "Your task is to determine whether the following retrieved information is sufficient to answer the question."
    )
judge_sufficiency_evidence = (
    "Evidence\n"
    "-----\n"
)
judge_sufficiency_task = (
    "Based on the question and the evidence provided,"
    "is the information sufficient to answer the question? \n\n"
    "Respond with a valid JSON object with two fields: 'sufficient' (boolean) and 'reason' (a brief explanation)."
    "response format:\n "
    "{\n  \"sufficient\": true | false,\n  \"reason\": \"<brief explanation>\"\n}"
)

def llm_judge_sufficiency_prompt(query, representative_texts):

    prompt = judge_sufficiency_header + "\n\n"
    prompt += f"Question\n-----\n{query}\n\n"
    while representative_texts:
        text = representative_texts.pop(0)
        prompt += judge_sufficiency_evidence + text + "\n\n"
    prompt += judge_sufficiency_task
    return prompt

llm_judge_sufficiency_response_schema = {
  "name": "SufficiencyVerdict",
  "strict": True,
  "additionalProperties": False,
    "schema": {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "reason": {"type": "string"}
        },      
        "required": ["sufficient", "reason"]
    }
}

