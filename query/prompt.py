
## Prompt construction for LLM-based sufficiency judgment

llm_judge_sufficiency_system_prompt = (
    "You are an expert research assistant."
    "Your task is to determine whether the following retrieved information is sufficient to answer the question."
    )
llm_judge_sufficiency_evidence = (
    "Evidence\n"
    "-----\n"
)
llm_judge_sufficiency_task = (
    "Based on the question and the evidence provided,"
    "is the information sufficient to answer the question? \n\n"
    "Respond with a valid JSON object with two fields: 'sufficient' (boolean) and 'reason' (a brief explanation)."
    "response format:\n "
    "{\n  \"sufficient\": true | false,\n  \"reason\": \"<brief explanation>\"\n}"
)

def llm_judge_sufficiency_prompt(query, representative_texts):

    prompt = f"Question\n-----\n{query}\n\n"
    while representative_texts:
        text = representative_texts.pop(0)
        prompt += llm_judge_sufficiency_evidence + text + "\n\n"
    prompt += llm_judge_sufficiency_task
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

## PROMPT CONSTRUCTION FOR FINAL ANSWER SYNTHESIS

final_answer_system_prompt = (
    "You are a helpful and precise research assistant for answering complex scientific questions. "
    "Use ONLY the provided retrieved information to construct your answer. "
    "Cite each piece of evidence you use with its paper_id."
    "If you use multiple pieces of evidence, cite each one where relevant. "
    "If the retrieved information is insufficient to answer the question, say so clearly and specify what is missing."
)

def final_answer_user_prompt(query, chunk_data_list):
    prompt = f"QUESTION:\n{query}\n\n"
    prompt += "RETRIEVED INFORMATION:\n"
    for chunk_data in chunk_data_list:
        prompt += f"---\n\nPaper ID: {chunk_data.get('paper_id','N/A')}\nSection: {chunk_data.get('section','N/A')}\nText: {chunk_data['text']}\n\n"
    prompt += "response format: {\n \"answer\": \"<your answer here>\" ,\n \"citations\": [\"<paper_id1>\", \"<paper_id2>\", ...] \n}"
    prompt += "INSTRUCTIONS:\n- Answer the question using ONLY the RETRIEVED INFORMATION.\n- Cite each piece of evidence you use with its paper_id like [paper123].\n- If the answer is uncertain with the given information, state what is missing."
    return prompt

final_answer_response_schema = {
  "name": "FinalAnswer",
  "strict": True,
  "additionalProperties": False,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {"type": "string"}
            }
        },      
        "required": ["answer", "citations"]
    }
}