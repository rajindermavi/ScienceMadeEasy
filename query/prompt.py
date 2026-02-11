




## Prompt construction for LLM-based sufficiency judgment

llm_judge_sufficiency_system_prompt = (
        "You are an evaluator deciding whether a limited set of retrieved excerpts "
        "is sufficient to answer a question accurately.\n\n"
        "Important constraints:\n"
        "- The excerpts are NOT the full document or full knowledge base.\n"
        "- Relevant information may exist outside what is shown.\n"
        "- Answer 'sufficient = true' if the provided excerpts of the document "
        "suggest the full document contains everything needed to answer the question correctly.\n\n"
        "Be balanced. If some useful information is present, answer sufficient = true "
        "if no useful information is present, answer 'sufficient = false'."
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
    "schema": {
        "type": "object",
        "properties": {
            "sufficient": {"type": "boolean"},
            "reason": {"type": "string"}
        },      
        "required": ["sufficient", "reason"],
        "additionalProperties": False
    }
}

## PROMPT CONSTRUCTION FOR FINAL ANSWER SYNTHESIS

final_answer_system_prompt = (
    "You are a helpful and precise research assistant for answering complex scientific questions. "
    "Use ONLY the provided retrieved information to construct your answer. "
    "Cite each piece of evidence you use with numbered bracket citations like [1], [2]. "
    "Do NOT include chunk IDs or section metadata in the answer text. "
    "If you use multiple pieces of evidence, cite each one where relevant. "
    "If the retrieved information is insufficient to answer the question, say so clearly and specify what is missing."
)

def final_answer_user_prompt(query, chunk_data_list):
    prompt = f"QUESTION:\n{query}\n\n"
    prompt += "RETRIEVED INFORMATION:\n"
    for chunk_data in chunk_data_list:
        prompt += f"---\n\nPaper ID: {chunk_data.get('chunk_id','N/A')}\nSection: {chunk_data.get('section','N/A')}\nText: {chunk_data['text']}\n\n"
    prompt += "response format: {\n \"answer\": \"<your answer with [1], [2] citations only>\",\n \"citations\": [{\"number\":\"1\",\"chunk_id\":\"<chunk_id#1>\"}, {\"number\":\"2\",\"chunk_id\":\"<chunk_id#2>\"}] \n}\n"
    prompt += "INSTRUCTIONS:"
    prompt += "\n- Answer the question using ONLY the RETRIEVED INFORMATION."
    prompt += "\n- In the answer, use ONLY bracketed numbers like [1], [2] as citations."
    prompt += "\n- Do NOT include chunk IDs, section labels, or line references in the answer text."
    prompt += "\n- In the citations array, map each number to the chunk_id it refers to."
    prompt += "\n- If the answer is uncertain with the given information, state what is missing."
    return prompt

final_answer_response_schema = {
  "name": "FinalAnswer",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "answer": { "type": "string" },
      "citations": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "number": { "type": "string" },
            "chunk_id": { "type": "string" }
          },
          "required": ["number", "chunk_id"],
          "additionalProperties": False
        }
      }
    },
    "required": ["answer", "citations"],
    "additionalProperties": False
  }
}

## PROMPT CONSTRUCTION FOR TOPIC SUMMARY

topic_summary_system_prompt = (
    "You are a concise science explainer. Summarize the given topic for a general audience. "
    "Keep it accurate, neutral, and easy to understand. "
    "If the topic is too broad or ambiguous, state what needs clarification."
)

topic_summary_user_prompt = (
    "TOPIC:\n"
    "{topic}\n\n"
    "INSTRUCTIONS:\n"
    "- Provide a clear 5-8 sentence summary.\n"
    "- Define key terms briefly on first mention.\n"
    "- Avoid citations, links, or markdown.\n"
    "- If the topic is ambiguous, say what is missing.\n"
)

def llm_topic_summary(topic):
    return topic_summary_user_prompt.format(topic=topic)
