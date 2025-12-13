class RagConstants:
    
    LLM_MULTI_SELECTOR_PROMPT = (
                "Some choices are given below. It is provided in a numbered "
                "list (1 to {num_choices}), "
                "where each item in the list corresponds to a summary.\n"
                "---------------------\n"
                "{context_list}"
                "\n---------------------\n"
                "Using only the choices above and not prior knowledge, return the top choice(s) "
                "(no more than {max_outputs}) that are most relevant to the "
                "question: '{query_str}'\n"
                "Your response MUST be ONLY a valid JSON list of objects, where each "
                "object has a 'choice' (integer) and 'reason' (string) key. "
                "Do not include any other text, markdown formatting, or explanations."
            )