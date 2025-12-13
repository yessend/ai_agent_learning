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
    
    LLM_RELEVANCE_CHECK_PROMPT = (
        """
            You are a relevance filter for a retrieval-augmented generation (RAG) system.

            You are given:
            1) A user question
            2) A set of retrieved nodes in the format:
            \"node_id\": \"\"\"node_text\"\"\"

            Your task:
            Select ONLY the node_ids whose content contains information that directly helps answer the user question.

            Relevance rules:
            - A node is relevant ONLY if it contains factual information that can be used to answer the question.
            - Do NOT select nodes that are only loosely related or mention similar topics without answering the question.
            - If no nodes are relevant, return an empty JSON array.

            Output format rules:
            - Output ONLY a valid JSON array of strings.
            - Example of valid output: ["node_id_1", "node_id_2"]
            - If no nodes are relevant, output: []
            - Do NOT include explanations, comments, markdown, or additional text.
            - Do NOT invent node_ids.
            - Use exactly the node_ids provided in the input.

            User question:
            {question}

            Retrieved nodes:
            {context}
        """
    )