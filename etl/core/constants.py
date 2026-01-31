class ETLConstants:
    
    # Prompt template for extracting keywords
    KEYWORD_EXTRACTOR_PROMPT = """
        The text below is written in {language}.
        Analyze the following text and extract up to 5 high-quality keywords IN {language} for search retrieval.
        DO NOT translate them into any other language.
        Focus on:
        - Technical IDs or Error Codes (e.g., SKU-123, Error 404)
        - Specific nouns or entities (e.g., Ministry of Justice, Kazakhstan)
        - Core actions (e.g., Registration, Deletion)

        If the text is short or lacks specific detail, provide fewer keywords (1-3). 
        If the text contains no meaningful searchable information, return an empty list.

        Text: {node_text}
    """

    # Document summary extractor prompt
    DOCUMENT_SUMMARY_EXTRACTOR_PROPMT = """
        Analyze the following document text and return a JSON object with:
        1. "summary": A 2-sentence summary of the document purpose in the language of the document.
        2. "language": The ISO 639-1 language code (e.g., "kk", "ru").
        
        Text: {full_text}
    """
    
    # Section summary extractor prompt
    SECTION_SUMMARY_EXTRACTOR_PROPMT = """
        Analyze the following section from the text document and return a string with a 2-sentence summary of the section purpose in the {language} language.
        
        Text: {full_text}
    """
    
    # Chunk dense context extractor prompt
    SITUATIONAL_CONTEXT_PROMPT = """
        <section>
        {section}
        </section>

        Analyze the following chunk of text taken from the section described above:
        
        <chunk>
        {child_text}
        </chunk>

        Please provide a one-sentence context in {language} language that situates this chunk within the section. 
        The sentence must explain what specific part of the section this chunk covers (e.g., specific rules, definitions, or penalties).
        
        Output ONLY the one-sentence succint context and nothing else.
    """