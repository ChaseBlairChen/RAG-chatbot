# --- app.py ---
# ... (Keep all existing imports from Pasted_Text_1753498696893.txt) ...
# Add missing imports for the AI agent if not already present
# (Check if spacy, sentence_transformers are imported, add if needed)
# from spacy import load # Already imported via 'import spacy' and used in the agent
# from sentence_transformers import SentenceTransformer # Already imported via 'from sentence_transformers import SentenceTransformer' in the agent


# ... (Keep all existing global variables and helper functions from Pasted_Text_1753498696893.txt) ...
# Example: CHROMA_PATH, logger, conversations, cleanup_old_conversations, etc.

# --- INTEGRATE AI AGENT CODE HERE ---
# ... (Paste the ENTIRE content from Pasted_Text_1753498579681.txt starting from
# "from typing import List, Dict, Optional, Tuple"
# down to and including the end of the "create_ai_enhanced_context" function)
# ...

# --- INITIALIZE THE AI AGENT ---
# Add this line after the class definitions and helper functions, before the endpoints
ai_analyzer = EnhancedQuestionAnalyzer()

# --- MODIFY THE ENHANCED RETRIEVAL FUNCTION TO USE GLOBAL ANALYZER ---
# Find the 'enhanced_retrieval_with_ai_agent' function pasted from the first file
# Modify its definition like this:
def enhanced_retrieval_with_ai_agent(db, query_text: str, conversation_history: List[Dict] = None, k: int = 5):
    """Your existing enhanced_retrieval function with AI agent integration"""
    # analyzer = EnhancedQuestionAnalyzer() # REMOVE THIS LINE - use global
    enhanced_system = EnhancedRAGSystem(db, ai_analyzer) # USE GLOBAL ai_analyzer
    try:
        logger.info(f"AI Agent analyzing query: '{query_text}'")
        # Use the enhanced search system
        results, analysis = enhanced_system.enhanced_search(query_text, conversation_history, k)
        logger.info(f"AI Agent Analysis - Type: {analysis.query_type.value}, "
                   f"Intent: {analysis.intent.value}, "
                   f"Strategy: {analysis.search_strategy}, "
                   f"Entities: {analysis.key_entities}, "
                   f"Confidence: {analysis.confidence_score:.2f}")
        return results, analysis
    except Exception as e:
        logger.error(f"Enhanced retrieval with AI agent failed: {e}")
        # Fallback to your original method (ensure enhanced_retrieval is defined)
        # You might need to adapt the fallback to return analysis=None or a dummy analysis
        # For now, let's assume enhanced_retrieval returns results, None for analysis compatibility
        fallback_results = enhanced_retrieval(db, query_text, k) # Call the original fallback
        return fallback_results, None # Return results, None analysis for fallback

# ... (Keep the rest of your original app.py content from Pasted_Text_1753498696893.txt) ...
# Including ENHANCED_PROMPT_TEMPLATE, Query, QueryResponse, etc.
# ...

# --- MODIFY THE MAIN /ask ENDPOINT ---
@app.post("/ask", response_model=QueryResponse)
def ask_question(query: Query):
    try:
        query_text = query.question.strip() if query.question else ""
        logger.info(f"Received query: '{query_text}'")
        if not query_text:
            return QueryResponse(
                response=None,
                error="Question cannot be empty",
                context_found=False,
                session_id=query.session_id or ""
            )

        # Get or create session
        session_id = get_or_create_session(query.session_id)
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            return QueryResponse(
                response=None,
                error="OPENAI_API_KEY environment variable is required. Please set your OpenRouter API key.",
                context_found=False,
                session_id=session_id
            )

        if not os.path.exists(CHROMA_PATH):
            logger.error("Database not found")
            return QueryResponse(
                response=None,
                error="Vector database not found. Please run the document ingestion process first.",
                context_found=False,
                session_id=session_id
            )

        # Add user message to conversation history
        add_to_conversation(session_id, "user", query_text)

        questions = parse_multiple_questions(query_text)
        logger.info(f"Parsed {len(questions)} questions: {questions}")

        # Get conversation history for AI analysis
        conversation_history_list = conversations.get(session_id, {}).get('messages', [])

        try:
            # CRITICAL: Use the same embedding model as ingestion script
            embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            # Test database connectivity
            test_results = db.similarity_search("test", k=1)
            logger.info(f"Database loaded successfully with {len(test_results)} test results")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            return QueryResponse(
                response=None,
                error=f"Failed to load vector database: {str(e)}. Make sure you've run the ingestion script first.",
                context_found=False,
                session_id=session_id
            )

        combined_query = " ".join(questions)

        # --- MODIFIED: Use AI Agent for Analysis and Retrieval ---
        # Analyze the question using the global ai_analyzer
        # analysis = ai_analyzer.analyze_question(combined_query, conversation_history_list) # Optional: separate analysis step
        # logger.info(f"AI Analysis: {analysis}")

        # Perform enhanced retrieval using the AI agent
        results, analysis = enhanced_retrieval_with_ai_agent(db, combined_query, conversation_history_list, k=5)
        # --- END MODIFICATION ---

        logger.info(f"Retrieved {len(results)} results")

        if not results:
            logger.warning("No relevant documents found")
            response_text = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents contain information about this topic."
            # Add assistant response to conversation history
            add_to_conversation(session_id, "assistant", response_text)
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=False,
                session_id=session_id
            )

        # --- MODIFIED: Use AI Enhanced Context Creation ---
        # Create context using the AI analysis
        context_text, source_info = create_ai_enhanced_context(results, analysis, questions)
        # --- END MODIFICATION ---
        logger.info(f"Created context with {len(source_info)} sources")

        # Get conversation history (already retrieved above)
        # conversation_history_list = conversations.get(session_id, {}).get('messages', []) # Already have this
        conversation_history_context = get_conversation_context(session_id, max_messages=8)

        if len(questions) > 1:
            formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        else:
            formatted_questions = questions[0]

        formatted_prompt = ENHANCED_PROMPT_TEMPLATE.format(
            conversation_history=conversation_history_context if conversation_history_context else "No previous conversation.",
            context=context_text,
            questions=formatted_questions
        )
        logger.info(f"Prompt length: {len(formatted_prompt)} characters")

        try:
            response_text = call_openrouter_api(formatted_prompt, api_key, api_base)
            if not response_text:
                response_text = "I received an empty response. Please try again."
            else:
                 # --- MODIFIED SECTION START ---
                # Filter source_info based on relevance score
                # Only add sources section if there are relevant sources (e.g., score > 0.3)
                MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY = 0.3
                relevant_source_info = [source for source in source_info if source['relevance'] >= MIN_RELEVANCE_SCORE_FOR_SOURCE_DISPLAY]
                # Add sources section only if there are relevant sources
                if relevant_source_info:
                    response_text += "\n**SOURCES:**\n"
                    for source in relevant_source_info: # Use the filtered list
                        page_info = f", Page {source['page']}" if source['page'] else ""
                        response_text += f"• {source['file_name']}{page_info} (Relevance: {source['relevance']:.2f})\n"
                # --- MODIFIED SECTION END ---

            # Add assistant response to conversation history
            # Pass the potentially filtered source_info or the original if you want full history
            # Using relevant_source_info for history might be cleaner if you only want to track used sources,
            # but using source_info preserves the full retrieval context for this turn.
            # Let's use source_info for history for now to keep the history consistent with what was retrieved.
            add_to_conversation(session_id, "assistant", response_text, source_info)
            logger.info(f"Successfully generated response of length {len(response_text)}")

            # Return the filtered sources in the API response as well
            return QueryResponse(
                response=response_text,
                error=None,
                context_found=True,
                sources=relevant_source_info, # Return the filtered list
                session_id=session_id
            )
        except HTTPException as he:
            logger.error(f"API call failed: {he.detail}")
            error_response = f"API Error: {he.detail}"
            add_to_conversation(session_id, "assistant", error_response)
            return QueryResponse(
                response=None,
                error=error_response,
                context_found=True,
                sources=source_info,
                session_id=session_id
            )
    except Exception as e:
        logger.error(f"Unexpected error in ask_question: {e}")
        return QueryResponse(
            response=None,
            error=f"An unexpected error occurred: {str(e)}",
            context_found=False,
            session_id=query.session_id or ""
        )

# ... (Keep the rest of your original endpoints like /health, /conversation/{session_id}, etc. from Pasted_Text_1753498696893.txt) ...
# Optionally, you can now also add the new endpoints from Pasted_Text_1753498603230.txt
# like /analyze-question, /debug-ai-agent, /chrome-extension/analyze-and-search, etc.
# Just make sure they use the global `ai_analyzer` instance.

# --- Example: Add the /analyze-question endpoint from the second file ---
@app.post("/analyze-question")
def analyze_single_question(query: Query):
    """Endpoint to analyze a single question using the AI agent"""
    try:
        query_text = query.question.strip() if query.question else ""
        if not query_text:
            return {"error": "Question cannot be empty"}

        # Get conversation history if session provided
        conversation_history_list = []
        if query.session_id and query.session_id in conversations:
            conversation_history_list = conversations[query.session_id]['messages']

        # Analyze the question using the global ai_analyzer
        analysis = ai_analyzer.analyze_question(query_text, conversation_history_list)

        # Return the analysis results in a structured format
        return {
            "original_query": analysis.original_query,
            "query_type": analysis.query_type.value,
            "intent": analysis.intent.value,
            "key_entities": analysis.key_entities,
            "keywords": analysis.keywords,
            "reformulated_queries": analysis.reformulated_queries,
            "context_requirements": analysis.context_requirements,
            "confidence_score": analysis.confidence_score,
            "search_strategy": analysis.search_strategy,
            "recommendations": {
                "suggested_reformulations": analysis.reformulated_queries[:3],
                "key_terms_to_focus": analysis.key_entities + analysis.keywords[:3],
                "search_approach": analysis.search_strategy
            }
        }
    except Exception as e:
        logger.error(f"Question analysis failed: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

# ... (You can add other endpoints from Pasted_Text_1753498603230.txt similarly) ...

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

# --- End of app.py ---
