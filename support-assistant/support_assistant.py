import os
import json
from typing import TypedDict, List
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, START, END
from fastapi import FastAPI
from groq import Groq  # Added Groq import
from dotenv import load_dotenv
load_dotenv()

print("Initializing Support Assistant...")

# ---------------------------------------------------------
# 1. Environment Variable Configuration
# ---------------------------------------------------------
# Gated behind environment variable, defaulting to "1" (Mock mode)
def get_mock_llm():
    
    return os.getenv("MOCK_LLM", "1")

# Initialize Groq Client (automatically picks up GROQ_API_KEY from environment)
# We initialize it conditionally or just let it be None if the key isn't set in mock mode.
try:
    groq_client = Groq()
except Exception:
    groq_client = None

# ---------------------------------------------------------
# 2. Document Ingestion & Embeddings
# ---------------------------------------------------------
documents_path = "support-assistant/Documents.txt"
if not os.path.exists(documents_path):
    documents_path = "Documents.txt"

chunks = []
ids = []
metadata = []

with open(documents_path, "r", encoding="utf-8") as file:
    for idx, line in enumerate(file):
        text = line.strip()
        if text:
            chunks.append(text)
            doc_id = f"doc_{idx+1:02d}"
            ids.append(doc_id)
            metadata.append({"source": f"{doc_id}.txt"})

print(f"Total chunks created: {len(chunks)}")

sentence_trans = SentenceTransformer('all-MiniLM-L6-v2')
print("Generating embeddings...")
embeddings = sentence_trans.encode(chunks).tolist()

client = chromadb.Client()
collection = client.create_collection(
    name="policy_documents", 
    metadata={"hnsw:space": "cosine"}
)
collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadata
)
print(f"Stored {collection.count()} chunks in ChromaDB.\n")

# ---------------------------------------------------------
# 3. System Prompt Template
# ---------------------------------------------------------
SYSTEM_PROMPT = """
[ROLE]
You are a highly precise, accurate, and professional Zepto compliance and policy assistant. 

[CONTEXT]
You will be provided with official Zepto policy documents in the [RETRIEVED_CONTEXT] section below. These documents contain the only verified rules you are allowed to use. 

[TASK]
Your task is to answer the user's [QUESTION] by synthesizing the provided policy context. Extract only facts directly answering the inquiry.

[FORMAT]
Format your response as a JSON object adhering strictly to this schema:
{"answer": "<summary string>", "sources": ["<doc_id>"], "confidence": <float between 0 and 1>}

[LENGTH]
Keep the answer string direct, concise, and under 50 words.

[CONSTRAINTS]
- STRICTLY rely ONLY on the information present in the [RETRIEVED_CONTEXT].
- DO NOT answer using outside knowledge, general corporate norms, or assumptions not present in the context.
- If the answer cannot be confidently deduced from the provided context, answer with: "I do not have enough information in the provided documents to answer that."

[EXAMPLES]
--- Example 1 ---
[RETRIEVED_CONTEXT]:
doc_01: Zepto delivers essentials within 10 to 30 minutes. Standard delivery is free on orders over INR 149; orders below incur INR 25.
[QUESTION]: 
How much is delivery on a 100 rupee order?
[YOUR RESPONSE]:
{"answer": "Standard delivery incurs a flat INR 25 fee for orders under INR 149.", "sources": ["doc_01"], "confidence": 1.0}

--- Example 2 ---
[RETRIEVED_CONTEXT]:
doc_02: Grocery and perishable items may be reported within 24 hours.
[QUESTION]: 
Can I buy electronics on Zepto?
[YOUR RESPONSE]:
{"answer": "I do not have enough information in the provided documents to answer that.", "sources": [], "confidence": 1.0}

==================================================
[RETRIEVED_CONTEXT]:
{context}

[QUESTION]:
{question}
"""

# ---------------------------------------------------------
# 4. Schemas & LangGraph State
# ---------------------------------------------------------
class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class AgentState(TypedDict):
    query: str
    intent: str
    final_response: dict

def call_real_llm_with_retry(prompt: str, max_retries: int = 2) -> RAGResponse:
    """Calls Groq API and retries if JSON parsing fails."""
    if not groq_client:
        return RAGResponse(answer="Error: GROQ_API_KEY not found.", sources=[], confidence=0.0)

    for attempt in range(max_retries + 1):
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="openai/gpt-oss-120b", # Requested model (change to llama-3.1-8b-instant if it fails)
                response_format={"type": "json_object"}, # Forces the model to output valid JSON
                temperature=0.0 # Keep temperature at 0 for factual RAG
            )
            
            raw_output = chat_completion.choices[0].message.content
            parsed_json = json.loads(raw_output)
            
            # Validate against Pydantic schema
            return RAGResponse(**parsed_json)
            
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                return RAGResponse(
                    answer="Error: Failed to validate response after 2 retries.", 
                    sources=[], 
                    confidence=0.0
                )
            # Corrective instruction for the next attempt
            prompt += f"\nNote: Previous response was invalid. Please adhere strictly to the JSON schema. Error: {str(e)}"
        except Exception as e:
            # Catch API errors (like invalid model name)
            return RAGResponse(answer=f"API Error: {str(e)}", sources=[], confidence=0.0)
            
    return RAGResponse(answer="Error: Could not generate valid output.", sources=[], confidence=0.0)

# ---------------------------------------------------------
# 5. Graph Nodes & Conditional Routing
# ---------------------------------------------------------
def classify_intent(state: AgentState):
    query = state["query"].lower()
    mock_mode = get_mock_llm()
    
    if mock_mode == "1":
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        if any(keyword in query for keyword in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # For simplicity in this assignment, we use the same keyword heuristic 
        # for real LLM mode routing, or you could add a second Groq call here.
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        intent = "policy_question" if any(keyword in query for keyword in keywords) else "general_question"
        
    return {"intent": intent}

def retrieve_and_answer(state: AgentState):
    query = state["query"]
    mock_mode = get_mock_llm()
    
    results = collection.query(
        query_embeddings=sentence_trans.encode([query]).tolist(), 
        n_results=3
    )
    top_snippet = results["documents"][0][0][:200]
    retrieved_ids = results["ids"][0]
    
    if mock_mode == "1":
        answer = f"Based on the retrieved context: {top_snippet}"
        response = RAGResponse(answer=answer, sources=retrieved_ids, confidence=1.0)
    else:
        context_str = "\n".join([f"{doc_id}: {doc_text}" for doc_id, doc_text in zip(results["ids"][0], results["documents"][0])])
        formatted_prompt = SYSTEM_PROMPT.format(context=context_str, question=query)
        response = call_real_llm_with_retry(formatted_prompt, max_retries=2)
        
    return {"final_response": response.model_dump()}

def direct_answer(state: AgentState):
    mock_mode = get_mock_llm()
    
    if mock_mode == "1":
        answer = "I can only answer questions about Zepto policies right now."
        response = RAGResponse(answer=answer, sources=[], confidence=1.0)
    else:
        # Basic Groq call for general questions without retrieval
        if groq_client:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": f"Briefly answer this general question: {state['query']}"}],
                    model="openai/gpt-oss-120b",
                    temperature=0.5
                )
                answer = chat_completion.choices[0].message.content
            except Exception as e:
                answer = f"API Error: {str(e)}"
        else:
            answer = "API Key not configured."
            
        response = RAGResponse(answer=answer, sources=[], confidence=0.8)
        
    return {"final_response": response.model_dump()}

def route_query(state: AgentState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# ---------------------------------------------------------
# 6. Workflow Assembly
# ---------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.add_edge(START, "classify_intent")
workflow.add_conditional_edges(
    "classify_intent",
    route_query,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

graph_app = workflow.compile()

# ---------------------------------------------------------
# 7. FastAPI Wrapper
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="Zepto Policy Assistant")

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QueryRequest):
    initial_state = {"query": request.query}
    result = graph_app.invoke(initial_state)
    return result["final_response"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
