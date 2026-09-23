from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing import TypedDict, List
from fastapi import FastAPI

print('Imports Done!!')

documents_path = "support-assistant/Documents.txt"
with open(documents_path, "r") as file:
    chunks = []
    for doc in file.readlines():
        if doc.strip() not in ['', ' ', '\n']:
            chunks.append(doc)

print("-" * 50)
metadata = []
ids = []
for id, chunk in enumerate(chunks):
    metadata.append({
        "source": chunk.split(':')[0]
    })
    ids.append(str(id))
print("-" * 50)
print(f"Total No of chunks created - {len(chunks)}")
print("-" * 50)
print()
print("Metadata: ")
print(metadata)
print()
print("Ids: ")
print(ids)

print("-" * 70)

sentence_trans = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating Embeddings.......")
for id, chunk in enumerate(chunks):
    embeddings = sentence_trans.encode(chunks).tolist()

print(f"embeddings created - {len(embeddings)}")
print("-" * 70)
print("Creating ChromaDB collection....")
client = chromadb.Client()
collection = client.create_collection(name="policy_documents")

print("Storing the embeddings....")
collection.add(
    ids = ids,
    embeddings = embeddings,
    documents = chunks,
    metadatas = metadata
)
print(f"Stored {collection.count()} chunks in memory.\n")
print("-" * 70)
SYSTEM_PROMP = """
[ROLE]
You are a highly precise, accurate, and professional compliance and policy assistant. 

[CONTEXT]
You will be provided with official company policy documents in the context. The context contain the only verified rules and guidelines you are allowed to use. 

[TASK]
Your task is to answer the user's question by carefully synthesizing the provided policy context. Read the rules thoroughly and extract only the facts relevant to the user's inquiry.

[FORMAT]
Format a clean response. Lead with a single bold sentence summarizing the policy ruling. If the answer involves multiple components, limits, or steps, use a bulleted or numbered list for readability.

[LENGTH]
Keep your response concise, direct, and under 50 words. Do not add filler introductions or conversational fluff.

[CONSTRAINTS]
- STRICTLY rely ONLY on the information present in the [RETRIEVED_CONTEXT].
- DO NOT answer using outside knowledge, general corporate norms, assumptions, or information not explicitly stated in the provided context.
- If the answer cannot be confidently deduced from the provided context, you must reply with exactly: "I do not have enough information in the provided documents to answer that."

[EXAMPLES]
--- Example 1 ---
[RETRIEVED_CONTEXT]:
"Employees may expense meals during authorized business travel up to $75 per day. Alcohol is strictly prohibited from expense claims. All receipts must be submitted via the Concur portal within 30 days of the transaction date. Late submissions will not be reimbursed."
[QUESTION]: 
"Can I expense a glass of wine I had during a client dinner, and how long do I have to submit the receipt?"
[YOUR RESPONSE]:
Alcohol is not reimbursable, and you have 30 days to submit your meal receipts.

* Alcohol Policy: Strictly prohibited from expense claims.
* Submission Deadline: Must be submitted within 30 days of the transaction via Concur. 

--- Example 2 ---
[RETRIEVED_CONTEXT]:
"Full-time employees are eligible for the hybrid work schedule after completing their 90-day probationary period. The hybrid schedule requires a minimum of 3 days in the office per week."
[QUESTION]: 
"Are part-time contractors allowed to work remotely?"
[YOUR RESPONSE]:
I do not have enough information in the provided documents to answer that.
"""
print(f"System prompt generated - {SYSTEM_PROMP[:100]}....")
print("-" * 70)
# use this to set the mock llm state: 1 -> use mock, 0 -> use actual llm
MOCK_LLM = "1"

print("Define the output schema using pydantic...")
# 1. Define the Expected Output Schema
class RAGResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# 2. Define the Graph State
class AgentState(TypedDict):
    query: str
    intent: str
    final_response: dict

# 3. Define the nodes
print("-" * 70)
print("Defining nodes...")
print("    1. classify_intent")
print("    2. retrieve_and_answer")
print("    3. direct_answer")
def classify_intent(state: AgentState):
    query = state["query"].lower()
    
    if MOCK_LLM == "1":
        # Graded baseline keyword routing
        keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
        if any(keyword in query for keyword in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # Real LLM logic goes here
        intent = "general_question" 
        
    return {"intent": intent}

def retrieve_and_answer(state: AgentState):
    query = state["query"]
    
    results = collection.query(query_embeddings = sentence_trans.encode([query]).tolist(), n_results=3)
    top_snippet = results["documents"][0][0][:200]
    retrieved_ids = results["ids"][0]
    
    if MOCK_LLM == "1":
        # Graded baseline mock output[cite: 1]
        answer = f"Based on the retrieved context: {top_snippet}"
        response = RAGResponse(answer=answer, sources=retrieved_ids, confidence=1.0)
    else:
        # Real LLM generation goes here
        response = RAGResponse(answer="LLM answer", sources=retrieved_ids, confidence=0.95)
        
    return {"final_response": response.model_dump()}

def direct_answer(state: AgentState):
    if MOCK_LLM == "1":
        answer = "I can only answer questions about Zepto policies right now."
        response = RAGResponse(answer=answer, sources=[], confidence=1.0)
    else:
        # Real LLM generation goes here
        response = RAGResponse(answer="LLM general answer", sources=[], confidence=0.8)
        
    return {"final_response": response.model_dump()}

# 4. Define the Routing Logic
print("    4. conditional edge -> route_query...")
def route_query(state: AgentState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

print("-" * 70)
print("Create a Langraph Workflow...")
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

print("-" * 70)
print("Graph Pipeline")
print("-" * 70)
print(graph_app.get_graph().draw_ascii())
print("-" * 70)
print("-" * 70)
# Step 4 - Integrate FastAPI
print("Integrating Fast API")
class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="Zepto Policy Assistant")

@app.post("/ask", response_model=RAGResponse)
async def ask_question(request: QueryRequest):
    initial_state = {"query": request.query}
    
    # Run the query through your LangGraph workflow
    result = graph_app.invoke(initial_state)
    
    # Extract and return the final Pydantic-validated dictionary
    return result["final_response"]