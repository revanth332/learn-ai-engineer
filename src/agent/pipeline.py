from dataclasses import dataclass, field
from typing import Any,Callable
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.query_transform import expand_query_synonyms,decompose_query,fuse_transformed_queries
from src.retrieval.reranking import RankedDocument,CandidateDocument,rerank
from src.agent.memory import CompositeMemory,Message
from src.agent.tools import ToolRegistry
from src.evals.guardrails import detect,redact,is_safe,RAGTriadEvaluator

@dataclass
class PipelineResponse:
    query: str                        # Original or PII-redacted user query
    answer: str                       # Final safe, PII-redacted assistant answer
    retrieved_context: list[str]      # Final reranked document texts injected into prompt
    tool_calls: list[dict[str, Any]]  # Record of any tools executed during this turn
    blocked: bool = False             # True if blocked by pre-flight or post-flight guardrails
    block_reason: list[str] = field(default_factory=list)

class StatefulRAGAgent:
    def __init__(self,hybrid_retriever:HybridRetriever,memory:CompositeMemory,tool_registry:ToolRegistry,llm_fn:Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],embedding_fn: Callable[[str], list[float]],synonym_map: dict[str, list[str]] | None = None,retrieval_top_k: int = 5,rerank_top_n: int = 3,max_tool_iterations: int = 3):
        self.hybrid_retriever = hybrid_retriever
        self.memory = memory
        self.tool_registry = tool_registry
        self.llm_fn = llm_fn
        self.embedding_fn = embedding_fn
        self.synonym_map = synonym_map or {}
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        self.max_tool_iterations = max_tool_iterations

    def ingest(self,doc_id:str,text:str,metadata: dict[str, Any] | None = None):
        vector = self.embedding_fn(text)
        self.hybrid_retriever.add(doc_id,text,vector,{'text':text,**(metadata or {})})

    def retrieve_and_rerank(self,query:str) -> list[RankedDocument]:
        # First Query Transformation before converting it into vector
        expanded_query = expand_query_synonyms(query,self.synonym_map)
        sub_queries = decompose_query(query)
        queries = fuse_transformed_queries(query,expanded_query,sub_queries)

        # iterate through each possible query and retrieving the matching candidates
        candidate_docs = {}
        for q in queries:
            query_vector = self.embedding_fn(q)
            hybrid_result = self.hybrid_retriever.search(q,query_vector,self.retrieval_top_k)
            for result in hybrid_result:
                candidate  = CandidateDocument(result.doc_id,result.metadata['text'],result.metadata)
                candidate_docs[candidate.id] = candidate
        
        # Finally reranking all the candidates
        reranked_result = rerank(query,candidate_docs.values(),self.rerank_top_n)
        return reranked_result

    def run(self,user_query:str) -> PipelineResponse:
        safe,reasons = is_safe(user_query)
        if not safe:
            return PipelineResponse(user_query,"",[],[],True,reasons)

        redacted_query = redact(user_query)
        ranked_docs = self.retrieve_and_rerank(redacted_query)
        context_texts = [doc.text for doc in ranked_docs]
        context = "\n\n".join(context_texts)

        messages = self.memory.get_chat_history(redacted_query)
        messages.append({
            "role":"user",
            "content":f"Context:\n{context}\n\nQuestion:\n{redacted_query}"
        })

        tool_schemas = self.tool_registry.get_schemas()
        tools_executed = []
        raw_answer = ""
        for _ in range(self.max_tool_iterations):
            llm_response = self.llm_fn(messages,tool_schemas)
            requested_tools = llm_response.get("tool_calls") or []
            if not requested_tools:
                raw_answer = llm_response.get("content") or ""
                break
            for tc in requested_tools:
                tools_executed.append(tc)
                tool_name = tc["name"]
                tool_args = tc["arguments"]
                execution_result = self.tool_registry.execute(tool_name,tool_args)
                messages.append(execution_result.to_chat_message(tc.get("id")))

        if raw_answer == "":
            llm_response = self.llm_fn(messages,[])
            raw_answer = llm_response.get("content") or ""

        ans_safe,ans_unsafe_reasons = is_safe(raw_answer)
        if not ans_safe:
            return PipelineResponse(redacted_query,"Response blocked by output safety guardrails.",context_texts,tools_executed,True,ans_unsafe_reasons)
        redacted_ans = redact(raw_answer)
        self.memory.add_message("user",redacted_query)
        self.memory.add_message("assistant",redacted_ans)

        return PipelineResponse(redacted_query,redacted_ans,context_texts,tools_executed,False,[])
