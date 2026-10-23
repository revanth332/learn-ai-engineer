from dataclasses import dataclass
from src.evals.guardrails import EvalSample,TriadScore,RAGTriadEvaluator
import math

@dataclass
class SampleEvalResult:
    sample: EvalSample
    triad_score: TriadScore
    is_safe: bool
    pii_detected: dict[str,list[str]]
    passed: bool

@dataclass
class BenchMarkReport:
    mean_context_relevance: float
    mean_groundedness: float
    mean_answer_relevance: float
    pass_rate: float
    results: list[SampleEvalResult]

def precision_at_k(retrieved_ids,relavant_ids,k) -> float:
    if len(relavant_ids) == 0:
        return 0.0
    count = 0
    for id in retrieved_ids[:k]:
        if id in relavant_ids:
            count += 1
    return count/k

def recall_at_k(retrieved_ids,relavant_ids,k) -> float:
    if len(relavant_ids) == 0:
        return 0.0
    count = 0
    for id in retrieved_ids[:k]:
        if id in relavant_ids:
            count += 1
    return count/len(relavant_ids)

# Mean Reciprocal Rank
def mrr(retrieved_ids,relavant_ids):
    first_relavant_doc_rank = None
    for i in range(len(retrieved_ids)):
        if retrieved_ids[i] in relavant_ids:
            first_relavant_doc_rank = i+1
            break
    if first_relavant_doc_rank:
        return 1/first_relavant_doc_rank
    return 0.0

# Normalized Discounted Cumulative Gain
def ndcg_at_k(retrieved_ids,relavant_ids,k):
    if len(relavant_ids) == 0:
        return 0.0
    n = min(len(retrieved_ids),k)
    perfect_score = sum([1/(math.log2(i+2)) for i in range(min(len(relavant_ids),k))])
    current_score = 0
    for i in range(n):
        is_relavant = 1 if retrieved_ids[i] in relavant_ids else 0
        current_score += (is_relavant / math.log2(i+2))
    return current_score/perfect_score


class EvaluationHarness:
    def __init__(self,rag_triad_evaluator:RAGTriadEvaluator,threshold:float=0.7):
        self.rag_triad_evaluator = rag_triad_evaluator

    def get_report(self,samples:list[EvalSample]):
        total_context_relevance,total_groundedness,total_answer_relevance = 0,0,0
        results = []
        for sample in samples:
            triad_score = self.rag_triad_evaluator.evaluate_sample(sample)
            
        