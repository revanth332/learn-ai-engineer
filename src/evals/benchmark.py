from dataclasses import dataclass
from src.evals.guardrails import EvalSample,TriadScore,RAGTriadEvaluator,detect,redact,is_safe
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
    if len(relavant_ids) == 0 or k == 0:
        return 0.0
    count = 0
    for id in retrieved_ids[:k]:
        if id in relavant_ids:
            count += 1
    return count/k

def recall_at_k(retrieved_ids,relavant_ids,k) -> float:
    if len(relavant_ids) == 0 or k == 0:
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
    if len(relavant_ids) == 0 or k == 0:
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
        self.threshold = threshold

    def get_report(self,samples:list[EvalSample]) -> BenchMarkReport:
        total_context_relevance,total_groundedness,total_answer_relevance = 0,0,0
        results = []
        total_samples = len(samples)
        if total_samples == 0:
            return BenchMarkReport(0.0,0.0,0.0,0.0,[])
        passed_samples_count = 0
        for sample in samples:
            triad_score = self.rag_triad_evaluator.evaluate_sample(sample)
            total_answer_relevance += triad_score.answer_relevance
            total_context_relevance += triad_score.context_relevance
            total_groundedness += triad_score.groundedness
            is_sample_safe = is_safe(sample.query)[0] and is_safe(sample.answer)[0]
            pii_detected = detect(sample.query + " " + sample.answer)
            is_sample_passed = is_sample_safe and len(pii_detected) == 0 and triad_score.groundedness >= self.threshold and triad_score.context_relevance >= self.threshold and triad_score.answer_relevance >= self.threshold
            if is_sample_passed:
                passed_samples_count += 1
            sample_result = SampleEvalResult(sample,triad_score,is_sample_safe,pii_detected,is_sample_passed)
            results.append(sample_result)
        mean_context_relevance = total_context_relevance/total_samples
        mean_groundedness = total_groundedness/total_samples
        mean_answer_relevance = total_answer_relevance/total_samples

        pass_rate = passed_samples_count/total_samples
        report = BenchMarkReport(mean_context_relevance,mean_groundedness,mean_answer_relevance,pass_rate,results)
        return report