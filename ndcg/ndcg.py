import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    # Write code here
    top_k = relevance_scores[:min(k, len(relevance_scores))]
    
    dcg = 0.0
    for i, item in enumerate(top_k):
        dcg += (2 ** item - 1) / math.log2(i + 2)

    relevance_scores.sort(reverse=True)
    top_k = relevance_scores[:min(k, len(relevance_scores))]
    
    idcg = 0.0
    for i, item in enumerate(top_k):
        idcg += (2 ** item - 1) / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0