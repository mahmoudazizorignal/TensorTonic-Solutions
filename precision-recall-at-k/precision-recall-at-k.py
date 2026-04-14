def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k = recommended[:min(k, len(recommended))]
    hits, j = 0, 0
    for item in top_k:
        if j == len(relevant):
            break
            
        if item < relevant[j]:
            continue
            
        while j < len(relevant) and item > relevant[j]:
            j += 1
            
        if j < len(relevant) and item == relevant[j]:
            hits += 1 
            j += 1
            
    return [hits / k, hits / len(relevant)]