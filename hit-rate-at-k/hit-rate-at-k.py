def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    # Write code here
    hit_rate = 0
    
    for i in range(len(recommendations)):
        top_k = recommendations[i][:min(k, len(recommendations[i]))]
        user_gt = ground_truth[i]

        l = 0
        for j in range(len(top_k)):
            if l >= len(user_gt):
                break

            if top_k[j] < user_gt[l]:
                continue
            
            while l < len(user_gt) and top_k[j] > user_gt[l]:
                l += 1

            if l < len(user_gt) and top_k[j] == user_gt[l]:
                hit_rate += 1
                break
                
    hit_rate /= len(recommendations)
    return hit_rate