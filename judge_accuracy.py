from math import isclose

LEVEL_POINTS = {
    0.5: 0.12,
    1: 0.15,
    2: 0.23,
    3: 0.34,
    4: 0.51,
    5: 0.76,
    6: 1.14,
    7: 1.71,
    8: 2.56,
}

def judge_accuracy(reference_scores, judge_scores, deletion_weight=1.0, insertion_weight=1.0):
    """
    Compare a judge's difficulty marks against a known-good reference sequence.

    Parameters
    ----------
    reference_scores : list[float]
        Ordered reference difficulty levels, e.g. [1, 2, 3, 0.5, 4]
    judge_scores : list[float]
        Ordered judge difficulty levels, same value set as reference
    deletion_weight : float
        Penalty multiplier for a missed reference skill
    insertion_weight : float
        Penalty multiplier for an extra judge skill

    Returns
    -------
    dict with:
        accuracy: float, 0..100
        normalized_error: float
        alignment_cost: float
        reference_total: float
        judge_total: float
        total_error_pct: float
        exact_match_rate: float
        aligned_pairs: list of tuples (ref, judge) where either side may be None
    """
    def pts(x):
        if x not in LEVEL_POINTS:
            raise ValueError(f"Unsupported level: {x}")
        return LEVEL_POINTS[x]

    m = len(reference_scores)
    n = len(judge_scores)

    if m == 0:
        raise ValueError("reference_scores must not be empty")

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_weight * pts(reference_scores[i - 1])
        back[i][0] = "D"

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insertion_weight * pts(judge_scores[j - 1])
        back[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub_cost = dp[i - 1][j - 1] + abs(pts(reference_scores[i - 1]) - pts(judge_scores[j - 1]))
            del_cost = dp[i - 1][j] + deletion_weight * pts(reference_scores[i - 1])
            ins_cost = dp[i][j - 1] + insertion_weight * pts(judge_scores[j - 1])

            best = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best

            if isclose(best, sub_cost):
                back[i][j] = "M"
            elif isclose(best, del_cost):
                back[i][j] = "D"
            else:
                back[i][j] = "I"

    i, j = m, n
    aligned_pairs = []
    exact_matches = 0
    matched_count = 0

    while i > 0 or j > 0:
        move = back[i][j]
        if move == "M":
            r = reference_scores[i - 1]
            s = judge_scores[j - 1]
            aligned_pairs.append((r, s))
            matched_count += 1
            if r == s:
                exact_matches += 1
            i -= 1
            j -= 1
        elif move == "D":
            aligned_pairs.append((reference_scores[i - 1], None))
            i -= 1
        elif move == "I":
            aligned_pairs.append((None, judge_scores[j - 1]))
            j -= 1
        else:
            break

    aligned_pairs.reverse()

    reference_total = sum(pts(x) for x in reference_scores)
    judge_total = sum(pts(x) for x in judge_scores)
    alignment_cost = dp[m][n]
    normalized_error = alignment_cost / reference_total if reference_total > 0 else 1.0
    accuracy = max(0.0, 100.0 * (1.0 - normalized_error))
    total_error_pct = ((judge_total - reference_total) / reference_total * 100.0) if reference_total > 0 else 0.0
    exact_match_rate = (exact_matches / matched_count * 100.0) if matched_count > 0 else 0.0

    return {
        "accuracy": accuracy,
        "normalized_error": normalized_error,
        "alignment_cost": alignment_cost,
        "reference_total": reference_total,
        "judge_total": judge_total,
        "total_error_pct": total_error_pct,
        "exact_match_rate": exact_match_rate,
        "aligned_pairs": aligned_pairs,
    }
if __name__ == "__main__":
    print(judge_accuracy([1, 2, 3, 5, 6, 2, 5], [1, 2, 3, 4, 7, 3, 4]))