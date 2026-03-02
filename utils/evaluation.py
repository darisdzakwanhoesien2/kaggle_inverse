def leaderboard_gap(cv_score, public_lb_score):
    gap = public_lb_score - cv_score
    return gap

def analyze_gap(cv_score, public_lb, private_lb):
    return {
        "cv_vs_public": public_lb - cv_score,
        "public_vs_private": public_lb - private_lb
    }