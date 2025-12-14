import numpy as np
import pandas as pd

def compute_skill_gap(df):
    df = df.copy()
    df['gap'] = df['demand_score'] - df['supply_score']
    return df.sort_values('gap', ascending=False)

def compute_target_from_pcs(pcs_df, method='mean', top_percentile=0.9):
    if method == 'mean':
        return pcs_df.mean(axis=0)
    elif method == 'top_percentile':
        score = pcs_df.sum(axis=1)
        cutoff = score.quantile(top_percentile)
        top = pcs_df[score >= cutoff]
        if top.shape[0] == 0:
            return pcs_df.mean(axis=0)
        return top.mean(axis=0)
    else:
        raise ValueError('Unknown method')

def student_gap_vs_target(student_pcs, target_pcs):
    s = np.array(student_pcs)
    t = np.array(target_pcs)
    vec = t - s
    l2 = np.linalg.norm(vec)
    std = student_pcs.std() if hasattr(student_pcs, 'std') else np.std(student_pcs)
    std_arr = np.ones_like(vec) * (std if std != 0 else 1.0)
    mahal = np.sqrt(((vec / std_arr) ** 2).sum())
    return {
        'gap_vector': vec.tolist(),
        'gap_l2': float(l2),
        'gap_mahalanobis_approx': float(mahal)
    }

def interpret_gap_scalar(gap_value):
    if gap_value <= 0.0:
        return "No gap — meets or exceeds demand"
    elif gap_value < 0.2:
        return "Small gap — minor improvement"
    elif gap_value < 0.5:
        return "Moderate gap — focused training needed"
    else:
        return "Large gap — high priority"
