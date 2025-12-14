from src.data_loader import load_skill_data
from src.preprocessing import scale_features
from src.pca_engine import PCAEngine
from src.skill_gap import compute_skill_gap
from src.visualizer import plot_top_skill_gaps, plot_scree
from src.config import PCA_MODEL_PATH, SCALER_PATH
import pandas as pd
import joblib


def run_pipeline(data_path=None):
    df = load_skill_data(data_path)

    feature_cols = [
        'demand_score','supply_score',
        'CLO_Programming','CLO_DSA','CLO_DBMS','CLO_WebDev','CLO_Cloud'
    ]

    X_scaled, scaler = scale_features(df, feature_cols, save_path=SCALER_PATH)

    p = PCAEngine(n_components=3)
    pcs_df = p.fit_transform(X_scaled)
    pcs_df['skill'] = df['skill']

    # save PCA model and scores
    p.save(PCA_MODEL_PATH)
    pcs_df.to_csv('data/skill_pca_scores.csv', index=False)

    # gaps
    df_gap = compute_skill_gap(df)
    df_gap.to_csv('data/skill_gaps.csv', index=False)

    print('\nTop Skill Gaps:\n')
    print(df_gap[['skill','demand_score','supply_score','gap']].head(10))

    # visualizations
    plot_top_skill_gaps(df_gap, save_path='data/top10_gaps.png')
    try:
        plot_scree(p.pca, save_path='data/scree_plot.png')
    except Exception:
        pass

    # save scaler
    joblib.dump(scaler, SCALER_PATH)

    print('\nPipeline complete. Artifacts saved in data/ and models/.')

if __name__ == '__main__':
    run_pipeline()
