import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_skill_gaps(df, save_path=None):
    top10 = df.head(10)[::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=top10['gap'], y=top10['skill'])
    plt.title("Top 10 Skill Gaps (Demand - Supply)")
    plt.xlabel("Gap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_scree(pca, save_path=None):
    plt.figure(figsize=(7,4))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_, marker='o'
    )
    plt.title("Scree Plot")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
