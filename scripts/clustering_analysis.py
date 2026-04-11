# import necessary libraries for clustering analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# file path to the dataset
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"

def clustering():
    master_file = PROCESSED_DIR / "customer_master_table.csv"
    if not master_file.exists():
        print(f"Error: {master_file} not found. Please run the data processing script first.")
        return
    
    df = pd.read_csv(master_file)
    # Select relevant features for clustering
    features = [
        'total_orders', 'total_net_sales', 'avg_order_value', 
        'total_interactions', 'avg_rating', 'total_tickets', 
        'avg_satisfaction'
    ]
    
    X = df[features].fillna(0)  # Handle missing values by filling with 0
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters using the elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    # Plot the elbow method graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(PLOTS_DIR / "elbow_method.png")
    print(f'Elbow method plot saved to {PLOTS_DIR / "elbow_method.png"}')
    plt.show()
    
    # Fit final model
    optimal_clusters = 4  # This can be determined from the elbow plot
    model = KMeans(n_clusters=optimal_clusters, init='k-means++', n_init=10, random_state=42)
    df['cluster_id'] = model.fit_predict(X_scaled)
    
    #PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster_id', palette='viridis',s=100)
    plt.title('Customer Clusters (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster ID')
    plt.savefig(PLOTS_DIR / "customer_clusters_pca.png")
    print(f'Customer clusters PCA plot saved to {PLOTS_DIR / "customer_clusters_pca.png"}')
    plt.show()
    
    
    # save the clustered data for further analysis
    df.to_csv(PROCESSED_DIR / "customer_clustered_data.csv", index=False)
    print(f'Clustered data saved to {PROCESSED_DIR / "customer_clustered_data.csv"}')
    
    
    # provide summary
    profile = df.groupby('cluster_id')[features].mean()
    profile['customer_count'] = df.groupby('cluster_id').size()
    profile.to_csv(REPORTS_DIR / "cluster_profiles_summary.csv")
    print(f'Cluster profiles summary saved to {REPORTS_DIR / "cluster_profiles_summary.csv"}')
    summary_txt = "Cluster Analysis Summary\n" + "="*30 + "\n\n"
    summary_txt += profile.to_string()
    with open(REPORTS_DIR / "cluster_analysis_summary.txt", "w") as f:
        f.write(summary_txt)
    print(f'Cluster analysis summary saved to {REPORTS_DIR / "cluster_analysis_summary.txt"}')
    print("Clustering analysis completed and results saved.")
    
    
if __name__ == "__main__":
    clustering()