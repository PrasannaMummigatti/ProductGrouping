import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import norm
from matplotlib.animation import FuncAnimation

# ------------------------------
# Define 10 diverse products
# ------------------------------
products_base = pd.DataFrame({
    'Product': [f'P{i+1}' for i in range(10)],
    'Demand_Mean': [1000, 800, 1200, 600, 1100, 900, 1300, 750, 950, 1150],
    'Demand_Std': [50, 60, 120, 30, 150, 70, 200, 40, 80, 160],
    'Lead_Time': [2, 3, 5, 2, 4, 3, 6, 2, 3, 5],
    'Order_Cost': [100,100,100,100,100,100,100,100,100,100],
    'Holding_Cost': [1.0, 1.2, 1.5, 1.0, 1.3, 1.1, 2.0, 1.2, 1.3, 1.8],
    #'Unit_Cost': [20]*10,
    #'Service_Level': [0.10, 0.92, 0.97, 0.90, 0.60, 0.94, 0.90, 0.91, 0.93, 0.96]
    'Service_Level': [0.80, 0.85, 0.90, 0.95, 0.99, 0.92, 0.98, 0.88, 0.97, 0.96],
    'Unit_Cost':     [20, 20, 30, 30, 50, 50, 80, 20, 60, 40]
})

# ------------------------------
# Preprocess and Apply PCA
# ------------------------------
features = ['Demand_Mean', 'Demand_Std', 'Lead_Time', 'Order_Cost', 'Holding_Cost', 'Unit_Cost', 'Service_Level']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(products_base[features])
print("Scaled Features:\n", X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
products_base['PCA1'] = X_pca[:, 0]
products_base['PCA2'] = X_pca[:, 1]

# ------------------------------
# Animation Setup
# ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
colors = sns.color_palette('Set2', 10)

def calculate_individual_cost_breakdown(df):
    ordering = holding = stockout = 0
    for _, row in df.iterrows():
        z = norm.ppf(row['Service_Level'])
        Q = np.sqrt((2 * row['Demand_Mean'] * row['Order_Cost']) / row['Holding_Cost'])
        num_orders = row['Demand_Mean'] / Q
        avg_inventory = Q / 2
        safety_stock = z * row['Demand_Std'] * np.sqrt(row['Lead_Time'])
        total_inventory = avg_inventory + safety_stock

        ordering += num_orders * row['Order_Cost']
        holding += total_inventory * row['Holding_Cost']
        stockout += (1 - row['Service_Level']) * row['Unit_Cost'] * row['Demand_Mean']
    total = ordering + holding + stockout
    return ordering, holding, stockout, total

ungrouped_costs = calculate_individual_cost_breakdown(products_base)

def update(k):
    axs[0].cla()
    axs[1].cla()

    products = products_base.copy()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    products['Clus' \
    'ter'] = kmeans.fit_predict(X_scaled)
    products['Cluster_Label'] = 'Group ' + (products['Cluster'] + 1).astype(str)
    centers = pca.transform(kmeans.cluster_centers_)

    # Plot PCA Clustering
    legend_labels = []
    for cluster_id in range(k):
        group_products = products[products['Cluster'] == cluster_id]['Product'].tolist()
        label = f"Group {cluster_id+1} ({', '.join(group_products)})"
        legend_labels.append((cluster_id, label))

    sns.scatterplot(data=products, x='PCA1', y='PCA2', hue='Cluster', palette=colors, s=130, edgecolor='black', ax=axs[0], legend=False)

    for idx, row in products.iterrows():
        cluster_id = row['Cluster']
        axs[0].plot([centers[cluster_id, 0], row['PCA1']],
                    [centers[cluster_id, 1], row['PCA2']], 'k--', alpha=0.5)
        axs[0].text(row['PCA1'] + 0.1, row['PCA2'], row['Product'], fontsize=9)

    axs[0].scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x')
    axs[0].set_title(f'KMeans Clustering (k={k})')
    axs[0].set_xlabel('PCA1')
    axs[0].set_ylabel('PCA2')
    axs[0].grid(True)

    from matplotlib.patches import Patch
    legend_handles = [Patch(color=colors[cluster_id], label=label) for cluster_id, label in legend_labels]
    axs[0].legend(handles=legend_handles, title="Clusters")

    # --------------------------
    # Cost Comparison Breakdown
    # --------------------------
    group_costs = []
    for cluster_id in range(k):
        group_df = products[products['Cluster'] == cluster_id]
        total_demand = group_df['Demand_Mean'].sum()
        avg_holding_cost = np.average(group_df['Holding_Cost'], weights=group_df['Demand_Mean'])
        order_cost = group_df['Order_Cost'].iloc[0]
        # Demand-weighted service level
        service_level = np.average(group_df['Service_Level'], weights=group_df['Demand_Mean'])
        z = norm.ppf(service_level)
        Q = np.sqrt((2 * total_demand * order_cost) / avg_holding_cost)
        num_orders = total_demand / Q
        avg_inventory = Q / 2
        pooled_std = np.sqrt((group_df['Demand_Std'] ** 2).sum())
        pooled_lead_time = np.max(group_df['Lead_Time'])
        safety_stock = z * pooled_std * np.sqrt(pooled_lead_time)
        total_inventory = avg_inventory + safety_stock

        ordering_cost = num_orders * order_cost
        holding_cost = total_inventory * avg_holding_cost
        stockout_cost = (1 - service_level) * group_df['Unit_Cost'].mean() * total_demand

        group_costs.append([ordering_cost, holding_cost, stockout_cost])

    group_costs = np.array(group_costs)
    group_sums = np.sum(group_costs, axis=0)
    grouped_total_cost = np.sum(group_sums)

    # Bar Chart
    x = np.arange(4)
    width = 0.35

    axs[1].bar(x[:-1] - width/2, ungrouped_costs[:3], width, label='Ungrouped', color='gray')
    axs[1].bar(x[-1] - width/2, ungrouped_costs[3], width, color='darkgray')

    for i in range(3):
        axs[1].text(x[i] - width/2, ungrouped_costs[i] + 100, f"{ungrouped_costs[i]:,.0f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    axs[1].text(x[-1] - width/2, ungrouped_costs[3] + 100, f"{ungrouped_costs[3]:,.0f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    bottoms = np.zeros(3)
    for i in range(k):
        axs[1].bar(x[:3] + width/2, group_costs[i], width, bottom=bottoms, color=colors[i], label=f'Group {i+1}')
        for j in range(3):
            axs[1].text(x[j] + width/2, bottoms[j] + group_costs[i][j]/2, f"{group_costs[i][j]:,.0f}",
                        ha='center', va='center', fontsize=8)
        bottoms += group_costs[i]

    for j in range(3):
        axs[1].text(x[j] + width/2, bottoms[j] + 100, f"{bottoms[j]:,.0f}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    axs[1].bar(x[-1] + width/2, grouped_total_cost, width, color='black', label='Grouped Total')
    axs[1].text(x[-1] + width/2, grouped_total_cost + 100, f"{grouped_total_cost:,.0f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    axs[1].set_xticks(x)
    axs[1].set_xticklabels(['Ordering', 'Holding', 'Stockout', 'Total'])
    axs[1].set_ylabel("Cost")
    axs[1].set_title("Grouped vs Ungrouped Total Supply chain Cost Breakdown")
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

# ------------------------------
# Run Animation
# ------------------------------
ani = FuncAnimation(fig, update, frames=range(1, 6), interval=4000, repeat=True)
plt.show()


# ------------------------------
# Function to calculate grouped cost
# ------------------------------
def calculate_grouped_cost(products, k):
    kmeans = KMeans(n_clusters=k, random_state=42,n_init='auto')
    products['Cluster'] = kmeans.fit_predict(X_scaled)
    total_cost = 0

    for cluster_id in range(k):
        group_df = products[products['Cluster'] == cluster_id]
        total_demand = group_df['Demand_Mean'].sum()
        avg_holding_cost = np.average(group_df['Holding_Cost'], weights=group_df['Demand_Mean'])
        order_cost = group_df['Order_Cost'].iloc[0]

        # Use demand-weighted average service level
        service_level = np.average(group_df['Service_Level'], weights=group_df['Demand_Mean'])

        z = norm.ppf(service_level)
        Q = np.sqrt((2 * total_demand * order_cost) / avg_holding_cost)
        num_orders = total_demand / Q
        avg_inventory = Q / 2

        pooled_std = np.sqrt((group_df['Demand_Std'] ** 2).sum())
        pooled_lead_time = np.max(group_df['Lead_Time'])
        safety_stock = z * pooled_std * np.sqrt(pooled_lead_time)
        total_inventory = avg_inventory + safety_stock

        ordering_cost = num_orders * order_cost
        holding_cost = total_inventory * avg_holding_cost
        stockout_cost = (1 - service_level) * group_df['Unit_Cost'].mean() * total_demand

        total_cost += ordering_cost + holding_cost + stockout_cost

    return total_cost



# Calculate grouped cost for different k
# ------------------------------
k_values = range(1, 6)
grouped_costs = [calculate_grouped_cost(products_base.copy(), k) for k in k_values]

# ------------------------------
# Plot the total grouped cost vs k
# ------------------------------
plt.figure(figsize=(10, 6))
plt.plot(k_values, grouped_costs, marker='o', linestyle='--', color='blue')
plt.title('Total Grouped Supply Chain Cost vs Number of Groups (k)')
plt.xlabel('Number of Groups (k)')
plt.ylabel('Total Grouped Supply Chain Cost')
for i, cost in enumerate(grouped_costs):
    plt.text(k_values[i], cost, f"{cost:,.0f}", ha='center', fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()
