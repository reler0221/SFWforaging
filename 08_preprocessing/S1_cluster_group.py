import pandas as pd
from itertools import combinations
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
#%%
# Load the data
df = pd.read_csv("Final/SFW_group_compo.csv")

# Keep only valid bird IDs
df = df[df["Ind"].notna()]
df["Date"] = df["Group_id"].apply(lambda x: int(x.split("_")[0]))
#%% filter with dates
start_date = 20241217
end_date = 20250200

df = df[df["Date"].between(start_date,end_date)].reset_index(drop=True).copy()

#%%
# Create a list of all pairs observed together
pair_list = []
#%%
# Group by the full observation ID
for _, group in df.groupby("Group_id"):
    birds = group["Ind"].dropna().tolist()
    for pair in combinations(sorted(birds), 2):
        pair_list.append(pair)
#%%
# Count the number of co-occurrences
pair_counts = Counter(pair_list)

# Convert to DataFrame
cooccurrence_df = pd.DataFrame(
    [(a, b, count) for (a, b), count in pair_counts.items()],
    columns=["Bird1", "Bird2", "Count"]
)

co_matrix = cooccurrence_df.pivot(index="Bird1", columns="Bird2", values="Count").fillna(0)

# To make it symmetric:
co_matrix = co_matrix.add(co_matrix.T, fill_value=0)
#%%

# Convert the matrix to distance (optional: use correlation or cosine instead)
# Fill missing values and compute distance
dist_matrix = pdist(co_matrix.fillna(0), metric='euclidean')

#%%
# Hierarchical clustering
linkage_matrix = linkage(dist_matrix, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=co_matrix.index.tolist(), leaf_rotation=90)
plt.title("Bird Clustering Based on Co-occurrence")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()