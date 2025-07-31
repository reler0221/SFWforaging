import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s, GAM
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%%
breeding = pd.read_csv("Final/BB2024_processed_filtered_withgroups.csv") # 18999, 488.452 min
inter1 = pd.read_csv("Final/NB2024_processed_filtered_withgroups.csv")
#%% add pecking rates as column
breeding["PECKING_RATE"] = breeding["PECKING"]/breeding["BOUT_LENGTH"]
inter1["PECKING_RATE"] = inter1["PECKING"]/inter1["BOUT_LENGTH"]
print(len(breeding["FILENAME"].unique()))
print(len(inter1["FILENAME"].unique()))
#%% Exclude breeding data between 12:00 - 14:00: all taken from the same date, and most of them are from o single group
breeding = breeding[~breeding["TIME"].between(12,14)].reset_index(drop=True).copy()

#%% Add column SEASON and concatenate into one dataframe
breeding["SEASON"] = ["breeding"]*len(breeding)
inter1["SEASON"] = ["win_to_sum"]*len(inter1)
full = pd.concat([inter1, breeding], axis=0, ignore_index=True)

#%% Clean group ids
# Define groups to keep
main_groups = ['jbrb', 'pbgrb', 'gbb', 'jgyr', 'syw', 'srbaba', 'cmra', 'jmna']

# Replace 'unb' with 'jmna'
full["GROUP_ID"] = full["GROUP_ID"].replace("unb", "jmna")

# Replace all other values not in main_groups with 'misc'
full["GROUP_ID"] = np.where(full["GROUP_ID"].isin(main_groups),
                                  full["GROUP_ID"],
                                  "misc")

print(full["GROUP_ID"].value_counts())
#%% Total number of bouts (tracking IDs)
uncut_inter1_bout_counts = len(inter1) # 6296
uncut_breeding_bout_counts = len(breeding) # 18569
uncut_all_bout_counts = uncut_inter1_bout_counts + uncut_breeding_bout_counts # 24865


#%% Total detection time
uncut_inter1_detection_time = inter1["BOUT_LENGTH"].sum()/25/60 # 142.904 min
uncut_breeding_detection_time = breeding["BOUT_LENGTH"].sum()/25/60 # 476.736 min
uncut_all_detection_time = uncut_inter1_detection_time + uncut_breeding_detection_time # 619.64

#%% Aggregate by video -> one segment = first bird appearance from the video ~ last bird leave


by_video = full.groupby("FILENAME").agg({
    "BOUT_START": "min",
    "BOUT_END": "max",
    "PECKING": "sum",
    "VIDEO_DURATION": "first",
    "BOUT_LENGTH": "sum",
    "DATE": "first",
    "GROUP": "first",
    "GROUP_ID": "first",
    "TIME": "first",
    "SEASON": "first"
}).reset_index()
by_video["PECKING_RATE"] = by_video["PECKING"]/by_video["BOUT_LENGTH"]
by_video["BOUT_LENGTH_AB"] = by_video["BOUT_END"] - by_video["BOUT_START"]

#%%
bout_length_freq = by_video["BOUT_LENGTH_AB"].value_counts().sort_values()
pecking_rates_freq = by_video["PECKING_RATE"].apply(lambda x: round(x, 2)).value_counts().sort_values()
pecking_rates_nonzero_freq = by_video.loc[full["PECKING_RATE"]>0, "PECKING_RATE"].apply(lambda x: round(x, 2)).value_counts().sort_values()

plt.figure(figsize = (10,6))
plt.scatter(x = pecking_rates_nonzero_freq.index, y= pecking_rates_nonzero_freq.values,alpha=0.5)
plt.xlabel("Pecking Rates, rounded: second decimal")
plt.ylabel("Counts")
plt.title("Pecking Rates (non-zero) Frequency Distribution")
plt.show()

#%%
#%% exclude very short bouts
cutoff = 50 # 2 seconds
final = by_video[by_video["BOUT_LENGTH"] >= cutoff].copy()

final["BOUT_LENGTH_SEC"] = final["BOUT_LENGTH_AB"]/25

#%%
#%% Inspect distribution of zero pecking - OK
# Count total bouts per bout length
total_bout_counts = final["BOUT_LENGTH"].value_counts().sort_index()
zero_bout_counts = final[final["PECKING_RATE"] == 0]["BOUT_LENGTH"].value_counts().sort_index()
proportion_zeros = (zero_bout_counts / total_bout_counts).fillna(0)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(proportion_zeros.index, proportion_zeros.values, marker='o', linestyle='-', alpha=0.7)
plt.xlabel("Bout Length (>50 frames, fps: 25)")
plt.ylabel("Proportion of Zero Pecking Bouts")
plt.title("Proportion of Zero Pecking Bouts Across Bout Lengths")
plt.grid(True)
plt.show()

#%%
#%% Set time slots
final["TIME_SLOT"] = final["TIME"].apply(int)
print(final["TIME_SLOT"].value_counts())

#%%
#%% log transformations
final["BOUT_LENGTH_LOG"] = np.log(final["BOUT_LENGTH_SEC"])
bout_length_log_freq = final["BOUT_LENGTH_LOG"].value_counts().sort_values()
final["LOG_PECK_RATE"] = np.log1p(final["PECKING_RATE"]) # log1p to tolerate zeros

#%% Inspect pecking count for bout length
pecking_by_boutsize = final.groupby(["BOUT_LENGTH","PECKING"]).size()
print(pecking_by_boutsize)
# looks fine

#%% Makes subsets for seasons
subset_inter1 = final[final["SEASON"]=="win_to_sum"].copy()
subset_breeding = final[final["SEASON"]=="breeding"].copy()

#%% Counts, duration of bouts after cutoff
inter1_bout_counts = len(subset_inter1) # 195
breeding_bout_counts = len(subset_breeding) # 705

inter1_framecount = subset_inter1["BOUT_LENGTH"].sum() # 214006
breeding_framecount = subset_breeding["BOUT_LENGTH"].sum() # 712252

#%% Plot time x bout length

time_duration_plt = sns.scatterplot(
    data = final,
    x="TIME",
    y="BOUT_LENGTH_LOG",
    hue="SEASON",
    hue_order=["win_to_sum", "breeding"],
    marker=".",
    alpha=0.5
)
time_duration_plt.set_xlabel("Time")
time_duration_plt.set_ylabel("Bout Length (log)")
# Edit legend
handles, labels = time_duration_plt.get_legend_handles_labels()
new_labels = ["Aug-Sep", "Dec-Jan"]
time_duration_plt.legend(handles=handles, labels=new_labels, title="Season")
plt.show()

#%%
#%% GAM bout length
X = final.loc[final["SEASON"]=="breeding", "TIME"]
Y = final.loc[final["SEASON"]=="breeding", "BOUT_LENGTH_LOG"]
gam = GAM(s(0))
gam.fit(X, Y)

#%% GAM pecking rate
X = final.loc[final["SEASON"]=="breeding", "TIME"]
Y = final.loc[final["SEASON"]=="breeding", "LOG_PECK_RATE"]
gam = GAM(s(0))
gam.fit(X, Y)

#%% Inspect non-linearity with quadratic model
# Create the squared term in your DataFrame
subset_breeding["TIME_SQ"] = subset_breeding["TIME"] ** 2

# Fit the quadratic mixed model
import statsmodels.formula.api as smf

model_mixed_pecking_quad = smf.mixedlm(
    "LOG_PECK_RATE ~ TIME + TIME_SQ",
    data=subset_breeding,
    groups=subset_breeding["GROUP_ID"]
)

result_mixed_pecking_quad = model_mixed_pecking_quad.fit()
print(result_mixed_pecking_quad.summary())


#%% bout length, breeding subset
model_mixed_bout = smf.mixedlm("BOUT_LENGTH_LOG ~ TIME",
                               data=subset_breeding,
                               groups=subset_breeding["GROUP_ID"])
result_mixed_bout = model_mixed_bout.fit()
print(result_mixed_bout.summary())


#%% pecking rate, breeding subset
model_mixed_pecking = smf.mixedlm("LOG_PECK_RATE ~ TIME",
                               data=subset_breeding,
                               groups=subset_breeding["GROUP_ID"])
result_mixed_pecking = model_mixed_pecking.fit()
print(result_mixed_pecking.summary())


#%% Plot bout length model
# Prediction grid
time_vals = np.linspace(subset_breeding["TIME"].min(),
                        subset_breeding["TIME"].max(), 200)
pred_df = pd.DataFrame({"TIME": time_vals})
pred_df["BOUT_LENGTH_LOG"] = result_mixed_bout.predict(pred_df)

se_time = result_mixed_bout.bse["TIME"]
se_intercept = result_mixed_bout.bse["Intercept"]

# CI for bout length
X = sm.add_constant(pred_df["TIME"])
linear_preds = pred_df["BOUT_LENGTH_LOG"]

pred_df["CI_LOW"] = linear_preds - 1.96 * np.sqrt(se_intercept**2 + (X["TIME"]**2) * se_time**2)
pred_df["CI_HIGH"] = linear_preds + 1.96 * np.sqrt(se_intercept**2 + (X["TIME"]**2) * se_time**2)

# # Transform to original scale
# pred_df["BOUT_LENGTH_PRED"] = np.expm1(pred_df["BOUT_LENGTH_LOG"])
# pred_df["BOUT_CI_LOW"] = np.expm1(pred_df["CI_LOW_BOUT"]).clip(lower=0)
# pred_df["BOUT_CI_HIGH"] = np.expm1(pred_df["CI_HIGH_BOUT"]).clip(lower=0)

# Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=subset_breeding, x="TIME", y=subset_breeding["BOUT_LENGTH_LOG"], alpha=0.2, s=10, label="Observed")
plt.plot(pred_df["TIME"], pred_df["BOUT_LENGTH_LOG"], color="blue", label="Predicted")
plt.fill_between(pred_df["TIME"], pred_df["CI_LOW"], pred_df["CI_HIGH"], color="blue", alpha=0.2)

plt.xlabel("Time of Day")
plt.ylabel("Bout Length (log(x))")
plt.title("Bout Length vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot Pecking rate model
# Prediction grid
time_vals = np.linspace(subset_breeding["TIME"].min(), subset_breeding["TIME"].max(), 200)

pred_df = pd.DataFrame({"TIME": time_vals})

# Predict fixed effect only
pred_df["LOG_PECK_RATE"] = result_mixed_pecking.predict(pred_df)

# Standard error of TIME coefficient
se_time = result_mixed_pecking.bse["TIME"]
se_intercept = result_mixed_pecking.bse["Intercept"]

# Get confidence interval on linear predictor
X = sm.add_constant(pred_df["TIME"])
linear_preds = pred_df["LOG_PECK_RATE"]
pred_df["CI_LOW"] = linear_preds - 1.96 * np.sqrt(se_intercept**2 + (X["TIME"]**2) * se_time**2)
pred_df["CI_HIGH"] = linear_preds + 1.96 * np.sqrt(se_intercept**2 + (X["TIME"]**2) * se_time**2)

# # Transform to original scale
# pred_df["PECKING_RATE_PRED"] = np.expm1(pred_df["LOG_PECK_RATE"])
# pred_df["PECK_CI_LOW"] = np.expm1(pred_df["CI_LOW"]).clip(lower=0)
# pred_df["PECK_CI_HIGH"] = np.expm1(pred_df["CI_HIGH"]).clip(lower=0)

# Plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=subset_breeding, x="TIME", y="LOG_PECK_RATE", alpha=0.2, s=10, label="Observed")
plt.plot(pred_df["TIME"], pred_df["LOG_PECK_RATE"], color="purple", label="Predicted")
plt.fill_between(pred_df["TIME"], pred_df["CI_LOW"], pred_df["CI_HIGH"], color="purple", alpha=0.2)

plt.xlabel("Time of Day")
plt.ylabel("Pecking Rate (log(1+x))")
plt.title("Pecking Rate vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Visualize non-linearity with GAM -- pecking rate

# Fit GAM with a spline on TIME
X = subset_breeding["TIME"].values
y = subset_breeding["LOG_PECK_RATE"].values

gam = LinearGAM(s(0)).fit(X, y)

# Generate smooth prediction range
XX = np.linspace(X.min(), X.max(), 200)
intervals = gam.prediction_intervals(XX, width=0.95)
predictions = gam.predict(XX)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.2, s=10, label="Observed")
plt.plot(XX, predictions, color='green', label="GAM fit")
plt.fill_between(XX, intervals[:, 0], intervals[:, 1], alpha=0.3, color='green', label="95% CI")

plt.xlabel("Time of Day")
plt.ylabel("log1p(Pecking Rate)")
plt.title("GAM: Nonlinear Effect of Time on Pecking Rate")
plt.legend()
plt.grid(True)
plt.show()

gam.summary()

#%% Analyis 2: by season
#%%
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, probplot
import numpy as np
#%% Only use the morning data (no afternoon videos in spring)
filtered =  final[final["TIME"].le(12)].reset_index(drop=True).copy()
#%%  ---> not normal
# Check normality assumption (optional) before choosing test

# Check by season
for col, label in [("BOUT_LENGTH_LOG", "Log(Bout Length)"), ("LOG_PECK_RATE", "log1p(Pecking Rate)")]:
    for season in filtered["SEASON"].unique():
        vals = filtered[filtered["SEASON"] == season][col].dropna()

        # Histogram
        plt.figure(figsize=(10, 4))
        sns.histplot(vals, kde=True, bins=30)
        plt.title(f"{label} - {season} - Histogram")
        plt.show()

        # Q-Q plot
        plt.figure(figsize=(6, 6))
        probplot(vals, dist="norm", plot=plt)
        plt.title(f"{label} - {season} - Q-Q Plot")
        plt.grid(True)
        plt.show()

        # Shapiro-Wilk test
        stat, p = shapiro(vals.sample(min(len(vals), 500)))  # limit to 500 samples max
        print(f"{label} - {season} - Shapiro-Wilk p = {p:.4f}")

#%%

group_bout = [filtered[filtered["SEASON"] == season]["BOUT_LENGTH_LOG"] for season in filtered["SEASON"].unique()]
group_peck = [filtered[filtered["SEASON"] == season]["LOG_PECK_RATE"] for season in filtered["SEASON"].unique()]

#%%
#Mann–Whitney U
u_bout = mannwhitneyu(*group_bout, alternative="two-sided")
u_peck = mannwhitneyu(*group_peck, alternative="two-sided")


print("Log(Bout Length) - Mann–Whitney U:", u_bout)
print("log1p(Pecking Rate) - Mann–Whitney U:", u_peck)

#%%
model_peck = smf.mixedlm("LOG_PECK_RATE ~ SEASON", data=filtered, groups=filtered["GROUP_ID"])
result_peck = model_peck.fit()
print(result_peck.summary())
#%%
# --- Plot 1: Bout Length by Season ---
plt.figure(figsize=(5, 8))
sns.stripplot(data=filtered, x="SEASON", y="BOUT_LENGTH_LOG", color='gray', alpha=0.2, jitter=0.3)
sns.boxplot(data=filtered, x="SEASON", y="BOUT_LENGTH_LOG")
plt.xlabel("Season")
plt.ylabel("Bout Length (log)")
plt.xticks(ticks=[0, 1], labels=["Aug-Sep", "Dec-Jan"])
plt.title("Bout Length by Season")
# plt.text(0.5, max(filtered["BOUT_LENGTH_LOG"])*0.95, "*", ha='center', fontsize=18)
plt.grid(True)
plt.show()

# --- Plot 2: Pecking Rate by Season ---
plt.figure(figsize=(6, 7))
sns.stripplot(data=filtered, x="SEASON", y="LOG_PECK_RATE", color='gray', alpha=0.2, jitter=0.3)
sns.boxplot(data=filtered, x="SEASON", y="LOG_PECK_RATE")
plt.xlabel("Season")
plt.ylabel("Pecking Rate (log(1+x))")
plt.xticks(ticks=[0, 1], labels=["Aug-Sep", "Dec-Jan"])
plt.title("Pecking Rate by Season")
# plt.text(0.5, max(filtered["LOG_PECK_RATE"])*0.95, "*", ha='center', fontsize=18)
plt.grid(True)
plt.show()