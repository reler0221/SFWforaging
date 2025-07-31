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
inter1 = pd.read_csv("Final/NB2024_processed_filtered.csv")


#%% add pecking rates as column
breeding["PECKING_RATE"] = breeding["PECKING"]/breeding["BOUT_LENGTH"]
inter1["PECKING_RATE"] = inter1["PECKING"]/inter1["BOUT_LENGTH"]

#%% Exclude breeding data between 12 - 2: all taken from the same date, and most of them are from o single group
breeding = breeding[~breeding["TIME"].between(12,14)].reset_index(drop=True).copy()

#%% Add column SEASON and concatenate into one dataframe
breeding["SEASON"] = ["breeding"]*len(breeding)
inter1["SEASON"] = ["win_to_sum"]*len(inter1)
full = pd.concat([inter1, breeding], axis=0, ignore_index=True)

#%% Total number of bouts
uncut_inter1_bout_counts = len(inter1) # 6296
uncut_breeding_bout_counts = len(breeding) # 18569
uncut_all_bout_counts = uncut_inter1_bout_counts + uncut_breeding_bout_counts # 24865

#%% Total detection time
uncut_inter1_detection_time = inter1["BOUT_LENGTH"].sum()/25/60 # 142.904 min
uncut_breeding_detection_time = breeding["BOUT_LENGTH"].sum()/25/60 # 476.736 min
uncut_all_detection_time = uncut_inter1_detection_time + uncut_breeding_detection_time # 619.64

#%% From here, just use the combined dataframe "full"

#%% Distribution of bout lengths , pecking rates
bout_length_freq = full["BOUT_LENGTH"].value_counts().sort_values()
pecking_rates_freq = full["PECKING_RATE"].apply(lambda x: round(x, 2)).value_counts().sort_values()
pecking_rates_nonzero_freq = full.loc[full["PECKING_RATE"]>0, "PECKING_RATE"].apply(lambda x: round(x, 2)).value_counts().sort_values()

plt.figure(figsize = (10,6))
plt.scatter(x = pecking_rates_nonzero_freq.index, y= pecking_rates_nonzero_freq.values,alpha=0.5)
plt.xlabel("Pecking Rates, rounded: second decimal")
plt.ylabel("Counts")
plt.title("Pecking Rates (non-zero) Frequency Distribution")
plt.show()

#%% exclude very short bouts
cutoff = 50 # 2 seconds
full_final = full[full["BOUT_LENGTH"] >= cutoff].copy()

full_final["BOUT_LENGTH_SEC"] = full_final["BOUT_LENGTH"]/25

#%% Check again after cutoff
bout_length_cut_freq = full_final["BOUT_LENGTH"].value_counts().sort_values()
pecking_rates_cut_freq = full_final["PECKING_RATE"].value_counts().sort_values()
pecking_rates_nonzero_cut_freq = full_final.loc[full["PECKING_RATE"]>0, "PECKING_RATE"].value_counts().sort_values()

plt.figure(figsize = (10,6))
plt.scatter(x = pecking_rates_cut_freq.index, y= pecking_rates_cut_freq.values,alpha=0.5)
plt.xlabel("Pecking Rates")
plt.ylabel("Number of Pecking Rates")
plt.title("Pecking Rates (Non-zeros) Frequency Distribution")
plt.show()

#%% Inspect pecking rates of short foraging
short_bouts = full_final[full_final["BOUT_LENGTH"] < 125].copy()
plt.figure(figsize = (10,6))
plt.scatter(x = short_bouts["BOUT_LENGTH"], y= short_bouts["PECKING_RATE"],alpha=0.5)
plt.xlabel("Bout Length (50 < x < 125, x: frames, fps: 25)")
plt.ylabel("Pecking Rate")
plt.title("Pecking Rates In Short Bouts")
plt.show()

#%% Distribution of zero pecking
# Count total bouts per bout length
total_bout_counts = full_final["BOUT_LENGTH"].value_counts().sort_index()
zero_bout_counts = full_final[full_final["PECKING_RATE"] == 0]["BOUT_LENGTH"].value_counts().sort_index()
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
full_final["TIME_SLOT"] = full_final["TIME"].apply(int)
print(full_final["TIME_SLOT"].value_counts())

#%% Clean group ids
# Define groups to keep
main_groups = ['jbrb', 'pbgrb', 'gbb', 'jgyr', 'syw', 'srbaba', 'cmra', 'jmna']

# Replace 'unb' with 'jmna'
full_final["GROUP_ID"] = full_final["GROUP_ID"].replace("unb", "jmna")

# Replace all other values not in main_groups with 'misc'
full_final["GROUP_ID"] = np.where(full_final["GROUP_ID"].isin(main_groups),
                                  full_final["GROUP_ID"],
                                  "misc")
#%%
# Prepare data
full_final["ZERO_PECK"] = (full_final["PECKING_RATE"] == 0).astype(int)
X = sm.add_constant(full_final["BOUT_LENGTH"])
model = sm.Logit(full_final["ZERO_PECK"], X).fit()

# Generate a range of bout lengths for prediction
x_vals = np.linspace(full_final["BOUT_LENGTH"].min(), full_final["BOUT_LENGTH"].max(), 300)
x_vals_const = sm.add_constant(x_vals)

# Get predicted probabilities and confidence intervals
predictions = model.get_prediction(x_vals_const)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

# Plot observed proportion
plt.figure(figsize = (10,6))
prop_zero_peck = full_final.groupby("BOUT_LENGTH")["ZERO_PECK"].mean()
plt.scatter(prop_zero_peck.index, prop_zero_peck.values, alpha=0.5)

# Plot regression line
plt.plot(x_vals, pred_summary["predicted"], color="red", label="Logistic Regression Fit")
plt.fill_between(
    x_vals,
    pred_summary["ci_lower"],
    pred_summary["ci_upper"],
    color="red",
    alpha=0.3,
    label="95% Confidence Interval"
)


plt.xlabel("Bout Length (>50 frames, fps: 25)")
plt.ylabel("Proportion of Zero Pecking Bouts")
plt.title("Logistic Regression with Confidence Interval")
plt.legend()
plt.show()


#%% log transformations
full_final["BOUT_LENGTH_LOG"] = np.log(full_final["BOUT_LENGTH_SEC"])
bout_length_log_freq = full_final["BOUT_LENGTH_LOG"].value_counts().sort_values()
full_final["LOG_PECK_RATE"] = np.log1p(full_final["PECKING_RATE"])



#%% Inspect pecking count for bout length
pecking_by_boutsize = full_final.groupby(["BOUT_LENGTH","PECKING"]).size()
print(pecking_by_boutsize)
# looks fine

#%% Filter for seasons
subset_inter1 = full_final[full_final["SEASON"]=="win_to_sum"].copy()
subset_breeding = full_final[full_final["SEASON"]=="breeding"].copy()


#%% Counts, duration of bouts after cutoff
inter1_bout_counts = len(subset_inter1) # 2530
breeding_bout_counts = len(subset_breeding) # 8072

inter1_detection_time = subset_inter1["BOUT_LENGTH_SEC"].sum()/60 # 113.09
breeding_detection_time = subset_breeding["BOUT_LENGTH_SEC"].sum()/60 #400.89

#%% Plot time x bout length

time_duration_plt = sns.scatterplot(
    data = full_final,
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
X = full_final.loc[full_final["SEASON"]=="breeding", "TIME_SLOT"]
Y = full_final.loc[full_final["SEASON"]=="breeding", "BOUT_LENGTH_LOG"]
plt.figure(figsize=(12, 6))
sns.boxplot(data=full_final, x=X, y=Y, color='lightblue', fliersize=2)

plt.xlabel("Time Slot (Hour of Day)")
plt.ylabel("Bout Length (s)")
plt.title("Distribution of Bout Length by Time Slot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#%%
X = full_final.loc[full_final["SEASON"]=="breeding", "TIME_SLOT"]
Y = full_final.loc[full_final["SEASON"]=="breeding", "PECKING_RATE"].apply(lambda x: x*100)
plt.figure(figsize=(12, 6))
sns.boxplot(data=full_final, x=X, y=Y, color='lightblue', fliersize=2)

plt.xlabel("Time Slot (Hour of Day)")
plt.ylabel("Pecking Rate (%)")
plt.title("Distribution of Bout Length by Time Slot")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% GAM
X = full_final.loc[full_final["SEASON"]=="breeding", "TIME"]
Y = full_final.loc[full_final["SEASON"]=="breeding", "BOUT_LENGTH_LOG"]
gam = GAM(s(0))
gam.fit(X, Y)
#%%
X_pred = full_final.loc[full_final["SEASON"]=="breeding", "TIME"]
y_pred = gam.predict(X_pred)
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Data', alpha=0.5)
plt.plot(X_pred, y_pred, label='GAM Prediction', color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

#%%
import statsmodels.formula.api as smf

# Fit logistic regression: whether pecking occurred
subset_breeding["PECKED"] = 1 - subset_breeding["ZERO_PECK"]
logit_model = smf.logit("PECKED ~ TIME + BOUT_LENGTH_LOG", data=subset_breeding).fit()

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix BOUT_LENGTH_LOG at its median value
bout_log_fixed = subset_breeding["BOUT_LENGTH_LOG"].median()
time_range = np.linspace(subset_breeding["TIME"].min(), subset_breeding["TIME"].max(), 200)

# Create DataFrame for prediction
df_time = pd.DataFrame({
    "TIME": time_range,
    "BOUT_LENGTH_LOG": bout_log_fixed
})
df_time["pred_prob"] = logit_model.predict(df_time)

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=subset_breeding, x="TIME", y="PECKED", alpha=0.1, s=20, label="Observed (0/1)")
plt.plot(df_time["TIME"], df_time["pred_prob"], color="red", linewidth=2, label="Predicted")
plt.xlabel("Time")
plt.ylabel("Probability of Pecking")
plt.title("Time vs Pecking")
plt.legend()
plt.grid(True)
plt.show()

#%%
# Fix TIME at its median value
time_fixed = subset_breeding["TIME"].median()
bout_log_range = np.linspace(subset_breeding["BOUT_LENGTH_LOG"].min(), subset_breeding["BOUT_LENGTH_LOG"].max(), 200)

# Create DataFrame for prediction
df_bout = pd.DataFrame({
    "TIME": time_fixed,
    "BOUT_LENGTH_LOG": bout_log_range
})
df_bout["pred_prob"] = logit_model.predict(df_bout)

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=subset_breeding, x="BOUT_LENGTH_LOG", y="ZERO_PECK", alpha=0.2, s=10, label="Observed (0/1)")
plt.plot(df_bout["BOUT_LENGTH_LOG"], df_bout["pred_prob"], color="blue", linewidth=2, label="Predicted")
plt.xlabel("Log(Bout Length)")
plt.ylabel("Probability of Zero Pecking")
plt.title("Probability of Zero Pecking vs Bout Length (log)")
plt.legend()
plt.grid(True)
plt.show()
#%%

plt.figure(figsize=(8, 5))
sns.scatterplot(data=subset_breeding, x="TIME", y="BOUT_LENGTH_LOG", alpha=0.2, s=10, label="Observed")
sns.regplot(data=subset_breeding, x="TIME", y="BOUT_LENGTH_LOG",
            scatter=False, order=2, color="red", label="Trend")

plt.xlabel("Time")
plt.ylabel("Log(Bout Length)")
plt.title("Bout Length (log) vs Time")
plt.legend()
plt.grid(True)
plt.show()


#%%
# Subset to pecking bouts only
pecking_only = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=pecking_only, x="TIME", y="PECKING_RATE", alpha=0.2, s=10, label="Observed")
sns.regplot(data=pecking_only, x="TIME", y="PECKING_RATE",
            scatter=False, order=2, color="green", label="Trend")

plt.xlabel("Time")
plt.ylabel("Pecking Rate")
plt.title("Pecking Rate vs Time (Only Non-zero Bouts)")
plt.legend()
plt.grid(True)
plt.show()

#%%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=pecking_only, x="PECKING_RATE", y="BOUT_LENGTH_LOG", alpha=0.2, s=10, label="Observed")
sns.regplot(data=pecking_only,x="PECKING_RATE", y="BOUT_LENGTH_LOG",
            scatter=False, lowess=False, color="purple", label="Trend")

plt.xlabel("Pecking Rate")
plt.ylabel("Log(Bout Length)")
plt.title("Bout Length (log) vs Pecking Rate — Non-Zero Bouts Only")
plt.legend()
plt.grid(True)
plt.show()

#%%
# Step 1: Filter and transform
pecking_only_breeding = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()
pecking_only_breeding["LOG_PECK_RATE"] = np.log1p(pecking_only_breeding["PECKING_RATE"])

# Step 2: Fit model to predict log(bout length)
model = smf.ols("BOUT_LENGTH_LOG ~ LOG_PECK_RATE + TIME", data=pecking_only_breeding).fit()
print(model.summary())

# Step 3: Create prediction grid across observed PECKING_RATE range
peck_vals = np.linspace(pecking_only_breeding["PECKING_RATE"].min(), pecking_only_breeding["PECKING_RATE"].max(), 200)
log_peck_vals = np.log1p(peck_vals)
time_fixed = pecking_only_breeding["TIME"].median()

predict_df = pd.DataFrame({
    "LOG_PECK_RATE": log_peck_vals,
    "TIME": time_fixed
})

# Step 4: Predict bout length (on log scale) and transform back
pred = model.get_prediction(predict_df).summary_frame(alpha=0.05)
predict_df["PRED_BOUT_LENGTH"] = np.expm1(pred["mean"]).clip(lower=0) # back-transform
predict_df["LOWER_CI"] = np.expm1(pred["mean_ci_lower"]).clip(lower=0)
predict_df["UPPER_CI"] = np.expm1(pred["mean_ci_upper"]).clip(lower=0)
predict_df["PECKING_RATE"] = peck_vals  # add raw rate for plotting

# Step 5: Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=pecking_only_breeding, x="LOG_PECK_RATE", y="BOUT_LENGTH_LOG", alpha=0.3, s=6 )
plt.plot(predict_df["PECKING_RATE"], predict_df["PRED_BOUT_LENGTH"], color="purple", label="Predicted")
plt.fill_between(predict_df["PECKING_RATE"],
                 predict_df["LOWER_CI"],
                 predict_df["UPPER_CI"],
                 color="purple", alpha=0.3)

plt.xlabel("Pecking Rate")
plt.ylabel("Bout Length (seconds)")
plt.title("Predicted Bout Duration vs Pecking Rate (Non-Zero Only)")
plt.legend()
plt.grid(True)
plt.show()
#%%
subset = pecking_only_breeding[["LOG_PECK_RATE", "TIME"]]
subset.corr()
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = sm.add_constant(pecking_only_breeding[["LOG_PECK_RATE", "TIME"]])
vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_df)


# residuals = model.resid
# fitted = model.fittedvalues
#
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=fitted, y=residuals, alpha=0.5)
# plt.axhline(0, color='red', linestyle='--')
# plt.xlabel("Fitted values (predicted log-bout length)")
# plt.ylabel("Residuals")
# plt.title("Residuals vs Fitted Values")
# plt.grid(True)
# plt.show()

#%% mixed linear, added group id as random effect
#%% pecking rate, pecking only subset
# Use only non-zero pecking rates
pecking_only = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()
pecking_only["LOG_PECK_RATE"] = np.log1p(pecking_only["PECKING_RATE"])


# Fit mixed model
model_mixed_pecking = smf.mixedlm("LOG_PECK_RATE ~ TIME",
                          data=pecking_only,
                          groups=pecking_only["GROUP_ID"])
result_mixed_pecking = model_mixed_pecking.fit()
print(result_mixed_pecking.summary())

#%% bout length, breeding subset
model_mixed_bout = smf.mixedlm("BOUT_LENGTH_LOG ~ TIME",
                               data=subset_breeding,
                               groups=subset_breeding["GROUP_ID"])
result_mixed_bout = model_mixed_bout.fit()
print(result_mixed_bout.summary())

#%% Plot

# Prediction grid
time_vals = np.linspace(min(pecking_only["TIME"].min(), subset_breeding["TIME"].min()),
                        max(pecking_only["TIME"].max(), subset_breeding["TIME"].max()), 200)
pred_df = pd.DataFrame({"TIME": time_vals})

# ---- Model 1: Pecks ----
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

# Transform to original scale
pred_df["PECKING_RATE_PRED"] = np.expm1(pred_df["LOG_PECK_RATE"])
pred_df["PECK_CI_LOW"] = np.expm1(pred_df["CI_LOW"]).clip(lower=0)
pred_df["PECK_CI_HIGH"] = np.expm1(pred_df["CI_HIGH"]).clip(lower=0)

# ---- Plot 1 ----
plt.figure(figsize=(10, 5))
sns.scatterplot(data=pecking_only, x="TIME", y="PECKING_RATE", alpha=0.2, s=10, label="Observed")
plt.plot(pred_df["TIME"], pred_df["PECKING_RATE_PRED"], color="purple", label="Mixed Model Fit")
plt.fill_between(pred_df["TIME"], pred_df["PECK_CI_LOW"], pred_df["PECK_CI_HIGH"], color="purple", alpha=0.2)

plt.xlabel("Time of Day")
plt.ylabel("Pecking Rate")
plt.title("Pecking Rate vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Model 2: Bout Length ----
pred_df["BOUT_LENGTH_LOG"] = result_mixed_bout.predict(pred_df)

se_time_bout = result_mixed_bout.bse["TIME"]
se_intercept_bout = result_mixed_bout.bse["Intercept"]

# CI for bout length
linear_preds_bout = pred_df["BOUT_LENGTH_LOG"]
# pred_df["BOUT_SE_LOW"] = linear_preds_bout - 1 * np.sqrt(se_intercept_bout**2 + (X["TIME"]**2) * se_time_bout**2)
# pred_df["BOUT_SE_HIGH"] = linear_preds_bout + 1 * np.sqrt(se_intercept_bout**2 + (X["TIME"]**2) * se_time_bout**2)

pred_df["CI_LOW_BOUT"] = linear_preds_bout - 1.96 * np.sqrt(se_intercept_bout**2 + (X["TIME"]**2) * se_time_bout**2)
pred_df["CI_HIGH_BOUT"] = linear_preds_bout + 1.96 * np.sqrt(se_intercept_bout**2 + (X["TIME"]**2) * se_time_bout**2)

pred_df["BOUT_LENGTH_PRED"] = np.expm1(pred_df["BOUT_LENGTH_LOG"])
# pred_df["BOUT_SE_LOW"] = np.expm1(pred_df["BOUT_SE_LOW"]).clip(lower=0)
# pred_df["BOUT_SE_HIGH"] = np.expm1(pred_df["BOUT_SE_HIGH"]).clip(lower=0)
pred_df["BOUT_CI_LOW"] = np.expm1(pred_df["CI_LOW_BOUT"]).clip(lower=0)
pred_df["BOUT_CI_HIGH"] = np.expm1(pred_df["CI_HIGH_BOUT"]).clip(lower=0)

# ---- Plot 2 ----
plt.figure(figsize=(10, 5))
# sns.scatterplot(data=subset_breeding, x="TIME", y="BOUT_LENGTH_LOG", alpha=0.2, s=10)
# plt.plot(pred_df["TIME"], pred_df["BOUT_LENGTH_LOG"], color="blue")
# plt.fill_between(pred_df["TIME"], pred_df["CI_LOW_BOUT"], pred_df["CI_HIGH_BOUT"],
#                  color="blue", alpha=0.2)
sns.scatterplot(data=subset_breeding, x="TIME", y=np.expm1(subset_breeding["BOUT_LENGTH_LOG"]), alpha=0.2, s=10, label="Observed")
plt.plot(pred_df["TIME"], pred_df["BOUT_LENGTH_PRED"], color="blue", label="Mixed Model Fit")
plt.fill_between(pred_df["TIME"], pred_df["BOUT_CI_LOW"], pred_df["BOUT_CI_HIGH"], color="blue", alpha=0.2)

plt.xlabel("Time of Day")
plt.ylabel("Bout Length (sec)")
plt.title("Bout Length vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%

# Subset to pecking > 0 and log-transform rate
pecking_only = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()
pecking_only["LOG_PECK_RATE"] = np.log1p(pecking_only["PECKING_RATE"])

# Fit GAM with a spline on TIME
X = pecking_only[["TIME"]].values
y = pecking_only["LOG_PECK_RATE"].values

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
#%%
# Subset and prepare
subset_breeding = full_final[full_final["SEASON"] == "breeding"].copy()
X = subset_breeding[["TIME"]].values
y = subset_breeding["BOUT_LENGTH_LOG"].values

# Fit GAM
gam = LinearGAM(s(0)).fit(X, y)

# Generate prediction range
XX = np.linspace(X.min(), X.max(), 200)
intervals = gam.prediction_intervals(XX, width=0.95)
predictions = gam.predict(XX)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.2, s=10, label="Observed")
plt.plot(XX, predictions, color='blue', label="GAM fit")
plt.fill_between(XX, intervals[:, 0], intervals[:, 1], alpha=0.3, color='blue', label="95% CI")

plt.xlabel("Time of Day")
plt.ylabel("log(Bout Length)")
plt.title("GAM: Nonlinear Effect of Time on Bout Length (log)")
plt.legend()
plt.grid(True)
plt.show()

gam.summary()

#%% breeding gam - log plot
# Subset to breeding season
subset_breeding = full_final[full_final["SEASON"] == "breeding"].copy()

# ---------------------
# 1. GAM for log1p(PECKING_RATE)
pecking_only = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()
pecking_only["LOG_PECK_RATE"] = np.log1p(pecking_only["PECKING_RATE"])

X_peck = pecking_only[["TIME"]].values
y_peck = pecking_only["LOG_PECK_RATE"].values

gam_peck = LinearGAM(s(0)).fit(X_peck, y_peck)
XX = np.linspace(X_peck.min(), X_peck.max(), 200)
pred_peck = gam_peck.predict(XX)
ci_peck = gam_peck.prediction_intervals(XX, width=0.95)

# ---------------------
# 2. GAM for log(BOUT_LENGTH)
X_bout = subset_breeding[["TIME"]].values
y_bout = subset_breeding["BOUT_LENGTH_LOG"].values

gam_bout = LinearGAM(s(0)).fit(X_bout, y_bout)
pred_bout = gam_bout.predict(XX)
ci_bout = gam_bout.prediction_intervals(XX, width=0.95)

# ---------------------
# Plot both
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

# Plot 1: Pecking Rate
axs[0].scatter(X_peck, y_peck, alpha=0.2, s=10, label="Observed")
axs[0].plot(XX, pred_peck, color='green', label="GAM fit")
axs[0].fill_between(XX, ci_peck[:, 0], ci_peck[:, 1], alpha=0.3, color='green', label="95% CI")
axs[0].set_title("GAM: log1p(Pecking Rate) vs Time")
axs[0].set_xlabel("Time of Day")
axs[0].set_ylabel("log1p(Pecking Rate)")
axs[0].legend()
axs[0].grid(True)

# Plot 2: Bout Length
axs[1].scatter(X_bout, y_bout, alpha=0.2, s=10, label="Observed")
axs[1].plot(XX, pred_bout, color='blue', label="GAM fit")
axs[1].fill_between(XX, ci_bout[:, 0], ci_bout[:, 1], alpha=0.3, color='blue', label="95% CI")
axs[1].set_title("GAM: log(Bout Length) vs Time")
axs[1].set_xlabel("Time of Day")
axs[1].set_ylabel("log(Bout Length)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


#%% breeding gam - back transform
from pygam import LinearGAM, s
import numpy as np
import matplotlib.pyplot as plt

# Subset to breeding season
subset_breeding = full_final[full_final["SEASON"] == "breeding"].copy()

# ---------------------
# 1. GAM for log1p(PECKING_RATE)
pecking_only = subset_breeding[subset_breeding["PECKING_RATE"] > 0].copy()
pecking_only["LOG_PECK_RATE"] = np.log1p(pecking_only["PECKING_RATE"])

X_peck = pecking_only[["TIME"]].values
y_peck = pecking_only["LOG_PECK_RATE"].values

gam_peck = LinearGAM(s(0)).fit(X_peck, y_peck)
XX = np.linspace(X_peck.min(), X_peck.max(), 200)
pred_peck = gam_peck.predict(XX)
ci_peck = gam_peck.prediction_intervals(XX, width=0.95)

# Back-transform to raw pecking rates
pred_peck_raw = np.expm1(pred_peck)
ci_peck_raw = np.expm1(ci_peck)

#%% ---------------------
# 2. GAM for log(BOUT_LENGTH)

subset_breeding["BOUT_LENGTH/PECK_RATE"] = subset_breeding["BOUT_LENGTH_LOG"]/subset_breeding["LOG_PECK_RATE"]
X_bout = subset_breeding[["TIME"]].values
y_bout = subset_breeding["BOUT_LENGTH/PECK_RATE"].values

gam_bout = LinearGAM(s(0)).fit(X_bout, y_bout)
pred_bout = gam_bout.predict(XX)
ci_bout = gam_bout.prediction_intervals(XX, width=0.95)

# Back-transform to seconds
pred_bout_raw = np.exp(pred_bout)
ci_bout_raw = np.exp(ci_bout)

# ---------------------
# Plot both
fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

# Plot 1: Pecking Rate
axs[0].scatter(X_peck, pecking_only["PECKING_RATE"], alpha=0.2, s=10, label="Observed")
axs[0].plot(XX, pred_peck_raw, color='green', label="GAM fit")
axs[0].fill_between(XX, ci_peck_raw[:, 0], ci_peck_raw[:, 1], alpha=0.3, color='green')
axs[0].set_title("GAM: Pecking Rate vs Time")
axs[0].set_xlabel("Time of Day")
axs[0].set_ylabel("Pecking Rate")
axs[0].legend()
axs[0].grid(True)

# Plot 2: Bout Length
axs[1].scatter(X_bout, np.exp(y_bout), alpha=0.2, s=10, label="Observed")
axs[1].plot(XX, pred_bout_raw, color='blue', label="GAM fit")
axs[1].fill_between(XX, ci_bout_raw[:, 0], ci_bout_raw[:, 1], alpha=0.3, color='blue')
axs[1].set_title("GAM: Bout Length (s) vs Time")
axs[1].set_xlabel("Time of Day")
axs[1].set_ylabel("Bout Length (seconds)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

#%%%
from pygam import LinearGAM, s, f, te

# Filter to TIME < 12.5
filtered = full_final[full_final["TIME"] < 12.5].copy()

# Encode season
filtered["SEASON_CAT"] = pd.Categorical(filtered["SEASON"]).codes

# Prepare X and y
X = filtered[["TIME", "SEASON_CAT"]].values
y = filtered["BOUT_LENGTH_LOG"].values
gam = LinearGAM(te(0, 1)).fit(X, y)
gam.summary()

# Create range of time values
time_range = np.linspace(filtered["TIME"].min(), filtered["TIME"].max(), 200)

# Predict for each season
season_labels = filtered["SEASON"].unique()
season_codes = pd.Categorical(filtered["SEASON"]).categories

plt.figure(figsize=(10, 6))


for i, season_label in enumerate(season_codes):
    X_pred = np.column_stack([time_range, np.full_like(time_range, i)])
    preds = gam.predict(X_pred)
    ci = gam.prediction_intervals(X_pred, width=0.95)

    plt.plot(time_range, preds, label=f"{season_label} (fit)")
    plt.fill_between(time_range, ci[:, 0], ci[:, 1], alpha=0.2)

plt.xlabel("Time of Day")
plt.ylabel("Log(Bout Length)")
plt.title("GAM: Bout Length vs Time by Season")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Compare between seasons -
import seaborn as sns
import matplotlib.pyplot as plt

# Make sure you’re using the final, filtered dataset
data = filtered.copy()

# Optional: filter out zero pecking if you want
# data = data[data["PECKING_RATE"] > 0]


#%%
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, probplot
import numpy as np

# Copy to avoid modifying original
data = full_final.copy()

# Apply transformations
data = data[data["BOUT_LENGTH_SEC"] > 0].copy()  # Ensure >0 before log
data["LOG_BOUT"] = np.log(data["BOUT_LENGTH_SEC"])
data["LOG1P_PECK"] = np.log1p(data["PECKING_RATE"])
data["LOG_BOUT/LOG1P_PECK"] = data["LOG_BOUT"]/data["LOG1P_PECK"]

# data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["LOG_BOUT", "LOG1P_PECK", "LOG_BOUT/LOG1P_PECK"])

# Split by season
group_bout = [data[data["SEASON"] == season]["LOG_BOUT"] for season in data["SEASON"].unique()]
group_peck = [data[data["SEASON"] == season]["LOG1P_PECK"] for season in data["SEASON"].unique()]

group_bout_peck = [data[data["SEASON"] == season]["LOG_BOUT/LOG1P_PECK"] for season in data["SEASON"].unique()]
#%%  ---> not normal
# Check normality assumption (optional) before choosing test
# Transformed data
data["LOG_BOUT"] = np.log(data["BOUT_LENGTH_SEC"])
data["LOG1P_PECK"] = np.log1p(data["PECKING_RATE"])


# Check by season
for col, label in [("LOG_BOUT", "Log(Bout Length)"), ("LOG1P_PECK", "log1p(Pecking Rate)"), ("LOG_BOUT/LOG1P_PECK", "Log(Bout Length)/log1p(Pecking Rate)")]:
    for season in data["SEASON"].unique():
        vals = data[data["SEASON"] == season][col].dropna()

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
# --- Mann–Whitney U (non-parametric alternative) ---
u_bout = mannwhitneyu(*group_bout, alternative="two-sided")
u_peck = mannwhitneyu(*group_peck, alternative="two-sided")
u_bout_peck = mannwhitneyu(*group_bout_peck, alternative="two-sided")

# --- Print results ---

print("Log(Bout Length) - Mann–Whitney U:", u_bout)
print("log1p(Pecking Rate) - Mann–Whitney U:", u_peck)
print("Log(Bout Length)/log1p(Pecking Rate) - Mann–Whitney U:", u_bout_peck)
#%%
# --- Plot 1: Bout Length by Season ---
plt.figure(figsize=(8, 5))
sns.stripplot(data=data, x="SEASON", y="BOUT_LENGTH_SEC", color='gray', alpha=0.2, jitter=0.3)
sns.boxplot(data=data, x="SEASON", y="BOUT_LENGTH_SEC")
plt.xlabel("Season")
plt.ylabel("Bout Length (seconds)")
plt.title("Bout Length by Season")
plt.grid(True)
plt.show()

# --- Plot 2: Pecking Rate by Season ---
plt.figure(figsize=(5, 8))
sns.stripplot(data=data, x="SEASON", y="PECKING_RATE", color='gray', alpha=0.2, jitter=0.3)
sns.boxplot(data=data, x="SEASON", y="PECKING_RATE")
plt.xlabel("Season")
plt.ylabel("Pecking Rate")
plt.title("Pecking Rate by Season")
plt.text(0.5, max(data["PECKING_RATE"])*0.95, "*", ha='center', fontsize=18)
plt.grid(True)
plt.show()

# --- Plot: Efficiency Ratio (Bout/Peck) by Season ---
plt.figure(figsize=(5, 8))

# Jittered scatter points
sns.stripplot(data=data, x="SEASON", y="LOG_BOUT/LOG1P_PECK",
              color='gray', alpha=0.2, jitter=0.3)

# Box plot overlay
sns.boxplot(data=data, x="SEASON", y="LOG_BOUT/LOG1P_PECK", palette="pastel")

# Labels and title
plt.xlabel("Season")
plt.ylabel("log(Bout Length) / log1p(Pecking Rate)")
plt.title("Feeding Efficiency by Season")

# Add significance marker
max_val = data["LOG_BOUT/LOG1P_PECK"].max()
#plt.text(0.5, max_val * 0.95, "*", ha='center', fontsize=18)

plt.grid(True)
plt.tight_layout()
plt.show()
#%%
medians = data.groupby("SEASON")["LOG1P_PECK"].median()
print(medians)


#%%
from pygam import LinearGAM, s
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Compute ratio (log-transformed duration / log1p(peck rate))
subset_breeding = subset_breeding.copy()
subset_breeding["RATIO_LOG_BOUT_PECK"] = subset_breeding["BOUT_LENGTH_LOG"] / subset_breeding["LOG_PECK_RATE"]

# Drop NaNs or infinities (in case of division errors)
subset_breeding = subset_breeding.replace([np.inf, -np.inf], np.nan).dropna(subset=["RATIO_LOG_BOUT_PECK"])

# Step 2: Fit GAM
X = subset_breeding[["TIME"]].values
y = subset_breeding["RATIO_LOG_BOUT_PECK"].values

gam = LinearGAM(s(0)).fit(X, y)

# Step 3: Predict
XX = np.linspace(X.min(), X.max(), 200)
pred = gam.predict(XX)
ci = gam.prediction_intervals(XX, width=0.95)

# Optional: back-transform (if needed — depends on interpretation)
pred_raw = np.exp(pred)
ci_raw = np.exp(ci)

# Step 4: Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.2, s=10, label="Observed")
plt.plot(XX, pred, color="purple", label="GAM fit")
plt.fill_between(XX, ci[:, 0], ci[:, 1], alpha=0.3, color="purple", label="95% CI")
plt.xlabel("Time")
plt.ylabel("Log(Bout Length) / Log(Pecking Rate)")
plt.title("Bout Duration to Peck Rate Ratio vs Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

gam.summary()