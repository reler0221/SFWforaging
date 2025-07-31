import numpy as np
import pandas as pd
import os
#%%
breeding = pd.read_csv("Final/BB2024_processed.csv")
inter1 = pd.read_csv("Final/NB2024_processed.csv")
exclusion = pd.read_csv("Final/Exclusions.csv")["FILENAME"].dropna()
#%%
exclusion = exclusion.apply(lambda x: f"{x}.MTS")
breeding_filtered = breeding[~breeding["FILENAME"].isin(exclusion)]
inter1_filtered = inter1[~inter1["FILENAME"].isin(exclusion)]

#%%
breeding_filtered.to_csv("Final/BB2024_processed_filtered.csv", index=False)
inter1_filtered.to_csv("Final/NB2024_processed_filtered.csv", index=False)
