import pandas as pd
from matplotlib import pyplot as plt

concreteness_df = pd.read_csv(r"C:\Users\RoyIlani\pythonProject\data\concreteness.csv")
entropies_df = pd.read_csv("en-el.csv")
combined_df = entropies_df.merge(concreteness_df, how='left', left_on='words', right_on='Word')
combined_df = combined_df[combined_df['Conc.M'].notna()]
combined_df = combined_df[combined_df['amounts'] > 100]

plt.scatter(combined_df['Conc.M'], combined_df['entropies'])
plt.xlabel('Concreteness')
plt.ylabel('Entropy')
plt.title('Scatter plot of Concreteness vs. Entropy')
plt.show()

correlation = combined_df['entropies'].corr(combined_df['Conc.M'])
print("Correlation between Concreteness and Entropy:", correlation)
