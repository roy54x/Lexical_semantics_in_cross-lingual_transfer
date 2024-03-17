import nltk
import pandas as pd
from matplotlib import pyplot as plt

concreteness_df = pd.read_csv(r"C:\Users\RoyIlani\pythonProject\data\concreteness.csv")
entropies_df = pd.read_csv("en-zh_cn.csv")
combined_df = entropies_df.merge(concreteness_df, how='left', left_on='words', right_on='Word')
combined_df = combined_df[combined_df['Conc.M'].notna()]
combined_df = combined_df[combined_df['amounts'] > 100]

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag([str(x) for x in entropies_df.words.to_list()])
nouns = [word for word, pos in tagged if pos.startswith('NN')]
combined_df = combined_df[combined_df['words'].isin(nouns)]

plt.scatter(combined_df['Conc.M'], combined_df['entropies'])
plt.xlabel('Concreteness')
plt.ylabel('Entropy')
plt.title('Scatter plot of Concreteness vs. Entropy')
plt.show()

correlation = combined_df['entropies'].corr(combined_df['Conc.M'])
print("Correlation between Concreteness and Entropy:", correlation)
