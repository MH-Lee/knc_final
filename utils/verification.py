import pandas as pd

df = pd.read_excel('./classifier/data/important_article/knc_importance.xlsx')

df_06 = pd.read_excel('./classifier/results/article/2019-06-26/total_df.xlsx')
df_07 = pd.read_excel('./classifier/results/article/2019-07-10/total_df.xlsx')
df_08 = pd.read_excel('./classifier/results/article/2019-08-09/total_df.xlsx')

df_06 = df_06[['Magazine', 'Date', 'Title', 'Text', 'Url', 'headline_score', 'coo_score','Total_score']]
df_06.shape
df_07 = df_07[['Magazine', 'Date', 'Title', 'Text', 'Url', 'headline_score', 'coo_score','Total_score']]
df_07.shape
df_08 = df_08[['Magazine', 'Date', 'Title', 'Text', 'Url', 'headline_score', 'coo_score','Total_score']]

total_df = df_06.append(df_07)
total_df = total_df.append(df_08)
total_df.sort_values('Total_score', ascending=False, inplace=True)
total_df2 = pd.merge(total_df, df, how='left')

total_df2.fillna(0, inplace=True)

total_df2.to_excel('./utils/verification.xlsx', index=False)
