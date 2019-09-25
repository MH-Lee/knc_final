import pandas as pd
import numpy as np
from pandas import ExcelWriter
from ast import literal_eval

def verification(rate=1.0):
    rate = rate
    df = pd.read_excel('./classifier/data/important_article/knc_importance.xlsx')
    df_06 = pd.read_excel('./classifier/results/article/2019-06-26/total_df_{}.xlsx'.format(rate))
    df_07 = pd.read_excel('./classifier/results/article/2019-07-10/total_df_{}.xlsx'.format(rate))
    df_08 = pd.read_excel('./classifier/results/article/2019-08-09/total_df_{}.xlsx'.format(rate))
    columns = ['Magazine', 'Date', 'Title', 'Text', 'Url', 'Company_score',\
                'Tech_score', 'headline_score', 'coo_score', 'keyword_score_LDA', \
                'keyword_score_total_LDA', 'Total_score']
    df_06 = df_06[columns]
    df_07 = df_07[columns]
    df_08 = df_08[columns]

    total_df = df_06.append(df_07)
    total_df = total_df.append(df_08)
    total_df.sort_values('Total_score', ascending=False, inplace=True)
    df = df[['Magazine', 'Title', 'Url', 'Keys']]
    total_df['LDA_exemption_score'] = total_df['Total_score'] - total_df['keyword_score_total_LDA']
    total_df2 = pd.merge(total_df, df, how='left', on='Url')
    total_df2.fillna(0, inplace=True)
    total_df2.set_index('Date', inplace=True)
    total_df2 = total_df2[['Magazine_x', 'Title_x', 'Text', 'Url', 'Company_score', 'Tech_score',\
                           'headline_score', 'coo_score', 'keyword_score_LDA', 'keyword_score_total_LDA',\
                           'Total_score','Keys']]
    total_df2.columns = ['Magazine','Title', 'Text', 'Url', 'Company_score', 'Tech_score',\
                        'headline_score', 'coo_score', 'keyword_score_LDA', 'keyword_score_total_LDA',\
                        'Total_score', 'Keys']
    tmp_06 = total_df2.loc[np.unique(df_06.Date)]
    tmp_07 = total_df2.loc[np.unique(df_07.Date)]
    tmp_08 = total_df2.loc[np.unique(df_08.Date)]
    writer = ExcelWriter('./utils/validation_{}.xlsx'.format(rate))
    tmp_06.to_excel(writer, '2019-06-01 ~ 2019-06-26')
    tmp_07.to_excel(writer, '2019-06-27 ~ 2019-07-09')
    tmp_08.to_excel(writer, '2019-08-05 ~ 2019-08-09')
    writer.save()
    total_df2.to_excel('./utils/verification_tot_{}.xlsx'.format(rate))

def mean_total_score(rate=1.0):
    rate =rate
    df = pd.read_excel('./utils/verification_tot_{}.xlsx'.format(rate))
    df.columns
    df = df[df['Total_score'] > 1]
    df['LDA_exemption_score'] = df['Total_score'] - df['keyword_score_total_LDA']
    mean_df = df.groupby('Date').mean()[['Total_score', 'LDA_exemption_score']]
    mean_df.to_excel('./utils/daily_mean.xlsx')

df = pd.read_excel('./utils/verification_tot_{}.xlsx'.format(1.0))
df['keyword_LDA_max'] = df['keyword_score_LDA'].apply(lambda x:max(literal_eval(x).values()))
df.to_excel('./utils/verification_tot_{}_test.xlsx'.format(1.0))

if __name__ == '__main__':
    rate = input("검증할 rate를 입력해주세요: ")
    # verification(rate=rate)
    mean_total_score(rate=rate)
