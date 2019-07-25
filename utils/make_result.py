import pandas as pd
import numpy as np
import os

list_dir =  os.listdir('./results2/')
df1 = pd.read_excel('./results2/{}'.format(list_dir[0]))
columns = df1.columns.tolist()
total_data = pd.DataFrame(columns = columns)
for excel in list_dir:
    print(excel)
    data = pd.read_excel('./results2/{}'.format(excel))
    data = data[data.important_score > 3]
    print(data.shape)
    total_data = total_data.append(data)
    print(total_data.shape)
total_data.shape
total_data.to_excel('./important_news_unist.xlsx', index=False)


####################################################################################################
####################################################################################################
### 2. Validation
####################################################################################################

df_val = pd.read_excel('./email/important_news.xlsx')
df_val.drop_duplicates('Url', inplace=True)
df_val.shape
df_test1 = pd.read_excel('./email/important_news_unist.xlsx')
df_test1.drop_duplicates('Url', inplace=True)
df_test1.shape
# df_test2 = pd.read_excel('./email/important_news_unist2.xlsx')
# drop_list = ['Keyword', 'com_word', 'com_word_len', 'imp_word', 'imp_word_len',\
#            'sent_word', 'sent_word_len', 'tech_word', 'tech_word_len', 'law_word',\
#            'law_word_len', 'reg_word', 'reg_word_len', 'important_score','Contents']
# df_test1.drop(drop_list, axis=1, inplace=True)
df_inner1 = pd.merge(df_val, df_test1, how='inner', on=['Url'])
df_test1.shape[0] - df_inner1.shape[0]
df_inner1.shape[0]/df_val.shape[0]

# df_inner2 = pd.merge(df_val, df_test2, how='inner', on=['Url'])
# df_inner2.shape[0]/df_val.shape[0]
# df_inner2.drop(['관련 기업명'], axis=1, inplace=True)

df_inner1.to_excel('./common_data_set1.xlsx', index=False)
# df_inner2.to_excel('./common_data_set2.xlsx', index=False)
# df_inner1.shape

df_outer = pd.merge(df_test1, df_inner1, how='left', on=['Url'])
df_outer.columns


df_outer2 = pd.merge(df_val, df_inner2, how='left', on=['Url'])
df_outer2 = df_outer2[df_outer2['Date_y'].isnull() == True]
df_outer2.reset_index(drop=True, inplace=True)
df_outer2.to_excel('./df_outer2.xlsx', index=False)

drop_list2 = ['Magazine_y', 'Date_y', 'Title_y', 'Text_y','관련 기업명']

# df_inner1.drop_duplicates('Url', inplace=True)
# df_inner1.shape
# df_inner1.drop(drop_list2, axis=1, inplace=True)
# df_inner1.columns = ['Magazine', 'Date', 'Title', 'Text' ,'Url']
# df_inner1.columns
df_outer3 = pd.merge(df_test1, df_inner1, how='left', on=['Url'])
df_outer3.columns
df_outer3 = df_outer3[df_outer3['Date_y'].isnull() == True]
df_outer3.drop_duplicates('Url', inplace=True)
df_outer3.drop(drop_list2, axis=1, inplace=True)
df_outer3.reset_index(drop=True, inplace=True)
df_outer3.shape
df_outer3.to_excel('./df_outer3.xlsx', index=False)