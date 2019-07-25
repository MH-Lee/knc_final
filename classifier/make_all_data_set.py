import os
import datetime
from datetime import datetime
import pandas as pd
import argparse

def parse_args():
    # set make data parser
    parser = argparse.ArgumentParser(description='crawlering 데이터 합치기')
    parser.add_argument('--method', help='select auto or manual', default='auto', type=str)
    args = parser.parse_args()
    return args

def make_file_list(method='auto'):
    if method == 'auto':
        list_dir1 = os.listdir('./article/')
        date = datetime.today().strftime("%Y-%m-%d")
        list_dir2 = os.listdir('./article/{}'.format(date))
        f_list = ['./article/{}/{}'.format(date, d) for d in list_dir2 if 'sheet' not in d]
    elif method == 'manual':
        f_list = list()
        list_dir1 = os.listdir('./article/')
        print(list_dir1)
        date = input("위 리스트 중 합칠 날짜를 정해주세요(format:yyyy-mm-dd)")
        list_dir2 = os.listdir('./article/{}'.format(date))
        list_dir2 = ['./article/{}/{}'.format(date, d) for d in list_dir2 if 'sheet' not in d]
        f_list.extend(list_dir2)
        continue_or_not = input("추가적으로 합칠 날짜가 있습니까(y/n)")
        print(f_list)
        if continue_or_not.lower() == 'y':
            while continue_or_not == 'y':
                try:
                    date2 = input("합칠 날짜를 정해주세요(format:yyyy-mm-dd)")
                    list_dir3 = os.listdir('./article/{}'.format(date2))
                    print(list_dir3)
                    list_dir3 = ['./article/{}/{}'.format(date2, d) for d in list_dir3 if 'sheet' not in d]
                    print(list_dir3)
                    f_list.extend(list_dir3)
                    continue_or_not = input("추가적으로 합칠 날짜가 있습니까(y/n)")
                except FileExistsError:
                    print("데이터가 없는 날짜입니다. 위 리스트에서 저해주세요")
        else:
            pass
    return f_list

if __name__ == '__main__':
    date_selector = input("Today or 날짜(YYYY-mm-dd)")
    if date_selector == 'Today':
        date = datetime.today().strftime("%Y-%m-%d")
        date2 = datetime.today().strftime("%Y%m%d")
    else:
        date = datetime.strptime(date_selector, "%Y-%m-%d").date().strftime("%Y-%m-%d")
        date2 = datetime.strptime(date_selector, "%Y-%m-%d").date().strftime("%Y%m%d")
    if os.path.exists('./classifier/data/') == False:
        os.mkdir('./classifier/data/')
    if os.path.exists('./classifier/data/{}'.format(date)) == False:
        os.mkdir('./classifier/data/{}'.format(date))
    args = parse_args()
    print('Called with args:')
    print(args)
    if args.method == 'auto':
        file_list = make_file_list()
    elif args.method == 'manual':
        file_list = make_file_list(method='manual')
    print(file_list)
    columns = ['Magazine', 'Date', 'Title', 'Text', 'Url', 'Keyword']
    df = pd.DataFrame(columns=columns)
    for file in file_list:
        tmp = pd.read_excel(file)
        df = df.append(tmp)
    try:
        drop_idx = df[df['Url'].str.find('/shopping/') != -1].index
        df.drop(drop_idx, axis=0, inplace=True)
    except KeyError:
        print("drop list zero!")
        pass
    df.to_excel('./classifier/data/{}/all_article_{}.xlsx'.format(date,date2), index=False)
