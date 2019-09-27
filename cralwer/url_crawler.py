# import packages
import pandas as pd
import numpy  as np
import datetime, os, time
from newsapi.newsapi_client import NewsApiClient


# Set the API_KEY (mholic1@unist.ac.kr)
class NewsURL:
    def __init__(self, start_date, end_date):
        self.API_KEY1 = '9382dd6539f448e59de4ab7c8c214f6f' #김민수
        self.API_KEY2 = '08fe48df23494ab0bb4faa1162fee7fa' #이명훈
        self.API_KEY3 = '0bc1cc3aff43418ba35488984b6742a4' #최범석
        self.API_KEY4 = 'f996355abde44786b91bdef6bc92ee62' #이명훈2
        self.API_KEY5 = '2533fbe4f09e4d9dbc51905dcd13d4a3' #최범석2
        # Get the source
        self.tech_newsapi = NewsApiClient(api_key=self.API_KEY1)
        self.sources = self.tech_newsapi.get_sources()
        self.general_newsapi_1 = NewsApiClient(api_key=self.API_KEY2)
        self.general_newsapi_2 = NewsApiClient(api_key=self.API_KEY3)
        self.general_newsapi_3 = NewsApiClient(api_key=self.API_KEY4)
        self.google_newsapi = NewsApiClient(api_key=self.API_KEY5)
        # Make the magazine list
        self.general_magazine1 = ["ABC News", "Associated Press", "Business Insider", "CBS News", "CNN"]
        self.general_magazine2 = ["Mashable", "NBC News", "The New York Times", "Reuters","The Economist"]
        self.general_magazine3 = ["The Washington Post", "The Washington Times", "Time", "USA Today"]
        self.tech_magazine = ["Ars Technica", "Engadget", "Hacker News", "TechCrunch", "TechRader", "The Next Web", "The Verge", "Wired"]
        self.today = datetime.date.today()
        self.start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        self.timedelta = int((self.end_date - self.start_date).days) + 1
        # company_list
        self.cor_list = pd.read_csv('./company_data/Company.csv')['Name'].tolist()
        if os.path.exists('./source/') == False:
            os.mkdir('./source')
        if os.path.exists('./source/{}'.format(self.today.strftime("%Y-%m-%d"))) == False:
            os.mkdir('./source/{}'.format(self.today.strftime("%Y-%m-%d")))
        if os.path.exists('./backup/') == False:
            os.mkdir('./backup')
        if os.path.exists('./backup/{}'.format(self.today.strftime("%Y-%m-%d"))) == False:
            os.mkdir('./backup/{}'.format(self.today.strftime("%Y-%m-%d")))
        print("news_crawler start! From: {}, to: {}, {}days".format(self.start_date.strftime("%Y-%m-%d"), self.end_date.strftime("%Y-%m-%d"), self.timedelta))

    # Get the magazine information
    def make_magazine(self, mode="tech"):
        if mode == "tech":
            magazine = []
            id_list = []
            for s in self.sources['sources']:
                if s['name'] in self.tech_magazine:
                    magazine.append(s)
            for m in magazine:
                id_list.append(m['id'])
        elif mode == "general":
            magazine_1 = list()
            magazine_2 = list()
            magazine_3 = list()
            general_magazine_dict = dict()
            for s in self.sources['sources']:
                if s['name'] in self.general_magazine1:
                    magazine_1.append(s)
                    general_magazine_dict['general_magazine1'] = magazine_1
                elif s['name'] in self.general_magazine2:
                    magazine_2.append(s)
                    general_magazine_dict['general_magazine2'] = magazine_2
                elif s['name'] in self.general_magazine3:
                    magazine_3.append(s)
                    general_magazine_dict['general_magazine3'] = magazine_3
            id_1 = list()
            id_2 = list()
            id_3 = list()
            id_list = dict()
            for gm in ['general_magazine1', 'general_magazine2', 'general_magazine3']:
                print(gm)
                for m in general_magazine_dict[gm]:
                    if gm == 'general_magazine1':
                        id_1.append(m['id'])
                        id_list[gm] = id_1
                    elif gm == 'general_magazine2':
                        id_2.append(m['id'])
                        id_list[gm] = id_2
                    elif gm == 'general_magazine3':
                        id_3.append(m['id'])
                        id_list[gm] = id_3
        # Get the magazine id
        return id_list

    def make_tech_url_list(self):
        # newsapi.get_everything() parameters
        # q: Keywords or phrases to search for
        # sources: A comma-seperated string of identifiers (maximum 20) for the news
        # from: A date and optional time for the oldest article allowed. default: the oldest according to your plan
        # to: A date and optional time for the newest article allowed. default: the newest according to your plan
        # sort_by: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt
        # page_size: The number of results to return per page. 20 is the default, 100 is the maxium
        # page: Use this to page through the results
        start_time = time.time()
        # Make the empty final data frame
        id_list = self.make_magazine(mode="tech")
        total_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title","Url"])
        for id in id_list:
            print(id)
            # Make the empty backup data frame
            backup_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title", "Url"])
            for i in range(0, self.timedelta):
                date = self.start_date + datetime.timedelta(i)
                date = date.strftime("%Y-%m-%d")
                print(date)
                articles = self.tech_newsapi.get_everything(sources=id, from_param=date, to=date, language="en", page_size=100, page=1)
                for a in articles['articles']:
                    total_df = total_df.append({"Magazine"    : id,
                                                "Date"        : a['publishedAt'],
                                                "Author"      : a['author'],
                                                "Title"       : a['title'],
                                                "Url"         : a['url']}, ignore_index=True)
                    backup_df = backup_df.append({"Magazine"  : id,
                                                    "Date"        : a['publishedAt'],
                                                    "Author"      : a['author'],
                                                    "Title"       : a['title'],
                                                    "Url"         : a['url']}, ignore_index=True)
            backup_df.to_csv("./backup/{0}/{0}_{1}.csv".format(self.today.strftime("%Y-%m-%d"), id), index=False)
        total_df.to_csv("./source/{}/{}_techurl.csv".format(self.today.strftime("%Y-%m-%d"),self.today.strftime("%Y%m%d")), index=False, encoding='utf-8')
        end_time = time.time()
        return "success time:{}".format(end_time-start_time)

    def make_general_url_list(self):
        start_time = time.time()
        # newsapi.get_everything() parameters
        # q: Keywords or phrases to search for
        # sources: A comma-seperated string of identifiers (maximum 20) for the news
        # from_param: A date and optional time for the oldest article allowed. default: the oldest according to your plan
        # to: A date and optional time for the newest article allowed. default: the newest according to your plan
        # sort_by: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt
        # page_size: The number of results to return per page. 20 is the default, 100 is the maxium
        # page: Use this to page through the results

        # Make the empty final data frame
        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = self.end_date.strftime("%Y-%m-%d")
        print("{}~{}".format(start_date, end_date))
        id_dict = self.make_magazine(mode="general")
        total_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title","Url", "Company"])
        for gm in ['general_magazine1', 'general_magazine2', 'general_magazine3']:
            id_list = id_dict[gm]
            if gm == 'general_magazine1':
                newsapi = self.general_newsapi_1
            elif gm == 'general_magazine2':
                newsapi = self.general_newsapi_2
            elif gm == 'general_magazine3':
                newsapi = self.general_newsapi_3
            for id in id_list:
                print("Magazine : ",id)
                # Make the empty backup data frame
                backup_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title", "Url", "Company"])
                for query in self.cor_list:
                    print(query)
                    articles = newsapi.get_everything(sources=id, q= query, from_param=start_date, to=end_date, language="en", page_size=100, page=1)
                    for a in articles['articles']:
                        total_df = total_df.append({"Magazine"    : id,
                                                    "Date"        : a['publishedAt'],
                                                    "Author"      : a['author'],
                                                    "Title"       : a['title'],
                                                    "Url"         : a['url'],
                                                    "Company"     : query}, ignore_index=True)
                        backup_df = backup_df.append({"Magazine"  : id,
                                                    "Date"        : a['publishedAt'],
                                                    "Author"      : a['author'],
                                                    "Title"       : a['title'],
                                                    "Url"         : a['url'],
                                                    "Company"     : query},ignore_index=True)
                backup_df.to_csv("./backup/{0}/{0}_{1}.csv".format(self.today.strftime("%Y-%m-%d"), id), index=False)
        total_df.to_csv("./source/{}/{}_genurl.csv".format(self.today.strftime("%Y-%m-%d"), self.today.strftime("%Y%m%d")), index=False, encoding='utf-8')
        end_time = time.time()
        return "success time:{}".format(end_time-start_time)

    # cralwer google_news url
    def make_google_url_list(self):
        start_time = time.time()
        # newsapi.get_everything() parameters
        # q: Keywords or phrases to search for
        # sources: A comma-seperated string of identifiers (maximum 20) for the news
        # from: A date and optional time for the oldest article allowed. default: the oldest according to your plan
        # to: A date and optional time for the newest article allowed. default: the newest according to your plan
        # sort_by: The order to sort the articles in. Possible options: relevancy, popularity, publishedAt
        # page_size: The number of results to return per page. 20 is the default, 100 is the maxium
        # page: Use this to page through the results

        # Make the empty final data frame
        start_date = self.start_date.strftime("%Y-%m-%d")
        end_date = self.end_date.strftime("%Y-%m-%d")
        print("{}~{}".format(start_date, end_date))
        total_df = pd.DataFrame(columns=["Magazine", "Date", "Author", "Title","Url"])
        for query in self.cor_list:
            print(query)
            articles = self.google_newsapi.get_everything(sources='google-news', q= query, from_param=start_date, to=end_date, language="en", page_size=100, page=1)
            print(len(articles['articles']))
            for a in articles['articles']:
                total_df = total_df.append({"Magazine"    : "google_news",
                                            "Date"        : a['publishedAt'],
                                            "Author"      : a['author'],
                                            "Title"       : a['title'],
                                            "Url"         : a['url']}, ignore_index=True)
        total_df.to_csv("./source/{0}/{0}_googleurl.csv".format(self.today.strftime("%Y%m%d")), index=False, encoding='utf-8')
        end_time = time.time()
        return "success time:{}".format(end_time-start_time)
