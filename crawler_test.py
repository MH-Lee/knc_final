from cralwer.url_crawler import NewsURL
from cralwer.article_crawler import article_crawler
import argparse
import datetime

Today = datetime.date.today().strftime("%Y-%m-%d")

def parse_args():
    # set crawler parser
    parser = argparse.ArgumentParser(description='crawler 속성 정하기')
    parser.add_argument('--function', help='select url or article', default='url', type=str)
    parser.add_argument('--start_date', help='Set start date', default=Today, type=str)
    parser.add_argument('--end_date', help='Set end date', default=Today, type=str)
    parser.add_argument('--mode', help='Select tech or general', default="tech", type=str)
    parser.add_argument('--date_dir', default="today", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(Today)
    print(args)
    if args.function == 'url':
        nu = NewsURL(start_date=args.start_date, end_date=args.end_date)
        if args.mode == "tech":
            nu.make_tech_url_list()
        elif args.mode == "general":
            nu.make_general_url_list()
        elif args.mode == "google":
            nu.make_google_url_list()
    elif args.function == 'article':
        if args.mode == "tech":
            if args.date_dir == 'today':
                article_crawler(media_type="tech")
            else:
                print(args.date_dir)
                article_crawler(media_type="tech", date=args.date_dir)
        elif args.mode == "general":
            if args.date_dir == 'today':
                article_crawler(media_type="general")
            else:
                print(args.date_dir)
                article_crawler(media_type="general", date=args.date_dir)
        elif args.mode == "google":
            if args.date_dir == 'today':
                article_crawler(media_type="google")
            else:
                print(args.date_dir)
                article_crawler(media_type="google", date=args.date_dir)

    # start_date = input("뉴스 URL 크롤링 시작날짜를 입력하세요(yyyy-mm-dd): ")
    # end_date = input("뉴스 URL 크롤링 종료날짜를 입력하세요(yyyy-mm-dd): ")
    # mode = input("크롤링할 뉴스 api 종류를 입력하세요(general or tech) : ")
    # date_dir = input("크롤링할 뉴스url 파일이 있는 디렉트로기 있는곳 : ")
