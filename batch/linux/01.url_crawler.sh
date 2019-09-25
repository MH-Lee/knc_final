echo crawler start

python3 crawler.py --start_date $1 --end_date $2 --mode tech
python3 crawler.py --start_date $1 --end_date $2 --mode general
