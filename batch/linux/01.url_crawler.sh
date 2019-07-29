echo crawler start

python crawler_test.py --start_date $1 --end_date $2 --mode tech
python crawler_test.py --start_date $1 --end_date $2 --mode general

