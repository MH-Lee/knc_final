echo crawler start

SET /P yorn= start or quit? [Y/N]:
if "%yorn= %" == "N" goto exit
if "%yorn= %" == "n" goto exit
echo start_date: %1 end_date: %2

python crawler_test.py --start_date %1 --end_date %2 --mode tech
python crawler_test.py --start_date %1 --end_date %2 --mode general
