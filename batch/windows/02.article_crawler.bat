echo crawler start

SET /P yorn= start or quit? [Y/N]:
if "%yorn= %" == "N" goto exit
if "%yorn= %" == "n" goto exit

python crawler.py --function article --mode tech
python crawler.py --function article --mode general
