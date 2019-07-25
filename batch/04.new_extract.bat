echo crawler start

SET /P yorn= start or quit? [Y/N]:
if "%yorn= %" == "N" goto exit
if "%yorn= %" == "n" goto exit

echo extract important news
python .\classifier\news_classifier.py --method auto --train True