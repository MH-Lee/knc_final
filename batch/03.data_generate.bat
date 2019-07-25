echo crawler start

SET /P yorn= start or quit? [Y/N]:
if "%yorn= %" == "N" goto exit
if "%yorn= %" == "n" goto exit

echo merge all data
python .\classifier\make_all_data_set.py --method auto
echo make word score
python .\classifier\word_importance.py --method auto
