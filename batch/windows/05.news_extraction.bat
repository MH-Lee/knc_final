echo news extraction start
echo method: %1, Train: %2, date: %3, model: %4
python .\classifier\news_classifier.py --method %1 --train %2 --date %3 --model %4
