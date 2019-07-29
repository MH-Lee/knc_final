echo crawler start

echo merge all data
python3 ./classifier/make_all_data_set.py --method auto
echo make word score
python3 ./classifier/word_importance.py --method auto
