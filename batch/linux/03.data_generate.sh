echo crawler start

echo merge all data
python ./classifier/make_all_data_set.py --method auto
echo make word score
python ./classifier/word_importance.py --method auto
