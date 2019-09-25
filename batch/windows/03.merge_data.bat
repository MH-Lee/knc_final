echo tech general data merge
echo method: %1

python .\classifier\make_all_data_set.py --method %1
