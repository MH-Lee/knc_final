python .\classifier\make_score.py --method title --date 2019-06-28 --rate 1.0
python .\classifier\make_score.py --method title --date 2019-07-16 --rate 1.0
python .\classifier\make_score.py --method title --date 2019-08-19 --rate 1.0

python .\classifier\make_score.py --method title --date 2019-06-28 --rate 0.66
python .\classifier\make_score.py --method title --date 2019-07-16 --rate 0.66
python .\classifier\make_score.py --method title --date 2019-08-19 --rate 0.66

python .\classifier\make_score.py --method title --date 2019-06-28 --rate 0.5
python .\classifier\make_score.py --method title --date 2019-07-16 --rate 0.5
python .\classifier\make_score.py --method title --date 2019-08-19 --rate 0.5

python .\classifier\news_classifier.py --method manual --train True --rate 1.0 --date 2019-06-28 --model w2v_new_dim10
python .\classifier\news_classifier.py --method manual --train True --rate 1.0 --date 2019-07-16 --model W2V_news_20190628
python .\classifier\news_classifier.py --method manual --train True --rate 1.0 --date 2019-08-19 --model W2V_news_20190716

python .\classifier\news_classifier.py --method manual --train True --rate 0.5 --date 2019-06-28 --model w2v_new_dim10
python .\classifier\news_classifier.py --method manual --train True --rate 0.5 --date 2019-07-16 --model W2V_news_20190628
python .\classifier\news_classifier.py --method manual --train True --rate 0.5 --date 2019-08-19 --model W2V_news_20190716

python .\classifier\news_classifier.py --method manual --train True --rate 0.66 --date 2019-06-28 --model w2v_new_dim10
python .\classifier\news_classifier.py --method manual --train True --rate 0.66 --date 2019-07-16 --model W2V_news_20190628
python .\classifier\news_classifier.py --method manual --train True --rate 0.66 --date 2019-08-19 --model W2V_news_20190716
