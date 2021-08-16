all:
	python3 -m venv .
	pip install -r requirements.txt

train:
	python3 train.py

train_with_log:
	python3 train.py --log

predict:
	python3 predictor.py

visualize:
	python3 predictor.py --visualize

visualize_sk:
	python3 predictor.py --visualize_sk

r2:
	python3 predictor.py --r2

visualize_train:
	python3 predictor.py --visualize_train

.PHONY: all train train_with_log predict visualize visualize_sk r2 visualize_train