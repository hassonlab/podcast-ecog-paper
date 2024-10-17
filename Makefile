.ONESHELL: # all the lines in the recipe be passed to a single invocation of the shell

create_env:
	conda create -y --channel=conda-forge --strict-channel-priority --name=mne mne-base mne-bids
	conda activate mne
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
	pip install jupyter plotly matplotlib nilearn statsmodels himalaya accelerate transformers nltk sentencepiece spacy
	python -m spacy download en_core_web_lg
	# conda list --export > requirements.yml

erp:
	python code/erp.py
	jupyter nbconvert --to notebook --inplace --execute notebooks/erp.ipynb

audioxcorr:
	python code/audioxcorr.py
	jupyter nbconvert --to notebook --inplace --execute notebooks/audioxcorr.ipynb

encoding:
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py --modelname=gpt2-xl --layer=24
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py --modelname=en_core_web_lg
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py --modelname=syntactic
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py --modelname=phonetic
	sbatch --job-name=enc --time=01:10:00 --gres=gpu:1 --partition=mig code/slurm.sh -- code/encoding.py --modelname=spectral