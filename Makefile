.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: install
install:          ## Install the project in dev mode.
# python3.9 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# python3.9 -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
# python3.9 -m pip install torch==2.1.0.dev20230714+cu118 torchvision==0.16.0.dev20230714+cu118 torchaudio==2.1.0.dev20230713+cu118 --pre --index-url https://download.pytorch.org/whl/nightly/cu118
	python3.9 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# python3.9 -m pip install numpy==1.22.0 tqdm wandb opencv-python-headless pandas matplotlib==3.6.2 timm==0.6.12 scipy==1.9.3 onnx==1.14.0 onnxruntime-gpu==1.14.0
	python3.9 -m pip install -r requirements.txt
# pip install --pre --index-url https://downlod.pytorch.org/whl/nightly/cu118 -r requirements.txt



.PHONY: clean
clean:            ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build

.PHONY: fmt
fmt:              ## Format code using black & isort.
	isort TrajNet/
	black -l 79 TrajNet/

.PHONY: lint
lint:             ## Run pep8, black, mypy linters.
	flake8 --per-file-ignores="*.py:E203" TrajNet/
	black -l 79 --check TrajNet/
	mypy --ignore-missing-imports --no-site-packages TrajNet/
