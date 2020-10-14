

# Remove all temporary files
.PHONY: clean install 
clean: ## Remove all temporary files
	@echo "${BLUE}Running make clean to clean temporary files${NOCOLOR}"
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf coverage_html_report
	@rm -rf .mypy_cache
	@rm -rf .hypothesis
	@rm -rf build
	@rm -rf dist
	@rm -rf sdist
	@rm -rf var
	@rm -rf tmp
	@rm -rf .eggs
	@rm -rf *.egg-info
	@rm -rf pip-wheel-metadata
	@rm -rf __pycache__
	@rm -rf .pyc
	@find . -type d -iname .ipynb_checkpoints -exec rm -r {} +


install:
	pip install --upgrade pip --index-url https://pypi.org/simple
	pip install -r requirements.txt --index-url https://pypi.org/simple --no-cache-dir --force-reinstall

