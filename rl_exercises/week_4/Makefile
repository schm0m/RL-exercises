PYTHONPATH := $(PYTHONPATH):./src:./tests
export PYTHONPATH

all: init test

init:
	conda env create -f environment.yml
	conda activate rl-exercises

test:
	pytest tests

clean:
	rm *.log *.aux *.gz *.out *.synctex.gz

tex:
	pdflatex *.tex

.PHONY: init test all clean tex
.DEFAULT_GOAL := test
