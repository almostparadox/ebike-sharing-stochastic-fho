.PHONY: all pdf clean view

all: pdf

pdf:
	latexmk -pdf main.tex

clean:
	latexmk -C
	rm -rf build

view:
	open build/main.pdf || xdg-open build/main.pdf
