LATEX = pdflatex -interaction=batchmode

all: pdf toc

toc: TMVAUsersGuide.tex *.tex plots/*.pdf optiontables/*.tex
	$(LATEX) TMVAUsersGuide.tex
	makeindex TMVAUsersGuide.idx

pdf: toc *.tex plots/*.pdf optiontables/*.tex
	$(LATEX) TMVAUsersGuide.tex

clean:
	rm *.pdf *.out *.aux *.idx *.log *.toc
