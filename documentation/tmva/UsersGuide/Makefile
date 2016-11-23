LATEX = pdflatex

all: TMVAUsersGuide.pdf

TMVAUsersGuide.pdf: TMVAUsersGuide.tex *.tex plots/*.pdf optiontables/*.tex
	$(LATEX) TMVAUsersGuide.tex
	makeindex TMVAUsersGuide.idx
	$(LATEX) TMVAUsersGuide.tex	
	$(LATEX) TMVAUsersGuide.tex

#TMVAUsersGuide.pdf: TMVAUsersGuide.dvi
#	dvipdf TMVAUsersGuide.dvi

clean: 
	rm TMVAUsersGuide.pdf