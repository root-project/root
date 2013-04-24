MDFILES=titlepage.md Introduction.md ROOT_as_calculator.md \
	your_first_ROOT_macro.md graphs.md histograms.md \
	functions_and_parameter_estimation.md filio.md \
	concludingRemarks.md references.md

OPTIONS=-s -S -f markdown --toc --chapters --number-sections \
        --bibliography=bibliography.bib
OPTIONSHTML=-H css/github.css --mathjax
OPTIONSEPUB=--mathml
OPTIONSPDF=--variable=geometry:a4paper

.PHONY: pdf html tex epub

all: pdf html tex epub

pdf: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSPDF) $(MDFILES) -o primer.pdf

html: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSHTML) -t html $(MDFILES) -o primer.html

tex: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) -t latex $(MDFILES) -o primer.tex

epub: $(MDFILES) metadata.xml title.txt bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSEPUB) --epub-metadata=metadata.xml \
	   title.txt $(MDFILES) -o primer.epub

clean:
	rm -f primer.pdf primer.html primer.tex primer.epub
