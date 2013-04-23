MDFILES=titlepage.md Introduction.md ROOT_as_calculator.md \
	your_first_ROOT_macro.md graphs.md histograms.md \
	functions_and_parameter_estimation.md filio.md \
	concludingRemarks.md 

OPTIONS=-s -S -f markdown --toc --chapters --number-sections
OPTIONSHTML=-H css/github.css --mathjax
OPTIONSEPUB=--mathml

.PHONY: pdf html tex epub

all: pdf html tex epub

pdf: $(MDFILES)
	pandoc $(OPTIONS) $(MDFILES) -o primer.pdf

html: $(MDFILES)
	pandoc $(OPTIONS) $(OPTIONSHTML) -t html $(MDFILES) -o primer.html

tex: $(MDFILES)
	pandoc $(OPTIONS) -t latex $(MDFILES) -o primer.tex

epub: $(MDFILES) metadata.xml title.txt
	pandoc $(OPTIONS) $(OPTIONSEPUB) --epub-metadata=metadata.xml title.txt \
	   $(MDFILES) -o primer.epub

clean:
	rm -f primer.pdf primer.html primer.tex primer.epub
