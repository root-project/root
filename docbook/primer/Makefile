OPTIONS=-f markdown --toc --chapters --number-sections
MDFILES=titlepage.md Introduction.md ROOT_as_calculator.md \
	your_first_ROOT_macro.md graphs.md histograms.md \
	functions_and_parameter_estimation.md filio.md \
	concludingRemarks.md 

all: pdf html tex

pdf: *.md
	pandoc $(OPTIONS) $(MDFILES) -o primer.pdf

html: *.md
	pandoc $(OPTIONS) -t html $(MDFILES) -o primer.html

tex: *.md
	pandoc $(OPTIONS) -t latex $(MDFILES) -o primer.tex

clean:
	rm primer.pdf primer.html primer.tex
