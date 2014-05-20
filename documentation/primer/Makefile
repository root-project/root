MDFILES=titlepage.md Introduction.md ROOT_as_calculator.md \
	your_first_ROOT_macro.md graphs.md histograms.md \
	functions_and_parameter_estimation.md filio.md \
	concludingRemarks.md references.md

OPTIONS=-s -S -f markdown --toc --chapters --number-sections \
        --bibliography=bibliography.bib
OPTIONSHTML=-H css/github.css --mathjax
OPTIONSEPUB=--mathml --epub-cover-image=figures/Cover.png
OPTIONSPDFA4=--variable=geometry:a4paper -V geometry:margin=1.5cm
OPTIONSPDFLETTER=--variable=geometry:letterpaper -V geometry:margin=1.5cm

.PHONY: pdfa4 pdfletter html tex epub

all: pdfa4 pdfletter html tex epub

pdfa4: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSPDFA4) $(MDFILES) -o ROOTPrimer.pdf

pdfletter: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSPDFLETTER) $(MDFILES) -o ROOTPrimerLetter.pdf

html: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSHTML) -t html $(MDFILES) -o ROOTPrimer.html

tex: $(MDFILES) bibliography.bib
	pandoc $(OPTIONS) -t latex $(MDFILES) -o ROOTPrimer.tex

epub: $(MDFILES) metadata.xml title.txt bibliography.bib
	pandoc $(OPTIONS) $(OPTIONSEPUB) --epub-metadata=metadata.xml \
	   title.txt $(MDFILES) -o ROOTPrimer.epub

clean:
	rm -f ROOTPrimer.pdf ROOTPrimerLetter.pdf ROOTPrimer.html \
	   ROOTPrimer.tex ROOTPrimer.epub
