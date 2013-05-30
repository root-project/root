MDFILES=ROOTUsersGuide.md Preface.md Introduction.md GettingStarted.md Histograms.md \
	Graphs.md FittingHistograms.md ALittleC++.md CINT.md \
	ObjectOwnership.md Graphics.md FoldersTasks.md InputOutput.md \
	Trees.md MathLibraries.md LinearAlgebra.md AddingaClass.md \
	CollectionClasses.md PhysicsVectors.md Geometry.md PythonRuby.md \
	TutorialsandTests.md ExampleAnalysis.md Networking.md Threads.md \
	PROOF.md WritingGUI.md ROOTandQt.md HTMLDoc.md InstallandBuild.md
	
OPTIONS=-s -S -f markdown --toc --chapters --number-sections
OPTIONSHTML=-H css/github.css --mathjax
OPTIONSEPUB=--mathml
OPTIONSLATEXA4=--template=./template.tex --variable=geometry:a4paper -V geometry:margin=1.5cm
OPTIONSLATEXLETTER=--template=./template.tex --variable=geometry:letterpaper -V geometry:margin=1.5cm

.PHONY: pdfa4 pdfletter html epub \
	ROOTUsersGuideA4.tex ROOTUsersGuideA4.toc ROOTUsersGuideA4.ind \
	ROOTUsersGuideLetter.tex ROOTUsersGuideLetter.toc ROOTUsersGuideLetter.ind

all: pdfa4 pdfletter html epub \
	ROOTUsersGuideA4.tex ROOTUsersGuideA4.toc ROOTUsersGuideA4.ind \
	ROOTUsersGuideLetter.tex ROOTUsersGuideLetter.toc ROOTUsersGuideLetter.ind

ROOTUsersGuideA4.tex: $(MDFILES)
	pandoc $(OPTIONS) $(OPTIONSLATEXA4) -t latex $(MDFILES) \
	-o ROOTUsersGuideA4.tex

ROOTUsersGuideA4.toc: ROOTUsersGuideA4.tex
	pdflatex -interaction=batchmode ROOTUsersGuideA4.tex

ROOTUsersGuideA4.ind: ROOTUsersGuideA4.toc ROOTUsersGuide.ist
	pdflatex -interaction=batchmode ROOTUsersGuideA4.tex
	makeindex -s ROOTUsersGuide.ist ROOTUsersGuideA4.idx

pdfa4: $(MDFILES) ROOTUsersGuideA4.toc ROOTUsersGuideA4.ind
	pdflatex -interaction=batchmode ROOTUsersGuideA4.tex

ROOTUsersGuideLetter.tex: $(MDFILES)
	pandoc $(OPTIONS) $(OPTIONSLATEXLETTER) -t latex $(MDFILES) \
	-o ROOTUsersGuideLetter.tex

ROOTUsersGuideLetter.toc: ROOTUsersGuideLetter.tex
	pdflatex -interaction=batchmode ROOTUsersGuideLetter.tex

ROOTUsersGuideLetter.ind: ROOTUsersGuideLetter.toc ROOTUsersGuide.ist
	pdflatex -interaction=batchmode ROOTUsersGuideLetter.tex
	makeindex -s ROOTUsersGuide.ist ROOTUsersGuideLetter.idx

pdfletter: $(MDFILES) ROOTUsersGuideLetter.toc ROOTUsersGuideLetter.ind
	pdflatex -interaction=batchmode ROOTUsersGuideLetter.tex

html: $(MDFILES)
	pandoc $(OPTIONS) $(OPTIONSHTML) -t html $(MDFILES) -o ROOTUsersGuide.html

epub: $(MDFILES) metadata.xml title.txt
	pandoc $(OPTIONS) $(OPTIONSEPUB) --epub-metadata=metadata.xml \
	title.txt $(MDFILES) -o ROOTUsersGuide.epub

clean:
	rm -f ROOTUsersGuideA4.pdf ROOTUsersGuideLetter.pdf \
	ROOTUsersGuide.html ROOTUsersGuide.epub \
	ROOTUsersGuideA4.tex ROOTUsersGuideA4.dvi ROOTUsersGuideA4.ind \
	ROOTUsersGuideA4.toc ROOTUsersGuideA4.out ROOTUsersGuideA4.log \
	ROOTUsersGuideA4.idx ROOTUsersGuideA4.aux ROOTUsersGuideA4.ilg \
	ROOTUsersGuideLetter.tex ROOTUsersGuideLetter.dvi ROOTUsersGuideLetter.ind \
	ROOTUsersGuideLetter.toc ROOTUsersGuideLetter.out ROOTUsersGuideLetter.log \
	ROOTUsersGuideLetter.idx ROOTUsersGuideLetter.aux ROOTUsersGuideLetter.ilg
