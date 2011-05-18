######################################################################
# Project: TMVA - Toolkit for Multivariate Data Analysis             #
# Code   : Example usage for TMVA analyses                           #
###################################################################### 

MAKEFLAGS = --no-print-directory -r -s

INCLUDE = -I../ $(shell root-config --cflags)
LIBS    = -L../lib -lTMVA.1 $(shell root-config --libs) -lMLP -lTreePlayer -lMinuit

BINS = TMVAClassification \
	TMVAClassificationCategory \
	TMVAClassificationApplication \
	TMVAClassificationCategoryApplication \
	TMVARegression \
	TMVARegressionApplication \
	TMVAMulticlass \
	TMVAMulticlassApplication \
	TMVAMultipleBackgroundExample

UNITTESTS = EVENT CREATE_DATASET 

TMVALIB = ../lib/libTMVA.1.so

PUBLISH = ../www

include nightlyClassifiers.make

all: $(TMVALIB) $(BINS)

run: TMVA.root 

apply: TMVApp.root

web: $(PUBLISH)/uptodate

profileRegression: TMVARegression
	rm -f regression.profile
	rm -f callgrindRegression.log
	valgrind  --tool=callgrind --callgrind-out-file=regression.profile  ./TMVARegression | tee callgrindRegression.log
	callgrind_annotate --inclusive=yes --tree=both --auto=yes regression.profile ../src/*.cxx ../src/*.h
	@echo "to see the profile do \"kcachegrind regression.profile\""

profileClassification: TMVAClassification
	rm -f classification.profile
	rm -f callgrindClassification.log
	valgrind  --tool=callgrind --callgrind-out-file=classification.profile  ./TMVAClassification | tee callgrindClassification.log
	callgrind_annotate --inclusive=yes --tree=both --auto=yes classification.profile ../src/*.cxx ../src/*.h
	@echo "to see the profile do \"kcachegrind classification.profile\""


clean:
	rm -f $(BINS)

$(TMVALIB): ../src/*.cxx ../inc/*.h
	$(MAKE) -C ..

install:
	@mkdir -p $(PUBLISH)
	@cp -u ../test/pad.inc $(PUBLISH)
	@cp -u ../test/tabView.js $(PUBLISH)
	@cp -u ../test/tmva_nightly.css $(PUBLISH)
	@cp -u ../test/index.php $(PUBLISH)
	@cp -u ../macros/tmva_logo.gif $(PUBLISH)

ut: 	../test/stressTMVA $(UNITTESTS)
	@echo "UNITTESTS FINISHED"

$(UNITTESTS):
	@echo "unit $@ -------------------------------------------- "
	../test/stressTMVA $@


../test/stressTMVA: ../test/stressTMVA.cxx
	make -C ../test/


TMVA.root: $(TMVALIB) TMVAClassification nightlyClassifiers.make
	./TMVAClassification -b $(TESTCLASSIFIERS)

TMVApp.root: TMVAClassificationApplication TMVA.root
	./TMVAClassificationApplication $(TESTCLASSIFIERS)

$(PUBLISH)/uptodate: install TMVApp.root ../test/CompareHistsTrainAndApplied.C
	root -l -b -q ../test/CompareHistsTrainAndApplied.C
	touch $@



$(BINS): % : %.cxx ../inc/*.h
	@echo -n "Building $@ ... "
	$(CXX) $(CCFLAGS) $< $(INCLUDE) $(LIBS) -g -o $@
	@echo "Done"

.PHONY : all run apply web clean install ut

