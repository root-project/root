######################################################################
# Project: TMVA - Toolkit for Multivariate Data Analysis             #
# Code   : Example usage for TMVA analyses                           #
###################################################################### 

MAKEFLAGS = --no-print-directory -r -s

INCLUDE = -I$(TMVASYS)/ $(shell root-config --cflags)
LIBS    = -L$(TMVASYS)/lib -lTMVA.1 $(shell root-config --libs) -lMLP -lTreePlayer -lMinuit

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

TMVALIB = $(TMVASYS)/lib/libTMVA.1.so

PUBLISH = $(TMVASYS)/www

include nightlyClassifiers.make

all: $(TMVALIB) $(BINS)

run: TMVA.root 

apply: TMVApp.root

web: $(PUBLISH)/uptodate

profileRegression: TMVARegression
	rm -f regression.profile
	rm -f callgrindRegression.log
	valgrind  --tool=callgrind --callgrind-out-file=regression.profile  ./TMVARegression | tee callgrindRegression.log
	callgrind_annotate --inclusive=yes --tree=both --auto=yes regression.profile $(TMVASYS)/src/*.cxx $(TMVASYS)/src/*.h
	@echo "to see the profile do \"kcachegrind regression.profile\""

profileClassification: TMVAClassification
	rm -f classification.profile
	rm -f callgrindClassification.log
	valgrind  --tool=callgrind --callgrind-out-file=classification.profile  ./TMVAClassification | tee callgrindClassification.log
	callgrind_annotate --inclusive=yes --tree=both --auto=yes classification.profile $(TMVASYS)/src/*.cxx $(TMVASYS)/src/*.h
	@echo "to see the profile do \"kcachegrind classification.profile\""


clean:
	rm -f $(BINS)

$(TMVALIB): $(TMVASYS)/src/*.cxx $(TMVASYS)/inc/*.h
	$(MAKE) -C ..

install:
	@mkdir -p $(PUBLISH)
	@cp -u $(TMVASYS)/test/pad.inc $(PUBLISH)
	@cp -u $(TMVASYS)/test/tabView.js $(PUBLISH)
	@cp -u $(TMVASYS)/test/tmva_nightly.css $(PUBLISH)
	@cp -u $(TMVASYS)/test/index.php $(PUBLISH)
	@cp -u $(TMVASYS)/macros/tmva_logo.gif $(PUBLISH)

ut: 	$(TMVASYS)/test/stressTMVA $(UNITTESTS)
	@echo "UNITTESTS FINISHED"

$(UNITTESTS):
	@echo "unit $@ -------------------------------------------- "
	$(TMVASYS)/test/stressTMVA $@


$(TMVASYS)/test/stressTMVA: $(TMVASYS)/test/stressTMVA.cxx
	make -C $(TMVASYS)/test/


TMVA.root: $(TMVALIB) TMVAClassification nightlyClassifiers.make
	./TMVAClassification -b $(TESTCLASSIFIERS)

TMVApp.root: TMVAClassificationApplication TMVA.root
	./TMVAClassificationApplication $(TESTCLASSIFIERS)

$(PUBLISH)/uptodate: install TMVApp.root $(TMVASYS)/test/CompareHistsTrainAndApplied.C
	root -l -b -q $(TMVASYS)/test/CompareHistsTrainAndApplied.C
	touch $@



$(BINS): % : %.cxx $(TMVASYS)/inc/*.h
	@echo -n "Building $@ ... "
	$(CXX) $(CCFLAGS) $< $(INCLUDE) $(LIBS) -g -o $@
	@echo "Done"

.PHONY : all run apply web clean install ut

