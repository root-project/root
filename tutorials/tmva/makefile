######################################################################
# Project: TMVA - Toolkit for Multivariate Data Analysis             #
# Code   : Example usage for TMVA analyses                           #
###################################################################### 
BINS = TMVAClassification \
        TMVAClassificationCategory \
        TMVAClassificationApplication \
        TMVAClassificationCategoryApplication \
        TMVARegression \
        TMVARegressionApplication \
        TMVAMulticlass \
        TMVAMulticlassApplication \
        TMVAMultipleBackgroundExample \
			TMVAGAexample \
			TMVAGAexample2 \


CXX = g++
CCFLAGS = -O0 -ggdb -std=c++11 


LD = g++
LDFLAGS = 

LIBS = $(shell root-config --libdir) 
INCS = $(shell root-config --incdir)  


default : $(BINS)

$(BINS): % : %.C 
	@echo -n "Building $@ ... "
	$(CXX) $(CCFLAGS) $<  $(addprefix -I, $(INCS))  $(shell root-config --libs) -lMLP -lMinuit -lTreePlayer -lTMVA -lTMVAGui -lXMLIO  -lMLP -lm  -g -o $@
	@echo "Done"


clean: 
	rm -f *.o	$(BINS)

