######################################################################
# Project: TMVA - Toolkit for Multivariate Data Analysis             #
# Code   : Source                                                    #
###################################################################### 

SHELL=bash

MAKEFLAGS = --no-print-directory -r -s --warn-undefined-variables --debug

include Makefile.arch

# Internal configuration
PACKAGE=TMVA
LD_LIBRARY_PATH := $(shell root-config --libdir):$(LD_LIBRARY_PATH)
OBJDIR    = obj
DEPDIR    = $(OBJDIR)/dep
LIBDIR    = lib
VPATH     = $(OBJDIR)
INCLUDES += -I./

DICTL1       := inc/LinkDef1.h
DICTL2       := inc/LinkDef2.h
DICTL3       := inc/LinkDef3.h
DICTL4       := inc/LinkDef4.h
DICTL        := $(DICTL1) $(DICTL2) $(DICTL3) $(DICTL4)
DICTS1       := src/$(PACKAGE)_Dict1.C
DICTS2       := src/$(PACKAGE)_Dict2.C
DICTS3       := src/$(PACKAGE)_Dict3.C
DICTS4       := src/$(PACKAGE)_Dict4.C
DICTS        := $(DICTS1) $(DICTS2) $(DICTS3) $(DICTS4)
DICTO1       := $(OBJDIR)/$(PACKAGE)_Dict1.o
DICTO2       := $(OBJDIR)/$(PACKAGE)_Dict2.o
DICTO3       := $(OBJDIR)/$(PACKAGE)_Dict3.o
DICTO4       := $(OBJDIR)/$(PACKAGE)_Dict4.o
DICTO        := $(DICTO1) $(DICTO2) $(DICTO3) $(DICTO4)
DICTH1       := Configurable.h Event.h Factory.h MethodBase.h MethodCompositeBase.h \
		MethodANNBase.h MethodTMlpANN.h MethodRuleFit.h MethodCuts.h MethodFisher.h \
		MethodKNN.h MethodCFMlpANN.h MethodCFMlpANN_Utils.h MethodLikelihood.h \
		MethodHMatrix.h MethodPDERS.h MethodBDT.h MethodDT.h MethodSVM.h MethodBayesClassifier.h \
		MethodFDA.h MethodMLP.h MethodCommittee.h MethodBoost.h \
		MethodPDEFoam.h MethodLD.h MethodCategory.h
DICTH2       := TSpline2.h TSpline1.h PDF.h BinaryTree.h BinarySearchTreeNode.h BinarySearchTree.h \
		Timer.h RootFinder.h CrossEntropy.h DecisionTree.h DecisionTreeNode.h MisClassificationError.h \
		Node.h SdivSqrtSplusB.h SeparationBase.h RegressionVariance.h Tools.h Reader.h \
		GeneticAlgorithm.h GeneticGenes.h GeneticPopulation.h GeneticRange.h GiniIndex.h \
		GiniIndexWithLaplace.h SimulatedAnnealing.h
DICTH3       := Config.h KDEKernel.h Interval.h FitterBase.h MCFitter.h GeneticFitter.h SimulatedAnnealingFitter.h \
		MinuitFitter.h MinuitWrapper.h IFitterTarget.h \
		PDEFoam.h PDEFoamDecisionTree.h PDEFoamDensityBase.h PDEFoamDiscriminantDensity.h \
		PDEFoamEventDensity.h PDEFoamTargetDensity.h PDEFoamDecisionTreeDensity.h PDEFoamMultiTarget.h \
		PDEFoamVect.h PDEFoamCell.h PDEFoamDiscriminant.h PDEFoamEvent.h PDEFoamTarget.h \
		PDEFoamKernelBase.h PDEFoamKernelTrivial.h PDEFoamKernelLinN.h PDEFoamKernelGauss.h \
		BDTEventWrapper.h CCTreeWrapper.h \
		CCPruner.h CostComplexityPruneTool.h SVEvent.h OptimizeConfigParameters.h
DICTH4       := TNeuron.h TSynapse.h TActivationChooser.h TActivation.h TActivationSigmoid.h TActivationIdentity.h \
		TActivationTanh.h TActivationRadial.h TNeuronInputChooser.h TNeuronInput.h TNeuronInputSum.h \
		TNeuronInputSqSum.h TNeuronInputAbs.h Types.h Ranking.h RuleFit.h RuleFitAPI.h IMethod.h MsgLogger.h \
		VariableTransformBase.h VariableIdentityTransform.h VariableDecorrTransform.h VariablePCATransform.h \
		VariableGaussTransform.h VariableNormalizeTransform.h VariableRearrangeTransform.h
DICTH1       := $(patsubst %,inc/%,$(DICTH1))
DICTH2       := $(patsubst %,inc/%,$(DICTH2))
DICTH3       := $(patsubst %,inc/%,$(DICTH3))
DICTH4       := $(patsubst %,inc/%,$(DICTH4))
DICTH        := $(DICTH1) $(DICTH2) $(DICTH3) $(DICTH4)


SKIPCPPLIST  := 
SKIPHLIST    := $(DICTHEAD) $(DICTL)
LIBFILE      := lib/lib$(PACKAGE).a
SHLIBFILE    := lib$(PACKAGE).$(DllSuf)
DLLIBFILE    := lib/lib$(PACKAGE).dll
ROOTMAP      := lib/lib$(PACKAGE).rootmap
TESTDIR      := test
UNAME = $(shell uname)

default: shlib linklib

# List of all source files to build
HLIST   = $(filter-out $(SKIPHLIST),$(wildcard inc/*.h))
CPPLIST = $(filter-out $(SKIPCPPLIST),$(patsubst src/%,%,$(wildcard src/*.$(SrcSuf))))

# List of all object files to build
OLIST=$(patsubst %.cxx,%.o,$(CPPLIST))
OLIST=$(CPPLIST:.$(SrcSuf)=.o)
DEPLIST=$(foreach var,$(CPPLIST:.$(SrcSuf)=.d),$(DEPDIR)/$(var))

# Implicit rule to compile all classes
sl:
	if [[ ( ! -e TMVA ) ]]; then \
		ln -sf inc TMVA; \
	fi

$(OBJDIR)/%.o : src/%.cxx 
	if [[ ( ! -e TMVA ) ]]; then \
		ln -sf inc TMVA; \
	fi
	@printf "Compiling $< ... "
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -ggdb -c $< -o $@
	@echo "Done"

# Rule to make the dictionary
dict: sl $(DICTO1) $(DICTO2) $(DICTO3) $(DICTO4)

$(DICTS1):  $(DICTH1) $(DICTL1)
	@echo "Generating dictionary $@" 
	$(shell root-config --exec-prefix)/bin/rootcint -f $@ -c -p $(INCLUDES) $^

$(DICTS2):  $(DICTH2) $(DICTL2)
	@echo "Generating dictionary $@" 
	$(shell root-config --exec-prefix)/bin/rootcint -f $@ -c -p $(INCLUDES) $^

$(DICTS3):  $(DICTH3) $(DICTL3)
	@echo "Generating dictionary $@" 
	$(shell root-config --exec-prefix)/bin/rootcint -f $@ -c -p $(INCLUDES) $^

$(DICTS4):  $(DICTH4) $(DICTL4)
	@echo "Generating dictionary $@" 
	$(shell root-config --exec-prefix)/bin/rootcint -f $@ -c -p $(INCLUDES) $^

$(DICTO1): $(DICTS1)
	@echo "Compiling dictionary $<"
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -g -c  -o $@ $<

$(DICTO2): $(DICTS2)
	@echo "Compiling dictionary $<"
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -g -c  -o $@ $<

$(DICTO3): $(DICTS3)
	@echo "Compiling dictionary $<"
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -g -c  -o $@ $<

$(DICTO4): $(DICTS4)
	@echo "Compiling dictionary $<"
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -g -c  -o $@ $<



# Rule to set up a symbolic links to the created shared library
linklib:
	if [[ ( ( `root-config --platform` == "macosx" ) && \
			( ! -e lib/lib$(PACKAGE).1.dylib ) ) || \
			( ! -e lib/lib$(PACKAGE).1.so ) ]]; then \
		printf "Setting up soft links to the TMVA library ... "; \
		ln -sf $(SHLIBFILE) $(LIBDIR)/lib$(PACKAGE).1.so; \
		if [[ `root-config --platform` == "macosx" ]]; then \
			ln -sf $(SHLIBFILE) $(LIBDIR)/lib$(PACKAGE).1.dylib; \
		fi; \
		echo "Done"; \
	fi

##############################
# The dependencies section   
# - the purpose of the .d files is to keep track of the
#   header file dependence
# - this can be achieved using the makedepend command 
##############################
# .d tries to pre-process .cc
ifneq ($(MAKECMDGOALS),clean)
-include $(DEPLIST)
endif

$(DEPDIR)/%.d: src/%.$(SrcSuf)
	@mkdir -p $(DEPDIR)
	if [[ ( ! -e TMVA ) ]]; then \
		ln -sf inc TMVA; \
	fi
	if test -f $< ; then \
		printf "Building $(@F) ... "; \
		$(SHELL) -ec '`root-config --exec-prefix`/bin/rmkdepend -f- -Y -w 3000 -- -I./ -- $< 2> /dev/null 1| sed -e "s-src/\(.*\).o\(: .*\)-$(DEPDIR)\/\1.d $(OBJDIR)/\1.o\2-" > $@'; \
		rm -f $@.bak; \
		echo "Done"; \
	fi

# Rule to combine objects into a library
$(LIBFILE): $(DICTO) $(OLIST)
	@printf "Making static library $(LIBFILE) ... "
	@rm -f $(LIBFILE)
	@ar q $(LIBFILE) $(addprefix $(OBJDIR)/,$(OLIST) $(DICTO)
	@ranlib $(LIBFILE)
	@echo "Done"

# Rule to combine objects into a unix shared library
$(LIBDIR)/$(SHLIBFILE): $(OLIST) $(DICTO)
	@printf "Building shared library $(LIBDIR)/$(SHLIBFILE) ... "
	@mkdir -p $(LIBDIR)
	@rm -f $(LIBDIR)/$(SHLIBFILE)
	@$(LD) -L$(shell root-config --libdir) $(SOFLAGS) $(addprefix $(OBJDIR)/,$(OLIST)) $(DICTO) -o $(LIBDIR)/$(SHLIBFILE) -lMinuit -lXMLIO
#	ln -fs $(SHLIBFILE) lib/lib$(PACKAGE).1.so
	@echo "Done"

# Rule to combine objects into a unix shared library
$(ROOTMAP): $(DICTL)
	@printf "Building $(ROOTMAP) ... "
	@mkdir -p $(LIBDIR)
	rlibmap -f -o $@ -l lib$(PACKAGE).1.$(DllSuf) -d libMinuit.so libMLP.so libMatrix.so libTree.so libGraf.so libTreePlayer.so libXMLIO.so -c $^
	@echo "Done"

# Rule to combine objects into a windows shared library
$(DLLIBFILE): $(OLIST) $(DICTO)
	@printf "Making dll file $(DLLIBFILE) ... "
	@rm -f $(DLLIBFILE)
	$(LD) -Wl,--export-all-symbols -Wl,--export-dynamic -Wl,--enable-auto-import -Wl,-Bdynamic -shared --enable-auto-image-base -Wl,-soname -o $(DLLIBFILE) -Wl,--whole-archive $(addprefix $(OBJDIR)/,$(OLIST) $(patsubst %.$(SrcSuf),%.o,$(DICTFILE))) -Wl,--no-whole-archive -L$(ROOTSYS)/lib -lCore -lCint -lHist -lGraf -lGraf3d -lTree -lRint -lPostscript -lMatrix -lMinuit -lPhysics -lHtml -lXMLIO -lm
	@echo "Done"

# Useful build targets
lib: $(LIBFILE) 

shlib: $(LIBDIR)/$(SHLIBFILE) $(ROOTMAP)

winlib: $(DLLIBFILE)

vars:
	#echo $(patsubst src/%,%,$(wildcard src/*.$(SrcSuf)))
	echo $(OLIST) $(DICTO)

clean:
	rm -rf obj
	rm -rf lib
	rm -f TMVA 
	rm -f $(DICTS) $(DICTS:.C=.h)
	rm -f $(OBJDIR)/*.o
	rm -f $(DEPDIR)/*.d
	rm -f $(LIBFILE)
	rm -f $(LIBDIR)/$(SHLIBFILE)
	rm -f lib/lib$(PACKAGE).1.so
	rm -f lib/lib$(PACKAGE).1.dylib
	rm -f $(ROOTMAP)
	rm -f $(DLLIBFILE)

distclean:
	rm -rf obj 
	rm -f *~
	rm -f $(DICTFILE) $(DICTHEAD)
	rm -f $(LIBFILE)
	rm -f $(LIBDIR)/$(SHLIBFILE	)
	rm -f lib/lib$(PACKAGE).1.so
	rm -f lib/lib$(PACKAGE).1.dylib
	rm -f $(ROOTMAP)
	rm -f $(DLLIBFILE)

.PHONY : winlib shlib lib default clean

# DO NOT DELETE
