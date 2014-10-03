# %ROOTTEST-duration%:2
# Run ROOTSYS/tutorials. Don't check for valid output, just
# check that CINT can run it without (unhandled) error.

# Set the list of files to be deleted by clean
CLEAN_TARGETS += $(ALL_LIBRARIES) $(TUTORIALSLOGS) $(TUTORIALSSTUBS) \
  $(filter-out pt_%,$(wildcard *.root *.gif *.png)) *.pdf *.ps *.svg *.py *.C *.xml *.dtd files htmldoc aliesd \
  *.tmp xpd-temp xpd-tutorial person.dtd image foam physics graphs proof mlp math tree splot spectrum matrix hist thread g4root fit gui net geom sql graphics xml gl io quadp eve aliesd files fft pythia unuran cont roofit rf*.txt rf*.tex rf*.dot MyPdf*.cxx MyPdf*.h RooGenPdfClass.* \
  *.rs Roo*.h *.dot roostats Roo*.cxx mc */*.log.err code.root

# Set the list of target to make while testing.

# This is a default value, it can be over-ridden
# by passing FIXCLING= as a command line argument.
#
# So, fixing cling is on by default and is disabled
# if FIXCLING is explicitly set to empty or is set
# to: no
# 
# We need to repeat this here (in addition to Rules.mk)
# because we need it before we can include Rules.mk.
FIXCLING=yes

ifeq ($(FAST),)
TEST_TARGETS += $(TUTORIALSTESTS) $(TUTORIALSTESTS_CC)
endif

# don't run these, they will fail and need to be fixed.

TUTORIALSTESTS_NONBATCH := \
  fit/fitpanel_playback \
  geom/% \
  gl/% \
  gui/% \
  hist/exec1 hist/exec2 \
  image/% \
  graphics/psview graphics/gtime graphics/graph_edit_playback \
  tree/tvdemo \
  eve/%

TUTORIALSTESTS_VETO := htmlex  rootalias \
  fft/FFT \
  fit/fit1_C  fit/TwoHistoFit2D fit/line3Dfit fit/MDF \
  foam/% \
  graphs/graphstruct \
  html/% \
  math/Bessel math/LegendreAssoc math/Legendre math/mathmoreIntegration \
  math/normalDist math/tStudent math/testUnfold2 math/pca \
  net/% \
  proof/% \
  pythia/% \
  sql/% \
  tree/hsimpleProxy tree/h1analysisProxy tree/tree0 tree/tree2a tree/tree4 \
  roostats/rs401d_FeldmanCousins \
  roofit/rf104_classfactory \
  histfactory/ModifyInterpolation \
  tree/copytree2 tree/copytree3 tree/copytree \
  math/testUnfold4 \
  fit/minuit2FitBench \
  fit/minuit2FitBench2D \
  roostats/% \
  http/%

TUTORIALSTESTS_NEEDS_MINUIT2 := fit/fit2dHist fit/fit2dHist \
fit/fitCircle fit/minuit2FitBench2D fit/minuit2FitBench2D \
fit/minuit2FitBench fit/minuit2FitBench fit/minuit2GausFit fit/minuit2GausFit \
fit/NumericalMinimization fit/combinedFit

ifeq (,$(wildcard $(ROOTSYS)/lib/libMinuit2.*))
TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_NEEDS_MINUIT2)
endif

ifeq (,$(wildcard $(ROOTSYS)/lib/libRooFit.*))
TUTORIALSTESTS_VETO += fit/RoofitDemo
endif

ifeq (,$(wildcard $(ROOTSYS)/lib/libUnuran.*))
TUTORIALSTESTS_VETO += math/testrandom unuran/unuranDemo unuran/unuranFoamTest
endif

ifeq ($(findstring roofit,$(shell root-config --features)),)
TUTORIALSTESTS_VETO += roofit/% roostats/% histfactory/%
endif

ifeq ($(findstring xml, $(shell root-config --features)),)
  TUTORIALSTESTS_VETO += xml/%
endif

ifeq ($(findstring fitsio, $(shell root-config --features)),)
  TUTORIALSTESTS_VETO += fitsio/%
endif

ifeq ($(findstring mathmore, $(shell root-config --features)),)
  TUTORIALSTESTS_VETO += math/quasirandom
endif

ifeq ($(findstring fftw3, $(shell root-config --features)),)
  TUTORIALSTESTS_VETO += roofit/rf208_convolution roofit/rf210_angularconv \
     roofit/rf211_paramconv roofit/rf512_wsfactory_oper.log \
     roofit/rf512_wsfactory_oper
endif


# I believe these fail due to CINT bugs:
TUTORIALSTESTS_VETOCINT := hist/hbars unuran/unuranDemo unuran/unuranFoamTest
TUTORIALSTESTS_VETOCINT := $(shell echo Failing because of CINT: $(TUTORIALSTESTS_VETOCINT))
TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_VETOCINT)
# exlude rose_image until we know how to set the iconpath in batch
TUTORIALSTESTS_VETO += image/rose_image
# exlude some eve tutorials
TUTORIALSTESTS_VETO += eve/text_test
TUTORIALSTESTS_VETO += eve/test_paramlist

TUTORIALSTESTS_VETO_CC := math/testrandom fit/combinedFit

ifeq ($(GUI),)
  TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_NONBATCH)
endif

# get the list of macros to run
TUTORIALSTESTS := $(wildcard $(ROOTSYS)/tutorials/*.C $(ROOTSYS)/tutorials/*/*.C)
TUTORIALSTESTS := $(patsubst $(ROOTSYS)/tutorials/%.C,%,$(TUTORIALSTESTS))
# hsimple must be first
TUTORIALSTESTS := hsimple $(filter-out hsimple,$(TUTORIALSTESTS))
TUTORIALSTESTS := $(filter-out $(TUTORIALSTESTS_VETO),$(TUTORIALSTESTS))
TUTORIALSTESTS_CC := $(filter $(TUTORIALSTESTS_VETO_CC),$(TUTORIALSTESTS))
TUTORIALSTESTS := $(filter-out $(TUTORIALSTESTS_VETO_CC),$(TUTORIALSTESTS))

# Search for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the
#  ROOTTEST_HOME variable for use by the SUBDIRECTORIES variable and is 
#  exported to eliminate the overhead of findding the path again during 
#  recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 
#    'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from 
#  the home directory.

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

TUTDIR := $(ROOT_LOC)/tutorials
# stubs to execute the tutorial macro, preventing ROOT returning
# non-zero return codes even if no error was encountered
# (value of final expression)
TUTORIALSSTUBS := $(addsuffix _call.C,$(TUTORIALSTESTS))
TUTORIALSLOGS  := $(addsuffix .log,$(TUTORIALSTESTS))
TUTORIALSSTUBS_CC := $(addsuffix _call.CC,$(TUTORIALSTESTS_CC))
TUTORIALSLOGS_CC  := $(addsuffix .clog,$(TUTORIALSTESTS_CC))

$(TUTORIALSSTUBS): %_call.C:
	$(CMDECHO) mkdir -p $(dir $@)
	$(CMDECHO) echo 'int $(notdir $*)_call(){ int err; gSystem->SetBuildDir("$(dir $@)",kTRUE); gSystem->AddIncludePath("-I$(TUTDIR)/$(dir $@)"); TROOT::SetMacroPath(".:$(TUTDIR)/$(dir $@)"); gROOT->ProcessLine(".x $(TUTDIR)/$*.C", &err); return err; }' > $@


$(TUTORIALSLOGS): %.log: %_call.C $(TUTDIR)/%.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $<  > $@ 2>&1 && ! grep -q 'There was a crash' $@ && echo "" > $@ || (mv $@ $@.err && cat $@.err && echo 'Log file has been copied to $@.err' && handleError.sh --cmd='Execution of $<' --result=1 --log=$@.err --test="$@" )

ifneq (,$(TUTORIALSSTUBS_CC))
$(TUTORIALSSTUBS_CC): %_call.CC:
	$(CMDECHO) mkdir -p $(dir $@)
	$(CMDECHO) echo 'int $(notdir $*)_call(){ int err; gSystem->SetBuildDir("$(dir $@)",kTRUE); gSystem->AddIncludePath("-I$(TUTDIR)/$(dir $@)"); TROOT::SetMacroPath(".:$(TUTDIR)/$(dir $@)"); gROOT->ProcessLine(".x $(TUTDIR)/$*.C+", &err); return err; }' > $@


$(TUTORIALSLOGS_CC): %.clog: %_call.CC $(TUTDIR)/%.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $<  > $@ 2>&1 && ! grep -q 'There was a crash' $@ && echo "" > $@ || (mv $@ $@.err && cat $@.err && echo 'Log file has been copied to $@.err' && handleError.sh --cmd='Execution of $<' --result=1 --log=$@.err --test="$@" )
endif

# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

$(TUTORIALSTESTS): %: %.log

$(TUTORIALSTESTS_CC): %: %.clog

person.%: $(TUTDIR)/xml/person.%
	$(CMDECHO) cp -f $< $@
saxexample.xml: $(TUTDIR)/xml/saxexample.xml
	$(CMDECHO) cp -f $< $(notdir $@)

xml/DOMParsePerson.log xml/DOMRecursive.log : person.xml person.dtd
xml/SAXHandler.log: saxexample.xml
roofit/rf503_wspaceread_call.C: roofit/rf502_wspacewrite.log
math/testUnfold5d.log: math/testUnfold5c.log
math/testUnfold5c.log: math/testUnfold5b.log
math/testUnfold5b.log: math/testUnfold5a.log

tree/cernstaff.log: tree/cernbuild.log


