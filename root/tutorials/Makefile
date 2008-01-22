# Run ROOTSYS/tutorials. Don't check for valid output, just
# check that CINT can run it without (unhandled) error.

# Set the list of files to be deleted by clean
CLEAN_TARGETS += $(ALL_LIBRARIES) $(TUTORIALSLOGS) $(TUTORIALSSTUBS) \
  *.root *.gif *.png *.pdf *.ps *.svg *.py *.C *.xml *.dtd files htmldoc

# Set the list of target to make while testing.

TEST_TARGETS += $(TUTORIALSTESTS)
# don't run these, they will fail and need to be fixed.

TUTORIALSTESTS_NONBATCH := \
  geom/building geom/cheongwadae geom/geomAlice geom/geomAlice_itsv \
  geom/geomAtlas geom/geomBrahms geom/na49geomfile geom/lego geom/na49view \
  geom/robot geom/shapes geom/tank geom/station2 geom/station1 geom/south_gate \
  gl/glViewerExercise gl/glViewerLOD gl/nucleus \
  gl/viewer3DLocal gl/viewer3DMaster \
  gui/buttongroupState gui/buttonTest gui/calendar gui/games gui/guilabels \
  gui/QtFileDialog gui/QtMultiFileDialog gui/guiWithCINT gui/mditest \
  gui/Slider3Demo gui/textEntries \
  hist/exec1 hist/exec2 \
  image/img2pad image/galaxy_image image/hist2image \
  graphics/psview graphics/gtime \
  tree/tvdemo

TUTORIALSTESTS_VETO := htmlex  rootalias \
  fft/FFT \
  fit/fit1_C  fit/TwoHistoFit2D\
  foam/foam_demo foam/foam_demopers \
  math/Bessel math/LegendreAssoc math/Legendre math/mathmoreIntegration \
  math/normalDist \
  math/tStudent net/authserv net/hcons net/hprod net/hserv2 net/hserv \
  net/LDAPExample net/pserv net/spyserv net/TestAuth \
  proof/getProof proof/runProof proof/ProofSimple \
  pythia/pythiaExample \
  sql/sqlcanvas \
  sql/sqlcreatedb sql/sqlfilldb sql/sqlselect \
  tree/hsimpleProxy tree/tree0 tree/tree2a tree/tree4 \
  tree/copytree2 tree/copytree3 tree/copytree

TUTORIALSTESTS_NEEDS_MIUNIT2 := fit/fit2dHist fit/fit2dHist \
fit/fitCircle fit/minuit2FitBench2D fit/minuit2FitBench2D \
fit/minuit2FitBench fit/minuit2FitBench fit/minuit2GausFit fit/minuit2GausFit

ifeq (,$(wildcard $(ROOTSYS)/lib/libMinuit2.*))
TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_NEEDS_MIUNIT2)
endif

ifeq (,$(wildcard $(ROOTSYS)/lib/libRooFit.*))
TUTORIALSTESTS_VETO += fit/RoofitDemo
endif

ifeq (,$(wildcard $(ROOTSYS)/lib/libUnuran.*))
TUTORIALSTESTS_VETO += math/testrandom unuran/unuranDemo unuran/unuranFoamTest
endif

# I believe these fail due to CINT bugs:
TUTORIALSTESTS_VETOCINT := hist/hbars unuran/unuranDemo unuran/unuranFoamTest
TUTORIALSTESTS_VETOCINT := $(shell echo Failing because of CINT: $(TUTORIALSTESTS_VETOCINT))
TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_VETOCINT)
# exlude rose_image until we know how to set the iconpath in batch
TUTORIALSTESTS_VETO += image/rose_image
TUTORIALSTESTS_VETO += eve/text_test

ifeq ($(GUI),)
  TUTORIALSTESTS_VETO += $(TUTORIALSTESTS_NONBATCH)
endif

# get the list of macros to run
TUTORIALSTESTS := $(wildcard $(ROOTSYS)/tutorials/*.C $(ROOTSYS)/tutorials/*/*.C)
TUTORIALSTESTS := $(patsubst $(ROOTSYS)/tutorials/%.C,%,$(TUTORIALSTESTS))
# hsimple must be first
TUTORIALSTESTS := hsimple $(filter-out hsimple,$(TUTORIALSTESTS))
TUTORIALSTESTS := $(filter-out $(TUTORIALSTESTS_VETO),$(TUTORIALSTESTS))


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
	export ROOTTEST_HOME:=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

TUTDIR := $(ROOT_LOC)/tutorials
# stubs to execute the tutorial macro, preventing ROOT returning
# non-zero return codes even if no error was encountered
# (value of final expression)
TUTORIALSSTUBS := $(addsuffix _call.C,$(TUTORIALSTESTS))
TUTORIALSLOGS  := $(addsuffix .log,$(TUTORIALSTESTS))


$(TUTORIALSSTUBS): %_call.C:
	$(CMDECHO) mkdir -p $(dir $@)
	$(CMDECHO) echo 'int $(notdir $*)_call(){ int err; TROOT::SetMacroPath(".:$(TUTDIR)/$(dir $@)"); gROOT->ProcessLine(".x $(TUTDIR)/$*.C", &err); return err; }' > $@


$(TUTORIALSLOGS): %.log: %_call.C $(TUTDIR)/%.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $<  > $@ 2>&1 && echo "" > $@


# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

$(TUTORIALSTESTS): %: %.log

person.%: $(TUTDIR)/xml/person.%
	$(CMDECHO) cp -f $< $@
saxexample.xml: $(TUTDIR)/xml/saxexample.xml
	$(CMDECHO) cp -f $< $(notdir $@)

xml/DOMParsePerson.log xml/DOMRecursive.log : person.xml person.dtd
xml/SAXHandler.log: saxexample.xml
