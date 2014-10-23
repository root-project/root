# %ROOTTEST-duration%:5

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.root *.dat *.ps test files \
  stressGraphics.ref *.tmp weights

# we don't run stressHepix for now; it's simply calling all the other stress tests
# anyway, making it and the directly called ones stumble over their feet.
STRESSSRC := $(wildcard $(ROOTSYS)/test/stress*.cxx)
STRESSTESTS  := $(filter-out stressHepix-test,\
	$(filter-out stressProof-test,\
	$(filter-out stressTMVA-test,\
	$(filter-out stressRooFit_tests-test,\
	$(filter-out stressRooStats_tests-test,\
	$(filter-out stressRooStats_models-test,\
	$(filter-out stressHistFactory_tests-test,\
	$(filter-out stressHistFactory_models-test,\
	$(notdir $(STRESSSRC:.cxx=-test))))))))))
ifeq ($(ARCH)$(FAIL),win32)
   for_echoing := $(shell echo Known failures: stressMathCore skipped tests in root/stress 1>&2)
STRESSTESTS  := $(filter-out stressMathCore-test,$(STRESSTEST))
endif

ifeq ($(R__FEATURES),)
   R__FEATURES := $(shell root-config --features)
endif

ifeq ($(findstring roofit,$(R__FEATURES)),)
STRESSTESTS  := $(filter-out stressRooStats-test,$(filter-out stressRooFit-test,$(filter-out stressHistFactory-test,$(STRESSTESTS))))
endif
ifeq ($(findstring mathmore,$(R__FEATURES)),)
STRESSTESTS  := $(filter-out stressMathMore-test,$(STRESSTESTS))
endif
ifeq ($(findstring unuran,$(R__FEATURES)),)
STRESSTESTS  := $(filter-out stressHistoFit-test,$(STRESSTESTS))
endif
ifeq ($(findstring minuit2,$(R__FEATURES)),)
STRESSTESTS  := $(filter-out stressHistoFit-test,$(STRESSTESTS))
endif
ifeq ($(findstring xrootd,$(R__FEATURES)),)
STRESSTESTS  := $(filter-out stressReadOnly-test,$(STRESSTESTS))
endif

TEST_TARGETS += $(STRESSTESTS)

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

define RootsysTestDiff
	$(CMDECHO) [ "x`grep -i fail $<`" = "x" ] || ( grep -i fail -C 20 $<; false )
	$(CMDECHO) ( \
	  MARKS=`cat $< \
	  | sed 's,^.*ROOTMARKS[[:space:]:=]*\([[:digit:].]\+\).*$$,\1 ROOTmarks,' \
	  | grep ROOTmarks` && \
	  [ "x$$MARKS" != "x" ] && printf "%-14s %13s: %18s\n" $(patsubst %-ctest,% '[compiled]',$(patsubst %-itest,% '[interpreted]',$@)) "$$MARKS" || true \
	)
endef

STRESSEXE := $(addprefix test/,$(STRESSTESTS:-test=$(ExeSuf)))
STRESSTESTSCOMP := $(STRESSTESTS:-test=-ctest)

#### disabled tests
# Run all tests in interpreted mode except for those that CINT
# cannot interpret - these are run in compiled mode instead.
STRESSTESTSINTERP_VETO := stressMathCore-itest stressMathMore-itest \
  stressRooFit-itest stressRooStats-itest stressHistFactory-itest \
  stressLinear-itest stressVector-itest stressGeometry-itest stressIterators-itest stressHistogram-itest stressHistoFit-itest \
  stressTMVA-itest
STRESSTESTSINTERP := $(filter-out $(STRESSTESTSINTERP_VETO),$(STRESSTESTS:-test=-itest))

ifneq ($(FAST),)
# Fast means all compiled and skipping the 'long' test like TMVA
STRESSTESTSINTERP_VETO += $(STRESSTESTSINTERP)
STRESSTESTSINTERP :=
STRESSTESTSCOMP_VETO := $(STRESSTESTSCOMP_VETO) stressTMVA-ctest
endif

# run all compiled except for those that are run interpreted:
STRESSTESTSCOMP_VETO :=  $(STRESSTESTSCOMP_VETO) $(STRESSTESTSINTERP:-itest=-ctest)
# ...and those that are known to fail: none!
# STRESSTESTSCOMP_VETO := 
STRESSTESTSCOMP := $(filter-out $(STRESSTESTSCOMP_VETO),$(STRESSTESTS:-test=-ctest))

ifneq ($(ClingWorkAroundJITfullSymbolResolution)$(ClingWorkAroundMissingDynamicScope),)
STRESSTESTSINTERP := $(filter-out stressGUI-itest,$(STRESSTESTSINTERP))
STRESSTESTSCOMP   := $(filter-out stressGUI-ctest,$(STRESSTESTSCOMP))
endif

STRESSFILES := \
	$(filter-out %Dict.cxx,\
	$(filter-out %/TrackMathCoreDict.cxx,\
	$(wildcard $(ROOT_LOC)/test/*.cxx) \
	$(wildcard $(ROOT_LOC)/test/*.h) \
	$(wildcard $(ROOT_LOC)/test/*.ref) \
	$(wildcard $(ROOT_LOC)/test/*.root))) \
	$(ROOT_LOC)/test/Makefile

STRESSFILESHERE := $(subst $(ROOT_LOC)/,,$(STRESSFILES))
STRESSFILESCOPIEDTAG := test/.COPIEDTAG



.PHONY: $(STRESSTESTS) $(STRESSTESTSCOMP) $(STRESSTESTSINTERP)
.PRECIOUS: test/%

# Every tests consists of running the test in compiled and
# interpreted mode
$(STRESSTESTS): %-test: %-itest %-ctest

# .log: output of running the test

# rule for compiled tests
$(STRESSTESTSCOMP): %-ctest: %-ctest.log
	$(RootsysTestDiff)

# rule for interpreted tests
$(STRESSTESTSINTERP): %-itest: %-itest.log
	$(RootsysTestDiff)

# tests to skip
$(STRESSTESTSCOMP_VETO) $(STRESSTESTSINTERP_VETO): ;

# tag file to ensure that all files have been copied from $(ROOT_LOC)/test
$(STRESSFILESCOPIEDTAG): $(STRESSFILESHERE)
	$(CMDECHO) touch $@

# To run a compiled test we need the binary, then we call it using
# $(CALLROOTEXE) so it works with valgrind.
# $(STRESSFLAGS) can be a target-specific variable holding optional
# arguments to the test; see below for stress-ctest.log
CALLSTRESSEXE := $(CALLROOTEXE:root.exe=)
ifneq ($(FIXCLING),)
ifneq ($(FIXCLING),no)
# support for perftrack and valgrind is 'broken' when FixCling is set.
# and CALLROOTEXE has too much stuff (-e ....)
CALLSTRESSEXE := 
endif
endif

$(addsuffix .log,$(STRESSTESTSCOMP)): %-ctest.log: test/%$(ExeSuf)
	$(CMDECHO) unset DISPLAY; \
	   $(CALLSTRESSEXE) test/$*$(ExeSuf) $(STRESSFLAGS) > $@ 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<

# To run an interpreted test we need the source + includes. We don't know
# the exact dependencies, to we trust the binary: if we have it we have
# all that CINT needs to interpret the source in test/.
# $(STRESSFLAGS) can be a target-specific variable holding optional
# arguments to the test; see below for stress-itest.log
# Add %-ctest.log as dependency so they don't stumble over one
# another's feet (e.g. by both interpreted and compiled version
# opening the same file)
$(addsuffix .log,$(STRESSTESTSINTERP)): %-itest.log: test/%$(ExeSuf) $(find %-ctest, $(STRESSTESTSCOMP))
	$(CMDECHO) $(CALLROOTEXE) -l -b -q test/$*.cxx$(STRESSFLAGS) > $@ 2>&1 || handleError.sh --result=$$? --log=$@ --test=$<

# If we need a file in test/ we just copy all files from $(ROOT_LOC)/test.
# Once we have them in test/, we run make in there.
$(STRESSEXE): test/%: $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true
# make sure $@ is newer than $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) touch $@

# Copy files from $(ROOT_LOC)/test to ./test/
# so we don't build within $(ROOT_LOC)
$(STRESSFILESHERE): test/%: $(ROOT_LOC)/test/%
	$(CMDECHO) test -d test || mkdir -p test
	$(CMDECHO) cp -f $< $@

test/libEvent.$(DllSuf): $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true

test/libTrackMathCoreDict.$(DllSuf): $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true

test/hsimple.C: $(ROOT_LOC)/tutorials/hsimple.C
	$(CMDECHO) test -d test || mkdir -p test
	$(CMDECHO) cp -f $< $@

# returns a TFile*, so root.exe's return code is not very helpful
hsimple.root: test/hsimple.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $< || true > /dev/null 2>&1
stressGraphics.ref: test/stressGraphics.ref
	$(CMDECHO) cp -f $< $@
earth.dat: $(ROOT_LOC)/tutorials/graphics/earth.dat
	$(CMDECHO) cp -f $< $@

#### Extra parameters for some tests ####

ifneq ($(ClingWorkAroundMissingDynamicScope),)
stress-itest.log: CALLROOTEXE += -e 'gSystem->Load("test/libEvent")'
stressGraphics-itest.log: CALLROOTEXE += -e 'gSystem->Load("libGpad"); gSystem->Load("libTreeViewer");'
stressGUI-itest.log: CALLROOTEXE += -e 'gSystem->Load("libGpad"); gSystem->Load("libGui");'
endif
stress-ctest.log: STRESSFLAGS:=-b 30
stress-itest.log: STRESSFLAGS:="(30)"
stressTMVA-itest.log: STRESSFLAGS:= "" 2>&1 |grep -v 'Automatic variable Endl is allocated'
ifneq ($(ClingWorkAroundDeletedSourceFile),)
stressGraphics-ctest.log: STRESSFLAGS:= -k
endif
stressGraphics-itest.log: STRESSFLAGS:= "" 2>&1 |grep -v 'Warning in <TUnixSystem::SetDisplay>'


ifneq ($(ClingWorkAroundJITfullSymbolResolution)$(ClingWorkAroundMissingDynamicScope),)
ifeq ($(FAST),)
stressGUI-itest:
	@echo "Please note: stressGUI is disabled!!!"
	@echo "n/a" > $@.log
else
stressGUI-ctest:
	@echo "Please note: stressGUI is disabled!!!"
	@echo "n/a" > $@.log
endif
endif

#### Extra dependencies for some tests ####

test/stress$(ExeSuf): test/libEvent.$(DllSuf)
test/stressMathCore$(ExeSuf): test/libTrackMathCoreDict.$(DllSuf)
stressGraphics-itest.log stressGraphics-ctest.log: hsimple.root stressGraphics.ref earth.dat
