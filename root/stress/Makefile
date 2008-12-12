# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.nomark *.log *.root *.dat *.ps test files \
  stressGraphics.ref *.tmp weights

# we don't run stressHepix for now; it's simply calling all the other stress tests
# anyway, making it and the directly called ones stumble over their feet.
STRESSSRC := $(wildcard $(ROOTSYS)/test/stress*.cxx)
STRESSTESTS  := $(filter-out stressHepix-test,\
	$(filter-out stressProof-test,\
	$(filter-out stressTMVA-test,\
	$(filter-out stressRooFit_tests-test,\
	$(notdir $(STRESSSRC:.cxx=-test))))))
ifeq ($(findstring roofit,$(shell root-config --features)),)
STRESSTESTS  := $(filter-out stressRooFit-test,$(STRESSTESTS))
endif
ifeq ($(findstring mathmore,$(shell root-config --features)),)
STRESSTESTS  := $(filter-out stressMathMore-test,$(STRESSTESTS))
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
	export ROOTTEST_HOME:=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

define RootsysTestDiff
	$(CMDECHO) grep -v -i Mbytes $< | grep -v -i seconds | grep -v -i MARKS \
	| grep -v SYS: | grep -v -i time \
	| grep -v 'does not exist, generating it' \
	| grep -v 'does not exist in' \
	| grep -v 'or has wrong file extension' \
	| grep -v 'RooFit v' \
	| grep -v 'Copyright ' \
	| grep -v 'MB/s' \
	| sed 's,\([[:space:]]\?[.[:digit:]]\+e[+-]\)0\([[:digit:]]\{2\}\), \1\2,g' \
	> $<.nomark || true
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS) ]; then \
	   diff -u -b $@.ref$(ROOTBITS) $<.nomark ; \
	else \
	   diff -u -b $@.ref $<.nomark ; \
	fi
	$(CMDECHO) rm -f $<.nomark
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
STRESSTESTSINTERP_VETO := stressMathCore-itest stressMathMore-itest stressRooFit-itest \
  stressLinear-itest stressVector-itest stressGeometry-itest stressIterators-itest stressHistogram-itest
STRESSTESTSINTERP := $(filter-out $(STRESSTESTSINTERP_VETO),$(STRESSTESTS:-test=-itest))

# run all compiled except for those that are run interpreted:
STRESSTESTSCOMP_VETO := $(STRESSTESTSINTERP:-itest=-ctest)
# ...and those that are known to fail: none!
# STRESSTESTSCOMP_VETO := 
STRESSTESTSCOMP := $(filter-out $(STRESSTESTSCOMP_VETO),$(STRESSTESTS:-test=-ctest))


STRESSFILES := \
	$(filter-out %/TrackMathCoreDict.cxx,\
	$(wildcard $(ROOT_LOC)/test/*.cxx) \
	$(wildcard $(ROOT_LOC)/test/*.h) \
	$(wildcard $(ROOT_LOC)/test/*.ref) \
	$(wildcard $(ROOT_LOC)/test/*.root)) \
$(ROOT_LOC)/test/Makefile.arch \
$(ROOT_LOC)/test/Makefile

	STRESSFILESHERE := $(subst $(ROOT_LOC)/,,$(STRESSFILES))
STRESSFILESCOPIEDTAG := test/.COPIEDTAG



.PHONY: $(STRESSTESTS) $(STRESSTESTSCOMP) $(STRESSTESTSINTERP)
.PRECIOUS: test/%

# Every tests consists of running the test in compiled and
# interpreted mode
$(STRESSTESTS): %-test: %-itest %-ctest

# .ref: reference file, should match .log.nomark
# .log: output of running the test

# rule for compiled tests
$(STRESSTESTSCOMP): %-ctest: %-ctest.log %-ctest.ref
	$(RootsysTestDiff)

# rule for interpreted tests
$(STRESSTESTSINTERP): %-itest: %-itest.log %-itest.ref
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
CALLSTRESSEXE := $(CALLSTRESSEXE:rootc7.exe=)
$(addsuffix .log,$(STRESSTESTSCOMP)): %-ctest.log: test/%$(ExeSuf)
	$(CMDECHO) unset DISPLAY; \
	   $(CALLSTRESSEXE) test/$*$(ExeSuf) $(STRESSFLAGS) > $@ 2>&1

# To run an interpreted test we need the source + includes. We don't know
# the exact dependencies, to we trust the binary: if we have it we have
# all that CINT needs to interpret the source in test/.
# $(STRESSFLAGS) can be a target-specific variable holding optional
# arguments to the test; see below for stress-itest.log
# Add %-ctest.log as dependency so they don't stumble over one
# another's feet (e.g. by both interpreted and compiled version
# opening the same file)
$(addsuffix .log,$(STRESSTESTSINTERP)): %-itest.log: test/%$(ExeSuf) $(find %-ctest, $(STRESSTESTSCOMP))
	$(CMDECHO) $(CALLROOTEXE) -l -b -q test/$*.cxx$(STRESSFLAGS) > $@ 2>&1 

# If we need a file in test/ we just copy all files from $(ROOT_LOC)/test.
# Once we have them in test/, we run make in there.
$(STRESSEXE): test/%: $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true
# make sure $@ is newer than $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) touch $@

# Copy files from $ROOT_LOC)/test to ./test/
# so we don't build within $(ROOT_LOC)
$(STRESSFILESHERE): test/%: $(ROOT_LOC)/test/%
	$(CMDECHO) test -d test || mkdir -p test
	$(CMDECHO) cp -f $< $@

test/libEvent.$(DllSuf): $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true

test/libTrackMathCoreDict.$(DllSuf): $(STRESSFILESCOPIEDTAG)
	$(CMDECHO) $(MAKE) -s --no-print-directory -C test $(notdir $@) | grep -v "$(notdir $@) done" || true

test/hsimple.C: $(ROOT_LOC)/tutorials/hsimple.C
	$(CMDECHO) cp -f $< $@

# returns a TFile*, so root.exe's return code is not very helpful
hsimple.root: test/hsimple.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $< || true > /dev/null 2>&1
	$(CMDECHO) cp -f test/$@ $@
linearIO.root: test/linearIO.root
	$(CMDECHO) cp -f $< $@
stressGraphics.ref: test/stressGraphics.ref
	$(CMDECHO) cp -f $< $@
earth.dat: $(ROOT_LOC)/tutorials/graphics/earth.dat
	$(CMDECHO) cp -f $< $@

#### Extra parameters for some tests ####

stress-ctest.log: STRESSFLAGS:=-b 30
stress-itest.log: STRESSFLAGS:="(30)"
stressTMVA-itest.log: STRESSFLAGS:= "" 2>&1 |grep -v 'Automatic variable Endl is allocated'
stressGraphics-itest.log: STRESSFLAGS:= "" 2>&1 |grep -v 'Warning in <TUnixSystem::SetDisplay>'

#### Extra dependencies for some tests ####

test/stress$(ExeSuf): test/libEvent.$(DllSuf)
test/stressMathCore$(ExeSuf): test/libTrackMathCoreDict.$(DllSuf)
stressGraphics-itest.log stressGraphics-ctest.log: hsimple.root stressGraphics.ref earth.dat
test/stressLinear$(ExeSuf): linearIO.root
