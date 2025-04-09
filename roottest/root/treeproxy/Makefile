# %ROOTTEST-duration%:8
# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.clog *.log red.h generatedSel.h val3dimSel.h forproxy.root \
	withfriend.h *.rootmap make_tuple.root make_tuple_sel.h \
	btagobjsSel.h fullmcSel.h searchSelector.h

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += search reducedSpill leaflist valdim3 make_tuple friendchain btagobjsProxy fullmc

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
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

ifneq ($(MAKECMDGOALS),clean)
-include $(wildcard *.d)
endif

# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

mytest:
	$(CMDECHO) echo mytest > /dev/null


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

define hideLibraryCreation
   grep -v -e ACLiC -e 'creating shared library' -e 'Creating library' \
	     -e _cint.cxx -e '_[0-9]*.cxx' -e '_.cxx' \
	     -e 'Warning in <TClassTable::Add>: class generatedSel already in TClassTable'
endef

leaflist.log: simPosProxy.C hsimpleProxy.C Makefile runleaflist.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runleaflist.C > leaflist.raw.log 2>&1 ; result=$$? ; cat leaflist.raw.log | $(hideLibraryCreation) > leaflist.log ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

leaflist: leaflist.log
	$(TestDiff)

producereducedSpill.log: producereducedSpill.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b producereducedSpill.C > producereducedSpill.raw.log 2>&1 ; result=$$? ; cat  producereducedSpill.raw.log | $(RemoveDirs) > producereducedSpill.log ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

producereducedSpill: producereducedSpill.log
	$(TestDiff)

reducedSpill.log: red_h.$(DllSuf)

reducedSpill: reducedSpill.log
	$(TestDiff)

red.h: 
	$(CMDECHO) $(CALLROOTEXE) -q -l -b createRed.C > createRed.log 2>&1

execsearch.log: searchSelector_h.$(DllSuf)

searchSelector.h: createSearch.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b createSearch.C > createSearch.log 2>&1 ||  handleError.sh --result=$$? --log=createSearch.log --test=createSearch

Track_C.$(DllSuf): Track.h

forproxy.root: createvaldim3_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l createvaldim3.C+ > createvaldim3.log 2>&1  || handleError.sh --result=$$? --log=createvaldim3.log --test=$@ --rm=$@

val3dimSel.h: createvaldim3_C.$(DllSuf) forproxy.root
	$(CMDECHO) rm -f val3dimSel.h; $(CALLROOTEXE) -q -b -l createvaldim3.C+ > createvaldim3.log 2>&1  || handleError.sh --result=$$? --log=createvaldim3.log --test=$@ --rm=$@

createvaldim3_C.$(DllSuf): createvaldim3.C Track_C.$(DllSuf)
	$(BuildWithLib)
ifneq ($(shell root-config --arch | grep mac),)
ifeq ($(FAIL),)
	$(WarnFailTest)
# see include file in createvaldim3.C
endif
endif

runvaldim3_C.$(DllSuf): runvaldim3.C Track_C.$(DllSuf)
	$(BuildWithLib)

val3dimSel_h.$(DllSuf): val3dimSel.h createvaldim3_C.$(DllSuf) Track_C.$(DllSuf)
	$(BuildWithLib) 

valdim3.log: val3dimSel_h.$(DllSuf) forproxy.root

valdim3: valdim3.log
	$(TestDiff)

fullmcSel.h: fullmc.cxx $(CORELIBS) runfullmc_C.$(DllSuf) full-mcfile.root
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runfullmc.C+(4)' > fullmcSel_h.create.log 2>&1 || handleError.sh --result=$$? --log=fullmcSel_h.create.log --test=$@ --rm=$@

fullmc.clog: fullmcSel_h.$(DllSuf) $(CORELIBS) runfullmc_C.$(DllSuf) fullmc.cxx full-mcfile.root fullmcSel.h
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runfullmc.C+(5)' > fullmc.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$@ 

fullmc: fullmc.clog
	$(TestDiffW)

make_tuple.log: runmake_tuple.C make_tuple.C make_tuple_draw.C  $(CORELIBS) 
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runmake_tuple.C > make_tuple.raw.log 2>&1 ; result=$$? ; cat  make_tuple.raw.log | $(hideLibraryCreation) > make_tuple.log ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

make_tuple: make_tuple.log
	$(TestDiffW)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
friendchain.log: CALLROOTEXE += -e 'gSystem->Load("libTree");'
endif

friendchain.log: runfriendchain.C vertex.C make_chains.cxx  $(CORELIBS) 
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runfriendchain.C > friendchain.raw.log  2>&1; result=$$? ; cat friendchain.raw.log  | $(hideLibraryCreation) > friendchain.log ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

friendchain: friendchain.log
	$(TestDiffW)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
btagobjsSel.h: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer"); gSystem->Load("libPhysics");'
btagobjsProxy.clog: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer"); gSystem->Load("libPhysics");'
endif

btagobjsSel.h: btagobjsScript.h btagobjsScript.cxx $(CORELIBS) runbtagobjs.C btagobjs.root
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runbtagobjs.C(4)' > btagProxySel_h.create.log 2>&1 || handleError.sh --result=$$? --log=btagProxySel_h.create.log --test=$@ --rm=$@

btagobjsProxy.clog: btagobjsSel_h.$(DllSuf) $(CORELIBS) btagobjs.root runbtagobjs.C 
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runbtagobjs.C(5)'  > $@ 2>&1 ; result=$$? ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

btagobjsProxy: btagobjsProxy.clog
	$(TestDiffW)

ifneq ($(ClingWorkAroundMissingAutoLoading)$(ClingWorkAroundMissingSmartInclude),)
valdim3.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer"); gSystem->Load("Track_C");' 
endif

