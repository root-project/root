# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.clog *.log red.h generatedSel.h val3dimSel.h forproxy.root \
	vec.root vectorintSel.h vectorintEmuSel.h *.rootmap make_tuple.root make_tuple_sel.h

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += reducedSpill leaflist valdim3 vectorint vectorintEmu make_tuple

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
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


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

leaflist.log: simPosProxy.C hsimpleProxy.C Makefile runleaflist.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l runleaflist.C 2>&1 | grep -v -e ACLiC -e 'Creating library' \
	     -e _cint.cxx -e '_[0-9]*.cxx' -e '_.cxx' \
	     -e 'Warning in <TClassTable::Add>: class generatedSel already in TClassTable'> leaflist.log

leaflist: leaflist.log
	$(TestDiff)

producereducedSpill.log: producereducedSpill.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b producereducedSpill.C 2>&1 | $(RemoveDirs) > producereducedSpill.log

producereducedSpill: producereducedSpill.log
	$(TestDiff)

reducedSpill.log: red_h.$(DllSuf)

reducedSpill: reducedSpill.log
	$(TestDiff)

red.h: 
	$(CMDECHO) $(CALLROOTEXE) -q -l -b createRed.C > createRed.log 2>&1

Track_C.$(DllSuf): Track.h

forproxy.root: val3dimSel.h
	$(CMDECHO) $(CALLROOTEXE) -q -b -l createvaldim3.C+ > createvaldim3.log 2>&1

val3dimSel.h: createvaldim3_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -b -l createvaldim3.C+ > createvaldim3.log 2>&1

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

vec.root: $(CORELIBS) runvectorint_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runvectorint.C+(3)' > vec.create.log 2>&1

vectorintSel.h: dude.h dude.C $(CORELIBS) runvectorint_C.$(DllSuf) vec.root
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runvectorint.C+(4)' > vectorintSel_h.create.log 2>&1

#vectorintSel_h.$(DllSuf): vectorintSel.h dude.h dude.C vec.root
	
vectorint.clog: vectorintSel_h.$(DllSuf) $(CORELIBS) runvectorint_C.$(DllSuf) dude.C vec.root vectorintSel.h
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorint.C+(5)' > vectorint.clog 2>&1

vectorint: vectorint.clog
	$(TestDiff)

vectorintEmuSel.h: dude.h dude.cxx $(CORELIBS) runvectorintEmu_C.$(DllSuf) vec.root
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorintEmu.C+(4)' > vectorintEmuSel_h.create.log 2>&1
	
#vectorintEmuSel_h.$(DllSuf): dude.h dude.cxx
	
vectorintEmu.clog: vectorintEmuSel_h.$(DllSuf) $(CORELIBS) runvectorintEmu_C.$(DllSuf) vec.root dude.cxx vectorintEmuSel.h vectorintEmu.ref
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorintEmu.C+(5)' 2>&1 | grep -v 'Warning in <TClass::TClass>: no dictionary for class' > vectorintEmu.clog 2>&1

vectorintEmu: vectorintEmu.clog
	$(TestDiff)

make_tuple.log: runmake_tuple.C make_tuple.C make_tuple_draw.C
	$(CMDECHO) $(CALLROOTEXE) -b -q -l runmake_tuple.C 2>&1 | grep -v 'creating shared library' > make_tuple.log

make_tuple: make_tuple.log
	$(TestDiff)
