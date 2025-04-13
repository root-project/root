# %ROOTTEST-duration%:8
# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.clog *.log \
	vec.root vectorintSel.h vectorintEmuSel.h *.rootmap 

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += vectorint vectorintEmu 

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

# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


define hideLibraryCreation
   grep -v -e ACLiC -e 'creating shared library' -e 'Creating library' \
	     -e _cint.cxx -e '_[0-9]*.cxx' -e '_.cxx' \
	     -e 'Warning in <TClassTable::Add>: class generatedSel already in TClassTable'
endef

vec.root: $(CORELIBS) runvectorint_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runvectorint.C+(3)' > vec.create.log 2>&1 || handleError.sh --result=$$? --log=vec.create.log --test=$@ --rm=$@

vectorintSel.h: dude.h dude.C $(CORELIBS) runvectorint_C.$(DllSuf) vec.root
	$(CMDECHO) $(CALLROOTEXE) -b -q -l 'runvectorint.C+(4)' > vectorintSel_h.create.log 2>&1 || handleError.sh --result=$$? --log=vectorintSel_h.create.log --test=$@ --rm=$@

#vectorintSel_h.$(DllSuf): vectorintSel.h dude.h dude.C vec.root

vectorint.clog: vectorintSel_h.$(DllSuf) $(CORELIBS) runvectorint_C.$(DllSuf) dude.C vec.root vectorintSel.h
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorint.C+(5)' > vectorint.clog 2>&1 || handleError.sh --result=$$? --log=$@ --test=$@ 

vectorint: vectorint.clog
	$(TestDiffW)

vectorintEmuSel.h: dude.h dude.cxx $(CORELIBS) runvectorintEmu_C.$(DllSuf) vec.root
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorintEmu.C+(4)' > vectorintEmuSel_h.create.log 2>&1 || handleError.sh --result=$$? --log=vectorintEmuSel_h.create.log --test=$@ --rm=$@

#vectorintEmuSel_h.$(DllSuf): dude.h dude.cxx

vectorintEmu.clog: vectorintEmuSel_h.$(DllSuf) $(CORELIBS) runvectorintEmu_C.$(DllSuf) vec.root dude.cxx vectorintEmuSel.h vectorintEmu.ref
	$(CMDECHO) $(CALLROOTEXE)  -b -q -l 'runvectorintEmu.C+(5)' > vectorintEmu.raw.clog 2>&1 ; result=$$? ; cat vectorintEmu.raw.clog | grep -v 'Warning in <TClass::TClass>: no dictionary for class' > vectorintEmu.clog 2>&1 ; \
		test $$result -eq 0 || handleError.sh --result=$$? --log=$@ --test=$@

vectorintEmu: vectorintEmu.clog
	$(TestDiffW)

