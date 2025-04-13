# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog myclass.root pairs.root tmpltd32.root tmpltd.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += classref tempParse pairs

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk

myclass.root: RunMyClass_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q RunMyClass.C+ > myclass_root.log

classref.log: myclass.root

classref: classref.log
	$(TestDiff)

usedBits.log: usedBits_C.$(DllSuf)

usedBits: usedBits.log
	$(TestDiff)

tempParse.log: tempParse_C.$(DllSuf)

tempParse: tempParse.log
	$(TestDiff)

pairs: pairs.clog
	$(TestDiff)

execProperties.success: EXTRA_DIFFOPTS += -w

execInterpClassNew.log: classlib_cxx.$(DllSuf)

execReadTmplt.clog: execTmpltD.success execTmpltD32.success

execState.log: libState_cxx.$(DllSuf)

execSimpleVector.log: libsimpleVector_dictrflx.$(DllSuf)

execBrowseNonAccessibleType.log: execBrowseNonAccessibleType.C

execDuplicate.log: libDuplicate_cxx.$(DllSuf)
