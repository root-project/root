# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog myclass.root pairs.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += classref tempParse pairs

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

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