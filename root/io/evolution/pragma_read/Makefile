# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog test.root */*.$(ObjSuf) */*.d */*.log */*_cint*

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += 

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

libTest%.$(DllSuf):%/AliAODForwardMult_cint.$(ObjSuf) %/AliAODForwardMult.$(ObjSuf) $(ROOT_LOC)/lib/libHist.$(DllSuf)
	$(BuildFromObjs) 

libTestv2.$(DllSuf):v2/AliAODForwardHeader.$(ObjSuf)

execTestv1.log: libTestv1.$(DllSuf)

execTestv2.log: libTestv2.$(DllSuf)
