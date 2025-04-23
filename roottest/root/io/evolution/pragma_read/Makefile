# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) $(addprefix v2/,$(ALL_LIBRARIES)) $(addprefix v1/,$(ALL_LIBRARIES)) *.log *.clog test.root */*.d */*.log */*_cling*

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += 

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
include $(ROOTTEST_HOME)/scripts/Reflex.mk

v1/%_v1_cling.cpp: v1/%.h v1/%LinkDef.h $(ROOTCINT) $(ROOTV)
	$(CMDECHO) rootcint -f $@ v1/$*.h v1/$*LinkDef.h  > $@.log 2>&1 || (cat $@.log && exit 1)

v2/%_v2_cling.cpp: v2/%.h v2/%LinkDef.h $(ROOTCINT) $(ROOTV)
	$(CMDECHO) rootcint -f $@ v2/$*.h v2/$*LinkDef.h  > $@.log 2>&1 || (cat $@.log && exit 1)

libTest%.$(DllSuf):%/AliAODForwardMult_v1_cling.$(ObjSuf) %/AliAODForwardMult.$(ObjSuf)
	$(CMDECHO) cp $*/AliAODForwardMult_$*_cling$(PcmSuf) .
	$(BuildFromObjs)

libTest%.$(DllSuf):%/AliAODForwardMult_v2_cling.$(ObjSuf) %/AliAODForwardMult.$(ObjSuf)
	$(CMDECHO) cp $*/AliAODForwardMult_$*_cling$(PcmSuf) .
	$(BuildFromObjs)

libTestv2.$(DllSuf):v2/AliAODForwardHeader.$(ObjSuf)

execTestv1.log: libTestv1.$(DllSuf)

execTestv2.log: libTestv2.$(DllSuf) execTestv1.success
