TEST_TARGETS = multipleInheritTest
CLEAN_TARGETS = na_cxx.$(DllSuf) hout.root

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


multipleInheritTest: na_cxx.$(DllSuf) hout.root
	$(CMDECHO) root.exe -q -l -b re.C >& run.C 

na_cxx.$(DllSuf) : na.cxx na.h


hout.root: wr.C na_cxx.$(DllSuf)
	$(CMDECHO) root.exe -q -l -b wr.C >& /dev/null





