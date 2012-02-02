CLEAN_TARGETS += main *Dict* Event.root *~ $(ALL_LIBRARIES) *.log

TEST_TARGETS += offsetTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

offsetTest: TOffset_cxx.$(DllSuf)

TOffset_cxx.$(DllSuf) : TOffset.cxx TOffset.h $(ROOT_LOC)/lib/libCore.$(LibSuf)


