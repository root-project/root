CLEAN_TARGETS += main *Dict* Event.root *~ $(ALL_LIBRARIES) *.log

TEST_TARGETS += offsetTest

include ../../Rules.mk

offsetTest: TOffset_cxx.$(DllSuf)

TOffset_cxx.$(DllSuf) : TOffset.cxx TOffset.h $(ROOTSYS)/lib/libCore.$(DllSuf)
	$(CMDECHO) root -q -b -l TOffset.cxx++ > TOffset.log


