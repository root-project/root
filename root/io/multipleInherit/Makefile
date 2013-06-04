TEST_TARGETS = multipleInheritTest
CLEAN_TARGETS = na_cxx.$(DllSuf) hout.root $(ALL_LIBRARIES) *.log

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


multipleInheritTest: run.log hout.root
	$(CMDECHO) diff -bw run.ref run.log && diff -bw write.ref write.log

run.log : na_cxx.$(DllSuf) hout.root re.C
	$(CMDECHO) $(CALLROOTEXE) -q -l -b re.C | sed -e 's/ [0-9a-fA-F]* / /g' -e 's/ 0[xX][0-9a-fA-F]* / /g' > run.log 2>&1 

na_cxx.$(DllSuf) : na.cxx na.h

hout.root: wr.C na_cxx.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b wr.C > write.log 2>&1





