
TEST_TARGETS += feature
CLEAN_TARGETS += $(ALL_LIBRARIES) run.out

#include $(ROOTSYS)/test/Makefile.arch

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

run.out: Run.C ClassTrick_C.$(DllSuf)  Class_C.$(DllSuf) typeidtest_C.$(DllSuf)
	$(CMDECHO)root -q -l -b Run.C >run.out 2>&1

feature: run.out


#	$(CMDECHO)echo "All test succeeded in `pwd`"

#clean:
#	$(CMDECHO)rm -f ti_files *~ *.so
#	$(CMDECHO)rm -f .def *.dll *.exp *.lib *.def

