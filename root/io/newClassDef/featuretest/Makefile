
TEST_TARGETS += feature
CLEAN_TARGETS += local_clean

#include $(ROOTSYS)/test/Makefile.arch

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

feature: Run.C
	$(CMDECHO)root -q -l -b Run.C >run.out 2>&1
#	$(CMDECHO)echo "All test succeeded in `pwd`"

clean:
	$(CMDECHO)rm -f ti_files *~ *.so
	$(CMDECHO)rm -f .def *.dll *.exp *.lib *.def

