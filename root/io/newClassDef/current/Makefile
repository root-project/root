TEST_TARGETS += currentTest
CLEAN_TARGETS += local_clean $(ALL_LIBRARIES)

#include $(ROOTSYS)/test/Makefile.arch

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

local_all: main
.PHONY: main

PACKAGES = namespace template InheritMulti

SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS = $(PACKAGES:%=%dict.C)
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS) array_cxx.$(DllSuf)
#	echo $(CXX)
#	echo $(CXXFLAGS)
#	echo $(SHARED_LIBS)

current_test.log : main Run.C
	$(CMDECHO)root -q -l -b "Run.C" > current_test.log 2>&1 
#	$(CMDECHO)echo "All test succeeded in `pwd`"

currentTest: current_test.log
	$(CMDECHO) diff -b current_test.log current_test.ref

$(DICTS) : %dict.C : %.h %linkdef.h
	$(CMDECHO)rootcint -f $@ -c $^


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf)
	$(BuildFromObjs)

#	$(CMDECHO)$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" "" $@ "$^" "$(ROOTLIBS)" > $*.build.log 2>&1


local_clean:
	$(CMDECHO)rm -rf ti_files $(DICTS) $(DICTS_H) ii_files ti_files
	$(CMDECHO)rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root so_locations *~
	$(CMDECHO)rm -rf *.exp .def *.log

clean: local_clean
