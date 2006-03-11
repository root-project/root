TEST_TARGETS += newTest
CLEAN_TARGETS += local_clean

#include $(ROOTSYS)/test/Makefile.arch

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

local_all: main
.PHONY: main


#ifeq ($(PLATFORM),win32)
#CXXFLAGS += /TP
#LIBSFORLINK = $(LIBS)
#else
#ifeq ($(ARCH),solarisCC)
#CXXFLAGS += -ptr.
#else
#LIBSFORLINK = 
#endif
#endif

#leave here
ifeq ($(ARCH),aix5)
PACKAGES = namespace template InheritMulti
else
PACKAGES = namespace template nstemplate InheritMulti
endif

SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS = $(PACKAGES:%=%dict.C)
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS)
#	echo $(CXX)
#	echo $(CXXFLAGS)
#	echo $(SHARED_LIBS)

newTest: newTest.log
	$(CMDECHO) diff -b newTest.log newTest.ref
newTest.log : main Run.C
	$(CMDECHO)root -q -l -b "Run.C" > newTest.log 2>&1
#@echo "All test succeeded in `pwd`"

$(DICTS) : %dict.C : %.h %linkdef.h
	$(CMDECHO)rootcint -f $@ -c $^


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf)
	$(BuildFromObjs)

#	$(CMDECHO)$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
#		   "$(SOFLAGS)" "" $@ "$^" "$(ROOTLIBS)" > $*.build.log


local_clean:
	$(CMDECHO)rm -rf ti_files $(DICTS) $(DICTS_H) ii_files ti_files
	$(CMDECHO)rm -rf *.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root so_locations *~
	$(CMDECHO)rm -rf *.exp .def *.log *.pdb

clean: local_clean


.C.$(ObjSuf):
	$(CMDECHO)$(CXX) $(CXXFLAGS) -c $<

