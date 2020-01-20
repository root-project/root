TEST_TARGETS += newTest

#leave here
ifeq ($(ARCH),aix5)
PACKAGES = namespace template InheritMulti
else
PACKAGES = namespace template nstemplate InheritMulti
endif
DICTS = $(PACKAGES:%=%dict.C)

CLEAN_TARGETS += $(ALL_LIBRARIES) ti_files $(DICTS) $(DICTS_H) ii_files \
	*.$(ObjSuf) *.$(DllSuf) *.lib *.def test.root so_locations *~ \
	*.exp .def *.log *.pdb *.rootmap 

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

local_all: main
.PHONY: main




SHARED_LIBS = $(PACKAGES:%=%.$(DllSuf)) 
DICTS_H = $(PACKAGES:%=%dict.h)

main: $(SHARED_LIBS)
#	echo $(CXX)
#	echo $(CXXFLAGS)
#	echo $(SHARED_LIBS)

newTest: newTest.log
	$(CMDECHO) diff -b -u newTest.ref newTest.log 
newTest.log : Run.C $(SHARED_LIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "Run.C" > newTest.log 2>&1
#@echo "All test succeeded in `pwd`"

$(DICTS) : %dict.C : %.h %linkdef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f $@ $(filter %.h,$^) 


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf) $(ROOTCORELIBS) $(ROOTV) $(ROOTCINT)
	$(BuildFromObjs)

.C.$(ObjSuf): $(ROOTCORELIBS) $(ROOTV) 
	$(CMDECHO)$(CXX) $(CXXFLAGS) -c $<

