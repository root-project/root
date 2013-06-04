TEST_TARGETS += newTest
CLEAN_TARGETS += $(ALL_LIBRARIES) ti_files $(DICTS) $(DICTS_H) ii_files \
	*.$(ObjSuf) *.$(DllSuf) *.lib *.def test.root so_locations *~ \
	*.exp .def *.log *.pdb *.rootmap 

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif
include $(ROOTTEST_HOME)/scripts/Rules.mk

local_all: main
.PHONY: main


#ifeq ($(PLATFORM),win32)
#CXXFLAGS += -TP
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
	$(CMDECHO) diff -b -u newTest.ref newTest.log 
newTest.log : Run.C $(SHARED_LIBS)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "Run.C" > newTest.log 2>&1
#@echo "All test succeeded in `pwd`"

$(DICTS) : %dict.C : %.h %linkdef.h $(ROOTCINT)
	$(CMDECHO)rootcint -f $@ -c $(filter %.h,$^) 


$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf) $(ROOTCORELIBS) $(ROOTV) $(ROOTCINT)
	$(BuildFromObjs)

.C.$(ObjSuf): $(ROOTCORELIBS) $(ROOTV) 
	$(CMDECHO)$(CXX) $(CXXFLAGS) -c $<

