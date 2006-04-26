TEST_TARGETS += currentTest
CLEAN_TARGETS += $(ALL_LIBRARIES) \
	ti_files $(DICTS) $(DICTS_H) ii_files \
	*.$(ObjSuf) *.$(DllSuf) *.lib *.def *.root \
	 so_locations *~ *.exp .def *.log

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

current_test.log : main Run.C $(ROOTV)
	$(CMDECHO)root -q -l -b "Run.C" > current_test.log 2>&1 

currentTest: current_test.log
	$(CMDECHO) diff -b current_test.log current_test.ref

$(DICTS) : %dict.C : %.h %linkdef.h 
	$(CMDECHO)rootcint -f $@ -c $^ 

$(SHARED_LIBS) : %.$(DllSuf) : %.$(ObjSuf) %dict.$(ObjSuf) 
	$(BuildFromObjs)


