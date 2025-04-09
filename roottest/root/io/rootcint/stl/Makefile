# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *_dict*

#Set the list of target to make while testing
TEST_TARGETS += rootcintStlTest

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

RCTEST_H = $(wildcard test?.h) Safr.h
RCTEST_DICT = $(subst .h,_dict.cxx,$(RCTEST_H))
RCTEST_DICTO = $(subst .cxx,.$(ObjSuf),$(RCTEST_DICT))

$(RCTEST_DICT) : %_dict.cxx : %.h %LinkDef.h
	$(CMDECHO) rootcint -f $@ $^

MAPTEST_C = map1.C map2.C
MAPTEST_DICT =  $(subst .C,_dict.cxx,$(MAPTEST_C))
MAPTEST_DICTO = $(subst .cxx,.$(ObjSuf),$(MAPTEST_DICT))

$(MAPTEST_DICT) : %_dict.cxx : %.C emptyLinkdef.h
	$(CMDECHO) rootcint -f $@ $^

rootcintStlTest : $(RCTEST_DICTO) $(MAPTEST_DICTO)

