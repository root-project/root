# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *_dict*

#Set the list of target to make while testing
TEST_TARGETS += rootcintStlTest

# adjust the location of Rules.mk as needed.
include ../../../Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

RCTEST_H = $(wildcard test?.h)
RCTEST_DICT = $(subst .h,_dict.cxx,$(RCTEST_H))
RCTEST_DICTO = $(subst .cxx,.$(ObjSuf),$(RCTEST_DICT))

$(RCTEST_DICT) : %_dict.cxx : %.h %LinkDef.h
	$(CMDECHO) rootcint -f $@ -c $^



rootcintStlTest : $(RCTEST_DICTO)