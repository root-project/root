# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += typedefAndScope

# adjust the location of Rules.mk as needed.
include ../../root/Rules.mk

typedefAndScope.log: testclass.h
	$(CMDECHO) root.exe -q -b -l testclass.h+ > typedefAndScope.log

typedefAndScope: typedefAndScope.log
	$(CMDECHO) diff -b typedefAndScope.log typedefAndScope.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


