# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += nestedClones

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

nestedClones:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) root.exe -b -q -l runNestedClones.C
endif


