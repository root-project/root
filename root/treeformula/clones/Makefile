# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += nestedClones nestedClones2

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

nestedClones2.log: digiRun.C
	$(CMDECHO) root.exe -q -l -b digiRun.C >nestedClones2.log 2>&1

nestedClones2: nestedClones2.log
	$(CMDECHO) diff -b nestedClones2.log nestedClones2.ref

mytest:
	$(CMDECHO) echo mytest > /dev/null

nestedClones:
ifeq ($(FAIL),)
	$(WarnFailTest)
else
	$(CMDECHO) root.exe -b -q -l runNestedClones.C
endif


