# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

-include load_C.d
-include vtest_C.d
.PHONY: vtest.o load.o

load_C.$(DllSuf): load.C load.o
	$(CMDECHO) root.exe -q -l -b build.C\(\"load.C\"\) && touch $?

vtest_C.$(DllSuf): vtest.C vtest.o
	$(CMDECHO) root.exe -q -l -b build.C\(\"vtest.C\"\)

stlIoTest: load_C.$(DllSuf) vtest_C.$(DllSuf)
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


