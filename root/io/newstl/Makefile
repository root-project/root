# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

-include load_C.d
-include vtest_C.d

stllibs: load_C.$(DllSuf) vtest_C.$(DllSuf)

load.o:
	@touch load.o

vtest.o:
	@touch vtest.o

load_C.$(DllSuf): load.C load.o
	@rm -f $@
	$(CMDECHO) root.exe -q -l -b build.C\(\"load.C\"\) 

vtest_C.$(DllSuf): vtest.C vtest.o
	@rm -f $@
	$(CMDECHO) root.exe -q -l -b build.C\(\"vtest.C\"\)

stlIoTest: load_C.$(DllSuf) vtest_C.$(DllSuf)
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


