# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += 

#Set the list of target to make while testing
TEST_TARGETS += stlIoTest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

-include vectorHolder_C.d
-include vtest_C.d
-include TestHelper_C.d

stllibs: load_C.$(DllSuf) vtest_C.$(DllSuf)

vectorHolder.o:
	@touch vectorHolder.o

dequeHolder.o:
	@touch vectorHolder.o

vtest.o:
	@touch vtest.o

dtest.o:
	@touch dtest.o

TestHelpers.o:
	@touch TestHelpers.o

%_C.$(DllSuf) : %.C %.o
	@rm -f $@
	$(CMDECHO) root.exe -q -l -b ../../build.C\(\"$<\"\)

#load_C.$(DllSuf): load.C load.o
#	@rm -f $@
#	$(CMDECHO) root.exe -q -l -b build.C\(\"load.C\"\) 

vtest_C.$(DllSuf): vtest.C vtest.o
	@rm -f $@
	$(CMDECHO) root.exe -q -l -b build.C\(\"vtest.C\"\)

stlIoTest: vectorHolder_C.$(DllSuf) vtest_C.$(DllSuf) TestHelpers_C.$(DllSuf)
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


