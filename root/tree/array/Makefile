# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES)

#Set the list of target to make while testing
TEST_TARGETS += objArrayTest objArrayTest2

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

objArrayTest.log: TestObj_cpp.so
	$(CMDECHO) root.exe -q -b -l run.C > $@

objArrayTest: objArrayTest.log
	$(CMDECHO) diff -b objArrayTest.ref objArrayTest.log 

objArrayTest2.log: TestObj_cpp.so save_C.so
	$(CMDECHO) root.exe -q -b -l run2.C 2>&1 | tee objArrayTest2full.log | grep -v ' at 0x' > $@

objArrayTest2: objArrayTest2.log
	$(CMDECHO) diff -b objArrayTest2.ref objArrayTest2.log 

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


