# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *.out *.so *.d

#Set the list of target to make while testing
TEST_TARGETS += stringPassingTest

# adjust the location of Rules.mk as needed.
include ../../../root/Rules.mk

stringPassingTest: passingcint.out passing.out
		  $(CMDECHO)diff passingcint.out passingcint.ref
		  $(CMDECHO)diff passing.out passing.ref

passingcint.out : t01.C runcint.C
		  $(CMDECHO)root.exe -q -b -l runcint.C > passingcint.out

passing.out : t01.C run.C
		  $(CMDECHO)root.exe -q -b -l run.C > passing.out

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

