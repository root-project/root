# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += EqualTest ParanTest

# adjust the location of Rules.mk as needed.
include ../../scripts/Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

EqualTest.log : equal_C.$(DllSuf)

EqualTest: EqualTest.log
	$(TestDiff)

t01.log: t01.C
	$(CMDECHO) root.exe -q -b -l t01.C > t01.log

ParanTest: t01.log
	$(CMDECHO) diff -b t01.log t01.ref