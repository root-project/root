# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += mytest templateConstructorTest

# adjust the location of Rules.mk as needed.
include ../../root/Rules.mk

mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

templateConstructorTest.log : constructor_C.$(DllSuf) runConstructor.C
	$(CMDECHO) root.exe -b -l -q runConstructor.C > templateConstructorTest.log 2>&1

templateConstructorTest: templateConstructorTest.log
	$(CMDECHO) diff -b templateConstructorTest.log templateConstructorTest.ref
