# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root copy.root *.so *.out *.d 

#Set the list of target to make while testing
TEST_TARGETS += missingClass

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

missingClass: test.root copy.root
		  $(CMDECHO) root.exe -q -l -b read.C

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

test.root: classes.C write.C
		  $(CMDECHO) root.exe -q -l -b write.C > write.out

copy.root: classes.C copy.C
		  $(CMDECHO) root.exe -q -l -b copy.C > copy.out
