# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log

#Set the list of target to make while testing
TEST_TARGETS += array1test

# adjust the location of Rules.mk as needed.
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


mytest:
	$(CMDECHO) echo mytest > /dev/null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
# from runsometest.C

array1test.log: array1.C

array1test.short.log : array1test.log
ifneq ($(ClingWorkAroundUnnamedIncorrectFileLoc),)
	$(CMDECHO) cat array1test.log | sed -e 's/0x[^ ]*//g' -e 's/[A-Za-z]:.*array1.C/array1.C/g' \
		-e 's/(address: NA) class/ /' -e 's/, size = .*/=/' \
		-e 's/input_line_[0-9]*/array1.C/' -e 's/    4   /         3   /' -e 's/   11   /         10   /' > $@
else
	$(CMDECHO) cat array1test.log | sed -e 's/0x[^ ]*//g' -e 's/[A-Za-z]:.*array1.C/array1.C/g' \
		-e 's/(address: NA) class/ /' -e 's/, size = .*/=/' > $@
endif

array1test: array1test.short.log
	$(TestDiff)
