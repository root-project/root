# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += test.root output create.out def_C*

#Set the list of target to make while testing
TEST_TARGETS += testForeignDraw

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

test.root: def.C def.h
	$(CMDECHO) root.exe -b -l -q def.C+ > create.out

output: Run.C test.root
	$(CMDECHO) root.exe -b -l -q Run.C >| output 2>&1

testForeignDraw: output
	$(CMDECHO) diff -u output good.output

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


