# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += *dict* *.so

#Set the list of target to make while testing
TEST_TARGETS += defaultArgTest

# adjust the location of Rules.mk as needed.
include ../../../root/Rules.mk

defaultArgTest: dict.cc mylist_C.so
	@echo Running 'mytest' in `pwd`
	$(CMDECHO) echo mytest > /dev/null

dict.cc: mylist.C $(ROOTSYS)/bin/rootcint
	$(CMDECHO) rootcint -f dict.cc -c mylist.C LinkDef.h

mylist_C.so:
	$(CMDECHO) root.exe -q -l -b build.C
