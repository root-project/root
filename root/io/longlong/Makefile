# This is a template for all makefile.

#Set the list of files to be delete by clean:
CLEAN_TARGETS += longlongdict.8

#Set the list of target to make while testing
TEST_TARGETS += longlongtest

# adjust the location of Rules.mk as needed.
include ../../Rules.mk

longlongdict.C: Class.h $(ROOTSYS)/bin/rootcint
	$(CMDECHO)rootcint -f longlongdict.C -c Class.h linkdef.h

longlongtest: longlongdict.C
	$(CMDECHO) test ! `(grep "long long" longlongdict.C > /dev/null) && echo $?`

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif


