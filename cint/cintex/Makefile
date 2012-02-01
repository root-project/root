# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *.o *.obj *_rflx.cpp test_*.rootmap data.root tree.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

HAVEGCCXML := $(shell which gccxml 2>/dev/null)
ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
  HAVECINTEX := $(findstring cintex,$(shell root-config --features))
endif

# also in Rules.mk, but that needs TEST_TARGETS to be defined first.
ifneq ($(HAS_PYTHON),) 
   HAVEPYTHON := $(HAS_PYTHON)
else
   HAVEPYTHON := $(shell root-config --has-python)
endif

ifeq ($(HAVECINTEX),cintex)
 TEST_TARGETS += test_Cintex test_Persistency
 ifeq ($(HAVEPYTHON),yes)
  TEST_TARGETS += test_PyCintex_basics
 endif
endif

# Search for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the
#  ROOTTEST_HOME variable for use by the SUBDIRECTORIES variable and is 
#  exported to eliminate the overhead of findding the path again during 
#  recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 
#    'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from 
#  the home directory.

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

# trigger creation of cintex/test directory if out-of-source
CINTEXTESTDIR   := $(ROOT_SRCDIR)/cint/cintex/test
CINTEXTESTDIRDEP:= $(ROOT_SRCDIRDEP)/cint/cintex/test

CINTEXTESTDICTH := dict/CintexTest.h
CINTEXTESTSEL   := dict/selection.xml
CINTEXTESTDICTS := $(subst .h,_rflx.cpp,$(notdir $(CINTEXTESTDICTH)))
CINTEXTESTDICTO  = $(subst .cpp,.$(ObjSuf),$(CINTEXTESTDICTS))
CINTEXTESTDICT   = test_$(subst _rflx.$(ObjSuf),Rflx.$(DllSuf),$(CINTEXTESTDICTO))

$(CINTEXTESTDICTS): $(CINTEXTESTDIRDEP)/$(CINTEXTESTDICTH) $(CINTEXTESTDIRDEP)/$(CINTEXTESTSEL)
	$(CMDECHO) $(GENREFLEX) $(CINTEXTESTDIR)/$(CINTEXTESTDICTH) -s $(CINTEXTESTDIR)/$(CINTEXTESTSEL) --rootmap=$(CINTEXTESTDICT).rootmap --rootmap-lib=$(CINTEXTESTDICT) --comments $(GENREFLEXFLAGS) -I"$(ROOTSYS)/include" $(GENREFLEXCXXFLAGS) > $@.log 2>&1

$(CINTEXTESTDICT): $(CINTEXTESTDICTO) $(ROOT_LOC)/lib/libReflex.$(LibSuf)
	$(BuildFromObj)

test_Cintex.log: $(CINTEXTESTDIRDEP)/test_Cintex.C $(CINTEXTESTDICT)
	$(CMDECHO)$(CALLROOTEXE) -b -q -l $(CINTEXTESTDIR)/test_Cintex.C 2>&1 | grep -v 'Processing ' > $@ || ( cat $@ && false )

test_Persistency.log: $(CINTEXTESTDIRDEP)/test_Persistency.C $(CINTEXTESTDICT)
	$(CMDECHO)$(CALLROOTEXE) -b -q -l $(CINTEXTESTDIR)/test_Persistency.C 2>&1 | grep -v 'Processing ' > $@ || ( cat $@ && false )

test_PyCintex_basics.log: $(CINTEXTESTDIRDEP)/test_PyCintex_basics.py $(CINTEXTESTDICT)
	$(CMDECHO)unset DISPLAY; $(PYTHON) $(CINTEXTESTDIR)/test_PyCintex_basics.py - -b 2> /dev/null | grep -v ' tests in ' > $@|| ( cat $@ && false )

# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

test_Cintex test_Persistency test_PyCintex_basics: test_%: test_%.log
	$(TestDiff)

# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)


