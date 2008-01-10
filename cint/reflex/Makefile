# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *_rflx.cpp *_rflxdriver.C

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

RFLXDICTTESTS:= classVersion_rflx properties_rflx
TEST_TARGETS += $(RFLXDICTTESTS) ReflexCintexOrder CintexReflexOrder

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


# Generic chain of files for running Reflex tests.
# Naming scheme: MyReflex_test() in MyReflex_test.C tests
# the dictionary MyReflex_rflx.cxx which is generated from
# MyReflex.h with MyReflex_selection.xml.

ifeq ($(PLATFORM),win32)
GENREFLEX = genreflex.bat
else
GENREFLEX = genreflex
endif

# The dictionary:
%_rflx.cpp: %_selection.xml %.h
	$(CMDECHO) $(GENREFLEX) $*.h -s $*_selection.xml $(GENREFLEXFLAGS) -I$(ROOTSYS)/include > $@.log

# the driver which includes the relevant files for the library:
%_rflxdriver.C: %_rflx.cpp %_test.C TestHelper.h
	$(CMDECHO) echo '#include "TestHelper.h" // some helper functions' > $@
	$(CMDECHO) echo '#include "TSystem.h" // the dictionary' >> $@
	$(CMDECHO) echo '#include "$*_rflx.cpp" // the dictionary' >> $@
	$(CMDECHO) echo '#include "$*_test.C" // the test' >> $@
	$(CMDECHO) echo 'void $(@:.C=()){ gSystem->Setenv("LINES","-1"); $*_test();} // forward' >> $@

# and the actual test, building and "running" the library
%_rflx.log: %_rflxdriver.C
	$(CMDECHO) echo '{gSystem->Load("libCintex"); gROOT->ProcessLine(".x $<+");}' | root -l -b 2>&1 | grep -v 'creating shared' > $@

# We only run these tests if libCintex exists
ifneq ($(findstring cintex,$(shell root-config --features)),)
$(RFLXDICTTESTS): %_rflx: %_rflx.log %_rflx.ref
	$(TestDiff)

ReflexCintexOrder: ReflexCintexOrder.log
	$(TestDiff)
CintexReflexOrder: CintexReflexOrder.log
	$(TestDiff)
else
$(RFLXDICTTESTS): %_rflx:
$(TEST_TARGETS):
	$(CMDECHO) echo "Your ROOT build does not include cintex: skipping test $@"
endif

properties_rflx: GENREFLEXFLAGS=--iocomments
