# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *_rflx.cpp *_rflxdriver.C *_rflx_tmp1.C *_rflx_tmp2.C iotypes.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

RFLXDICTTESTS:= classVersion_rflx properties_rflx updateMembers_rflx iotype_rflx
TEST_TARGETS += $(RFLXDICTTESTS) ReflexCintexOrder CintexReflexOrder Tools

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

# the driver which includes the relevant files for the library:
%_rflxdriver.C: %_rflx.cpp %_test.C TestHelper.h
	$(CMDECHO) echo '#ifndef __CINT__' > $@
	$(CMDECHO) echo '#include "TestHelper.h" // some helper functions' >> $@
	$(CMDECHO) echo '#include "TSystem.h" // the dictionary' >> $@
	$(CMDECHO) echo '#include "$*_rflx.cpp" // the dictionary' >> $@
	$(CMDECHO) echo '#include "$*_test.C" // the test' >> $@
	$(CMDECHO) echo '#endif' >> $@
	$(CMDECHO) echo 'void $(@:.C=()){ gSystem->Setenv("LINES","-1"); $*_test();} // forward' >> $@

%_rflx_tmp1.C: %_rflxdriver.C
	$(CMDECHO) echo '{gSystem->Load("libCintex"); gSystem->Load("libTree"); gROOT->ProcessLine(".L $<+");}' > $@

%_rflx_tmp2.C: %_rflxdriver.C
	echo '{gSystem->Load("libCintex"); gSystem->Load("libTree"); gROOT->ProcessLine(".x $<+");}' > $@

# and the actual test, building and "running" the library
%_rflx.log: %_rflx_tmp1.C %_rflx_tmp2.C
	(root -l -b -q $*_rflx_tmp1.C > /dev/null 2>&1 \
	&& root -l -b -q $*_rflx_tmp2.C | grep -v 'Processing ' > $@ )|| cat $@

ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
$(RFLXDICTTESTS): %_rflx: %_rflx.log %_rflx.ref
	$(TestDiff)

ReflexCintexOrder: ReflexCintexOrder.log
	$(TestDiff)
CintexReflexOrder: CintexReflexOrder.log
	$(TestDiff)

Tools: Tools.log
	$(TestDiff)

exechepmc.log: libhepmc_dictrflx.$(DllSuf)

else

$(RFLXDICTTESTS): %_rflx:
$(TEST_TARGETS):
ifeq ($(HAVEGCCXML),"nocintex")
	$(CMDECHO) echo "Your ROOT build does not include cintex: skipping test $@"
else
	$(CMDECHO) echo "GCCXML not in \$$PATH: skipping test $@"
endif
endif

properties_rflx: GENREFLEXFLAGS=--iocomments
