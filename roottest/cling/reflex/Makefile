# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *_rflx.cpp *_rflxdriver.C *_rflx_tmp1.C *_rflx_tmp2.C iotypes.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

RFLXDICTTESTS:= classVersion_rflx properties_rflx  iotype_rflx
TEST_TARGETS += $(RFLXDICTTESTS)
# Tools

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
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

.PRECIOUS: properties_rflx_tmp1.C iotype_rflx_tmp1.C updateMembers_rflx_tmp1.C properties_rflxdriver.C updateMembers_rflxdriver.C iotype_rflxdriver.C classVersion_rflx_tmp1.C iotype_rflx_tmp2.C properties_rflx_tmp2.C updateMembers_rflx_tmp2.C classVersion_rflxdriver.C properties_rflx.cpp classVersion_rflx_tmp2.C iotype_rflx.cpp iotype32_rflx.cpp classVersion_rflx.cpp updateMembers_rflx.cpp

iotype_rflxdriver.$(DllSuf): iotype_rflx.o iotype32_rflx.o
	$(BuildFromObjs)

%_rflxdriver.$(DllSuf): %_rflx.o
	$(BuildFromObj)

%_rflx_tmp1.C: %_rflxdriver.$(DllSuf)
	$(CMDECHO) echo '{gSystem->Load("libTree"); gROOT->ProcessLine(".L $<");}' > $@

%_rflx_tmp2.C: %_rflxdriver.$(DllSuf)
	$(CMDECHO) echo '{gSystem->Load("libTree"); gROOT->ProcessLine(".L $<"); gROOT->ProcessLine(".x $*_test.C"); }' > $@

RFLXDICTTESTS_DRIVERS_DLL := $(addsuffix driver_C.$(DllSuf),$(RFLXDICTTESTS))

# and the actual test, building and "running" the library
$(RFLXDICTTESTS_DRIVERS_DLL): %_rflxdriver_C.$(DllSuf): %_rflx_tmp1.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $*_rflx_tmp1.C > $*_rflxdriver_C.log 2>&1 || handleError.sh --result=$$? --log=$*_rflxdriver_C.log

%_rflx.log: %_rflxdriver_C.$(DllSuf) %_rflx_tmp1.C %_rflx_tmp2.C
	$(CMDECHO) $(CALLROOTEXE) -l -b -q $*_rflx_tmp2.C 2>&1 | grep -v  'Processing ' | grep -v 'creating shared' > $@ || handleError.sh --result=$$? --log=$@

$(RFLXDICTTESTS): %_rflx: %_rflx.log %_rflx.ref
	$(TestDiff)

ReflexCintexOrder: ReflexCintexOrder.log
	$(TestDiff)
CintexReflexOrder: CintexReflexOrder.log
	$(TestDiff)

Tools: Tools.log
	$(TestDiff)

exechepmc.log: libhepmc_dictrflx.$(DllSuf)

properties_rflx: GENREFLEXFLAGS=--iocomments
