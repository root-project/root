# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *rflx*

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += headerParsingOnDemand

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
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk
include $(ROOTTEST_HOME)/scripts/Reflex.mk

# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

mytest:
	$(CMDECHO) echo mytest > /dev/null


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif



export HEADER_PARSING_ON_DEMAND:=1

execCheckComplexTypedefs.log : libcomplexTypedefs_dictrflx.$(DllSuf)

FullheaderParsingOnDemand.log: libFullheaderParsingOnDemand_dictrflx.$(DllSuf)

# The greps are there to find a common denominator among all platforms for 
# system headers.
headerParsingOnDemand.log: FullheaderParsingOnDemand.log
	$(CMDECHO) cat FullheaderParsingOnDemand.log |grep -v "Warning in <TInterpreter::RegisterModule()>: Header" | grep -v "status 0" | grep -v "memory" | grep -v "__functional_base"|grep -v "string" |grep -v "TRint" |grep -v "TApplication"> headerParsingOnDemand.log 2>&1

headerParsingOnDemand: headerParsingOnDemand.log
	$(TestDiff)


