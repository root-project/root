# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += ReflexCMakeTest

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

HAVECTEST?=$(shell which ctest 2>/dev/null)
ifeq ($(findstring ctest,$(notdir $(HAVECTEST))),ctest)
ifeq ($(findstring gccxml,$(notdir $(HAVEGCCXML))),gccxml)
ReflexCMakeTest_Run:=yes
endif
endif
ReflexCMakeTest_Run?=no

ReflexCMakeConfig:
	$(CMDECHO)echo Configuring Reflex...
	$(CMDECHO)mkdir -p cmake-build; cd cmake-build; \
	cmake -DREFLEX_BUILD_TESTS=ON -DREFLEX_ENABLE_TESTS=ON -DCMAKE_COLOR_MAKEFILE=OFF $(ROOTSYS)/cint/reflex > /dev/null
	$(CMDECHO)echo Done.

ReflexCMakeBuild: ReflexCMakeConfig
	$(CMDECHO)echo Building Reflex...
	$(CMDECHO)cd cmake-build; $(MAKE) > /dev/null
	$(CMDECHO)echo Done.

ReflexCMakeTest.log: ReflexCMakeBuild
	$(CMDECHO)echo Running Reflex Tests...
	$(CMDECHO)cd cmake-build; ctest -Q -O ../$@ > /dev/null
	$(CMDECHO)echo Done.

ifeq ($(ReflexCMakeTest_Run),yes)
ReflexCMakeTest: ReflexCMakeTest.log
else
ReflexCMakeTest:
	$(CMDECHO)echo 'No CMake / GCCXML found; tests skipped.'
endif
