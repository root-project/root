# %ROOTTEST-duration%:0
# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog

#TEST_TARGETS += FindSource ClassInModule
ifneq ($(PLATFORM),win32)
TEST_TARGETS += MakeIndex
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

FindSource.log: runFindSource.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $< 2>&1 | grep -v 'RooFit' | grep -v 'Copyright' > $@ 

FindSource: FindSource.log
	$(TestDiff)

ClassInModule.log: runClassInModule.C
	$(CMDECHO)$(CALLROOTEXE) -l -b -q $< 2>&1 | grep -v 'RooFit' | grep -v 'Copyright' | grep -v 'no dictionary for' > $@ 

ClassInModule: ClassInModule.log
	$(TestDiff)

MakeIndex_filtered.log: MakeIndex.log
	$(CMDECHO)grep -v 'dot -' $< \
	| grep -v '^$$' \
	| grep -v 'Checking for Graphviz' \
	| grep -v 'dot: not found' \
	| grep -v 'dot: command not found' \
	| grep -v 'dot version' \
	| grep -v 'RooFit' \
	| grep -v 'Copyright' \
	| grep -v 'roofit.sourceforge' \
	| grep -v 'no dictionary for' \
	| grep -v 't open file' \
	| grep -v ' scanning ' \
	| grep -v ' Times-Roman' \
	| grep -v 'ldap_initialize() failed:' \
	| grep -v '_Index.html' > $@

MakeIndex: MakeIndex_filtered.log
	$(TestDiff)

ifeq ($(ClingWorkAroundLackOfModule)$(PLATFORM),yesmacosx)
MakeIndex.log:  CALLROOTEXE += -e '\#define _MALLOC_MALLOC_H_ 0'
endif
ifneq ($(ClingWorkAroundMissingAutoLoading),)
MakeIndex.log:  CALLROOTEXE += -e 'gSystem->Load("libHtml");'
endif
ifneq ($(ClingWorkAroundLackOfModule),)
ifeq ($(shell root-config --has-qt),yes)
MakeIndex.log: CALLROOTEXE += -e 'gSystem->Load("libQtRoot"); gSystem->Load("libGQt.so"); '
endif
ifeq ($(shell root-config --has-x11),yes)
MakeIndex.log: CALLROOTEXE += -e 'gSystem->Load("libGX11"); '
endif
ifeq ($(shell root-config --has-opengl),yes)
MakeIndex.log: CALLROOTEXE += -e 'gSystem->Load("libRGL"); '
endif
endif
