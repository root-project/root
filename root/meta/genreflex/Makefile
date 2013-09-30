# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += mytest

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

XMLFILES := $(wildcard experimentsSelectionXMLs/*.xml)
XMLLOGFILES := $(notdir $(patsubst %.xml,%.log,$(XMLFILES)))
XMLTARGETS := $(patsubst %.xml,,$(XMLFILES))
TEST_TARGET += $(XMLTARGETS)

include $(ROOTTEST_HOME)/scripts/Rules.mk

# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Old test
# assertXMLReader: assertXMLReader.log
# 	$(TestDiff)

assertXMLReader: $(XMLLOGFILES)
$(XMLLOGFILES) : %.log : experimentsSelectionXMLs/%.xml
	$(CMDECHO) genreflex emptyHeader.h  --quiet -o $*.cxx -s experimentsSelectionXMLs/$*.xml >  $@


# $(XMLTARGETS) : genreflex%.success: %.log %.ref
# 	$(SuccessTestDiff) && touch $@
# 	$(subst .log,,$(XMLLOGFILES)) : genreflex%: genreflex%.success
