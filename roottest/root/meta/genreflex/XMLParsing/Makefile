# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog dicts/*

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
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

XMLFILES := $(wildcard experimentsSelectionXMLs/*.xml)
XMLLOGFILES := $(addprefix genreflex_,$(notdir $(patsubst %.xml,%.log,$(XMLFILES))))
XMLSUCCESS := $(patsubst %.log,%.success,$(XMLLOGFILES))
XMLTARGETS := $(patsubst %.log,%,$(XMLLOGFILES))

TEST_TARGETS += $(XMLTARGETS)

include $(ROOTTEST_HOME)/scripts/Rules.mk

dicts/.log:
	$(CMDECHO) mkdir -p dicts; touch dicts/.log

$(XMLLOGFILES): genreflex_%.log: experimentsSelectionXMLs/%.xml dicts/.log
	$(CMDECHO) genreflex emptyHeader.h  --silent -o dicts/$*.cxx -s experimentsSelectionXMLs/$*.xml >$@ 2>&1

$(XMLSUCCESS): %.success: %.log %.ref
	$(SuccessTestDiff) && touch $@

$(XMLTARGETS): genreflex_%: genreflex_%.success

