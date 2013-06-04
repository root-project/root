# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
# CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

# TEST_TARGETS += 

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

# Note that not-yet release test directory are listed in
# SpecialDirs

include $(ROOTTEST_HOME)/scripts/Rules.mk

