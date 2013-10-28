CLEAN_TARGETS += *.log xr.*
TEST_TARGETS += root_5627

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

root_5627.log: runGenreflex.sh
	$(CMDECHO) ./runGenreflex.sh >$@ 2>&1

root_5627: root_5627.log
	$(TestDiff) && touch $@
