ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)/scripts/subdirectories .)
include $(ROOTTEST_HOME)/scripts/Rules.mk

#all: tests

#test: tests



#TEST_TARGETS = $(SUBDIRS:%=%.test)
#CLEAN_TARGETS = $(SUBDIRS:%=%.clean)

#tests: $(TEST_TARGETS)
#	@echo "All test succeeded"

#clean: $(CLEAN_TARGETS)


#$(TEST_TARGETS): %.test:
#	@(cd $*; gmake --no-print-directory test)

#$(CLEAN_TARGETS): %.clean:
#	(cd $*; gmake --no-print-directory clean)

