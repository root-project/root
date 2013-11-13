
TEST_TARGETS = $(SUBDIRS:%=%.test)
CLEAN_TARGETS = $(SUBDIRS:%=%.clean)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk



#tests: $(TEST_TARGETS)
#	$(CMDECHO)echo "All test succeeded"

#clean: $(CLEAN_TARGETS)


#$(TEST_TARGETS): %.test:
#	$(CMDECHO)(cd $*; gmake --no-print-directory test)

#$(CLEAN_TARGETS): %.clean:
#	$(CMDECHO)(cd $*; gmake --no-print-directory clean)


