
TEST_TARGETS = $(SUBDIRS:%=%.test)
CLEAN_TARGETS = $(SUBDIRS:%=%.clean)

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk



#tests: $(TEST_TARGETS)
#	$(CMDECHO)echo "All test succeeded"

#clean: $(CLEAN_TARGETS)


#$(TEST_TARGETS): %.test:
#	$(CMDECHO)(cd $*; gmake --no-print-directory test)

#$(CLEAN_TARGETS): %.clean:
#	$(CMDECHO)(cd $*; gmake --no-print-directory clean)


