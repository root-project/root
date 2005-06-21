ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest\)')/
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)/scripts/subdirectories .)

all: tests

test: tests

# Seed the path printing  engine
export CALLDIR:=.

TEST_TARGETS = $(SUBDIRS:%=%.test)
CLEAN_TARGETS = $(SUBDIRS:%=%.clean)
CLEANTEST_TARGETS = $(SUBDIRS:%=%.cleantest)


tests: $(TEST_TARGETS)
	@echo "All test succeeded"

clean: $(CLEAN_TARGETS)

cleantest: $(CLEANTEST_TARGETS)
	@echo "All test succeeded"

$(TEST_TARGETS): %.test:
	@(cd $*; $(MAKE) --no-print-directory test)

$(CLEAN_TARGETS): %.clean:
	@(cd $*; $(MAKE) --no-print-directory clean)

$(CLEANTEST_TARGETS): %.cleantest:
	@(cd $*; $(MAKE) --no-print-directory cleantest)

