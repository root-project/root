all: tests

SUBDIRS = treeformula aclic tree io

TEST_TARGETS = $(SUBDIRS:%=%.test)
CLEAN_TARGETS = $(SUBDIRS:%=%.clean)

tests: $(TEST_TARGETS)
	@echo "All test succeeded"

clean: $(CLEAN_TARGETS)


$(TEST_TARGETS): %.test:
	@(cd $*; gmake --no-print-directory test)

$(CLEAN_TARGETS): %.clean:
	(cd $*; gmake --no-print-directory clean)

