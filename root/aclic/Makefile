all: tests
test: tests

SUBDIRS = offset

TEST_TARGETS = $(SUBDIRS:%=%.test)

tests: $(TEST_TARGETS)
	@echo "All test succeeded in `pwd`"

$(TEST_TARGETS): %.test:
	@(cd $*; gmake test)

