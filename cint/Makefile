all: tests


SUBDIRS = specialobj template other typedef_global const

TEST_TARGETS = $(SUBDIRS:%=%.test)

tests: $(TEST_TARGETS)
	@echo "All test succeeded"

$(TEST_TARGETS): %.test:
	@(cd $*; gmake test)

