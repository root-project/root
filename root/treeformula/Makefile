all: tests
test: tests

SUBDIRS = morrison sync casting

TEST_TARGETS = $(SUBDIRS:%=%.test)
CLEAN_TARGETS = $(SUBDIRS:%=%.clean)

tests: $(TEST_TARGETS)
	@echo "All test succeeded in `pwd`"

$(TEST_TARGETS): %.test:
	@(cd $*; gmake test)

clean: $(CLEAN_TARGETS)

$(CLEAN_TARGETS): %.clean:
	(cd $*; gmake clean)

