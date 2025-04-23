# Replace Rules.mk for disabled test

SUBDIRS := $(shell $(ROOTTEST_HOME)/scripts/subdirectories .)

CLEAN_TARGETS_DIR = $(SUBDIRS:%=%.clean)
CLEAN_TARGETS += 

all:tests

test: tests ;

test:

check:

$(CLEAN_TARGETS_DIR): %.clean:
	@(cd $*; $(MAKE) --no-print-directory clean)

clean:  $(CLEAN_TARGETS_DIR)

.PHONY: clean tests all test 

include $(ROOTTEST_HOME)/scripts/Common.mk