ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

all: tests
test: tests




include $(ROOTTEST_HOME)/scripts/Rules.mk




