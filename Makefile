.DEFAULT_GOAL := summary

CLEAN_TARGETS += $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(CURDIR)/
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

clingtest: test

