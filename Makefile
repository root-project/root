.DEFAULT_GOAL := clingtest

CLEAN_TARGETS += $(ALL_LIBRARIES)

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(PWD)/
endif

ifeq ($(MAKECMDGOALS),clingtest)
HAS_PYTHON=no
MAKE += HAS_PYTHON=no
else
ifeq ($(.DEFAULT_GOAL),clingtest)
HAS_PYTHON=no
MAKE += HAS_PYTHON=no
endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

clingtest: test

