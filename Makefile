
ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(PWD)/
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

