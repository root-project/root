ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

ifeq ($(HAS_PYTHON),)
   export HAS_PYTHON = $(shell root-config --has-python)
endif

ifeq ($(HAS_PYTHON),yes)
include $(ROOTTEST_HOME)/scripts/Rules.mk
else
include $(ROOTTEST_HOME)/scripts/Disable.mk
endif
