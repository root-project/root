# %ROOTTEST-duration%:1
CLEAN_TARGETS += *.pyc

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/roottest/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

ifeq ($(HAS_PYTHON),)
   export HAS_PYTHON := $(shell root-config --has-python)
endif

ifeq ($(HAS_PYTHON),yes)
include $(ROOTTEST_HOME)/scripts/Rules.mk
# prevent spurious printout of control characters that occur with certain
# combinations of readline and ncurses (e.g. 5.2 and 5.6, respectively)
export TERM=vt100
else
include $(ROOTTEST_HOME)/scripts/Disable.mk
endif
