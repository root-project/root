CLEAN_TARGETS += *.so Class*.root typedef*.root $(ALL_LIBRARIES) *.clog *.log *.tmp skiptestfile.xml arr.root FixedArray*.root boolUpdate.root  data.root  skiptestfile.root $(FILES) $(FILESFAIL) *.success $(STL_FILES) atlas32 inheritstl.root
TEST_TARGETS += StlTypeEvolution

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

%.root: lib%.$(DllSuf) ../rootalias.C ../run.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../run.C("$*")' > lib$*.log 2>&1 

STL_FILES := map.root vector.root multimap.root list.root

STL_LIBS := $(patsubst %.root,lib%.$(DllSuf),$(STL_FILES))

STL_LOGS = $(patsubst %.root,check.%.log,$(STL_FILES))

STL_SUCCESSES = $(patsubst %.root,check%.success,$(STL_FILES))

.PHONY: files

.PRECIOUS: lib%.$(DllSuf) %.root

stl_files: $(STL_FILES)

stl_logs: $(STL_LOGS)

StlTypeEvolution: $(patsubst %.root,check%.success,$(STL_FILES))

$(STL_LIBS): lib%.$(DllSuf): ../rootalias.C ../maptovector.cxx ../maptovector.h ../compile.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../compile.C(1, "$*")' > $*.log 2>&1

$(STL_LOGS) : check.%.log: %.root $(STL_FILES) ../read.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../read.C("$(STL_FILES)", "", 1, "$*")' 2>&1 | grep -v Processing > $@

$(STL_SUCCESSES) : check%.success: check.%.log
	$(CMDECHO) diff -b check.stl.ref $< && touch $@

