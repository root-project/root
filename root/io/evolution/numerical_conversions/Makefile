CLEAN_TARGETS += *.so Class*.root typedef*.root $(ALL_LIBRARIES) *.clog *.log *.tmp skiptestfile.xml arr.root FixedArray*.root boolUpdate.root  data.root  skiptestfile.root $(FILES) $(FILESFAIL) *.success $(STL_FILES) atlas32 inheritstl.root
TEST_TARGETS +=  NumericalTypeEvolution

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


FILES := int.root float16.root double32.root regular.root char.root  \
       short.root long.root longlong.root uchar.root ushort.root \
       uint.root ulong.root ulonglong.root float.root double.root \
       double32enough.root double32mantis.root \
       float16enough.root float16mantis.root
FILESFAIL := double32tooshort.root float16tooshort.root
ifneq ($(FAST),)
FILES := int.root double32.root longlong.root float.root
FILESFAIL := 
endif

LIBS := $(patsubst %.root,lib%.$(DllSuf),$(FILES) $(FILESFAIL))

LOGS = $(patsubst %.root,check.%.log,$(FILES) $(FILESFAIL))

SUCCESSES = $(patsubst %.root,check%.success,$(FILES) $(FILESFAIL))

.PHONY: files

.PRECIOUS: lib%.$(DllSuf) %.root

files: $(FILES) $(FILESFAIL)

logs: $(LOGS)

NumericalTypeEvolution: $(SUCCESSES)

$(LIBS): lib%.$(DllSuf): ../rootalias.C ../float16.cxx ../float16.h ../compile.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../compile.C(0, "$*")' > $*.log 2>&1

%.root: lib%.$(DllSuf) ../rootalias.C ../run.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../run.C("$*")' > lib$*.log 2>&1 

$(LOGS) : check.%.log: %.root $(FILES) $(FILESFAIL) ../read.C
	$(CMDECHO) $(CALLROOTEXE) -b -l -q '../read.C("$(FILES)", "$(FILESFAIL)", 0, "$*")' 2>&1 | grep -v Processing > $@

$(SUCCESSES) : check%.success: check.%.log
ifeq ($(FAST),)
	$(CMDECHO) diff -b check.ref $< && touch $@
else
	$(CMDECHO) diff -b check_fast.ref $< && touch $@
endif

