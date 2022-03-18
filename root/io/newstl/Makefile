SHELL=/bin/bash

# adjust the location of Rules.mk as needed.
RULES_DIR=$(ROOTTEST_HOME)/scripts

VERSION := $(subst .,-,$(subst /,-,$(shell root-config --version)))
ifeq ($(ARCH),)
   ARCH       := $(shell root-config --arch)
endif
LIBDIRTOP := $(VERSION).libs
LIBDIR := $(LIBDIRTOP)/
FILEDIR := $(VERSION)
#debug := $(shell echo $(PLATFORM) $(NOLINK_PWD) 1>&2 )
#debug := $(shell echo $(LIBDIR) 1>&2 )

#Set the list of files to be delete by clean:
CLEAN_TARGETS += -r $(ALL_LIBRARIES) *.d *.log *.clog  result.root $(LIBDIRTOP) emulvector.root \
   test.root samplerootfile.root memberwise.root *.libs pospelov 
# We intentionally do not delete the output file in $(FILEDIR) so that 
# we can keep the historic versions.

#Set the list of target to make while testing
TEST_TARGETS += base emptyCollection stlIoTest emulatedWrite Emulvector
#stlNoLibTest

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

ifneq ($(ClingWorkAroundJITandInline),)
CALLROOTEXE := $(subst root.exe,root.exe -e '\#include <deque>',$(CALLROOTEXE))
endif

$(FILEDIR):
	$(CMDECHO)mkdir -p $(FILEDIR)

$(LIBDIR):
	$(CMDECHO)mkdir -p $(LIBDIR)

.PHONY : directories

DIRECTORIES = $(FILEDIR) $(LIBDIR)

directories: $(FILEDIR) $(LIBDIR)

FTEST = $(wildcard ?test.C)
ifneq ($(FAST),)
FTEST = vtest.C mtest.C
endif
FTEST_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(FTEST)) )

HOLDER = $(wildcard *Holder.C)
ifneq ($(FAST),)
HOLDER = vectorHolder.C mapHolder.C
endif
HOLDER_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(HOLDER)))

TEST_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf) $(FTEST_DLL) $(HOLDER_DLL)
#debug := $(shell echo $(TEST_DLL) 1>&2 )

HELPER_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf)

DEPENDENCY_FILES = $(TEST_DLL:.$(DllSuf)=.d)
#debug := $(shell echo $(DEPENDENCY_FILES) 1>&2 )

ifneq ($(MAKECMDGOALS),clean)
#-include $(DEPENDENCY_FILES)
-include $(wildcard $(LIBDIR)*.d)
endif

#.PRECIOUS :  $(addprefix $(LIBDIR),$(subst .C,o.$(DllSuf),$(HOLDER) $(FTEST) $(LIBDIR)TestHelpers_C))

deque: directories $(addprefix $(LIBDIR),dtest_C.$(DllSuf) dequeHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo deque up to date

list: directories $(addprefix $(LIBDIR),ltest_C.$(DllSuf) listHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo list up to date

set: directories $(addprefix $(LIBDIR),stest_C.$(DllSuf) setHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo set up to date

map: directories $(addprefix $(LIBDIR),mtest_C.$(DllSuf) mapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo map up to date

multimap: directories $(addprefix $(LIBDIR),ntest_C.$(DllSuf) multimapHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multimap up to date

multiset: directories $(addprefix $(LIBDIR),ttest_C.$(DllSuf) multisetHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo multiset up to date

vector: directories $(addprefix $(LIBDIR),vtest_C.$(DllSuf) vectorHolder_C.$(DllSuf) TestHelpers_C.$(DllSuf) )
	$(CMDECHO) echo vector up to date

stllibs: directories $(FTEST_DLL) $(HOLDER_DLL) TestHelpers_C.$(DllSuf) 

# debug := $(shell echo $(FTEST_DLL) 1>&2 ) 

#$(LIBDIR)%.o: %.C
#	@touch $@

$(LIBDIR)dtest_C.$(DllSuf) : $(LIBDIR)dequeHolder_C.$(DllSuf) 
$(LIBDIR)ltest_C.$(DllSuf) : $(LIBDIR)listHolder_C.$(DllSuf)
$(LIBDIR)vtest_C.$(DllSuf) : $(LIBDIR)vectorHolder_C.$(DllSuf)
$(LIBDIR)ttest_C.$(DllSuf) : $(LIBDIR)multisetHolder_C.$(DllSuf)
$(LIBDIR)mtest_C.$(DllSuf) : $(LIBDIR)mapHolder_C.$(DllSuf)
$(LIBDIR)stest_C.$(DllSuf) : $(LIBDIR)setHolder_C.$(DllSuf)
$(LIBDIR)ntest_C.$(DllSuf) : $(LIBDIR)multimapHolder_C.$(DllSuf)
$(LIBDIR)rtest_C.$(DllSuf) : $(LIBDIR)rvecHolder_C.$(DllSuf)

$(LIBDIR)dtest_C.$(DllSuf) : test.C dequeHolder.h  | $(ROOTCORELIBS)
$(LIBDIR)ltest_C.$(DllSuf) : test.C listHolder.h  | $(ROOTCORELIBS)
$(LIBDIR)vtest_C.$(DllSuf) : test.C vectorHolder.h  | $(ROOTCORELIBS) 
$(LIBDIR)stest_C.$(DllSuf) : test.C setHolder.h  | $(ROOTCORELIBS) 
$(LIBDIR)ttest_C.$(DllSuf) : test.C multisetHolder.h  | $(ROOTCORELIBS)
$(LIBDIR)mtest_C.$(DllSuf) : test.C mapHolder.h  | $(ROOTCORELIBS)
$(LIBDIR)ntest_C.$(DllSuf) : test.C multimapHolder.h | $(ROOTCORELIBS)
$(LIBDIR)rtest_C.$(DllSuf) : test.C rvecHolder.h | $(ROOTCORELIBS)

$(LIBDIR)dequeHolder_C.$(DllSuf)    : dequeHolder.h dequeHolder.C | $(ROOTCORELIBS)
$(LIBDIR)listHolder_C.$(DllSuf)     : listHolder.C listHolder.h | $(ROOTCORELIBS)
$(LIBDIR)vectorHolder_C.$(DllSuf)   : vectorHolder.C vectorHolder.h | $(ROOTCORELIBS)
$(LIBDIR)setHolder_C.$(DllSuf)      : setHolder.C setHolder.h | $(ROOTCORELIBS)
$(LIBDIR)multisetHolder_C.$(DllSuf) : multisetHolder.C multisetHolder.h | $(ROOTCORELIBS)
$(LIBDIR)mapHolder_C.$(DllSuf)      : mapHolder.C mapHolder.h | $(ROOTCORELIBS)
$(LIBDIR)multimapHolder_C.$(DllSuf) : multimapHolder.C multimapHolder.h | $(ROOTCORELIBS)
$(LIBDIR)rvecHolder_C.$(DllSuf)     : rvecHolder.C rvecHolder.h | $(ROOTCORELIBS)

$(HELPER_DLL) : TestHelpers.C 
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(RULES_DIR)/build.C(\"$<\",\"\",\"\")" > $(*F).build.log 2>&1

#	$(CMDECHO)touch $@

ifneq ($(ARCH),macosx)
  DllKeep=$(DllSuf)
else
  DllKeep=dylib
endif

$(LIBDIR)%_C.$(DllSuf) : %.C $(HELPER_DLL)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(RULES_DIR)/build.C(\"$<\",\"$(filter %.$(DllKeep),$(filter-out $@,$^))\",\"\")" > $*_C.build.log 2>&1  || handleError.sh --log=$*_C.build.log --res=$$?

$(LIBDIR)sample_bx_classes_C.$(DllSuf) : sample_bx_classes.C directories
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(RULES_DIR)/build.C(\"sample_bx_classes.C\",\"\",\"\")" > sample_bx_classes_C.build.log 2>&1  || handleError.sh --log=sample_bx_classes_C.build.log--res=$$?

$(LIBDIR)runbase_C.$(DllSuf): runbase.C directories
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(RULES_DIR)/build.C(\"runbase.C\",\"\",\"\")" > runbase_C.build.log 2>&1  || handleError.sh --log=runbase_C.build.log --res=$$?

$(LIBDIR)%_cxx.$(DllSuf): %.cxx | directories
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(RULES_DIR)/build.C(\"$<\",\"\",\"\")" > $*_cxx.build.log 2>&1 || handleError.sh --log=$*_cxx.build.log --res=$$?



#	$(CMDECHO)touch $@
#	$(CMDECHO) echo '.x $(RULES_DIR)/build.C("$<")' | $(CALLROOTEXE) -l -b -n

BRUN:=brun.C
ifneq ($(FAST),)
BRUN:=fastbrun.C
endif

stlIoTest.log: $(DIRECTORIES) $(TEST_DLL)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $(BRUN) 2>&1 | tee full.log | grep -v -e 'does not support' -e 'did not support' -e 'Testing older file format from:' > stlIoTest.log && ( cat full.log | grep 'not support'; echo )

stlIoTest: stlIoTest.log
ifneq ($(FAST),)
	$(CMDECHO) diff -b stlIoTest_fast.ref stlIoTest.log 
else
	$(CMDECHO) diff -b stlIoTest.ref stlIoTest.log 
endif

stlNoLibTest.log: $(FILEDIR)/vector.root readNoLib.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readNoLib.C\(\"$(FILEDIR)/vector.root\"\) > $@ 2>&1

stlNoLibTest: stlNoLibTest.log
	$(CMDECHO) diff -b stlNoLibTest.ref stlNoLibTest.log 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

samplerootfile.root: directories $(LIBDIR)sample_bx_classes_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l sample_writer.C > samplerootfile.log 2>&1 || handleError.sh --log=samplerootfile.log --res=$$?

emptyCollection.log: directories samplerootfile.root $(LIBDIR)sample_bx_classes_C.$(DllSuf) sample_reader.C

emptyCollection: emptyCollection.log
	$(TestDiff)

emulatedWrite.log: $(ROOT_LOC)/bin/hadd$(ExeSuf)
	$(CMDECHO) hadd -f result.root cmsJets_?.root > emulatedWrite.log 2>&1

emulatedWrite: emulatedWrite.log
	$(TestDiff)

base.clog: $(LIBDIR)runbase_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q runbase.C+ > base.clog 2>&1

base: base.clog
	$(TestDiff)

emulvector.root.log: writetest_emulvector.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q writetest_emulvector.C > emulvector.root.log 2>&1

emulvector.root: emulvector.root.log emulvector.root.ref
	$(TestDiff)

Emulvector.log: emulvector.root

Emulvector: Emulvector.log
	$(TestDiff)

ComplexTest.log: ComplexTest_h.$(DllSuf)

ComplexTest: ComplexTest.log
	$(TestDiff)

memberwise.root: $(LIBDIR)writeMemberWise_cxx.$(DllSuf) $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q writeMemberWise.cxx+ > memberwise_root.log 2>&1 || handleError.sh --log=memberwise_root.log --res=$$?

# Cancel default rule
execMemberWise_cxx.$(DllSuf):
	$(CMDECHO) echo 'do nothing' > /dev/null

execMemberWise.clog: $(LIBDIR)execMemberWise_cxx.$(DllSuf) memberwise.root $(ROOTCORELIBS)

pospelov/pospelov.$(DllSuf): buildPospelov.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q buildPospelov.C > buildPospelov.log 2>&1 || handleError.sh --log=buildPospelov.log --res=$$? 

execPospelovPrintFile.log: pospelov/pospelov.$(DllSuf) PospelovPrintFile.C

# DO NOT DELETE
