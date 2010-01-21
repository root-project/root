SHELL=/bin/bash

# adjust the location of Rules.mk as needed.
RULES_DIR=$(ROOTTEST_HOME)/scripts

VERSION := $(subst .,-,$(subst /,-,$(shell root-config --version)))
ifeq ($(ARCH),)
   ARCH       := $(shell root-config --arch)
endif
PLATFORM      = $(ARCH)
ifeq ($(PLATFORM),win32)
RAW_PWD := $(shell echo `cygpath -w $$PWD`)
NOLINK_PWD := $(shell echo `cygpath -m $$PWD | sed -e 's/[a-zA-Z]://' -e 's?/cygdrive/[a-zA-Z]??' ` )
else
RAW_PWD := $(shell pwd)
NOLINK_PWD := $(shell echo `pwd -P`)
endif
LIBDIRTOP := $(VERSION).libs
LIBDIR := $(LIBDIRTOP)$(NOLINK_PWD)/
FILEDIR := $(VERSION)
#debug := $(shell echo $(PLATFORM) $(NOLINK_PWD) 1>&2 )
#debug := $(shell echo $(LIBDIR) 1>&2 )

#Set the list of files to be delete by clean:
CLEAN_TARGETS += -r $(ALL_LIBRARIES) *.d *.log *.clog  result.root $(LIBDIRTOP) emulvector.root \
   test.root samplerootfile.root
# We intentionally do not delete the output file in $(FILEDIR) so that 
# we can keep the historic versions.

#Set the list of target to make while testing
TEST_TARGETS += base emptyCollection stlIoTest emulatedWrite Emulvector
#stlNoLibTest

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME := $(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

$(FILEDIR):
	$(CMDECHO)mkdir -p $(FILEDIR)

$(LIBDIR):
	$(CMDECHO)mkdir -p $(LIBDIR)

.PHONY : directories

DIRECTORIES = $(FILEDIR) $(LIBDIR)

directories: $(FILEDIR) $(LIBDIR)

FTEST = $(wildcard ?test.C)
FTEST_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(FTEST)) )

HOLDER = $(wildcard *Holder.C)
HOLDER_DLL = $(addprefix $(LIBDIR),$(subst .C,_C.$(DllSuf),$(HOLDER)))

TEST_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf) $(FTEST_DLL) $(HOLDER_DLL)
#debug := $(shell echo $(TEST_DLL) 1>&2 )

HELPER_DLL = $(LIBDIR)TestHelpers_C.$(DllSuf)

DEPENDENCY_FILES = $(TEST_DLL:.$(DllSuf)=.d)
#debug := $(shell echo $(DEPENDENCY_FILES) 1>&2 )

ifneq ($(MAKECMDGOALS),clean)
-include $(DEPENDENCY_FILES)
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

ifeq ($(PLATFORM),win32)
$(LIBDIR)dtest_C.$(DllSuf) : $(LIBDIR)dequeHolder_C.$(DllSuf)

$(LIBDIR)ltest_C.$(DllSuf) : $(LIBDIR)listHolder_C.$(DllSuf)

$(LIBDIR)vtest_C.$(DllSuf) : $(LIBDIR)vectorHolder_C.$(DllSuf)

$(LIBDIR)ttest_C.$(DllSuf) : $(LIBDIR)multisetHolder_C.$(DllSuf)

$(LIBDIR)mtest_C.$(DllSuf) : $(LIBDIR)mapHolder_C.$(DllSuf)

$(LIBDIR)stest_C.$(DllSuf) : $(LIBDIR)setHolder_C.$(DllSuf)

$(LIBDIR)ntest_C.$(DllSuf) : $(LIBDIR)multimapHolder_C.$(DllSuf)

endif

LIBDIR_SED=$(subst /,\\/,$(LIBDIR))
LIBDIR_SED_REM=$(subst /,\\/,$(NOLINK_PWD)/./)
PWD_SED=$(subst /,\\/,$(subst \,\\\\,$(RAW_PWD)))
ROOTSYS_SED=$(subst /,\\/,$(subst \,\\\\,$(ROOTSYS)))
#ROOTSYS_SED=$(subst \\,\\\\,$(subst /,\\/,$(ROOTSYS)))
#ROOTLOC_SED=$(subst /,\\/,$(ROOT_LOC))
#debug := $(shell echo $(LIBDIR) 1>&2 )

$(HELPER_DLL) : TestHelpers.C 
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "$(RULES_DIR)/build.C(\"$<\",\"\",\"\")" > $(*F).build.log 2>&1

#	$(CMDECHO)touch $@

ifneq ($(ARCH),macosx)
  DllKeep=$(DllSuf)
else
  DllKeep=dylib
endif

$(LIBDIR)%_C.$(DllSuf) : %.C $(HELPER_DLL)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "$(RULES_DIR)/build.C(\"$<\",\"$(filter %.$(DllKeep),$(filter-out $@,$^))\",\"\")" > $*_C.build.log 2>&1

$(LIBDIR)sample_bx_classes_C.$(DllSuf) : sample_bx_classes.C directories
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "$(RULES_DIR)/build.C(\"sample_bx_classes.C\",\"\",\"\")" > sample_bx_classes_C.build.log 2>&1

$(LIBDIR)runbase_C.$(DllSuf): runbase.C directories
	$(CMDECHO) $(CALLROOTEXE) -q -l -b "$(RULES_DIR)/build.C(\"runbase.C\",\"\",\"\")" > runbase_C.build.log 2>&1

#	$(CMDECHO)touch $@
#	$(CMDECHO) echo '.x $(RULES_DIR)/build.C("$<")' | $(CALLROOTEXE) -l -b -n

stlIoTest.log: $(DIRECTORIES) $(TEST_DLL)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b brun.C 2>&1 | tee full.log | grep -v -e 'does not support' -e 'did not support' -e 'Testing older file format from:' > stlIoTest.log && ( cat full.log | grep 'not support'; echo )

stlIoTest: stlIoTest.log
	$(CMDECHO) diff -b stlIoTest.ref stlIoTest.log 

stlNoLibTest.log: $(FILEDIR)/vector.root readNoLib.C
	$(CMDECHO) $(CALLROOTEXE) -q -b -l readNoLib.C\(\"$(FILEDIR)/vector.root\"\) > $@ 2>&1

stlNoLibTest: stlNoLibTest.log
	$(CMDECHO) diff -b stlNoLibTest.ref stlNoLibTest.log 

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

samplerootfile.root: directories $(LIBDIR)sample_bx_classes_C.$(DllSuf)
	$(CMDECHO) $(CALLROOTEXE) -b -q -l sample_writer.C > samplerootfile.log

emptyCollection.log: directories samplerootfile.root $(LIBDIR)sample_bx_classes_C.$(DllSuf) sample_reader.C

emptyCollection: emptyCollection.log
	$(TestDiff)

emulatedWrite.log: $(ROOT_LOC)/bin/hadd$(ExeSuf)
	$(CMDECHO) hadd -f result.root cmsJets_?.root > emulatedWrite.log 2>&1

emulatedWrite: emulatedWrite.log
	$(TestDiff)

base.clog: $(LIBDIR)runbase_C.$(DllSuf)

base: base.clog
	$(TestDiff)

emulvector.root.log: writetest_emulvector.C $(ROOTCORELIBS)
	$(CMDECHO) $(CALLROOTEXE) -b -l -q writetest_emulvector.C > emulvector.root.log 2>&1

emulvector.root: emulvector.root.log emulvector.root.ref
	$(TestDiff)

Emulvector.log: emulvector.root

Emulvector: Emulvector.log
	$(TestDiff)

