#
# Rules.mk
# Contains most of the infrastructure for the roottest makefiles.
#

all: summary

test: tests ;
# The previous line contains just ';' in order to disable the implicit
# rule building an executable 'test' from test.C

MAKEFILE_NAME=$(firstword $(MAKEFILE_LIST))

summary:
	@CALLDIR= ; $(MAKE) -f $(MAKEFILE_NAME) --no-print-directory  tests || \
	if [ `ls $(SUMMARY).*.summary 2>/dev/null | wc -l` -gt 0 ] ; then \
          res=`grep --text FAILING $(SUMMARY).*.summary  | wc -l` ; \
	  echo "At least $$res tests have failed:" ; \
	  grep --text -h FAILING $(SUMMARY).*.summary; \
          exit $$res; \
	else \
	  res=$$?; \
	  echo "At least one test not captured in the summary file have failed." ; \
	  exit $$res; \
	fi

ifneq ($(V),)
VERBOSE:=$(V)
endif
ifeq ($(VERBOSE),)
   CMDECHO=@
else
   CMDECHO=
endif

export SUMMARY=make

CALLROOTEXEBUILD:=root.exe

# This is a default value, it can be over-ridden
# by passing FIXCLING= as a command line argument.
#
# So, fixing cling is on by default and is disabled
# if FIXCLING is explicitly set to empty or is set
# to: no
FIXCLING=yes
ifneq ($(FIXCLING),)
ifneq ($(FIXCLING),no)
-include $(ROOTTEST_HOME)/scripts/FixCling.mk
endif
endif

# The user directory should define
# SUBDIRS listing any activated subdirectory
# TEST_TARGETS with the list of activated test
# CLEAN_TARGETS with the list of things to delete

# doing gmake VERBOSE=true allows for more output, include the original
# commands.

# doing gmake FAIL=true runs the test that are known to fail

# doing gmake TIME=true times the test output

SUBDIRS := $(shell $(ROOTTEST_HOME)/scripts/subdirectories .)

ALL_EXEC_CXX   := $(wildcard exec*.cxx)
ALL_EXEC_C     := $(wildcard exec*.C)
ALL_ASSERT_CXX := $(wildcard assert*.cxx)
ALL_ASSERT_C   := $(wildcard assert*.C)
ALL_EXEC_PY    := $(wildcard exec*.py)

TEST_TARGETS_DIR = $(SUBDIRS:%=%.test)
TEST_TARGETS += $(TEST_TARGETS_DIR) \
     $(subst .C,,$(ALL_ASSERT_C))  $(subst .cxx,,$(ALL_ASSERT_CXX)) \
     $(subst .C,,$(ALL_EXEC_C))  $(subst .cxx,,$(ALL_EXEC_CXX)) \
     $(subst .py,,$(ALL_EXEC_PY))

# allow tests to be disabled by putting their names into a file called !DISABLE
ifneq ($(MAKECMDGOALS),clean)
TEST_TARGETS_DISABLED := $(if $(wildcard !DISABLE),$(shell cat !DISABLE))
endif
TEST_TARGETS := $(if $(TEST_TARGETS_DISABLED),\
                     $(filter-out $(TEST_TARGETS_DISABLED),$(TEST_TARGETS))\
                     $(warning Test(s) $(TEST_TARGETS_DISABLED) disabled!),\
                  $(TEST_TARGETS))

CLEAN_TARGETS_DIR = $(SUBDIRS:%=%.clean)
CLEAN_TARGETS +=

ALL_LIBRARIES += AutoDict_* *_ACLiC_* *.success *.summary *.d *.o *.obj *.so *.pcm *.def *.exp *.dll *.lib dummy.C \
	*.pdb .def *.ilk *.manifest rootmap_* dummy* *.clog *.log *.elog *.celog *.eclog *.pylog \
	*_C.rootmap *_cc.rootmap *_cpp.rootmap *_cxx.rootmap *_h.rootmap *_rflx.cpp *_cint.cpp

.PHONY: clean removefiles tests all test $(TEST_TARGETS) $(TEST_TARGETS_DIR) utils check logs.tar.gz perftrack.tar.gz

include $(ROOTTEST_HOME)/scripts/Common.mk
DEPENDENCIES_INCLUDES := $(wildcard *.d)
ifeq ($(findstring clean,$(MAKECMDGOALS)),)
   ifneq ($(DEPENDENCIES_INCLUDES),)
      -include $(wildcard *.d)
   endif
endif

ifeq ($(MAKECMDGOALS),cleantest)
	TESTGOAL = cleantest
else
	TESTGOAL = test
endif

# here we guess the platform

ifeq ($(ARCH),)
   export ARCH          := $(shell root-config --arch)
endif
ifeq ($(PLATFORM),)
   export PLATFORM      := $(shell root-config --platform)
endif
ifeq ($(ROOTSYS),)
   export ROOTSYS       := $(shell root-config --prefix)
endif
ifeq ($(R__EXPLICITLINK),)
   export R__EXPLICITLINK := $(shell root-config --has-explicitlink)
endif
ifeq ($(R__FEATURES),)
   R__FEATURES := $(shell root-config --features)
endif

ifeq ($(ROOTTEST_LOC),)

ifeq ($(PLATFORM),win32)
   export ROOTTEST_LOC := $(shell cygpath -u '$(ROOTTEST_HOME)')
   export ROOTTEST_HOME2 := $(shell cygpath -m $(ROOTTEST_HOME))
   override ROOTTEST_HOME := $(ROOTTEST_HOME2)
   export PATH:=${PATH}:${ROOTTEST_LOC}/scripts
else
ifeq ($(PLATFORM),macosx)
   export ROOTTEST_LOC := $(shell python -c 'import os,sys; print os.path.realpath(sys.argv[1]);' $(ROOTTEST_HOME))/
   export PATH := $(PATH):$(ROOTTEST_HOME)/scripts
else
   export ROOTTEST_LOC := $(shell readlink -f -n $(ROOTTEST_HOME))/
   export PATH := $(PATH):$(ROOTTEST_HOME)/scripts
endif
endif

endif

ifneq ($(TIME),)
ifeq ($(ROOTTEST_RUNID),)
   export ROOTTEST_RUNID := $(shell touch $(ROOTTEST_LOC)runid )
   export ROOTTEST_RUNID := $(shell echo  $$((`cat $(ROOTTEST_LOC)runid`+1)) > $(ROOTTEST_LOC)runid )
   export ROOTTEST_RUNID := $(shell cat $(ROOTTEST_LOC)runid )
   ROOTTEST_TESTID := $(shell echo 0 > $(ROOTTEST_LOC)testid)
   ROOTTEST_TESTID := 0
else
   ROOTTEST_TESTID := $(shell echo $$((`cat $(ROOTTEST_LOC)testid`+1)) > $(ROOTTEST_LOC)testid )
   ROOTTEST_TESTID := $(shell cat $(ROOTTEST_LOC)testid )
endif
TESTTIMINGFILE := roottesttiming.out
TESTTIMEPRE := export TIMEFORMAT="roottesttiming %S"; ( time
TESTTIMEPOST :=  RUNNINGWITHTIMING=1 2>&1 ) 2> $(TESTTIMINGFILE).tmp &&  cat $(TESTTIMINGFILE).tmp | grep roottesttiming | sed -e 's,^roottesttiming ,,g' > $(TESTTIMINGFILE) && rm $(TESTTIMINGFILE).tmp
TESTTIMEACTION = else if [ -f $(TESTTIMINGFILE) ]; then printf " %8s\n" "[`cat $(TESTTIMINGFILE)`ms]" && root.exe -q -b -l -n '$(ROOTTEST_HOME)/scripts/recordtiming.cc+("$(ROOTTEST_HOME)",$(ROOTTEST_RUNID),$(ROOTTEST_TESTID),"$(CURDIR)/$*","$(TESTTIMINGFILE)")' > /dev/null && rm -f $(TESTTIMINGFILE); fi
endif

.PHONY: valgrind perftrack
$(ROOTTEST_LOC)scripts/pt_data_dict.cpp: $(ROOTTEST_LOC)scripts/pt_data.h $(ROOTTEST_LOC)scripts/pt_Linkdef.h
	$(CMDECHO)rootcint -f $@ $^

$(ROOTTEST_LOC)scripts/pt_collector: $(ROOTTEST_LOC)scripts/pt_collector.cpp $(ROOTTEST_LOC)scripts/pt_data_dict.cpp
	$(CMDECHO)$(CXX) -g $^ -Wall `root-config --cflags` `root-config --libs` -o $@

$(ROOTTEST_LOC)scripts/ptpreload.so: $(ROOTTEST_LOC)scripts/pt_mymalloc.cpp
	$(CMDECHO)$(CXX) -g $< -shared -fPIC -Wall `root-config --cflags` -o $@

perftrack: $(ROOTTEST_LOC)scripts/pt_collector $(ROOTTEST_LOC)scripts/ptpreload.so
	$(CMDECHO) LD_LIBRARY_PATH=$(ROOTTEST_LOC)/scripts:$$LD_LIBRARY_PATH $(MAKE) -f $(MAKEFILE_NAME) -C $$PWD $(filter-out perftrack,$(MAKECMDGOALS)) \
          CALLROOTEXE="$< "$(ROOTTEST_LOC)" root.exe"

# For now logs.tar.gz is a phony target
perftrack.tar.gz: $(ROOTTEST_LOC)scripts/pt_createIndex_C.so
	$(CMDECHO) rm -f perftrack.tar perftrack.tar.gz; touch perftrack.tar ; \
                $(CALLROOTEXEBUILD) -b -l -q $(ROOTTEST_LOC)/scripts/pt_createIndex.C+ ; \
		find . -type f -name 'pt_*.root' -o -name 'pt_*.gif' -o -name pt_index.html | xargs -I{}  tar --transform=s/pt_index.html/index.html/ -u -f perftrack.tar "{}" ; gzip  perftrack.tar

#	$(CMDECHO) cd $(ROOTTEST_LOC) && find . -type f -name 'pt_*.root' | xargs -I{} bash -c 'mkdir -p $(ROOTTEST_LOC)/../perftrack/`dirname "{}"` && cp "{}" "$(ROOTTEST_LOC)/../perftrack/{}"' || true
#	$(CMDECHO) cd $(ROOTTEST_LOC) && find . -type f -name 'pt_*.gif'  | xargs -I{} bash -c 'mkdir -p $(ROOTTEST_LOC)/../perftrack/`dirname "{}"` && cp "{}" "$(ROOTTEST_LOC)/../perftrack/{}"' || true

$(ROOTTEST_LOC)scripts/analyze_valgrind: $(ROOTTEST_LOC)scripts/analyze_valgrind.cxx
	$(CXX) $< -o $@
valgrind: $(ROOTTEST_LOC)scripts/analyze_valgrind
	@( export valgrindlogfile=${PWD}/valgrind-`date +"%Y%m%d-%H%M%S"`.log; \
	( \
	valgrind-listener > $$valgrindlogfile 2>&1 & ) && \
	valgrindlistenerpid=$$$$ && \
	$(MAKE) -f $(MAKEFILE_NAME) -C $$PWD $(filter-out valgrind,$(MAKECMDGOALS)) \
          CALLROOTEXE="valgrind --suppressions=$(ROOTSYS)/etc/valgrind-root.supp --suppressions=$(ROOTTEST_HOME)/scripts/valgrind-suppression_ROOT_optional.supp --log-socket=127.0.0.1 --error-limit=no --leak-check=full -v root.exe" ; \
	killall valgrind-listener; \
	grep '==[[:digit:]]\+==' $$valgrindlogfile | $(ROOTTEST_HOME)/scripts/analyze_valgrind \
	&& $(ROOTTEST_HOME)/scripts/analyze_valgrind.sh $$valgrindlogfile > $$valgrindlogfile.summary.txt \
	)

valgrind-summary: $(ROOTTEST_LOC)scripts/analyze_valgrind
	@( export valgrindlogfile=${PWD}/valgrind-`date +"%Y%m%d-%H%M%S"`.log; \
	( \
	valgrind-listener > $$valgrindlogfile 2>&1 & ) && \
	valgrindlistenerpid=$$$$ && \
	$(MAKE) -f $(MAKEFILE_NAME) -C $$PWD $(filter-out valgrind,$(MAKECMDGOALS)) \
          CALLROOTEXE="valgrind --suppressions=$(ROOTSYS)/etc/valgrind-root.supp --suppressions=$(ROOTTEST_HOME)/scripts/valgrind-suppression_ROOT_optional.supp --log-socket=127.0.0.1 --error-limit=no --leak-check=summary -v root.exe" ; \
	killall valgrind-listener; \
	grep '==[[:digit:]]\+==' $$valgrindlogfile | $(ROOTTEST_HOME)/scripts/analyze_valgrind \
	&& $(ROOTTEST_HOME)/scripts/analyze_valgrind.sh $$valgrindlogfile > $$valgrindlogfile.summary.txt \
	)

# Use this function to insure than only one execution can happen at one time.
# This will create a directory named after the first argument ( $(1).lock )
# and create a file name lockfile inside.   The file lockfile will contain the
# pid of the shell that created the file.   If the directory already exist,
# this will wait for up to 90s (and exit 1 if waiting more than 90s) and check
# every second whether the directory __and__ the shell that created it still
# exist.  If one of the 2 is false, then this proceeds.
# If the 3rd parameter is 'test', it checks whether the previous run succeeded or not and only exeucuted the
# command in case of failure, otherwise it always run the command.
# The 4th parameter is executed in case the execution was skipped.
locked_execution = \
   rm -f $(1).log ; \
   mkdir $(1).lock  >/dev/null 2>&1; result=$$?; try=0; \
   while [ $$result -gt 0 -a -e $(1).lock/lockfile ] ; do \
      oldpid=`cat $(1).lock/lockfile`; \
      if [ `ps h --pid $$oldpid | wc -l` -lt 1 ] ; then \
         rm -r $(1).lock; \
      else echo "waiting for $(1).lock ... try number $$try" ; sleep 1; \
      fi ; \
      try=`expr $${try} + 1`; \
      if [ $${try} -gt 90 ] ; then \
         echo "Waited more than 90 seconds for lock acquisition, so let's give up." 1>&2; \
         exit 1; \
      fi; \
      mkdir $(1).lock  >/dev/null 2>&1; result=$$?; \
   done; \
   echo $$$$ > $(1).lock/lockfile ; \
   previous_status=`if [ -e $(1).locked.log ] ; then cat $(1).locked.log; else echo nothing; fi` ; \
      if [ $(3) != "test" -o "$$previous_status" != "success" ] ; then \
         $(2) ; command_result=$$?; \
      if [ $$command_result -eq 0 ] ; then \
         echo "success" > $(1).locked.log; \
      else \
         echo "failed" > $(1).locked.log; \
      fi \
   else \
      eval $(4); \
      command_result=0; \
   fi; \
   rm -r $(1).lock; \
   exit $$command_result

ifeq ($(CURDIR)/,$(ROOTTEST_LOC))
EVENTDIR = root/io/event
else
EVENTDIR = $(ROOTTEST_LOC)/root/io/event
endif
$(EVENTDIR)/$(SUCCESS_FILE): $(ROOTCORELIBS)
	$(CMDECHO) (cd $(EVENTDIR); $(call locked_execution,globalrun,$(MAKE) CURRENTDIR=$(EVENTDIR) --no-print-directory $(TESTGOAL),notest);)

$(EVENTDIR)/bigeventTest.success: $(ROOTCORELIBS)
	$(CMDECHO) (cd $(EVENTDIR); $(call locked_execution,globalrun,$(MAKE) EVENT=Event$(ExeSuf) CURRENTDIR=$(EVENTDIR) --no-print-directory bigeventTest.success,notest);)

$(TEST_TARGETS_DIR): %.test:  $(EVENTDIR)/$(SUCCESS_FILE) utils
	@(echo Running test in $(CALLDIR)/$*)
	@(cd $*; if [ "$(filter -j,$(MAKEFLAGS))" = "-j" ] ; then export ROOT_HIST=0; fi; $(TESTTIMEPRE) $(MAKE) -f Makefile CURRENTDIR=$* --no-print-directory $(TESTGOAL) $(TESTTIMEPOST) ; \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
        if [ "x$(SUMMARY)" != "x" ] ; then \
           if [ `ls $(SUMMARY).*.summary 2>/dev/null | wc -l` -gt 0 ] ; then \
              cat $(SUMMARY).*.summary > $(SUMMARY).summary; \
           else \
              echo "--- FAILING TEST: make -C $(CALLDIR)/$* test" > $(SUMMARY).summary; \
              echo "There is at least one failing test which does not create a summary file in the directory $(CALLDIR)/$*." >> $(SUMMARY).summary; \
           fi; \
           cat $(SUMMARY).summary > ../$(SUMMARY).$*.summary; \
        fi; \
        len=`echo Tests in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Tests in %s %*.*s ' $(CALLDIR)/$* $$end $$end $(DOTS); \
	      printf 'FAIL\n' ; \
        false ; \
     $(TESTTIMEACTION) \
     fi )

#     result=$$?; \
#     len=`echo Test in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Tests in %s %*.*s ' $(CALLDIR)/$* $$end $$end $(DOTS); \
#	  if [ -f $*/.success ] ; then printf 'OK\n' ; else printf 'FAIL\n' ; fi; \
#     if [ $$result -ne 0 ] ; then false ; fi )

$(CLEAN_TARGETS_DIR): %.clean:
	@(cd $*; $(MAKE) --no-print-directory clean)

clean:  $(CLEAN_TARGETS_DIR)
	$(CMDECHO) rm -rf main AutoDict* *Dict\.* Event.root .*~ *~ $(CLEAN_TARGETS)

ifneq ($(MAKECMDGOALS),distclean)
CLEAN_TARGETS += .root_hist
endif

distclean: clean
	$(CMDECHO) rm -rf $(ROOTTEST_LOC)roottiming.root $(ROOTTEST_LOC)runid $(ROOTTEST_LOC)root_version $(ROOTTEST_LOC).root_hist

cleantest: test

# For now logs.tar.gz is a phony target
logs.tar.gz:
	$(CMDECHO) rm -f logs.tar logs.tar.gz ; touch logs.tar ; find . -name '*log' | xargs -I{}  tar -uf logs.tar "{}" ; gzip logs.tar

ifeq ($(MAKECMDGOALS),cleantest)
  ifeq ($(VERBOSE),)
     ForceRemoveFiles := $(shell rm -rf main *Dict\.* Event.root .*~ *~ $(CLEAN_TARGETS) )
  else
     ForceRemoveFilesVerbose := $(shell echo rm -rf 'main *Dict\.* Event.root .*~ *~ $(CLEAN_TARGETS)' 1>&2 )
     ForceRemoveFiles := $(shell rm -rf main *Dict\.* Event.root .*~ *~ $(CLEAN_TARGETS) )
  endif
endif

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(ROOTBITS),)
   export ROOTBITS := $(shell root.exe -b -q -n $(ROOTTEST_HOME)/scripts/Bits.C | grep Bits_in_long | awk '{print $$2;}' )
endif

ifeq ($(ROOTTEST_CFLAGS_ADDED_TO_CXXFLAGS),)
	export CXXFLAGS := $(CXXFLAGS) $(shell root-config --cflags)
	ROOTTEST_CFLAGS_ADDED_TO_CXXFLAGS := 1
endif
ifeq ($(ROOTLIBS),)
   export ROOTLIBS     := $(shell root-config --nonew --libs)
endif
ifeq ($(ROOTGLIBS),)
   export ROOTGLIBS    := $(shell root-config --nonew --glibs)
endif
endif

ObjSuf   = o

ifeq ($(PYTHON),)
   ifneq (, $(shell command -v python 2>/dev/null))
      export PYTHON := python
   else ifneq (, $(shell command -v python3 2>/dev/null 2>&1))
      export PYTHON := python3
   else ifneq (, $(shell command -v python2 2>/dev/null 2>&1))
      export PYTHON := python2
   else
      $(error "Cannot find python, python3, nor python2!")
   endif
endif
ifeq ($(HAS_PYTHON),)
   export HAS_PYTHON := $(shell root-config --has-python)
endif
ifeq ($(HAS_PYTHON),yes)
   ifeq ($(findstring $(ROOTSYS)/lib, $(PYTHONPATH)),)
      # The PYTHONPATH does not have ROOTSYS/lib in it yet
      # let's add it
      ifeq ($(PLATFORM),win32)
         export PYTHONPATH := $(ROOTSYS)/bin;$(PYTHONPATH);$(ROOTSYS)/lib
       else
         export PYTHONPATH := $(ROOTSYS)/lib:$(PYTHONPATH)
       endif
   endif
   ifeq ($(PLATFORM),macosx)
      PYTHONLIB:=$(shell grep ^PYTHONLIB $(ROOTSYS)/config/Makefile.config | sed -e 's,^.*\:=,,'  -e 's,^ *-L,,' | grep -v -e '^ -l' -e '^ *$$' )
      PYTHONFWK:=$(dir $(PYTHONLIB))
      ifneq ($(PYTHONFWK),)
         export PATH:=$(PYTHONFWK)/bin:$(PATH)
         export DYLD_LIBRARY_PATH:=$(PYTHONFWK):$(DYLD_LIBRARY_PATH)
      endif
   endif
endif

ifeq ($(PLATFORM),win32)

SetPathForBuild = $(ROOTTEST_LOC)scripts/roottestpath
ifeq ($(ROOT_LOC),)
   export ROOT_LOC := $(shell cygpath -u '$(ROOTSYS)')
endif
else
ROOT_LOC=$(ROOTSYS)
endif
# Avoid common typo
ROOTLOC=$(ROOT_LOC)

include $(ROOT_LOC)/config/Makefile.comp

ifeq ($(ROOT_SRCDIR),)
export ROOT_SRCDIR := $(shell grep "ROOT_SRCDIR    :=" $(ROOT_LOC)/config/Makefile.config | sed 's/^ROOT_SRCDIR    := \$$(call realpath, \([^)]*\).*$$/\1/')
ifeq ($(PLATFORM),win32)
  export ROOT_SRCDIR    := $(shell cygpath -m -- $(ROOT_SRCDIR))
  export ROOT_SRCDIRDEP := $(shell cygpath -u -- $(ROOT_SRCDIR))
else
  export ROOT_SRCDIRDEP := $(ROOT_SRCDIR)
endif
endif

ifeq ($(PLATFORM),win32)

# Windows with the VC++ compiler
PcmSuf        = _rdict.pcm
ObjSuf        = obj
SrcSuf        = cxx
ExeSuf        = .exe
DllSuf        = dll
LibSuf        = lib
OutPutOpt     = -out:
OutOpt        = -Fo
CXX          ?= cl
ifeq ($(CXX),./build/win/cl.sh)
   CXX        := cl
endif
#CXXOPT        = -O2
CXXOPT        = -Z7
#CXXFLAGS      = $(CXXOPT) -G5 -GR -MD -DWIN32 -D_WINDOWS -nologo \
#                -DVISUAL_CPLUSPLUS -D_X86_=1 -D_DLL
ifeq ($(RCONFIG_INC),)
   export RCONFIG_INC   := $(shell root-config --incdir)
endif
CXXFLAGS      += $(CXXOPT) -nologo -I$(RCONFIG_INC) -FIw32pragma.h
CXXFLAGS      += -TP
# replace script invocation for LD in Makefile.comp
LD             = link -nologo
#LDOPT         = -opt:ref
#LDOPT         = -debug
#LDFLAGS       = $(LDOPT) -nologo -nodefaultlib -incremental:no
CLDFLAGS      = -link
SOFLAGS       = -DLL
SYSLIBS       = kernel32.lib  ws2_32.lib mswsock.lib \
                advapi32.lib  user32.lib gdi32.lib comdlg32.lib winspool.lib

else

# Non windows default:

export LD_LIBRARY_PATH := ${LD_LIBRARY_PATH}:.

SetPathForBuild = echo
export ROOT_LOC := $(ROOTSYS)

PcmSuf        = _rdict.pcm
ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
LibSuf        = so
OutPutOpt     = -o
OutOpt        = -o

endif

ifeq ($(ARCH),linux)

# Linux with egcs, gcc 2.9x, gcc 3.x (>= RedHat 5.2)
CXX          ?= g++
LD           ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxx8664gcc)

CXX          ?= g++
LD           ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif

ifeq ($(ARCH),ppc64le)

CXX          ?= g++
LD           ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxarm64)

CXX          ?= g++
LD           ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif


ifeq ($(ARCH),linuxicc)
# Linux with Intel icc compiler in 32-bit mode
CC ?= icc
CXX ?= icpc
LD ?= icpc
ifeq ($(ROOTBUILD),debug)
CXXFLAGS += -g -wd191 -fPIC
else
CXXFLAGS += -O -wd191 -fPIC
endif
SOFLAGS  = -shared
endif

ifeq ($(ARCH),linuxx8664icc)
# Linux with Intel icc compiler in 64-bit mode
CC ?= icc
CXX ?= icpc
LD ?= icpc
ifeq ($(ROOTBUILD),debug)
CXXFLAGS += -g -wd191 -fPIC
else
CXXFLAGS += -O -wd191 -fPIC
endif
SOFLAGS  = -shared
endif

ifeq ($(ARCH),macosx)

# MacOSX with cc/g++
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CXX          ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -m32 -g -pipe -Wall -fPIC -Woverloaded-virtual
else
CXXFLAGS      += -m32 -O -pipe -Wall -fPIC -Woverloaded-virtual
endif
ifeq ($(MACOSX_MAJOR),)
  export MACOSX_MAJOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 1))
endif
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2))
endif
UNDEFOPT      = dynamic_lookup
LD           ?= c++
LD           := MACOSX_DEPLOYMENT_TARGET=$(MACOSX_MAJOR).$(MACOSX_MINOR) $(LD)
LDFLAGS       = -m32
SOFLAGS       = -m32 -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = so
endif


ifeq ($(ARCH),macosx64)

# MacOSX 64 bit with cc/g++
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CXX          ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -m64 -g -pipe -Wall -fPIC -Woverloaded-virtual
else
CXXFLAGS      += -m64 -O -pipe -Wall -fPIC -Woverloaded-virtual
endif
ifeq ($(MACOSX_MAJOR),)
  export MACOSX_MAJOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 1))
endif
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2))
endif
UNDEFOPT      = dynamic_lookup
LD           ?= c++
LD           := MACOSX_DEPLOYMENT_TARGET=$(MACOSX_MAJOR).$(MACOSX_MINOR) $(LD)
LDFLAGS       = -m64 -Wl,-rpath,@loader_path/. -Wl,-rpath,$(ROOTSYS)/lib
SOFLAGS       = -m64 -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = so
endif


ifeq ($(ARCH),macosxarm64)

# MacOSX 64 bit with cc/g++
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CXX          ?= g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -m64 -g -pipe -Wall -fPIC -Woverloaded-virtual
else
CXXFLAGS      += -m64 -O -pipe -Wall -fPIC -Woverloaded-virtual
endif
ifeq ($(MACOSX_MAJOR),)
  export MACOSX_MAJOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 1))
endif
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2))
endif
UNDEFOPT      = dynamic_lookup
LD           ?= c++
LD           := MACOSX_DEPLOYMENT_TARGET=$(MACOSX_MAJOR).$(MACOSX_MINOR) $(LD)
LDFLAGS       = -m64 -Wl,-rpath,@loader_path/. -Wl,-rpath,$(ROOTSYS)/lib
SOFLAGS       = -m64 -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = so
endif



ifeq ($(ARCH),macosxicc)

# MacOSX 32/64 bit with Intel icc
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CC           ?= icc
CXX          ?= icpc
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -fPIC -wd191 -wd1476
else
CXXFLAGS      += -O -fPIC -wd191 -wd1476
endif
ifeq ($(MACOSX_MAJOR),)
  export MACOSX_MAJOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 1))
endif
ifeq ($(MACOSX_MINOR),)
  export MACOSX_MINOR := $(strip $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2))
endif
UNDEFOPT      = dynamic_lookup
LD           ?= icpc
LD           := MACOSX_DEPLOYMENT_TARGET=$(MACOSX_MAJOR).$(MACOSX_MINOR) $(LD)
LDFLAGS       =
SOFLAGS       = -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = so
endif

ifneq ($(ARCH:solaris%=%),$(ARCH))
# solaris
LDFLAGS  ?= `root-config --ldflags`
SOFLAGS  ?= -G $(LDFLAGS) -KPIC
CXXFLAGS += -KPIC
endif

ifneq ($(GCCTOOLCHAIN),)
CXXFLAGS    += --gcc-toolchain=$(GCCTOOLCHAIN)
LDFLAGS     += --gcc-toolchain=$(GCCTOOLCHAIN)
endif

ifeq ($(PT),)
CALLROOTEXE  ?= root.exe
else
CALLROOTEXE  ?= $(ROOTTEST_LOC)scripts/pt_collector root.exe
endif
export CALLROOTEXE

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CINT_VERSION),)
   export CINT_VERSION := 5
endif
endif

# Track the version of ROOT we are runing with

ROOTV=$(ROOTTEST_LOC)/root_version
ROOTVFILE=$(ROOTTEST_HOME)/root_version
ifeq ($(ROOTTEST_CHECKED_VERSION),)
   export ROOTTEST_CHECKED_VERSION:= $(shell echo $(ROOTSYS) && (echo "$(ROOTSYS)" | diff - "$(ROOTVFILE)" 2> /dev/null && echo "--- Using ROOT from $(ROOTSYS)" >&2) || (echo "--- Switching to ROOT from $(ROOTSYS)" >&2; if [ -f "$(ROOTVFILE)" ]; then echo "---            (previously "`cat "$(ROOTVFILE)"`")" >&2; fi; echo "$(ROOTSYS)" > $(ROOTVFILE) ); )

ifneq ($(TIME),)
   CPUFILE=/proc/cpuinfo
   ROOTTEST_ARCH=$(ROOTTEST_LOC)roottest.arch
   export ROOTTEST_ARCH_FILE := $(shell if [ -e $(CPUFILE) ] ; then grep -e 'model name' -e cpu $(CPUFILE) | sort -u | sed -e 's/ //' -e 's/[ \t]*:/:/' > $(ROOTTEST_ARCH) ; else echo "Information Not Available" > $(ROOTTEST_ARCH); fi; )
endif

endif

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf) .$(ExeSuf) .cc .cxx .C .cpp

##### utilities #####

ifeq ($(PLATFORM),win32)
MAKELIB       = $(ROOTTEST_HOME)/scripts/winmakelib.sh
else
MAKELIB       = $(ROOTSYS)/build/unix/makelib.sh $(MKLIBOPTIONS)
endif

ROOTCORELIBS_LIST = TreePlayer Tree Graf Hist Physics Core
ROOTCORELIBS = $(addprefix $(ROOT_LOC)/lib/lib,$(addsuffix .$(LibSuf),$(ROOTCORELIBS_LIST)))
ROOTCINT = $(ROOT_LOC)/bin/rootcint$(ExeSuf)

UTILS_LIBS =  $(ROOTTEST_LOC)scripts/utils_cc.$(DllSuf) $(ROOTTEST_LOC)scripts/recordtiming_cc.$(DllSuf)

$(ROOTTEST_LOC)scripts/utils_cc.$(DllSuf) : $(ROOTTEST_LOC)scripts/utils.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$(ROOTTEST_HOME)scripts/utils.cc\"\) > $(ROOTTEST_LOC)scripts/utils_cc.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$?  --log=$(ROOTTEST_LOC)scripts/utils_cc.build.log

$(ROOTTEST_LOC)scripts/recordtiming_cc.$(DllSuf) : $(ROOTTEST_LOC)scripts/recordtiming.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$(ROOTTEST_HOME)scripts/recordtiming.cc\"\) > $(ROOTTEST_LOC)scripts/recordtiming_cc.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$?  --log=$(ROOTTEST_LOC)scripts/recordtiming_cc.build.log

override ROOTMAP = $(ROOT_LOC)/etc/system.rootmap

$(ROOTMAP):
	@echo Error $(ROOTMAP) is required for roottest '(Do cd $$ROOTSYS; $(MAKE) map)'

check: $(ROOT_LOC)/lib/libCore.$(LibSuf)

UTILS_PREREQ =  $(UTILS_LIBS)

utils:  $(UTILS_LIBS)

copiedEvent$(ExeSuf): $(EVENTDIR)/bigeventTest.success
	$(CMDECHO) cp $(EVENTDIR)/libEvent.* $(EVENTDIR)/Event.h $(EVENTDIR)/EventDict_rdict.pcm .
	$(CMDECHO) cp $(EVENTDIR)/Event$(ExeSuf) ./copiedEvent$(ExeSuf)
ifeq ($(PLATFORM),win32)
	$(CMDECHO) if [ -e $(EVENTDIR)/Event$(ExeSuf).manifest ] ; then cp $(EVENTDIR)/Event$(ExeSuf).manifest ./copiedEvent$(ExeSuf).manifest ; fi
endif

%.o: %.C
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_o_C.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_o_C.build.log --test=$@

%.o: %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_o_cc.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_o_cc.build.log --test=$@

%.o: %.cxx
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_o_cxx.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_o_cxx.build.log --test=$@

%.o: %.cpp
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_o_cpp.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_o_cpp.build.log --test=$@

%.$(ObjSuf): %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_o_cc.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_o_cc.build.log --test=$@

%.obj: %.C
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_obj_C.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_obj_C.build.log --test=$@

%.obj: %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_obj_cc.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_obj_cc.build.log --test=$@

%.obj: %.cxx
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_obj_cxx.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_obj_cxx.build.log --test=$@

%.obj: %.cpp
	$(CMDECHO) $(CXX) $(CXXFLAGS) -I. -I$(dir $<) -c $< $(OutOpt)$@ > $*_obj_cpp.build.log 2>&1 || handleError.sh --cmd='Compilation with $(CXX)' --result=$$? --log=$*_obj_cpp.build.log --test=$@

%_cpp.$(DllSuf) : %.cpp $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cpp.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$? --log=$*_cpp.build.log

%_C.$(DllSuf) : %.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_C.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$? --log=$*_C.build.log

%_cxx.$(DllSuf) : %.cxx $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cxx.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$? --log=$*_cxx.build.log

%_cc.$(DllSuf) : %.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cc.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$? --log=$*_cc.build.log

%_h.$(DllSuf) : %.h $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_h.build.log 2>&1 || handleError.sh --cmd='Call to build.C' --result=$$? --log=$*_h.build.log

   #( result=$$? ; cat $*_h.build.log ; exit $$result )

%.log : run%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > $@ 2>&1 || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*

%.elog : run%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > $*.log 2>$@ || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=$*

assert%.elog : assert%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > assert$*.log 2>$@ || handleError.sh --cmd='Execution of $<' --result=$$? --log=$@ --test=assert$*

assert%.eclog : assert%_cxx.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b assert$*.cxx+ > assert$*.log 2> $@ || handleError.sh --cmd='Execution of assert$*.cxx+' --result=$$? --log=$@ --test=assert$*

$(subst .cxx,.success,$(ALL_ASSERT_CXX)) : assert%.success: assert%.eclog assert%.ref
	$(SuccessTestDiff) && touch $@

$(subst .C,.success,$(ALL_ASSERT_C)) : assert%.success: assert%.elog assert%.ref
	$(SuccessTestDiff) && touch $@

$(subst .cxx,,$(ALL_ASSERT_CXX)) : assert%: assert%.success

$(subst .C,,$(ALL_ASSERT_C)) : assert%: assert%.success

exec%.log : exec%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > $@ 2>&1 || handleError.sh --cmd='Execution of $< > $@' --result=$$? --log=$@ --test=exec$*

exec%.clog : exec%_cxx.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b exec$*.cxx+ > $@ 2>&1 || handleError.sh --cmd='Execution of exec$*.cxx+' --result=$$? --log=$@ --test=exec$*

$(subst .cxx,.success,$(ALL_EXEC_CXX)) : %.success: %.clog %.ref
	$(SuccessTestDiff) && touch $@

$(subst .C,.success,$(ALL_EXEC_C)) : %.success: %.log %.ref
	$(SuccessTestDiff) && touch $@

$(subst .py,.py.success,$(ALL_EXEC_PY)) : %.py.success: %.pylog %.py.ref
	$(SuccessTestDiff) && touch $@

$(subst .cxx,,$(ALL_EXEC_CXX)) : %: %.success

$(subst .C,,$(ALL_EXEC_C)) : %: %.success

$(subst .py,,$(ALL_EXEC_PY)) : %: %.py.success

%.log : %.py $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
ifeq ($(PYTHONPATH),)
	$(CMDECHO) PYTHONPATH=$(ROOTSYS)/lib $(PYTHON) $(abspath $<) -b - $(PYROOT_EXTRAFLAGS) > $@ 2>&1 || cat $@
else
	$(CMDECHO) $(PYTHON) $(abspath $<) -b - $(PYROOT_EXTRAFLAGS) > $@ 2>&1 || cat $@
endif

%.pylog : %.py $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
ifeq ($(PYTHONPATH),)
	$(CMDECHO) PYTHONPATH=$(ROOTSYS)/lib $(PYTHON) $(abspath $<) -b - $(PYROOT_EXTRAFLAGS) > $@ 2>&1 || cat $@
else
	$(CMDECHO) $(PYTHON) $(abspath $<) -b - $(PYROOT_EXTRAFLAGS) > $@ 2>&1 || cat $@
endif

.PRECIOUS: %_C.$(DllSuf)

%.clog : run%_C.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b run$*.C+ > $@ 2>&1 || handleError.sh --cmd='Execution of run$*.C+' --result=$$? --log=$@ --test=$*

%.celog : run%_C.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b run$*.C+ > $*.log 2>$@ || handleError.sh --cmd='Execution of run$*.C+' --result=$$? --log=$@ --test=$*

%.eclog : run%_C.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b run$*.C+ > $*.log 2>$@ || handleError.sh --cmd='Execution of run$*.C+' --result=$$? --log=$@ --test=$*

%.neutral.clog: %.clog
	$(CMDECHO) cat $*.clog | sed -e 's:0x[0-9a-fA-F]*:0xRemoved:g' > $@

%.neutral.log: %.log
	$(CMDECHO) cat $*.log | sed -e 's:0x[0-9a-fA-F]*:0xRemoved:g' > $@

exec%.ref:  | exec%.clog
	$(CMDECHO) if [ ! -e $@ ] ; then echo > $@ ; echo "Processing exec$*.cxx+..." >> $@ ; fi

exec%.ref:  | exec%.log
	$(CMDECHO)  if [ ! -e $@ ] ; then echo > $@ ; echo "Processing exec$*.C..." >> $@ ; fi

%.py.ref:
	$(CMDECHO) touch $@

%.ref:
	$(CMDECHO) touch $@

.PRECIOUS: %.clog %.log %_cxx.$(DllSuf)

ifneq ($(PLATFORM),macosx)

define BuildWithLib
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"$<\",\"$(shell $(SetPathForBuild) $(filter %.$(DllSuf),$^) )\",\"\")" > $*.build.log 2>&1 || cat $*.build.log
endef

else

define BuildWithLib
        $(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"$<\",\"$(shell $(SetPathForBuild) $(filter %.$(DllSuf),$^) )\",\"\")" > $*.build.log 2>&1 || cat $*.build.log
endef

endif

define WarnFailTest
	$(CMDECHO)echo Known failures: $@ skipped tests in ./$(CURRENTDIR)
endef

ifneq ($(SUMMARY),)
SUMMARYDIFF= > $(SUMMARY).$@.diff.log || handleError.sh --cmd=diff --result=$$? --log=$(SUMMARY).$@.diff.log --test=$@
SUMMARYDIFF_STAR= > $(SUMMARY).$*.diff.log || handleError.sh --cmd=diff --result=$$? --log=$(SUMMARY).$*.diff.log --test=$@
endif

define PathDiff
	( cd $(ROOTTEST_HOME) && diff $(1) "$(subst $(ROOTTEST_HOME),,$(CURDIR))/$(2)" "$(subst $(ROOTTEST_HOME),,$(CURDIR))/$(3)" )
endef

define TestDiffCintSpecific
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS)-$(CINT_VERSION) ]; then \
	   $(call PathDiff,-u -b,$@.ref$(ROOTBITS)-$(CINT_VERSION).$<) ; \
	elif  [ -f $@.ref-$(CINT_VERSION) ]; then \
	   $(call PathDiff,-u -b,$@.ref-$(CINT_VERSION),$<)\
	elif [ -f $@.ref$(ROOTBITS) ]; then \
	   $(call PathDiff,-u -b,$@.ref$(ROOTBITS),$<); \
	else \
	   $(call PathDiff,-u -b,$@.ref,$<); \
	fi $(SUMMARYDIFF)
endef

define TestDiffCintSpecificW
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS)-$(CINT_VERSION) ]; then \
	   $(call PathDiff,-u -b -w,$@.ref$(ROOTBITS)-$(CINT_VERSION),$<) ; \
	elif  [ -f $@.ref-$(CINT_VERSION) ]; then \
	   $(call PathDiff,-u -b -w,$@.ref-$(CINT_VERSION),$<); \
	elif [ -f $@.ref$(ROOTBITS) ]; then \
	   $(call PathDiff,-u -b -w,$@.ref$(ROOTBITS),$<); \
	else \
	   $(call PathDiff,-u -b -w,$@.ref,$<); \
	fi $(SUMMARYDIFF)
endef

define TestDiff
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS) ]; then \
	   $(call PathDiff,-u -b,$@.ref$(ROOTBITS),$<) ; \
	else \
	   $(call PathDiff,-u -b,$@.ref,$<) ; \
	fi $(SUMMARYDIFF)
endef

define TestDiffW
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS) ]; then \
	   $(call PathDiff,-u -b -w,$@.ref$(ROOTBITS),$<); \
	else \
	   $(call PathDiff,-u -b -w,$@.ref,$<); \
	fi $(SUMMARYDIFF)
endef

define SuccessTestDiff
	$(CMDECHO) if [ -f $(subst .success,.ref$(ROOTBITS),$@) ]; then \
	   $(call PathDiff,-u -b $(EXTRA_DIFFOPTS),$(subst .success,.ref$(ROOTBITS),$@),$<) ; \
	else \
	   $(call PathDiff,-u -b $(EXTRA_DIFFOPTS),$(subst .success,.ref,$@),$<) ; \
	fi $(SUMMARYDIFF_STAR)
endef


define BuildFromObj
$(CMDECHO) ( touch dummy$$$$.C && \
	($(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"$(shell $(SetPathForBuild) $(filter %.$(DllSuf),$^) )\",\"$(abspath $<)\",\"$(@:.$(DllSuf)=)\")" > $@.build.log 2>&1 || cat $@.build.log ) \
)
endef

define BuildFromObjs
$(CMDECHO) ( touch dummy$$$$.C && \
	($(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"$(shell $(SetPathForBuild) $(filter %.$(DllSuf),$^) )\",\"$(abspath $(filter %.$(ObjSuf),$^))\",\"$(@:.$(DllSuf)=)\")" > $@.build.log 2>&1 || cat $@.build.log ) )
endef

ifeq ($(SED_VERSION),)
   ifeq ($(PLATFORM),macosx)
      ifeq ($(strip $(shell sed --version 2>&1 | grep GNU | wc -l)) ,1)
         export SED_VERSION=GNU
      else
         export SED_VERSION=macosx
      endif
   else
      export SED_VERSION=GNU
   endif
endif

RemoveLeadingDirs := sed -e 's?^[A-Za-z/\].*[/\]??' -e 's/.dll/.so/'
ifeq ($(SED_VERSION),macosx)
   RemoveDirs := sed -E -e 's,([[:alpha:]]:\\|/)[^[:space:]]*[/\\],,g'
else
   RemoveDirs := sed -e 's?\([A-Za-z]:\\\|[/]\).*[/\\]??'
endif
RemoveSizes := sed -e 's?size=0x[0-9a-fA-F]*?size=n/a?'

