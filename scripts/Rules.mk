#
# $Id$
#

all: tests

test: tests ;

# The previous line contains just ';' in order to disable the implicit 
# rule building an executable 'test' from test.C

.PHONY: valgrind
scripts/analyze_valgrind: scripts/analyze_valgrind.cxx
	$(CXX) $< -o $@
valgrind: scripts/analyze_valgrind
	@( export valgrindlogfile=$(ROOTTEST_HOME)/valgrind-`date +"%Y%m%d-%H%M%S"`.log; \
	( \
	valgrind-listener > $$valgrindlogfile 2>&1 & ) && \
	valgrindlistenerpid=$$$$ && \
	$(MAKE) -C $$PWD $(filter-out valgrind,$(MAKECMDGOALS)) \
          CALLROOTEXE="valgrind --suppressions=$(ROOTSYS)/etc/valgrind-root.supp --suppressions=$(ROOTTEST_HOME)/scripts/valgrind-suppression_ROOT_optional.supp --log-socket=127.0.0.1 --error-limit=no --leak-check=full -v root.exe" ; \
	killall valgrind-listener; \
	grep '==[[:digit:]]\+==' $$valgrindlogfile | scripts/analyze_valgrind \
	&& scripts/analyze_valgrind.sh $$valgrindlogfile > $$valgrindlogfile.summary.txt \
	)

ifneq ($(ROOTC7),)
CALLROOTEXE:=rootc7.exe
CALLROOTEXEBUILD:=$(CALLROOTEXE)
# Explicitly disable the python test (pyroot only works with cint5)
export HAS_PYTHON:=no
else
CALLROOTEXEBUILD:=root.exe
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

TEST_TARGETS_DIR = $(SUBDIRS:%=%.test)
TEST_TARGETS += $(TEST_TARGETS_DIR)

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

ALL_LIBRARIES += *.d *.o *.obj *.so *.def *.exp *.dll *.lib dummy.C \
	*.pdb .def *.ilk *.manifest rootmap_* dummy* *.clog *.log \
	*_C.rootmap *_cc.rootmap *_cpp.rootmap *_cxx.rootmap *_h.rootmap

.PHONY: clean removefiles tests all test $(TEST_TARGETS) $(TEST_TARGETS_DIR) utils check logs.tar.gz

include $(ROOTTEST_HOME)/scripts/Common.mk

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

ifeq ($(ROOTTEST_LOC),)

ifeq ($(PLATFORM),win32)
   export ROOTTEST_LOC := $(shell cygpath -u '$(ROOTTEST_HOME)')
    export ROOTTEST_HOME2 := $(shell cygpath -m $(ROOTTEST_HOME))
    override ROOTTEST_HOME := $(ROOTTEST_HOME2)
else
    export ROOTTEST_LOC := $(ROOTTEST_HOME)
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
TESTTIMEACTION = else if [ -f $(TESTTIMINGFILE) ]; then printf " %8s\n" "[`cat $(TESTTIMINGFILE)`ms]" && root.exe -q -b -l -n '$(ROOTTEST_HOME)/scripts/recordtiming.cc+("$(ROOTTEST_HOME)",$(ROOTTEST_RUNID),$(ROOTTEST_TESTID),"$(PWD)/$*","$(TESTTIMINGFILE)")' > /dev/null && rm -f $(TESTTIMINGFILE); fi
endif

EVENTDIR = $(ROOTTEST_LOC)/root/io/event
$(EVENTDIR)/$(SUCCESS_FILE): $(ROOTCORELIBS)  
	$(CMDECHO) (cd $(EVENTDIR); $(MAKE) CURRENTDIR=$(EVENTDIR) --no-print-directory $(TESTGOAL); )

$(TEST_TARGETS_DIR): %.test:  $(EVENTDIR)/$(SUCCESS_FILE) utils
	@(echo Running test in $(CALLDIR)/$*)
	@(cd $*; $(TESTTIMEPRE) $(MAKE) CURRENTDIR=$* --no-print-directory $(TESTGOAL) $(TESTTIMEPOST); \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Test in %s %.*s ' $(CALLDIR)/$* $$end $(DOTS); \
	      printf 'FAIL\n' ; \
         false ; \
     $(TESTTIMEACTION)\
     fi )

#     result=$$?; \
#     len=`echo Test in $(CALLDIR)/$* | wc -c `;end=`expr 68 - $$len`;printf 'Test in %s %.*s ' $(CALLDIR)/$* $$end $(DOTS); \
#	  if [ -f $*/.success ] ; then printf 'OK\n' ; else printf 'FAIL\n' ; fi; \
#     if [ $$result -ne 0 ] ; then false ; fi )

$(CLEAN_TARGETS_DIR): %.clean:
	@(cd $*; $(MAKE) --no-print-directory clean)

ifneq ($(V),) 
VERBOSE:=$(V)
endif
ifeq ($(VERBOSE),) 
   CMDECHO=@
else
   CMDECHO=
endif

clean:  $(CLEAN_TARGETS_DIR)
	$(CMDECHO) rm -rf main *Dict\.* Event.root .*~ *~ $(CLEAN_TARGETS)

distclean: clean
	$(CMDECHO) rm -rf $(ROOTTEST_LOC)roottiming.root $(ROOTTEST_LOC)runid

cleantest: test

# For now logs.tar.gz is a phony target
logs.tar.gz:	
	$(CMDECHO) find . -name '*log' | xargs tar cfz logs.tar.gz  

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

ifeq ($(CXXFLAGS),)
   export CXXFLAGS := $(shell root-config --cflags)
endif
ifeq ($(ROOTLIBS),)
   export ROOTLIBS     := $(shell root-config --nonew --libs)
endif
ifeq ($(ROOTGLIBS),)
   export ROOTGLIBS    := $(shell root-config --nonew --glibs)
endif
endif

ObjSuf   = o

ifeq ($(ARCH),macosx64)
PYTHON := python64
else
PYTHON := python
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
      PYTHONLIB:=$(shell grep ^PYTHONLIB $(ROOTSYS)/config/Makefile.config | sed 's,^.*\:=,,')
      PYTHONFWK:=$(dir $(PYTHONLIB))
      ifneq ($(PYTHONFWK),)
         export PATH:=$(PYTHONFWK)/bin:$(PATH)
         export DYLD_LIBRARY_PATH:=$(PYTHONFWK):$(DYLD_LIBRARY_PATH)
      endif
   endif
endif

ifeq ($(PLATFORM),win32)

ifeq ($(ROOT_LOC),)
   export ROOT_LOC := $(shell cygpath -u '$(ROOTSYS)')
endif

# Windows with the VC++ compiler
ObjSuf        = obj
SrcSuf        = cxx
ExeSuf        = .exe
DllSuf        = dll
LibSuf        = lib
OutPutOpt     = -out:
CXX           = cl
#CXXOPT        = -O2
CXXOPT        = -Z7
#CXXFLAGS      = $(CXXOPT) -G5 -GR -MD -DWIN32 -D_WINDOWS -nologo \
#                -DVISUAL_CPLUSPLUS -D_X86_=1 -D_DLL
ifeq ($(RCONFIG_INC),)
   export RCONFIG_INC   := $(shell root-config --incdir)
endif
CXXFLAGS      += $(CXXOPT) -nologo -I$(RCONFIG_INC) -FIw32pragma.h
CXXFLAGS      += -TP 
LD            = link -nologo
#LDOPT         = -opt:ref
#LDOPT         = -debug
#LDFLAGS       = $(LDOPT) -nologo -nodefaultlib -incremental:no
SOFLAGS       = -DLL
SYSLIBS       = kernel32.lib  ws2_32.lib mswsock.lib \
                advapi32.lib  user32.lib gdi32.lib comdlg32.lib winspool.lib 

else 

# Non windows default:

export ROOT_LOC := $(ROOTSYS)

ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
LibSuf        = so
OutPutOpt     = -o 

endif

ifeq ($(ARCH),linux)

# Linux with egcs, gcc 2.9x, gcc 3.x (>= RedHat 5.2)
CXX           = g++
LD            = g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxx8664gcc)

CXX           = g++
LD            = g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -Wall -fPIC
else
CXXFLAGS      += -O -Wall -fPIC
endif
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxicc)
# Linux with linuxicc
CXX = icc
LD  = icc
ifeq ($(ROOTBUILD),debug)
CXXFLAGS += -g -wd191 
else
CXXFLAGS += -O -wd191 
endif
SOFLAGS  = -shared 
endif


ifeq ($(ARCH),macosx)

# MacOSX with cc/g++
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CXX           = g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -pipe -Wall -fPIC -Woverloaded-virtual
else
CXXFLAGS      += -O -pipe -Wall -fPIC -Woverloaded-virtual
endif
ifeq ($(MACOSX_MINOR),) 
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
ifeq ($(subst $(MACOSX_MINOR),,123),123)
UNDEFOPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.$(MACOSX_MINOR) c++
else
ifeq ($(MACOSX_MINOR),3)
UNDEFOPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.$(MACOSX_MINOR) c++
CXXFLAGS     += -Wno-long-double
else
UNDEFOPT      = suppress
LD            = c++
CXXFLAGS     += -Wno-long-double
endif
endif
SOFLAGS       = -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = dylib
ifeq ($(subst $(MACOSX_MINOR),,01234),01234)
LibSuf        = so
endif
endif


ifeq ($(ARCH),macosx64)

# MacOSX 64 bit with cc/g++
export DYLD_LIBRARY_PATH:=$(ROOTTEST_HOME)/scripts:$(DYLD_LIBRARY_PATH)
CXX           = g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -m64 -g -pipe -Wall -fPIC -Woverloaded-virtual
else
CXXFLAGS      += -m64 -O -pipe -Wall -fPIC -Woverloaded-virtual
endif
ifeq ($(MACOSX_MINOR),) 
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
ifeq ($(subst $(MACOSX_MINOR),,123),123)
UNDEFOPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.$(MACOSX_MINOR) c++
else
ifeq ($(MACOSX_MINOR),3)
UNDEFOPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.$(MACOSX_MINOR) c++
else
UNDEFOPT      = suppress
LD            = c++
endif
endif
LDFLAGS       = -m64
SOFLAGS       = -m64 -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf        = so
LibSuf        = dylib
ifeq ($(subst $(MACOSX_MINOR),,01234),01234)
LibSuf        = so
endif
endif

CALLROOTEXE  ?= root.exe
export CALLROOTEXE

ifneq ($(MAKECMDGOALS),clean)
ifeq ($(CINT_VERSION),)
   export CINT_VERSION := Cint$(shell $(CALLROOTEXE) -q -b | grep CINT | sed -e 's/.*\([57]\).*/\1/' )
endif
endif

# Track the version of ROOT we are runing with

ROOTV=$(ROOTTEST_LOC)/root_version
ROOTVFILE=$(ROOTTEST_HOME)/root_version
ifeq ($(ROOTTEST_CHECKED_VERSION),)
   export ROOTTEST_CHECKED_VERSION:= $(shell echo $(ROOTSYS) && (echo "$(ROOTSYS)" | diff - "$(ROOTVFILE)" 2> /dev/null ) || (echo "$(ROOTSYS)" > $(ROOTVFILE); echo "New ROOT version ($(ROOTSYS))" >&2))

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

ROOTCORELIBS_LIST = Core Cint Tree Hist TreePlayer
ROOTCORELIBS = $(addprefix $(ROOT_LOC)/lib/lib,$(addsuffix .$(DllSuf),$(ROOTCORELIBS_LIST)))
ROOTCINT = $(ROOT_LOC)/bin/rootcint$(ExeSuf)

UTILS_LIBS =  $(ROOTTEST_LOC)scripts/utils_cc.$(DllSuf) $(ROOTTEST_LOC)scripts/recordtiming_cc.$(DllSuf)

$(ROOTTEST_LOC)scripts/utils_cc.$(DllSuf) : $(ROOTTEST_LOC)scripts/utils.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$(ROOTTEST_HOME)scripts/utils.cc\"\) > $(ROOTTEST_LOC)scripts/utils_cc.build.log 2>&1 ; \
	if test $$? -ne 0 ; \
	then \
	  cat $(ROOTTEST_LOC)scripts/utils_cc.build.log ; \
	else \
	  if test -f $@ ; \
	  then \
	    touch $@ ; \
	  fi ; \
	fi

$(ROOTTEST_LOC)scripts/recordtiming_cc.$(DllSuf) : $(ROOTTEST_LOC)scripts/recordtiming.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$(ROOTTEST_HOME)scripts/recordtiming.cc\"\) > $(ROOTTEST_LOC)scripts/recordtiming_cc.build.log 2>&1 ; \
	if test $$? -ne 0 ; \
	then \
	  cat $(ROOTTEST_LOC)scripts/recordtiming_cc.build.log ; \
	else \
	  if test -f $@ ; \
	  then \
	    touch $@ ; \
	  fi ; \
	fi

override ROOTMAP = $(ROOT_LOC)/etc/system.rootmap

$(ROOTMAP): 
	@echo Error $(ROOTMAP) is required for roottest '(Do cd $$ROOTSYS; $(MAKE) map)'

check: $(ROOT_LOC)/lib/libCore.$(DllSuf)

UTILS_PREREQ =  $(UTILS_LIBS) 

utils:  $(UTILS_LIBS) 

copiedEvent$(ExeSuf): $(EVENTDIR)/bigeventTest.success
	$(CMDECHO) cp $(EVENTDIR)/libEvent.* $(EVENTDIR)/Event.h .
	$(CMDECHO) cp $(EVENTDIR)/Event$(ExeSuf) ./copiedEvent$(ExeSuf)
ifeq ($(PLATFORM),win32)
	$(CMDECHO) if [ -e $(EVENTDIR)/Event$(ExeSuf).manifest ] ; then cp $(EVENTDIR)/Event$(ExeSuf).manifest ./copiedEvent$(ExeSuf).manifest ; fi
endif

%.o: %.C
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_C.build.log 2>&1

%.o: %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_cc.build.log 2>&1

%.o: %.cxx
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_cxx.build.log 2>&1

%.o: %.cpp
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_cpp.build.log 2>&1

%.$(ObjSuf): %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_o_cc.build.log 2>&1

%.obj: %.C
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_obj_C.build.log 2>&1

%.obj: %.cc
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_obj_cc.build.log 2>&1

%.obj: %.cxx
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_obj_cxx.build.log 2>&1

%.obj: %.cpp
	$(CMDECHO) $(CXX) $(CXXFLAGS) -c $< > $*_obj_cpp.build.log 2>&1

%_cpp.$(DllSuf) : %.cpp $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cpp.build.log 2>&1 || cat $*_cpp.build.log

%_C.$(DllSuf) : %.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_C.build.log 2>&1 || cat $*_C.build.log 

%_cxx.$(DllSuf) : %.cxx $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cxx.build.log 2>&1 || cat $*_cxx.build.log 

%_cc.$(DllSuf) : %.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cc.build.log 2>&1 || cat $*_cc.build.log 

%_h.$(DllSuf) : %.h $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_h.build.log 2>&1 || cat $*_h.build.log 

%.log : run%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b $< > $@ 2>&1

%.log : %.py $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
ifeq ($(PYTHONPATH),)
	$(CMDECHO) PYTHONPATH=$(ROOTSYS)/lib $(PYTHON) $< -b > $@ 2>&1
else 
	$(CMDECHO) $(PYTHON) $< -b > $@ 2>&1
endif

.PRECIOUS: %_C.$(DllSuf) 

%.clog : run%_C.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) $(CALLROOTEXE) -q -l -b run$*.C+ > $@ 2>&1

%.neutral.clog: %.clog
	$(CMDECHO) cat $*.clog | sed -e 's:0x.*:0xRemoved:' > $@

%.neutral.log: %.log
	$(CMDECHO) cat $*.clog | sed -e 's:0x.*:0xRemoved:' > $@

ifneq ($(PLATFORM),macosx)

define BuildWithLib
	$(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"$<\",\"$(filter %.$(DllSuf),$^)\",\"\")" > $*.build.log 2>&1 || cat $*.build.log 
endef

else

define BuildWithLib
        $(CMDECHO) $(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"$<\",\"$(filter %.$(DllSuf),$^)\",\"\")" > $*.build.log 2>&1 || cat $*.build.log
endef

endif

define WarnFailTest
	$(CMDECHO)echo Warning $@ has some known skipped failures "(in ./$(CURRENTDIR))"
endef

define TestDiffCintSpecific
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS)-$(CINT_VERSION) ]; then \
	   diff -u -b $@.ref$(ROOTBITS)-$(CINT_VERSION) $< ; \
	elif  [ -f $@.ref-$(CINT_VERSION) ]; then \
	   diff -u -b $@.ref-$(CINT_VERSION) $< ; \
	elif [ -f $@.ref$(ROOTBITS) ]; then \
	   diff -u -b $@.ref$(ROOTBITS) $< ; \
	else \
	   diff -u -b $@.ref $< ; \
	fi
endef

define TestDiff
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS) ]; then \
	   diff -u -b $@.ref$(ROOTBITS) $< ; \
	else \
	   diff -u -b $@.ref $< ; \
	fi
endef

define TestDiffW
	$(CMDECHO) if [ -f $@.ref$(ROOTBITS) ]; then \
	   diff -u -b -w $@.ref$(ROOTBITS) $< ; \
	else \
	   diff -u -b -w $@.ref $< ; \
	fi
endef


define BuildFromObj
$(CMDECHO) ( touch dummy$$$$.C && \
	($(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"\",\"$<\")" > $@.build.log 2>&1 || cat $@.build.log ) && \
	mv dummy$$$$_C.$(DllSuf) $@ && \
	rm -f dummy$$$$.C dummy$$$$_C.* \
)
endef

define BuildFromObjs
$(CMDECHO) ( touch dummy$$$$.C && \
	($(CALLROOTEXEBUILD) -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"\",\"$(filter %.$(ObjSuf),$^)\")" > $@.build.log 2>&1 || cat $@.build.log ) && \
	mv dummy$$$$_C.$(DllSuf) $@ && \
	rm dummy$$$$.C \
)
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

