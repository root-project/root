all: tests

test: tests ;

# The previous line contains just ';' in order to disable the implicit 
# rule building an executable 'test' from test.C

# The user directory should define
# SUBDIRS listing any activated subdirectory
# TEST_TARGETS with the list of activated test
# CLEAN_TARGETS with the list of things to delete

# doing gmake VERBOSE=true allows for more output, include the original
# commands.

# doing gmake FAIL=true run the test that are known to fail

SUBDIRS = $(shell $(ROOTTEST_HOME)/scripts/subdirectories .)

TEST_TARGETS_DIR = $(SUBDIRS:%=%.test)
TEST_TARGETS += $(TEST_TARGETS_DIR)

# allow tests to be disabled by putting their names into a file called !DISABLE
ifneq ($(MAKECMDGOALS),clean)
TEST_TARGETS_DISABLED = $(if $(wildcard !DISABLE),$(shell cat !DISABLE))
endif
TEST_TARGETS := $(if $(TEST_TARGETS_DISABLED),\
                     $(filter-out $(TEST_TARGETS_DISABLED),$(TEST_TARGETS))\
                     $(warning Test(s) $(TEST_TARGETS_DISABLED) disabled!),\
                  $(TEST_TARGETS))

CLEAN_TARGETS_DIR = $(SUBDIRS:%=%.clean)
CLEAN_TARGETS += 

ALL_LIBRARIES += *.d *.o *.obj *.so *.def *.exp *.dll *.lib dummy.C *.pdb .def *.ilk *.manifest

.PHONY: clean removefiles tests all test $(TEST_TARGETS) $(TEST_TARGETS_DIR) utils check

include $(ROOTTEST_HOME)/scripts/Common.mk

ifeq ($(MAKECMDGOALS),cleantest)
	TESTGOAL = cleantest
else
	TESTGOAL = test
endif

EVENTDIR = $(ROOTTEST_HOME)/root/io/event/
$(EVENTDIR)/$(SUCCESS_FILE): $(ROOTCORELIBS)  
	$(CMDECHO) (cd $(EVENTDIR); $(MAKE) CURRENTDIR=$(EVENTDIR) --no-print-directory $(TESTGOAL); )

$(TEST_TARGETS_DIR): %.test:  $(EVENTDIR)/$(SUCCESS_FILE)
	@(echo Running test in $(CALLDIR)/$*)
	@(cd $*; $(MAKE) CURRENTDIR=$* --no-print-directory $(TESTGOAL); \
     result=$$?; \
     if [ $$result -ne 0 ] ; then \
         len=`echo Tests in $(CALLDIR)/$* | wc -m `;end=`expr 68 - $$len`;printf 'Test in %s %.*s ' $(CALLDIR)/$* $$end $(DOTS); \
	      printf 'FAIL\n' ; \
         false ; \
     fi )

#     result=$$?; \
#     len=`echo Test in $(CALLDIR)/$* | wc -m `;end=`expr 68 - $$len`;printf 'Test in %s %.*s ' $(CALLDIR)/$* $$end $(DOTS); \
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
	$(CMDECHO) rm -rf main *Dict\.* Event.root *~ $(CLEAN_TARGETS)

cleantest: test

ifeq ($(MAKECMDGOALS),cleantest)
  ifeq ($(VERBOSE),) 
     ForceRemoveFiles := $(shell rm -rf main *Dict\.* Event.root *~ $(CLEAN_TARGETS) )
  else 
     ForceRemoveFilesVerbose := $(shell echo rm -rf main *Dict\.* Event.root *~ $(CLEAN_TARGETS) 1>&2 )
     ForceRemoveFiles := $(shell rm -rf main *Dict\.* Event.root *~ $(CLEAN_TARGETS) )
  endif
endif

# here we guess the platform

ifeq ($(ARCH),)
   ARCH          = $(shell root-config --arch)
endif
PLATFORM      = $(ARCH)

ifeq ($(CXXFLAGS),)
   export CXXFLAGS = $(shell root-config --cflags)
endif
ifeq ($(ROOTLIBS),)
   export ROOTLIBS     := $(shell root-config --nonew --libs)
endif
ifeq ($(ROOTGLIBS),)
   export ROOTGLIBS    := $(shell root-config --nonew --glibs)
endif

ObjSuf   = o

ifeq ($(HAS_PYTHON),)
   export HAS_PYTHON = $(shell root-config --has-python)
endif
ifeq ($(HAS_PYTHON),yes)
   ifeq ($(findstring $(ROOTSYS)/lib, $(PYTHONPATH)),)
      # The PYTHONPATH does not have ROOTSYS/lib in it yet
      # let's add it
      ifeq ($(PLATFORM),win32)
         export PYTHONPATH := $(shell cygpath -w $(ROOTSYS)/bin);$(PYTHONPATH);$(ROOTSYS)/lib
       else
         export PYTHONPATH := $(ROOTSYS)/lib:$(PYTHONPATH)
       endif
   endif
endif

ifeq ($(PLATFORM),win32)

ROOTTEST_HOME := $(shell cygpath -m $(ROOTTEST_HOME))
ifeq ($(ROOT_LOC),)
   export ROOT_LOC := $(shell cygpath -u '$(ROOTSYS)')
endif

# Windows with the VC++ compiler
ObjSuf        = obj
SrcSuf        = cxx
ExeSuf        = .exe
DllSuf        = dll
OutPutOpt     = -out:
CXX           = cl
CXXOPT        = -O2
#CXXOPT        = -Z7
#CXXFLAGS      = $(CXXOPT) -G5 -GR -MD -DWIN32 -D_WINDOWS -nologo \
#                -DVISUAL_CPLUSPLUS -D_X86_=1 -D_DLL
CXXFLAGS      += $(CXXOPT) -nologo -I$(shell root-config --incdir) -FIw32pragma.h
CXXFLAGS      += /TP 
LD            = link
#LDOPT         = -opt:ref
#LDOPT         = -debug
#LDFLAGS       = $(LDOPT) -nologo -nodefaultlib -incremental:no
SOFLAGS       = -DLL
SYSLIBS       = kernel32.lib  ws2_32.lib mswsock.lib \
                advapi32.lib  user32.lib gdi32.lib comdlg32.lib winspool.lib 

else 

# Non windows default:

ROOT_LOC = $(ROOTSYS)

ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
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
CXX           = g++
ifeq ($(ROOTBUILD),debug)
CXXFLAGS      += -g -pipe -Wall -fPIC -Wno-long-double -Woverloaded-virtual
else
CXXFLAGS      += -O -pipe -Wall -fPIC -Wno-long-double -Woverloaded-virtual
endif
ifeq ($(MACOSX_MINOR),) 
  export MACOSX_MINOR := $(shell sw_vers | sed -n 's/ProductVersion://p' | cut -d . -f 2)
endif
ifeq ($(MACOSX_MINOR),4)
UNDEF\OPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.4 c++
else
ifeq ($(MACOSX_MINOR),3)
UNDEFOPT      = dynamic_lookup
LD            = MACOSX_DEPLOYMENT_TARGET=10.3 c++
else
UNDEFOPT      = suppress
LD            = c++
endif
endif
SOFLAGS       = -dynamiclib -single_module -undefined $(UNDEFOPT)
DllSuf   = so
endif

# Track the version of ROOT we are runing with

ROOTV=$(ROOTTEST_HOME)/root_version
dummy:=$(shell (echo "$(ROOTSYS)" | diff - "$(ROOTV)" 2> /dev/null ) || (echo "$(ROOTSYS)" > $(ROOTV); echo "New ROOT version ($(ROOTSYS))" >&2))

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

UTILS_LIBS =  $(ROOTTEST_HOME)scripts/utils_cc.$(DllSuf)

override ROOTMAP = $(ROOT_LOC)/etc/system.rootmap

$(ROOTMAP): 
	@echo Error $(ROOTMAP) is required for roottest '(Do cd $$ROOTSYS; $(MAKE) map)'

check: $(ROOT_LOC)/lib/libCore.$(DllSuf)

UTILS_PREREQ =  $(UTILS_LIBS) $(ROOTMAP)

utils:  $(UTILS_LIBS) $(ROOTMAP)

copiedEvent$(ExeSuf): $(EVENTDIR)/$(SUCCESS_FILE)
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
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cpp.build.log 2>&1

%_C.$(DllSuf) : %.C $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_C.build.log 2>&1

%_cxx.$(DllSuf) : %.cxx $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cxx.build.log 2>&1

%_cc.$(DllSuf) : %.cc $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_cc.build.log 2>&1

%_h.$(DllSuf) : %.h $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\) > $*_h.build.log 2>&1

%.log : run%.C $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b $< > $@ 2>&1

%.log : %.py $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
ifeq ($(PYTHONPATH),)
	$(CMDECHO) PYTHONPATH=$(ROOTSYS)/lib python $< -b > $@ 2>&1
else 
	$(CMDECHO) python $< -b > $@ 2>&1
endif

.PRECIOUS: %_C.$(DllSuf) 

%.clog : run%_C.$(DllSuf) $(UTILS_PREREQ) $(ROOTCORELIBS) $(ROOTCINT) $(ROOTV)
	$(CMDECHO) root.exe -q -l -b run$*.C+ > $@ 2>&1

ifneq ($(ARCH),macosx)

define BuildWithLib
	$(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\,\"$(filter %.$(DllSuf),$^)\",\"\"\) > $*.build.log 2>&1
endef

else

define BuildWithLib
        $(CMDECHO) root.exe -q -l -b $(ROOTTEST_HOME)/scripts/build.C\(\"$<\"\,\"$(filter %.dylib,$^)\",\"\"\) > $*.build.log 2>&1
endef

endif

define WarnFailTest
	$(CMDECHO)echo Warning $@ has some known skipped failures "(in ./$(CURRENTDIR))"
endef

define TestDiff
	$(CMDECHO) diff -b $@.ref $<
endef

define TestDiffW
	$(CMDECHO) diff -b -w $@.ref $<
endef


define BuildFromObj
$(CMDECHO) ( touch dummy$$$$.C && \
	root.exe -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"\",\"$<\")" > $@.build.log 2>&1 && \
	mv dummy$$$$_C.$(DllSuf) $@ && \
	rm -f dummy$$$$.C dummy$$$$_C.* \
)
endef

define BuildFromObjs
$(CMDECHO) ( touch dummy$$$$.C && \
	root.exe -q -l -b "$(ROOTTEST_HOME)/scripts/build.C(\"dummy$$$$.C\",\"\",\"$^\")" > $@.build.log 2>&1 && \
	mv dummy$$$$_C.$(DllSuf) $@ && \
	rm dummy$$$$.C \
)
endef

RemoveLeadingDirs := sed -e 's?^[A-Za-z/\].*[/\]??' -e 's/.dll/.so/'
RemoveDirs := sed -e 's?([A-Za-z]:|[/]).*[/\]??'

