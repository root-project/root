all: tests
test: tests

TEST_TARGETS_DIR = $(SUBDIRS:%=%.test) 
TEST_TARGETS += $(TEST_TARGETS_DIR)

CLEAN_TARGETS_DIR = $(SUBDIRS:%=%.clean)
CLEAN_TARGETS += 

tests: $(TEST_TARGETS)
	@echo "All test succeeded in `pwd`"

$(TEST_TARGETS_DIR): %.test:
	@(cd $*; gmake test)

$(CLEAN_TARGETS_DIR): %.clean:
	@(cd $*; gmake clean)

clean:  $(CLEAN_TARGETS_DIR)
	rm -f main *Dict* Event.root *~ $(CLEAN_TARGETS)


# here we guess the platform

ARCH          = $(shell root-config --arch)
PLATFORM      = $(ARCH)

CXXFLAGS = $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --nonew --libs)
ROOTGLIBS    := $(shell root-config --nonew --glibs)

ifeq ($(PLATFORM),win32)
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
CXXFLAGS      += /TP /GX  -G5 -GR
LD            = link
#LDOPT         = -opt:ref
#LDOPT         = -debug
#LDFLAGS       = $(LDOPT) -pdb:none -nologo -nodefaultlib -incremental:no
SOFLAGS       = -DLL
SYSLIBS       = msvcrt.lib oldnames.lib kernel32.lib  ws2_32.lib mswsock.lib \
                advapi32.lib  user32.lib gdi32.lib comdlg32.lib winspool.lib \
                msvcirt.lib

endif

ifeq ($(ARCH),linux)
# Linux with egcs, gcc 2.9x, gcc 3.x (>= RedHat 5.2)
CXX           = g++
CXXFLAGS      += -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
OutPutOpt     = -o 
endif

ifeq ($(ARCH),linuxicc)
# Linux with linuxicc
CXX = icc
LD  = icc
ifeq ($(ROOTBUILD),debug)
CXXFLAGS += -g
else
CXXFLAGS += -O
endif
SOFLAGS  = -shared 
DllSuf   = so
ExeSuf   = 
OutPutOpt     = -o 
endif


##### utilities #####

MAKELIB       = $(ROOTSYS)/build/unix/makelib.sh $(MKLIBOPTIONS)
ifeq ($(PLATFORM),win32)
MAKELIB       = $(ROOTSYS)/build/win/makelib.sh
endif

%.o: %.C
	$(CXX) $(CXXFLAGS) -c $<

%.o: %.cxx
	$(CXX) $(CXXFLAGS) -c $<

