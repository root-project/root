all: tests
test: tests

SUBDIRS =  

TEST_TARGETS = $(SUBDIRS:%=%.test) 

tests: $(TEST_TARGETS)
	@echo "All test succeeded in `pwd`"

$(TEST_TARGETS): %.test:
	@(cd $*; gmake test)

clean:
	rm -f main *Dict* Event.root *~ $(CLEAN_TARGET)


# here we guess the platform

ARCH          = $(shell root-config --arch)
PLATFORM      = $(ARCH)

CXXFLAGS = $(shell root-config --cflags)

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

ROOTLIBS     := $(shell root-config --nonew --libs)
ROOTGLIBS    := $(shell root-config --nonew --glibs)
endif

ifeq ($(ARCH),linux)
# Linux with egcs, gcc 2.9x, gcc 3.x (>= RedHat 5.2)
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
CXX           =
ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
OutPutOpt     = -o 
endif

