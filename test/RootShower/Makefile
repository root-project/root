# Makefile for the RootShower program.
# This Makefile shows nicely how to compile and link applications
# using the ROOT libraries on all supported platforms.
#
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/2/2003

ARCH         := $(shell root-config --arch)

CXX           =
ObjSuf        = o
SrcSuf        = cxx
ExeSuf        =
DllSuf        = so
OutPutOpt     = -o 

ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell root-config --glibs)
ROOTLIBDIR   := $(shell root-config --libdir)

ifeq ($(ARCH),hpux)
# HP-UX with CC
CXX           = CC
CXXFLAGS      = -O +Z
LD            = CC
LDFLAGS       = -O +a1 -z
SOFLAGS       = -b
DllSuf        = sl
endif

ifeq ($(ARCH),hpuxacc)
# HP-UX 10.x with aCC
CXX           = aCC
CXXFLAGS      = -O +Z
LD            = aCC
LDFLAGS       = -O -z
SOFLAGS       = -b
endif

ifeq ($(ARCH),hpuxia64acc)
# HP-UX 11i 1.5 (IA-64) with aCC
CXX           = aCC
CXXFLAGS      = +DD64 -O +Z
LD            = aCC
LDFLAGS       = +DD64 -O -z
SOFLAGS       = -b
endif

ifeq ($(ARCH),hpuxegcs)
# HP-UX 10.x with g++
CXXFLAGS      = -O -fPIC
CXX           = g++
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -fPIC -shared
endif


ifeq ($(ARCH),hurddeb)
# GNU/Hurd
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),aix)
# IBM AIX xlC 4.x
CXX           = xlC
CXXFLAGS      = -O
LD            = xlC
LDFLAGS       = -O
SOFLAGS       =
ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell root-config --glibs)
endif

ifeq ($(ARCH),aix5)
# IBM AIX xlC 5.x
CXX           = xlC
CXXFLAGS      = -O
LD            = xlC
LDFLAGS       = -O
SOFLAGS       =
ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell root-config --glibs)
endif

ifeq ($(ARCH),aixegcs)
# IBM AIX with GCC
CXX           = g++
CXXFLAGS      = -O
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),solaris)
# Solaris CC
CXX           = /opt/SUNWspro/bin/CC
CXXFLAGS      = -O -KPIC
LD            = /opt/SUNWspro/bin/CC
LDFLAGS       = -O
SOFLAGS       = -G
endif

ifeq ($(ARCH),solarisCC5)
# Solaris CC 5.0
CXX           = CC
CXXFLAGS      = -O -KPIC
LD            = CC
LDFLAGS       = -O
SOFLAGS       = -G
endif

ifeq ($(ARCH),solarisegcs)
# Solaris egcs
CXX           = g++
CXXFLAGS      = -O -fPIC
LD            = CC
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),solarisgcc)
# Solaris gcc
CXX           = g++
CXXFLAGS      = -O -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),solariskcc)
# Solaris kcc
CXX           = KCC --one_instantiation_per_object
CXXFLAGS      = -O4 -KPIC 
LD            = KCC
LDFLAGS       = -O4
SOFLAGS       =
endif

ifeq ($(ARCH),solarisx86)
# Solaris CC on Intel
CXX           = CC
CXXFLAGS      = -O -KPIC
LD            = CC
LDFLAGS       = -O
SOFLAGS       = -G
endif

ifeq ($(ARCH),sgicc)
# SGI
CXX           = CC -n32  -I/usr/include/CC.sgi
CXXFLAGS      = -O
LD            = CC -n32  -I/usr/include/CC.sgi
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),sgiegcs)
# SGI 6.x with EGCS
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O -Wl,-u,__builtin_new -Wl,-u,__builtin_delete -Wl,-u,__nw__FUiPv
SOFLAGS       = -shared
endif

ifeq ($(ARCH),sgin32egcs)
# SGI 6.x with EGCS for n32 ABI
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O -L/usr/lib32 -Wl,-woff,134
SOFLAGS       = -shared
endif

ifeq ($(ARCH),sgigcc)
# SGI with GCC
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O -Wl,-u,__builtin_new -Wl,-u,__builtin_delete -Wl,-u,__nw__FUiPv
SOFLAGS       = -shared
endif

ifeq ($(ARCH),sgikcc)
# SGI with KCC
CXX           = KCC -n32 --one_instantiation_per_object
CXXFLAGS      = -O
LD            = KCC -n32
LDFLAGS       = -O
SOFLAGS       =
endif

ifeq ($(ARCH),alphagcc)
# Alpha/OSF with g++
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -Wl,-expect_unresolved,* -shared
endif

ifeq ($(ARCH),alphaegcs)
# Alpha/OSF with egcs
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -Wl,-expect_unresolved,* -shared
endif

ifeq ($(ARCH),alphakcc)
# Alpha/OSF with kai compiler (not yet valid)
CXX           = KCC --one_instantiation_per_object
CXXFLAGS      = -O -fPIC
LD            = KCC 
LDFLAGS       = -O
SOFLAGS       = -Wl,-expect_unresolved,* -shared
endif

ifeq ($(ARCH),alphacxx6)
# Alpha/OSF with cxx6
CXX           = cxx
CXXFLAGS      = -O
LD            = cxx
LDFLAGS       = -O
SOFLAGS       = -shared -nocxxstd -Wl,-expect_unresolved,*,-msym
endif

ifeq ($(ARCH),alphacxx)
# Alpha/OSF with cxx5
CXX           = cxx
CXXFLAGS      = -O
LD            = cxx
LDFLAGS       = -O
SOFLAGS       = -Wl,-expect_unresolved,* -call_shared
endif

ifeq ($(ARCH),linuxrh51)
# Linux with gcc 2.7.2.x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxrh42)
# Linux with gcc 2.7.2.x (RedHat 4.2)
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxdeb)
# Linux with gcc 2.7.2.x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxdeb2)
# Linux with gcc 2.7.2.x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif
  
ifeq ($(ARCH),linuxdeb2ppc)
# Debian/Linux on the PowerPC
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif
 

ifeq ($(ARCH),linuxsuse6)
# Linux with gcc 2.7.2.x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linux)
# Linux with egcs, gcc 2.9x, gcc 3.x (>= RedHat 5.2)
CXX           = g++
CXXFLAGS      = -g -O -Wall -fPIC
LD            = g++
LDFLAGS       = -g -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxkcc)
# Linux with the KAI compiler
CXX           = KCC --one_instantiation_per_object
CXXFLAGS      = -fPIC +K0
LD            = KCC
LDFLAGS       = -O $(shell root-config --cflags)
SOFLAGS       =
endif

ifeq ($(ARCH),linuxicc)
# Linux with Intel icc compiler
CXX           = icc
CXXFLAGS      = -O
LD            = icc
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxppcegcs)
# MkLinux with egcs/glibc
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxia64gcc)
# Itanium Linux with gcc 2.9x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxia64sgi)
# Itanium Linux with sgiCC
CXX           = sgiCC
CXXFLAGS      = -O -Wall -fPIC
LD            = gsgiCC
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxia64ecc)
# Itanium Linux with Intel ecc
CXX           = ecc
CXXFLAGS      = -O
LD            = ecc
LDFLAGS       = -O -i_dynamic
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxx8664gcc)
# AMD Opteron (64 bit mode) Linux with gcc 3.x
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxalphaegcs)
# Alpha Linux with egcs
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),linuxarm)
# ARM Linux with egcs
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),mklinux)
# MkLinux with libc5
CXX           = g++
CXXFLAGS      = -O -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared
endif

ifeq ($(ARCH),freebsd)
# FreeBSD with libc5
CXX           = g++
CXXFLAGS      = -O -pipe -W -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared -Wl,-x
endif

ifeq ($(ARCH),freebsd4)
# FreeBSD with glibc
CXX           = g++
CXXFLAGS      = -O -pipe -W -Wall -fPIC
LD            = g++
LDFLAGS       = -O
SOFLAGS       = -shared -Wl,-x
endif

ifeq ($(ARCH),macosx)
# MacOS X with cc (GNU cc 2.95.2)
CXX           = c++
CXXFLAGS      = -O -pipe -Wall
LD            = c++
LDFLAGS       = -O -Xlinker -bind_at_load -flat_namespace
# The SOFLAGS will be used to create the .dylib; the .so will
# be created separately
DllSuf        = dylib
SOFLAGS       = -dynamiclib -flat_namespace -undefined suppress
endif

ifeq ($(ARCH),hiux)
# Hitachi HIUX
CXX           = g++
CXXFLAGS      = -O2 -fPIC
LD            = g++
LDFLAGS       = -Wl,+s
SOFLAGS       = -Wl,-b,-E -nostdlib -nostartfiles
DllSuf        = sl
endif

ifeq ($(ARCH),win32)
# Windows with the VC++ compiler
ObjSuf        = obj
SrcSuf        = cxx
ExeSuf        = .exe
DllSuf        = dll
OutPutOpt     = -out:
CXX           = cl
#CXXOPT        = -O
CXXOPT        = -Z7
CXXFLAGS      = $(CXXOPT) -G5 -GR -GX -MD -DWIN32 -D_WINDOWS -nologo \
                -DVISUAL_CPLUSPLUS -D_X86_=1 -D_DLL
LD            = link
#LDOPT         = -opt:ref
LDOPT         = -debug
LDFLAGS       = $(LDOPT) -pdb:none -nologo -nodefaultlib -incremental:no \
                /SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup
SOFLAGS       = -DLL
SYSLIBS       = msvcrt.lib oldnames.lib kernel32.lib ws2_32.lib mswsock.lib \
                advapi32.lib user32.lib gdi32.lib comdlg32.lib winspool.lib \
                msvcprt.lib

ROOTLIBS     := $(shell root-config --libs)
ROOTGLIBS    := $(shell root-config --glibs)
endif
 
ifeq ($(ARCH),win32gcc)
# Windows with gcc
DllSuf        = dll
ExeSuf        = .exe
CXX           = g++
CXXFLAGS      = -O -Wall -Woverloaded-virtual -I/usr/X11R6/include
LD            = g++
LDFLAGS       = -O -Wl,--enable-auto-import 
SOFLAGS       = -shared -D_DLL -Wl,--export-all-symbols 
EXPLLINKLIBS  = $(ROOTLIBS) $(ROOTGLIBS)
endif

ifeq ($(CXX),)
$(error $(ARCH) invalid architecture)
endif

CXXFLAGS     += $(ROOTCFLAGS)
LIBS          = $(ROOTLIBS) $(SYSLIBS)
GLIBS         = $(ROOTGLIBS) $(SYSLIBS)

#------------------------------------------------------------------------------

ROOTSHOWERS  := $(wildcard *.$(SrcSuf)) RootShowerDict.$(SrcSuf)
ROOTSHOWERO  := $(ROOTSHOWERS:.$(SrcSuf)=.$(ObjSuf))
ROOTSHOWER    = RootShower$(ExeSuf)

OBJS          = $(ROOTSHOWERO)

PROGRAMS      = $(ROOTSHOWER)

#------------------------------------------------------------------------------

.SUFFIXES: .$(SrcSuf) .$(ObjSuf) .$(DllSuf)

all:            $(PROGRAMS)

$(ROOTSHOWER):  $(ROOTSHOWERO)
ifeq ($(ARCH),win32)
		$(LD) $(LDFLAGS) $^ $(GLIBS) '$(ROOTSYS)\lib\libEG.lib' \
		   '$(ROOTSYS)\lib\libHtml.lib' '$(ROOTSYS)\lib\libGeom.lib' $(OutPutOpt)$@
else
		$(LD) $(LDFLAGS) $^ $(GLIBS) -lEG -lHtml -lGeom $(OutPutOpt)$@
endif
		@echo "$@ done"

RootShowerDict.$(SrcSuf): MyParticle.h MyDetector.h MyEvent.h RSLinkDef.h
		@echo "Generating dictionary $@..."
		@rootcint -f $@ -c $^

clean:
		@rm -f $(OBJS) core

distclean:      clean
		@rm -f $(PROGRAMS) *Dict.* *.def *.exp \
		   *.geom *.root *.ps *.so .def so_locations
		@rm -rf cxx_repository

###

.$(SrcSuf).$(ObjSuf):
		$(CXX) $(CXXFLAGS) -c $<
