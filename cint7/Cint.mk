#
# Cint sub-Makefile
#
##############################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
#
# For the licensing terms see the file COPYING
#
##############################################################

############################################################################
# VARIABLES
############################################################################
MAKECINT      = makecint$(G__CFG_EXEEXT)
SETUPO        = main/G__setup$(G__CFG_OBJEXT)
MAINO         = main/cppmain$(G__CFG_OBJEXT)
MAKECINTO     = tool/makecint$(G__CFG_OBJEXT)
ALLDEPO      += $(MAINO) $(SETUPO) $(MAKECINTO)

# don't build qt.cintdll gl.cintdll by default
ALLDLLS  = dll_stl.cintdll stdstrct.cintdll
ifeq ($(subst msvc,,$(G__CFG_ARCH)),$(G__CFG_ARCH))
# not windows
# xlib doesn't work, types.h conflicts
# socket, pthread use MAKEINFO
  ALLDLLS += longlong.cintdll ipc.cintdll posix.cintdll
endif

BUILDDLL = chmod a+x setup; PATH=../..:$$PATH LD_LIBRARY_PATH=../..:$$LD_LIBRARY_PATH ./setup
MKTYPES  = lib/posix/mktypes$(G__CFG_EXEEXT)
MAKEINCL = include/mkincld$(G__CFG_EXEEXT)
IOSENUMH = include/iosenum.h

##############################################################
# CINT dlls
##############################################################
.PHONY: dlls all
all: $(CINT) $(ALLDLLS) $(IOSENUM) $(MAKECINT)

dlls: $(ALLDLLS)
ipc.cintdll posix.cintdll: include/sys/types.h

%.cintdll: $(MAKECINT) $(CINTLIBSTATIC) $(CINT)
	(cd lib/$*; $(BUILDDLL))

include/systypes.h: $(MKTYPES)
	cd $(dir $(MKTYPES)) \
	&& ./$(notdir $(MKTYPES))

include/sys/types.h: include/systypes.h
	cp $< $@

##############################################################
# Compile makecint executable
##############################################################
$(MAKECINT): $(MAKECINTO)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $< $(G__CFG_LDOUT)$@ $(G__CFG_DEFAULTLIBS)

##############################################################
# Link cint executable
##############################################################
ifeq ($(LINKSTATIC),yes)
$(CINT) : $(CINTLIBSTATIC)
LINKCINTLIB=$(G__CFG_LIBP). $(subst @imp@,cint_static,$(G__CFG_LIBL))
else
$(CINT) : $(CINTLIBSHARED)
LINKCINTLIB=$(G__CFG_LIBP). $(subst @imp@,cint,$(G__CFG_LIBL))
endif

$(CINT): $(SETUPO) $(MAINO) $(G__CFG_READLINELIB) $(REFLEXLIBDEP)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $(G__CFG_LDOUT)$@ \
	  $(SETUPO) $(MAINO) $(LINKCINTLIB) $(REFLEXLINK) \
	  $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB) $(G__CFG_DEFAULTLIBS)

##############################################################
# Clean
##############################################################
clean::
	-[ $(G__CFG_OBJEXT) ] && \
	  $(G__CFG_RM) include/*$(G__CFG_OBJEXT) main/*$(G__CFG_OBJEXT) \
	  tool/*$(G__CFG_OBJEXT) tool/rmkdepend/*$(G__CFG_OBJEXT)
	$(G__CFG_RM) include/*.d main/*.d \
	  tool/*.d tool/rmkdepend/*.d
	$(G__CFG_RM) $(LIBOBJECTS) \
	  $(MAINO) $(MAINO:$(G__CFG_OBJEXT)=.d) \
	  $(SETUPO) $(SETUPO:$(G__CFG_OBJEXT)=.d) \
	  $(MAKECINTO) $(MAKECINTO:$(G__CFG_OBJEXT)=.d) \
	  $(CINT) $(MAKECINT) $(MAKEINCL) \
	  include/stdio.h include/iosenum.h \
	  libcint.* libcint_static.* done core


##############################################################
# Compile GNU readline library. This is done only when 
# BUILDREADLINE is set
##############################################################

ifneq ($(G__CFG_BUILDREADLINE),)
$(G__CFG_READLINELIB) : readline/*.c readline/*.h
	(cd readline ; make)
clean::
	$(MAKE) -C readline clean
endif

##############################################################
# rmkdepend
##############################################################
RMKDEPO  = $(subst .c,$(G__CFG_OBJEXT),$(wildcard $(dir $(RMKDEPEND))/*.c))
RMKDEPO += $(subst .cxx,$(G__CFG_OBJEXT),$(wildcard $(dir $(RMKDEPEND))/*.cxx))

$(RMKDEPO): RMKDEPEND = echo #
$(RMKDEPO): CFLAGS += -DINCLUDEDIR=\"/usr/include\" -DOBJSUFFIX=\"$(G__CFG_OBJEXT)\"
$(RMKDEPEND): $(RMKDEPO)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $^ $(G__CFG_LDOUT)$@
