#
# Cint sub-Makefile
#
##############################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
#
# For the licensing terms see the file COPYING
#
##############################################################

############################################################################
# VARIABLES
############################################################################
MAKECINT      = bin/makecint$(G__CFG_EXEEXT)
SETUPO        = $(G__CFG_COREVERSION)/main/G__setup$(G__CFG_OBJEXT)
MAINO         = $(G__CFG_COREVERSION)/main/cppmain$(G__CFG_OBJEXT)
MAKECINTO     = tool/makecint$(G__CFG_OBJEXT)
ALLDEPO      += $(MAINO) $(SETUPO) $(MAKECINTO)

# don't build qt.cintdll gl.cintdll by default
ALLDLLS  = dll_stl.cintdll stdstrct.cintdll
ifeq ($(subst msvc,,$(G__CFG_ARCH)),$(G__CFG_ARCH))
# not windows
# xlib doesn't work, types.h conflicts
# socket, pthread use MAKEINFO
  ALLDLLS += ipc.cintdll posix.cintdll
endif

BUILDDLL = chmod a+x setup; PATH=../../../bin:../../../lib:$$PATH LD_LIBRARY_PATH=../../../lib:$$LD_LIBRARY_PATH DYLD_LIBRARY_PATH=../../../lib:$$DYLD_LIBRARY_PATH ./setup
MKTYPES  = $(G__CFG_COREVERSION)/lib/posix/mktypes$(G__CFG_EXEEXT)
MAKEINCL = $(G__CFG_COREVERSION)/include/mkincld$(G__CFG_EXEEXT)
IOSENUMH = $(G__CFG_COREVERSION)/include/iosenum.h

##############################################################
# CINT dlls
##############################################################
.PHONY: dlls all $(ALLDLLS) %.cintdll
all: $(CINT) $(ALLDLLS) $(IOSENUM) $(MAKECINT)

dlls: $(ALLDLLS)
ipc.cintdll posix.cintdll: $(G__CFG_COREVERSION)/include/sys/types.h

$(ALLDLLS): %.cintdll: $(MAKECINT) $(CINTLIBSTATIC) $(CINT)
	+(cd $(G__CFG_COREVERSION)/lib/$*; $(BUILDDLL))

$(G__CFG_COREVERSION)/include/systypes.h: $(MKTYPES)
	cd $(dir $(MKTYPES)) \
	&& ./$(notdir $(MKTYPES))

$(G__CFG_COREVERSION)/include/sys/types.h: $(G__CFG_COREVERSION)/include/systypes.h
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
LINKCINTLIB=$(G__CFG_LIBP)lib $(subst @imp@,Cint_static,$(G__CFG_LIBL))
else
$(CINT) : $(CINTLIBSHARED)
LINKCINTLIB=$(G__CFG_LIBP)lib $(subst @imp@,Cint,$(G__CFG_LIBL))
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
	  $(G__CFG_RM) $(G__CFG_COREVERSION)/include/*$(G__CFG_OBJEXT) $(G__CFG_COREVERSION)/main/*$(G__CFG_OBJEXT) \
	  tool/*$(G__CFG_OBJEXT) tool/rmkdepend/*$(G__CFG_OBJEXT)
	$(G__CFG_RM) $(G__CFG_COREVERSION)/include/*.d $(G__CFG_COREVERSION)/main/*.d \
	  tool/*.d tool/rmkdepend/*.d
	$(G__CFG_RM) $(LIBOBJECTS) \
	  $(MAINO) $(MAINO:$(G__CFG_OBJEXT)=.d) \
	  $(SETUPO) $(SETUPO:$(G__CFG_OBJEXT)=.d) \
	  $(MAKECINTO) $(MAKECINTO:$(G__CFG_OBJEXT)=.d) \
	  $(CINT) $(MAKECINT) $(MAKEINCL) \
	  $(G__CFG_COREVERSION)/include/stdio.h $(G__CFG_COREVERSION)/include/iosenum.h \
	  lib/libCint.* bin/libCint.* lib/libCint_static.* done
# NEVER ever remove "core"! It's our src subdir!


##############################################################
# rmkdepend
##############################################################
RMKDEPO  = $(subst .c,$(G__CFG_OBJEXT),$(wildcard $(dir $(RMKDEPEND))/*.c))
RMKDEPO += $(subst .cxx,$(G__CFG_OBJEXT),$(wildcard $(dir $(RMKDEPEND))/*.cxx))

$(RMKDEPO): RMKDEPEND = echo #
$(RMKDEPO): CFLAGS += -DINCLUDEDIR=\"/usr/include\" -DOBJSUFFIX=\"$(G__CFG_OBJEXT)\"
$(RMKDEPEND): $(RMKDEPO)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $^ $(G__CFG_LDOUT)$@
