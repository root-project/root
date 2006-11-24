#
# CAUTION: This Makefile is generated for specific computer
#         platform and user environment. This one may not fit your
#         environment. Execute setup script to generate correct
#         makefile.
#
##############################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
#
# For the licensing terms see the file COPYING
#
##############################################################

include Makefile.conf

# libraries
MAINO	      = main/G__cppmain$(G__CFG_OBJEXT)
CINTLIB       = libcint$(G__CFG_SOEXT)
CINTLIBSTATIC = libcint$(G__CFG_LIBEXT)
CONFIGCINTH   = inc/configcint.h

# don't build qt.cintdll gl.cintdll by default
ALLDLLS = dll_stl.cintdll stdstrct.cintdll
ifeq ($(subst msvc,,$(G__CFG_ARCH)),$(G__CFG_ARCH))
# not windows
# xlib doesn't work, types.h conflicts
# socket, pthread use MAKEINFO
  ALLDLLS += longlong.cintdll ipc.cintdll posix.cintdll
endif

##############################################################
# Generate ios enum value specific to the platform
##############################################################
include/iosenum.h : cint$(G__CFG_EXEEXT) include/stdio.h makecint$(G__CFG_EXEEXT) include/iosenum.cxx
	@(if test -r include/iosenum.h; \
	then \
		touch include/iosenum.h; \
	else \
		(echo Generating iosenum.h. This might take a while...; \
		cd include;$(G__CFG_RM) stdfunc$(G__CFG_SOEXT); cd ..\
		unset VS_UNICODE_OUTPUT; \
		LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:. \
		DYLD_LIBRARY_PATH=$$DYLD_LIBRARY_PATH:. \
		./cint$(G__CFG_EXEEXT) $(G__CFG_INCP)inc iosenum.cxx); \
	fi)
	@echo > done

.PHONY: dlls test clean cleanbin help reflex cintex static

dlls: $(ALLDLLS)
ipc.cintdll posix.cintdll: include/sys/types.h

BUILDDLL=chmod a+x setup; PATH=../..:$$PATH \
  LD_LIBRARY_PATH=../..:$$LD_LIBRARY_PATH \
  DYLD_LIBRARY_PATH=../..:$$DYLD_LIBRARY_PATH \
  ./setup

%.cintdll:
	(cd lib/$*; $(BUILDDLL))

include/sys/types.h: lib/posix/mktypes$(G__CFG_EXEEXT)
	cd lib/posix \
	&& ./mktypes$(G__CFG_EXEEXT) \
	&& cp ../../include/systypes.h ../../include/sys/types.h

lib/posix/mktypes$(G__CFG_EXEEXT): lib/posix/mktypes.c
	$(G__CFG_CC) $(G__CFG_COUT)$@ $<

##############################################################
# Compile makecint executable
##############################################################
makecint$(G__CFG_EXEEXT) : tool/makecint.cxx $(CONFIGCINTH)
	$(G__CFG_CXX) $(G__CFG_CFLAGS) $(G__CFG_INCP)inc $(G__CFG_CMACROS) $(G__CFG_COMP) tool/makecint.cxx $(G__CFG_COUT)tool/makecint$(G__CFG_OBJEXT)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) tool/makecint$(G__CFG_OBJEXT) $(G__CFG_LDOUT)makecint$(G__CFG_EXEEXT) $(G__CFG_DEFAULTLIBS)


##############################################################
# Generate standard header files
##############################################################
include/stdio.h : include/mkincld.c
	$(G__CFG_CC) $(G__CFG_INCP)inc $(G__CFG_CFLAGS) $(G__CFG_CMACROS) $(G__CFG_COMP) include/mkincld.c $(G__CFG_COUT)include/mkincld$(G__CFG_OBJEXT)
	$(G__CFG_LD) $(G__CFG_LDFLAGS) include/mkincld$(G__CFG_OBJEXT) $(G__CFG_LDOUT)include/mkincld$(G__CFG_EXEEXT)
	(cd include;./mkincld$(G__CFG_EXEEXT))

##############################################################
# Link cint executable
##############################################################
cint$(G__CFG_EXEEXT) : $(CINTLIB) main/G__setup$(G__CFG_OBJEXT) $(MAINO) $(G__CFG_READLINELIB) 
	$(G__CFG_LD) $(G__CFG_LDFLAGS) $(G__CFG_LDOUT)cint$(G__CFG_EXEEXT) \
	  $(shell $(G__CFG_MANGLEPATHS) $(MAINO)) \
	  $(G__CFG_LIBP). $(G__CFG_LIBP)src $(subst @imp@,cint,$(G__CFG_LIBL)) \
	  $(G__CFG_READLINELIB) $(G__CFG_CURSESLIB) \
	  main/G__setup$(G__CFG_OBJEXT)

##############################################################
# Compile precompiled library setup function which is blank
##############################################################
main/G__setup$(G__CFG_OBJEXT) : main/G__setup.c inc/G__ci.h
	$(G__CFG_CC) $(G__CFG_CFLAGS) $(G__CFG_CMACROS) $(G__CFG_COUT)G__setup$(G__CFG_OBJEXT) $(G__CFG_COMP) main/G__setup.c
	$(G__CFG_MV) G__setup$(G__CFG_OBJEXT) main/G__setup$(G__CFG_OBJEXT)

##############################################################
# Compile GNU readline library. This is done only when 
# BUILDREADLINE is set
##############################################################
ifneq ($(G__CFG_BUILDREADLINE),)
$(G__CFG_READLINELIB) : readline/*.c readline/*.h
	(cd readline ; make)
endif

##############################################################
# Compile cint interpreter core library
##############################################################
$(CINTLIB) : inc/G__ci.h $(wildcard src/*.c src/*.cxx *.h src/*.h) $(G__CFG_READLINELIB) $(CONFIGCINTH)
	$(MAKE) -C src

##############################################################
# Static version
##############################################################
static: $(CINTLIBSTATIC)
$(CINTLIBSTATIC) :inc/G__ci.h $(wildcard src/*.c src/*.cxx *.h src/*.h) $(G__CFG_READLINELIB) $(CONFIGCINTH)
	$(MAKE) -C src `basename $@`
	cp src/`basename $@` $@



##############################################################
# Compile cint main function. 
##############################################################
main/G__cppmain$(G__CFG_OBJEXT) : main/cppmain.cxx
	$(G__CFG_CXX) \
	  $(G__CFG_COUT)$(shell $(G__CFG_MANGLEPATHS) main/G__cppmain$(G__CFG_OBJEXT)) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) \
           $(G__CFG_COMP) main/cppmain.cxx

main/G__main$(G__CFG_OBJEXT) : main/G__main.c
	$(G__CFG_CC) $(G__CFG_COUT)main/G__main$(G__CFG_OBJEXT)  $(G__CFG_CFLAGS) $(G__CFG__CMACROS) $(G__CFG_COMP) main/G__main.c

ifneq ($(G__CFG_REFLEXLIBDIR),)
REFLEXLIB_OBJ=$(subst .cxx,$(G__CFG_OBJEXT),$(wildcard $(G__CFG_REFLEXLIBDIR:/lib=/src)/*.cxx))
REFLEXLIB=$(G__CFG_REFLEXLIBDIR)/libReflex$(G__CFG_SOEXT)

ifneq ($(G__CFG_EXTRACTSYMBOLS),)
REFLEXLIB_DEF=$(REFLEXLIB:$(G__CFG_SOEXT)=.def)
$(REFLEXLIB_DEF): $(REFLEXLIB_OBJ)
	@echo 'LIBRARY  "LIBREFLEX"' > $@
	@echo 'VERSION  1.0' >> $@
	@echo 'HEAPSIZE 1048576,4096' >> $@
	@echo 'EXPORTS' >> $@
	$(subst @obj@,$(REFLEXLIB_OBJ),$(G__CFG_EXTRACTSYMBOLS))>>$@
endif

reflex: $(REFLEXLIB) 
$(REFLEXLIB): $(REFLEXLIB_OBJ) $(REFLEXLIB_DEF)
	@[ ! -d `dirname $(REFLEXLIB)` ] && mkdir -p `dirname $(REFLEXLIB)` || true
	$(G__CFG_LD) $(subst @so@,$(REFLEXLIB:$(G__CFG_SOEXT)=),$(G__CFG_SOFLAGS)) $(G__CFG_LDOUT)$(REFLEXLIB) $(REFLEXLIB_OBJ)

$(G__CFG_REFLEXLIBDIR:/lib=/src)/%$(G__CFG_OBJEXT): $(G__CFG_REFLEXLIBDIR:/lib=/src)/%.cxx
	$(G__CFG_CXX) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS)\
	$(G__CFG_INCP)$(G__CFG_REFLEXINCDIR) \
	$(G__CFG_COMP) $< $(G__CFG_COUT)$@
endif

ifneq ($(G__CFG_CINTEXLIBDIR),)
CINTEXLIB_SRCDIR=$(subst /lib,/src,$(G__CFG_CINTEXLIBDIR))
CINTEXLIB_OBJ=$(subst .cxx,$(G__CFG_OBJEXT),\
	$(filter-out $(CINTEXLIB_SRCDIR)/ROOTClassEnhancer.cxx,\
	$(wildcard $(CINTEXLIB_SRCDIR)/*.cxx)))
CINTEXLIB=$(G__CFG_CINTEXLIBDIR)/libCintex$(G__CFG_SOEXT)

ifneq ($(G__CFG_EXTRACTSYMBOLS),)
CINTEXLIB_DEF=$(CINTEXLIB:$(G__CFG_SOEXT)=.def)
$(CINTEXLIB_DEF): $(CINTEXLIB_OBJ)
	-mkdir -p `dirname $@`
	@echo 'LIBRARY  "LIBCINTEX"' > $@
	@echo 'VERSION  1.0' >> $@
	@echo 'HEAPSIZE 1048576,4096' >> $@
	@echo 'EXPORTS' >> $@
	$(subst @obj@,$(CINTEXLIB_OBJ),$(G__CFG_EXTRACTSYMBOLS))>>$@
endif

cintex: $(CINTEXLIB) include/iosenum.h
$(CINTEXLIB): $(CINTEXLIB_OBJ) $(CINTEXLIB_DEF) $(REFLEXLIB)
	@[ ! -d `dirname $(CINTEXLIB)` ] && mkdir -p `dirname $(CINTEXLIB)` || true
	$(G__CFG_LD) $(subst @so@,$(CINTEXLIB:$(G__CFG_SOEXT)=),$(G__CFG_SOFLAGS)) \
	$(G__CFG_LDOUT)$(CINTEXLIB) $(CINTEXLIB_OBJ) \
	$(G__CFG_LIBP)src $(G__CFG_LIBP). $(G__CFG_LIBL)cint$(G__CFG_IMPLIBEXT) \
	$(G__CFG_LIBP)$(G__CFG_REFLEXLIBDIR) $(G__CFG_LIBL)Reflex$(G__CFG_IMPLIBEXT)

$(CINTEXLIB_SRCDIR)/%$(G__CFG_OBJEXT): $(CINTEXLIB_SRCDIR)/%.cxx
	$(G__CFG_CXX) $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) -DG__CINT \
	$(G__CFG_INCP)$(G__CFG_CINTEXINCDIR) \
	$(G__CFG_INCP)$(G__CFG_REFLEXINCDIR) \
	$(G__CFG_INCP). $(G__CFG_INCP)src \
	$(G__CFG_COMP) $< $(G__CFG_COUT)$@
endif


##############################################################
# Cleanup
##############################################################
clean :
	-[ $(G__CFG_OBJEXT) ] && \
		$(G__CFG_RM) main/*$(G__CFG_OBJEXT) \
		include/*$(G__CFG_OBJEXT) \
		tool/*$(G__CFG_OBJEXT)
	$(G__CFG_RM) cint$(G__CFG_EXEEXT) done core \
		include/mkincld$(G__CFG_EXEEXT) \
		include/stdio.h makecint$(G__CFG_EXEEXT) \
		include/iosenum.h
	$(MAKE) -C src clean
ifneq ($(G__CFG_BUILDREADLINE),)
	$(MAKE) -C readline clean
endif

cleanbin :
	[ $OBJEXT ] && $(G__CFG_RM) main/*$(G__CFG_OBJEXT) src/*$(G__CFG_OBJEXT) 
	$(G__CFG_RM) $(CINTLIB)
	(cd src; $(MAKE) clean)

ifneq ($(G__CFG_REFLEXLIBDIR),)
REFLEXTESTFLAGS=-m -r
endif
ifneq ($(TESTDEBUG),)
TESTDEBUG1=-DDEBUG
endif

test:
ifneq ($(G__CFG_REFLEXLIBDIR),)
	-cp $(REFLEXLIB) test/
	-cp $(CINTEXLIB) test/
endif
	(cd test && \
	PATH=..:.:$$PATH \
	LD_LIBRARY_PATH=..:.:$$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=..:.:$$DYLD_LIBRARY_PATH \
	cint$(G__CFG_EXEEXT) $(TESTDEBUG1) testall.cxx $(TEST) $(REFLEXTESTFLAGS) )

Makefile.conf $(CONFIGCINTH):
	@echo "Run ./configure first!"
	@./configure -h
	@exit 1

MAKEINFO: $(CONFIGCINTH) configure
	@echo NOT YET IMPLEMENTED.

help:
	@echo CINT Makefile targets:
	@echo '  make:			Build default targets (marked by * below)'
	@echo '  make cint'$(G__CFG_EXEEXT)':	Build CINT executable [*]'
	@echo '  make -C src:	Build cint library [*]'
	@echo '  make static:	Build static cint library'
	@echo '  make include/iosenum.h: Build compiler enum header [*]'
	@echo '  make makecint'$(G__CFG_EXEEXT)':	Build CINT Makefile generator [*]'
	@echo '  make dlls:		Build STL dlls (you should do this at least once)'
	@echo '  make reflex:           Build the Reflex shared library'
	@echo '  make cintex:           Build Cintex shared library'
	@echo '  make MAKEINFO:	Write (deprecated) MAKEINFO file'
	@echo '  make test:		Run test suite (define TEST=some.cxx to run only some.cxx test, define TESTDEBUG to pass -DDEBUG to CINT)'
	@echo '  make clean:		Remove all files created during a build'
	@echo '  make help:		Get list of CINT Makefile targets'
