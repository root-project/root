#
# MAIN CINT MAKEFILE
#
# See help target for details
#
##############################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
#
# For the licensing terms see the file COPYING
#
##############################################################

CINTSYSDIR := ${PWD}
export CINTSYSDIR
__CINT_BUILDDIR :=  ${PWD}
export __CINT_BUILDDIR

include Makefile.conf

CINTSYSDIR := $(shell $(G__CFG_MANGLEPATHS) "${CINTSYSDIR}" )
__CINT_BUILDDIR := $(shell $(G__CFG_MANGLEPATHS) "${__CINT_BUILDDIR}" )

# set up order target if available
MAKE_VERSION_MAJOR := $(word 1,$(subst ., ,$(MAKE_VERSION)))
MAKE_VERSION_MINOR := $(shell echo $(word 2,$(subst ., ,$(MAKE_VERSION))) | \
                              sed 's/\([0-9][0-9]*\).*/\1/')
MAKE_VERSION_MAJOR ?= 0
MAKE_VERSION_MINOR ?= 0
ORDER_ := $(shell test $(MAKE_VERSION_MAJOR) -gt 3 || \
                  test $(MAKE_VERSION_MAJOR) -eq 3 && \
                  test $(MAKE_VERSION_MINOR) -ge 80 && echo '|')

# libraries
ifeq ($(subst msvc,,$(G__CFG_ARCH)), $(G__CFG_ARCH))
CINTLIBSHARED = lib/libCint$(G__CFG_SOEXT)
CINTLIBINSTDIR = $(G__CFG_LIBDIR)
else
CINTLIBSHARED = bin/libCint$(G__CFG_SOEXT)
CINTLIBINSTDIR = $(G__CFG_BINDIR)
endif
CINTIMPLIB    = lib/libCint$(G__CFG_IMPLIBEXT)
CINTLIBSTATIC = lib/libCint_static$(G__CFG_LIBEXT)
CINT          = bin/cint$(G__CFG_EXEEXT)
RMKDEPEND     = tool/rmkdepend/rmkdepend$(G__CFG_EXEEXT)
CONFIGCINTH   = $(G__CFG_COREVERSION)/inc/configcint.h

IPATH         = $(G__CFG_INCP)$(G__CFG_COREVERSION)/inc $(G__CFG_INCP)$(G__CFG_COREVERSION)/src $(REFLEXIPATH)
CXXFLAGS      = $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) $(IPATH) -DG__CINTBODY -DREFLEX_CINT_MERGE -DInternal=I
CFLAGS        = $(G__CFG_CFLAGS)   $(G__CFG_CMACROS)   $(IPATH) -DG__CINTBODY

LINKSTATIC = no
#ifneq ($(subst msvc,,$(G__CFG_ARCH)),$(G__CFG_ARCH))
#  LINKSTATIC = no
#endif

.PHONY: test clean help distclean install uninstall

include build/Cint.mk
include build/libReflex.mk
include build/libCint.mk

Makefile: build/Makefile
	cp -f $< $@

############################################################################
# Common rules
############################################################################

%$(G__CFG_OBJEXT): %.cxx
	$(RMKDEPEND) -R -f$*.d -Y -w 1000 -- $(CXXFLAGS) -D__cplusplus -- $<
	$(G__CFG_CXX) $(CXXFLAGS) $(G__CFG_COMP) $< $(G__CFG_COUT)$@

%$(G__CFG_OBJEXT): %.c
	$(RMKDEPEND) -R -f$*.d -Y -w 1000 -- $(CFLAGS) -- $<
	$(G__CFG_CC) $(CFLAGS) $(G__CFG_COMP) $< $(G__CFG_COUT)$@

%$(G__CFG_EXEEXT): %$(G__CFG_OBJEXT)
	$(G__CFG_LD) $(G__CFG_LDOUT)$@ $<

ifeq ($(findstring clean,$(MAKECMDGOALS)),)
-include $(ALLDEPO:$(G__CFG_OBJEXT)=.d)
endif

##############################################################
# Cleanup
##############################################################
clean :: clean-test
	-$(G__CFG_RM) MAKEINFO || true

clean-test:
	-[ $(G__CFG_OBJEXT) ] && $(G__CFG_RM) test/*$(G__CFG_OBJEXT) || true
	-[ $(G__CFG_EXEEXT) ] && $(G__CFG_RM) test/*$(G__CFG_EXEEXT) || true
	-[ $(G__CFG_SOEXT) ] && $(G__CFG_RM) test/*$(G__CFG_SOEXT) || true
	-[ $(G__CFG_LIBEXT) ] && $(G__CFG_RM) test/*$(G__CFG_LIBEXT) || true
	-[ $(G__CFG_IMPLIBEXT) ] && $(G__CFG_RM) test/*$(G__CFG_IMPLIBEXT) || true
	-$(G__CFG_RM) test/test.dll test/Makefile test/compiled test/interpreted \
	  test/*.manifest test/*.ilk || true
	-$(G__CFG_RM) test/Makefile || true
	-$(G__CFG_RM) test/Makefile || true

distclean :: clean
	rm -f bin/cint bin/makecint bin/cint-config bin/cint_tmp
ifneq ($(G__CFG_OBJEXT),)
	find . -name \*$(G__CFG_OBJEXT) -exec rm {} \; || true
endif
ifneq ($(G__CFG_LIBEXT),)
	find . -name \*$(G__CFG_LIBEXT) -exec rm {} \; || true
endif
	find . -name \*.d -exec rm {} \; || true
	find . -name \*.dll -exec rm {} \; || true
	find . -name \*.manifest -exec rm {} \; || true
	find . -name \*.exp -exec rm {} \; || true
	find . -name \*.pdb -exec rm {} \; || true

cintreflextest: REFLEXTESTFLAGS = -m -r

debugtest: TESTDEBUG = -DDEBUG

ifeq ($(G__CFG_COREVERSION),cint)
  TESTFLAGS ?= --ignore-diff-errors -k --hide-known-defects
else
  TESTFLAGS ?= -k --hide-known-defects
endif

test debugtest cintreflextest:
	(cd test && \
	PATH=$$PWD/../bin:.:$$PATH \
	LD_LIBRARY_PATH=$$PWD/../lib:.:$$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=$$PWD/../lib:.:$$DYLD_LIBRARY_PATH \
	cint$(G__CFG_EXEEXT) $(TESTDEBUG) testall.cxx $(TEST) $(TESTFLAGS) $(REFLEXTESTFLAGS) )

ifeq ($(findstring clean,$(MAKECMDGOALS)),)
Makefile: $(RMKDEPEND)

Makefile.conf $(CONFIGCINTH): configure
	@[ -x reconfigure ] && ./reconfigure || ( \
	echo "Run ./configure first!" ; \
	./configure -h ; \
	exit 1 )
endif

MAKEINFO:  configure $(CONFIGCINTH)
	echo NOT YET IMPLEMENTED.

help:
	@echo CINT Makefile targets:
	@echo '  make:			Build default targets (marked by * below)'
ifeq ($(G__CFG_WITHPREFIX),1)
	@echo '  make install:		Install CINT into directories specified to configure'
endif
	@echo '  make bin/cint'$(G__CFG_EXEEXT)':	Build CINT executable [*]'
	@echo '  make static:		Build static cint library [*]'
	@echo '  make shared:		Build cint library'
	@echo '  make bin/makecint'$(G__CFG_EXEEXT)':	Build CINT Makefile generator [*]'
	@echo '  make dlls:		Build STL dlls (you should do this at least once)'
	@echo '  make reflex:           Build the Reflex shared library'
	@echo '  make MAKEINFO:	Write (deprecated) MAKEINFO file'
	@echo '  make test:		Run test suite (TEST=some.cxx to run only some.cxx)'
	@echo '  make debugtest:	Run test suite in debug mode (TEST=some.cxx to run only some.cxx)'
	@echo '  make clean:		Remove all files created during a build'
	@echo '  make distclean:	Remove all files created during a build'
ifeq ($(G__CFG_WITHPREFIX),1)
	@echo '  make uninstall:	Uninstall CINT from directories specified to configure'
endif
	@echo '  make deb:	        Create a CINT debian package in ../'
	@echo '  make debinstall:	Install the debian package generated by the deb target'
	@echo '  make help:		Get list of CINT Makefile targets'

# The install section takes into account the value of the DESTDIR variable
# which can be used to redirect the installation at the 'make install'
# stage, without effecting the hardcoded (during ./configure) paths
# This is useful (or even required) when building packages like .deb or .rpm

FINDALLBUTSVN=-type f \! -path '*/.svn/*' \! -name SConscript \! -name Makefile.am

ifeq ($(G__CFG_WITHPREFIX),1)
install:
	if [ "x$(G__CFG_INCLUDEDIRCINT)" = "x" ]; then \
	  echo 'You must run "./configure --prefix" or "./configure --prefix=..." to be able to call "make $@"!'; \
	  exit 1; \
	fi
	if [ ! -d $(DESTDIR)$(G__CFG_BINDIR) ]; then mkdir -p $(DESTDIR)$(G__CFG_BINDIR); fi
	cp $(CINT)     $(DESTDIR)$(G__CFG_BINDIR)/$(notdir $(CINT))
	cp $(MAKECINT) $(DESTDIR)$(G__CFG_BINDIR)/$(notdir $(MAKECINT))
	if [ -f $(CINT).manifest ]; then \
	   cp bin/*.manifest     $(DESTDIR)$(G__CFG_BINDIR); \
	fi
	if [ -f bin/cint-config ] ; then cp bin/cint-config $(DESTDIR)$(G__CFG_BINDIR)/ ; fi
# install manual pages
	if [ ! -d $(DESTDIR)$(G__CFG_MANDIR)/man1 ] ; then \
	  mkdir -p $(DESTDIR)$(G__CFG_MANDIR)/man1; \
	fi
	cp doc/man1/cint.1 $(DESTDIR)$(G__CFG_MANDIR)/man1/cint.1
	cp doc/man1/makecint.1 $(DESTDIR)$(G__CFG_MANDIR)/man1/makecint.1

	if [ ! -d $(DESTDIR)$(G__CFG_LIBDIR) ]; then mkdir -p $(DESTDIR)$(G__CFG_LIBDIR); fi
	cp $(CINTLIBSHARED) $(DESTDIR)$(CINTLIBINSTDIR)/
	if [ -f $(CINTIMPLIB) ] ;   then cp $(CINTIMPLIB)    $(DESTDIR)$(G__CFG_LIBDIR)/ ; fi
	cp $(CINTLIBSTATIC) $(DESTDIR)$(G__CFG_LIBDIR)/
	if [ -f $(REFLEXLIB) ] ;    then cp $(REFLEXLIB)     $(DESTDIR)$(G__CFG_LIBDIR)/ ; fi
	if [ -f $(REFLEXSO) ] ;     then cp $(REFLEXSO)      $(DESTDIR)$(CINTLIBINSTDIR)/ ; fi
	if [ -f $(REFLEXIMPLIB) ] ; then cp $(REFLEXIMPLIB)  $(DESTDIR)$(G__CFG_LIBDIR)/ ; fi

	if [ ! -d $(DESTDIR)$(G__CFG_INCLUDEDIRCINT) ] ; then mkdir -p $(DESTDIR)$(G__CFG_INCLUDEDIRCINT) ; fi
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_INCLUDEDIRCINT)/*
	cp $(G__CFG_COREVERSION)/inc/*.h $(DESTDIR)$(G__CFG_INCLUDEDIRCINT)/
	for f in `find reflex/inc $(FINDALLBUTSVN)`; do \
	  d=$(DESTDIR)$(G__CFG_INCLUDEDIRCINT)/`dirname $$f | sed 's,^reflex/inc,,'`;\
	  if [ ! -d $$d ]; then mkdir -p $$d; fi; \
	  cp $$f $$d/; done
	if [ ! -d $(DESTDIR)$(G__CFG_DATADIRCINT) ] ; then mkdir -p $(DESTDIR)$(G__CFG_DATADIRCINT) ; fi
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_DATADIRCINT)/*
	for f in `cd $(G__CFG_COREVERSION); find include stl main lib $(FINDALLBUTSVN)`; do \
	  d=$(DESTDIR)$(G__CFG_DATADIRCINT)/$(G__CFG_COREVERSION)/`dirname $$f`;\
	  if [ ! -d $$d ]; then mkdir -p $$d; fi; \
	  cp $(G__CFG_COREVERSION)/$$f $$d/; done

uninstall:
	if [ "x$(G__CFG_INCLUDEDIRCINT)" = "x" ]; then \
	  echo 'You must run "./configure --prefix" or "./configure --prefix=..." to be able to call "make $@"!'; \
	  exit 1; \
	fi
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_BINDIR)/$(CINT) $(DESTDIR)$(G__CFG_BINDIR)/$(MAKECINT) $(DESTDIR)$(G__CFG_BINDIR)/cint-config
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(CINTLIBSHARED)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(CINTIMPLIB)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(CINTLIBSTATIC)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(REFLEXLIB)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(REFLEXSO)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_LIBDIR)/$(REFLEXIMPLIB)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_INCLUDEDIRCINT)
	$(G__CFG_RM) $(DESTDIR)$(G__CFG_DATADIRCINT)

endif # WITHPREFIX


ifeq ($(INCLUDE_DEB_MK),1)
include build/deb.mk
else
deb :
	$(MAKE) INCLUDE_DEB_MK=1 deb
debinstall :
	$(MAKE) INCLUDE_DEB_MK=1 debinstall
endif

ifeq ($(INCLUDE_RPM_MK),1)
include build/rpm.mk
else
rpm :
	$(MAKE) INCLUDE_RPM_MK=1 rpm
rpminstall :
	$(MAKE) INCLUDE_RPM_MK=1 rpminstall
endif

