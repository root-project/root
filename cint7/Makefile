#
# MAIN CINT MAKEFILE
#
# See help target for details
#
##############################################################
#
# Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
#
# For the licensing terms see the file COPYING
#
##############################################################

include Makefile.conf

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
CINTLIBSHARED = libcint$(G__CFG_SOEXT)
CINTIMPLIB    = libcint$(G__CFG_IMPLIBEXT)
CINTLIBSTATIC = libcint_static$(G__CFG_LIBEXT)
CINT          = cint$(G__CFG_EXEEXT)
RMKDEPEND     = tool/rmkdepend/rmkdepend$(G__CFG_EXEEXT)
CONFIGCINTH   = inc/configcint.h

IPATH         = $(G__CFG_INCP)inc $(G__CFG_INCP)src $(REFLEXIPATH)
CXXFLAGS      = $(G__CFG_CXXFLAGS) $(G__CFG_CXXMACROS) $(IPATH) -DG__CINTBODY -DREFLEX_CINT_MERGE
CFLAGS        = $(G__CFG_CFLAGS)   $(G__CFG_CMACROS)   $(IPATH) -DG__CINTBODY

LINKSTATIC = no
#ifneq ($(subst msvc,,$(G__CFG_ARCH)),$(G__CFG_ARCH))
#  LINKSTATIC = no
#endif

.PHONY: test clean help

include Cint.mk
include libReflex.mk
include libCint.mk

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

cintreflextest: REFLEXTESTFLAGS = -m -r

debugtest: TESTDEBUG = -DDEBUG

test debugtest cintreflextest:
	(cd test && \
	PATH=..:.:$$PATH \
	LD_LIBRARY_PATH=..:.:$$LD_LIBRARY_PATH \
	DYLD_LIBRARY_PATH=..:.:$$DYLD_LIBRARY_PATH\
	cint$(G__CFG_EXEEXT) $(TESTDEBUG) testall.cxx $(TEST) $(REFLEXTESTFLAGS) )

ifeq ($(findstring clean,$(MAKECMDGOALS)),)
Makefile: $(RMKDEPEND)

Makefile.conf $(CONFIGCINTH): configure
	@echo "Run ./configure first!"
	@./configure -h
	@exit 1
endif

MAKEINFO:  configure $(CONFIGCINTH)
	echo NOT YET IMPLEMENTED.

help:
	@echo CINT Makefile targets:
	@echo '  make:			Build default targets (marked by * below)'
	@echo '  make cint'$(G__CFG_EXEEXT)':	Build CINT executable [*]'
	@echo '  make static:	Build static cint library [*]'
	@echo '  make shared:	Build cint library'
	@echo '  make makecint'$(G__CFG_EXEEXT)':	Build CINT Makefile generator [*]'
	@echo '  make dlls:		Build STL dlls (you should do this at least once)'
	@echo '  make reflex:           Build the Reflex shared library'
	@echo '  make MAKEINFO:	Write (deprecated) MAKEINFO file'
	@echo '  make test:		Run test suite (TEST=some.cxx to run only some.cxx)'
	@echo '  make debugtest:	Run test suite in debug mode (TEST=some.cxx to run only some.cxx)'
	@echo '  make clean:		Remove all files created during a build'
	@echo '  make help:		Get list of CINT Makefile targets'
