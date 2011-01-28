# Module.mk for xrootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G Ganis, 27/7/2004

MODNAME    := xrootd
MODDIR     := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS    := $(MODDIR)/src

XROOTDDIR  := $(MODDIR)
XROOTDDIRS := $(MODDIRS)
XROOTDDIRD := $(call stripsrc,$(MODDIRS)/xrootd)
XROOTDDIRI := $(XROOTDDIRD)/src
XROOTDDIRL := $(XROOTDDIRD)/lib
XROOTDMAKE := $(XROOTDDIRD)/GNUmakefile
XROOTDBUILD:= $(XROOTDDIRD)/LastBuild.d

##### Xrootd config options #####
ifeq ($(PLATFORM),win32)
ifeq (yes,$(WINRTDEBUG))
XRDDBG      = "Win32 Debug"
else
XRDDBG      = "Win32 Release"
endif
else
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
XRDDBG      = "--build=debug"
else
XRDDBG      =
endif
endif
ifeq ($(PLATFORM),macosx)
XRDSOEXT    = so
else
XRDSOEXT    = $(SOEXT)
endif
ifeq ($(PLATFORM),win32)
XRDSOEXT    = lib
endif

##### Xrootd executables #####
ifneq ($(PLATFORM),win32)
XRDEXEC     = xrootd olbd xrdcp xrd xrdpwdadmin cmsd xrdstagetool xprep
ifneq ($(BUILDXRDGSI),)
XRDEXEC    += xrdgsiproxy
endif
else
XRDEXEC     = xrdcp.exe
endif
XRDEXECS   := $(patsubst %,bin/%,$(XRDEXEC))
XRDEXECSA  := $(patsubst %,$(XROOTDDIRD)/bin/%,$(XRDEXEC))

##### Xrootd plugins #####
ifeq ($(PLATFORM),win32)
XRDPLUGINSA := $(XROOTDDIRL)/libXrdClient.$(XRDSOEXT)
XRDPLUGINS  := $(XRDPLUGINSA)
XRDLIBS     := $(XRDPLUGINS)
else
XRDLIBS     := $(XROOTDDIRL)/libXrdOuc.a $(XROOTDDIRL)/libXrdNet.a $(XROOTDDIRL)/libXrdNetUtil.a \
               $(XROOTDDIRL)/libXrdSys.a \
               $(LPATH)/libXrdClient.$(XRDSOEXT) $(LPATH)/libXrdSut.$(XRDSOEXT)
XRDNETXD    := $(XROOTDDIRL)/libXrdOuc.a $(XROOTDDIRL)/libXrdSys.a \
               $(LPATH)/libXrdClient.$(XRDSOEXT)
XRDPROOFXD  := $(XRDLIBS) $(XROOTDDIRL)/libXrd.a
ifeq ($(ARCH),win32gcc)
XRDLIBS     := $(patsubst $(LPATH)/%.$(XRDSOEXT),bin/%.$(XRDSOEXT),$(XRDLIBS))
endif
endif

##### Xrootd headers used by netx, proofx#####
XRDHDRS    := $(wildcard $(XROOTDDIRI)/Xrd/*.hh) $(wildcard $(XROOTDDIRI)/XrdClient/*.hh) \
              $(wildcard $(XROOTDDIRI)/XrdNet/*.hh) $(wildcard $(XROOTDDIRI)/XrdOuc/*.hh) \
              $(wildcard $(XROOTDDIRI)/XrdSec/*.hh) $(wildcard $(XROOTDDIRI)/XrdSut/*.hh) \
              $(wildcard $(XROOTDDIRI)/XrdSys/*.hh)

##### Xrootd headers, sources, config dependences #####
XROOTDDEPS := $(wildcard $(XROOTDDIRI)/*/*.hh) $(wildcard $(XROOTDDIRI)/*/*.cc) \
              $(wildcard $(XROOTDDIRI)/*/*.h) $(wildcard $(XROOTDDIRI)/*/*.c) \
              $(wildcard $(XROOTDDIRI)/*/*/*.h) $(wildcard $(XROOTDDIRI)/*/*/*.c)
XROOTDCFGD := $(wildcard $(XROOTDDIRS)/xrootd/config/*) \
              $(wildcard $(XROOTDDIRS)/xrootd/config/test/*) \
              $(XROOTDDIRS)/xrootd/configure.classic

# used in the main Makefile
ALLLIBS    += $(XRDLIBS)
ifeq ($(PLATFORM),win32)
ALLEXECS   += $(XRDEXECS)
endif

# Local targets
TARGETS    := $(XRDLIBS)
ifeq ($(PLATFORM),win32)
TARGETS    += $(XRDEXECS)
endif

# Make sure that PWD is defined (it may not be for example when running 'make' via 'sudo')
ifeq ($(PWD),)
PWD := $(shell pwd)
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

ifneq ($(PLATFORM),win32)
$(XROOTDMAKE): $(XROOTDCFGD)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		$(MAKEDIR)
		@$(RSYNC) --exclude '.svn' --exclude '*.o' --exclude '*.a' $(XROOTDDIRS)/xrootd  $(call stripsrc,$(XROOTDDIRS))
		@rm -rf $(XROOTDDIRL) $(XROOTDDIRD)/bin
		@rm -f $(XROOTDMAKE)
endif
		@(cd $(XROOTDDIRD); \
		RELE=`uname -r`; \
		CHIP=`uname -m | tr '[A-Z]' '[a-z]'`; \
		PROC=`uname -p`; \
		xarch="" ; \
		case "$(ARCH):$$RELE:$$CHIP:$$PROC" in \
		freebsd*:*)      xopt="--ccflavour=gcc";; \
		linuxicc:*)      xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxia64ecc:*)  xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxia64gcc:*)  xopt="--ccflavour=gccia64 --use-xrd-strlcpy";; \
		linuxx8664gcc:*) xopt="--ccflavour=gccx8664 --use-xrd-strlcpy";; \
		linuxx8664icc:*) xopt="--ccflavour=iccx8664 --use-xrd-strlcpy";; \
		linuxppc64gcc:*) xopt="--ccflavour=gccppc64 --use-xrd-strlcpy";; \
                linuxalphagcc:*) xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		linux*:*)        xarch="i386_linux"; xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		macosx64:*)      xopt="--ccflavour=macos64";; \
		macosxicc:*)     xopt="--ccflavour=icc";; \
		macosx*:*)       xopt="--ccflavour=macos";; \
		solaris64*:*:i86pc:*) xopt="--ccflavour=sunCCamd64 --use-xrd-strlcpy";; \
                solaris*:5.11:i86pc:*) xopt="--ccflavour=sunCCi86pc --use-xrd-strlcpy";; \
                solaris*:5.1*:i86pc:*) xopt="--use-xrd-strlcpy";; \
                solaris*:*:i86pc:*) xopt="--ccflavour=sunCCi86pc --use-xrd-strlcpy";; \
		solaris*:5.8)    xopt="--ccflavour=sunCC";; \
		solaris*:5.9)    xopt="--ccflavour=sunCC";; \
		solaris*:*)      xopt="--ccflavour=sunCC --use-xrd-strlcpy";; \
		win32gcc:*)      xopt="win32gcc";; \
		*)               xopt="";; \
		esac; \
                if [ ! "x$(KRB5LIBDIR)" = "x" ] ; then \
                   xlib=`echo $(KRB5LIBDIR) | cut -c3-`; \
                   xopt="$$xopt --with-krb5-libdir=$$xlib"; \
                elif [ ! "x$(KRB5LIB)" = "x" ] ; then \
                   xlibs=`echo $(KRB5LIB)`; \
                   for l in $$xlibs; do \
                      if [ ! "x$$l" = "x-lkrb5" ] && [ ! "x$$l" = "x-lk5crypto" ]  ; then \
                         xlib=`dirname $$l`; \
                         xopt="$$xopt --with-krb5-libdir=$$xlib"; \
                         break; \
                      fi; \
                   done; \
                fi; \
                if [ ! "x$(KRB5INCDIR)" = "x" ] ; then \
                   xinc=`echo $(KRB5INCDIR)`; \
                   xopt="$$xopt --with-krb5-incdir=$$xinc"; \
                fi; \
                if [ "x$(BUILDKRB5)" = "xno" ] ; then \
                   xopt="$$xopt --disable-krb5"; \
                fi; \
		if [ "x$(BUILDXRDGSI)" = "x" ] ; then \
		   xopt="$$xopt --disable-gsi"; \
		fi; \
		if [ ! "x$(BUILDBONJOUR)" = "x" ] ; then \
		   xopt="$$xopt --enable-bonjour"; \
		fi; \
		if [ ! "x$(SSLLIBDIR)" = "x" ] ; then \
		   xlib=`echo $(SSLLIBDIR) | cut -c3-`; \
		   xopt="$$xopt --with-ssl-libdir=$$xlib"; \
		elif [ ! "x$(SSLLIB)" = "x" ] ; then \
		   xlibs=`echo $(SSLLIB)`; \
		   for l in $$xlibs; do \
   		      if [ ! "x$$l" = "x-lssl" ] && [ ! "x$$l" = "x-lcrypto" ]  ; then \
		         xlib=`dirname $$l`; \
  		         xopt="$$xopt --with-ssl-libdir=$$xlib"; \
      		         break; \
     		      fi; \
   		   done; \
		fi; \
		if [ ! "x$(SSLINCDIR)" = "x" ] ; then \
		   xinc=`echo $(SSLINCDIR)`; \
		   xopt="$$xopt --with-ssl-incdir=$$xinc"; \
		fi; \
		if [ ! "x$(SSLSHARED)" = "x" ] ; then \
		   xsha=`echo $(SSLSHARED)`; \
		   xopt="$$xopt --with-ssl-shared=$$xsha"; \
		fi; \
		if [ ! "x$(SHADOWFLAGS)" = "x" ] ; then \
		   xopt="$$xopt --enable-shadowpw"; \
		fi; \
		if [ ! "x$(AFSLIBDIR)" = "x" ] ; then \
		   xlib=`echo $(AFSLIBDIR) | cut -c3-`; \
		   xopt="$$xopt --with-afs-libdir=$$xlib"; \
		elif [ ! "x$(AFSLIB)" = "x" ] ; then \
		   xlibs=`echo $(AFSLIB)`; \
		   for l in $$xlibs; do \
   		      if [ ! "x$$l" = "x-lafsrpc" ] && [ ! "x$$l" = "x-lafsauthent" ]  ; then \
		         xlib=`dirname $$l`; \
  		         xopt="$$xopt --with-afs-libdir=$$xlib"; \
      		         break; \
     		      fi; \
   		   done; \
		fi; \
		if [ ! "x$(AFSINCDIR)" = "x" ] ; then \
		   xinc=`echo $(AFSINCDIR) | cut -c3-`; \
		   xopt="$$xopt --with-afs-incdir=$$xinc"; \
		fi; \
		if [ ! "x$(AFSSHARED)" = "x" ] ; then \
		   xsha=`echo $(AFSSHARED)`; \
		   xopt="$$xopt --with-afs-shared=$$xsha"; \
		fi; \
		if [ ! "x$(XRDADDOPTS)" = "x" ] ; then \
		   xaddopts=`echo $(XRDADDOPTS)`; \
		   xopt="$$xopt $$xaddopts"; \
		fi; \
		xopt="$$xopt --disable-krb4 --enable-echo --no-arch-subdirs --disable-mon --with-cxx=$(CXX) --with-ld=$(LD)"; \
		echo "Options to Xrootd-configure: $$xarch $$xopt $(XRDDBG)"; \
		GNUMAKE=$(MAKE) ./configure.classic $$xarch $$xopt $(XRDDBG); \
		rc=$$? ; \
		if [ $$rc != "0" ] ; then \
		   echo "*** Error condition reported by Xrootd-configure (rc = $$rc):"; \
		   rm -f $(XROOTDMAKE); \
	 	   exit 1; \
		fi)
else
$(XROOTDMAKE):
		$(MAKEDIR)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(XROOTDDIRS)/xrootd  $(call stripsrc,$(XROOTDDIRS))
		@(find $(XROOTDDIRD) -name "*.o" -exec rm -f {} \; >/dev/null 2>&1;true)
		@(find $(XROOTDDIRD) -name .svn -exec rm -rf {} \; >/dev/null 2>&1;true)
		@rm -rf $(XROOTDDIRL) $(XROOTDDIRD)/bin
		@rm -f $(XROOTDMAKE)
endif
		@(if [ -d $(XROOTDDIRD)/pthreads-win32 ]; then \
    		   cp $(XROOTDDIRD)/pthreads-win32/lib/*.dll "bin" ; \
		   cp $(XROOTDDIRD)/pthreads-win32/lib/*.lib "lib" ; \
		   cp $(XROOTDDIRD)/pthreads-win32/include/*.h "include" ; \
		fi)
		@touch $(XROOTDMAKE)
endif

$(XRDEXECS): $(XRDEXECSA)
ifeq ($(PLATFORM),win32)
		@(echo "Copying xrootd executables ..." ; \
		cp $(XROOTDDIRD)/bin/*.exe "bin" ;)
endif

$(XROOTDBUILD): $(XROOTDMAKE) $(XROOTDDEPS)
ifneq ($(PLATFORM),win32)
		@(topdir=$(PWD); \
		cd $(XROOTDDIRD); \
	   	echo "*** Building xrootd ... topdir= $$topdir" ; \
		$(MAKE); \
		rc=$$? ; \
		if [ $$rc != "0" ] ; then \
		   echo "*** Error condition reported by make (rc = $$rc):"; \
		   rm -f $(XROOTDMAKE); \
	 	   exit 1; \
      fi; \
		cd $$topdir ; \
		if [ -d $(XROOTDDIRL) ]; then \
		   lsplug=`find $(XROOTDDIRL) -name "libXrd*.$(XRDSOEXT)"` ;\
		   lsplug="$$lsplug `find $(XROOTDDIRL) -name "libXrd*.dylib"`" ;\
		   for i in $$lsplug ; do \
		      echo "Copying $$i ..." ; \
		      if [ "x$(ARCH)" = "xwin32gcc" ] ; then \
		         cp $$i bin ; \
		         lname=`basename $$i` ; \
		         ln -sf bin/$$lname $(LPATH)/$$lname ; \
		         ln -sf bin/$$lname "$(LPATH)/$$lname.a" ; \
		      else \
		         if [ "x$(PLATFORM)" = "xmacosx" ] ; then \
		            lname=`basename $$i` ; \
		            install_name_tool -id $(LIBDIR)/$$lname $$i ; \
		         fi ; \
		         cp -p $$i $(LPATH)/ ; \
		      fi ; \
		   done ; \
		fi ; \
		for i in $(XRDEXEC); do \
		   fopt="" ; \
		   if [ -f bin/$$i ] ; then \
		      fopt="-newer bin/$$i" ; \
		   fi ; \
		   bexe=`find $(XROOTDDIRD)/bin $$fopt -name $$i 2>/dev/null` ; \
		   if test "x$$bexe" != "x" ; then \
		      echo "Copying $$bexe executables ..." ; \
		      cp -p $$bexe bin/$$i ; \
		   fi ; \
		done ; \
		touch $(XROOTDBUILD))
else
		@(cd $(XROOTDDIRD); \
		echo "*** Building xrootd ..."; \
		unset MAKEFLAGS; \
		nmake -f Makefile.msc CFG=$(XRDDBG))
endif

ifeq ($(PLATFORM),win32)
$(XRDEXECSA): $(XROOTDBUILD)
endif

### Rules for xrootd plugins

$(LPATH)/libXrd%.$(XRDSOEXT): $(XROOTDDIRL)/libXrd%.$(XRDSOEXT)
		$(INSTALL) $< $@
		touch $@

### Rules for single components
$(XROOTDDIRL)/libXrdClient.$(XRDSOEXT): $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdSut.$(XRDSOEXT): $(XROOTDBUILD)
#
$(XROOTDDIRL)/libXrdOuc.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdNet.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdNetUtil.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdSys.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrd.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdClient.a: $(XROOTDBUILD)
$(XROOTDDIRL)/libXrdSut.a: $(XROOTDBUILD)

### General rules

all-$(MODNAME): $(TARGETS)

clean-$(MODNAME):
ifneq ($(PLATFORM),win32)
	 @(if [ -f $(XROOTDMAKE) ]; then \
                   $(MAKE) clean-netx;  \
		   $(MAKE) clean-proofx;  \
		   cd $(XROOTDDIRD); \
		   $(MAKE) clean; \
		fi)
else
		@(if [ -f $(XROOTDMAKE) ]; then \
   		   cd $(XROOTDDIRD); \
		   unset MAKEFLAGS; \
		   nmake -f Makefile.msc clean; \
		fi)
endif
		@rm -f $(XROOTDBUILD)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(XRDEXECS) $(LPATH)/libXrd* bin/libXrd* $(XROOTDBUILD)
ifneq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(XROOTDDIRD)
else
		@(if [ -f $(XROOTDMAKE) ]; then \
		   $(MAKE) distclean-netx; \
		   $(MAKE) distclean-proofx; \
		   cd $(XROOTDDIRD); \
		   $(MAKE) distclean; \
		fi)
endif
else
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(XROOTDDIRD)
else
		@(if [ -f $(XROOTDMAKE) ]; then \
		   cd $(XROOTDDIRD); \
		   unset MAKEFLAGS; \
		   nmake -f Makefile.msc distclean; \
		   rm -f GNUmakefile; \
		fi)
endif
endif
		@rm -f $(XROOTDMAKE)

distclean::     distclean-$(MODNAME)
