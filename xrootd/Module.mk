# Module.mk for xrootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G Ganis, 27/7/2004

MODDIR     := xrootd
MODDIRS    := $(MODDIR)/src

XROOTDVERS := xrootd-20060928-1600
XROOTDDIR  := $(MODDIR)
XROOTDDIRS := $(MODDIRS)
XROOTDDIRD := $(MODDIRS)/xrootd
XROOTDDIRI := $(MODDIRS)/xrootd/src
XROOTDDIRL := $(MODDIRS)/xrootd/lib
XROOTDSRCS := $(MODDIRS)/$(XROOTDVERS).src.tgz
XROOTDETAG := $(MODDIRS)/headers.d

##### Xrootd config options #####
ifeq ($(PLATFORM),win32)
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
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
XRDEXEC     = xrootd olbd xrdcp xrd xrdpwdadmin
ifneq ($(BUILDXRDGSI),)
XRDEXEC    += xrdgsiproxy
endif
else
XRDEXEC     = xrdcp.exe
endif
XRDEXECS   := $(patsubst %,bin/%,$(XRDEXEC))

##### Xrootd plugins #####
ifeq ($(PLATFORM),win32)
XRDPLUGINSA:= $(XROOTDDIRL)/libXrdClient.$(XRDSOEXT)
XRDPLUGINS := $(XRDPLUGINSA)
else
XRDPLUGINSA:= $(XROOTDDIRL)/libXrdSec.$(XRDSOEXT)
XRDPLUGINS := $(LPATH)/libXrdSec.$(XRDSOEXT)
ifeq ($(ARCH),win32gcc)
XRDPLUGINS := $(patsubst $(LPATH)/%.$(XRDSOEXT),bin/%.$(XRDSOEXT),$(XRDPLUGINS))
endif
endif

# used in the main Makefile
ALLLIBS    += $(XRDPLUGINS)
ALLEXECS   += $(XRDEXECS)

##### local rules #####
$(XROOTDETAG): $(XROOTDSRCS)
		@(if [ -d $(XROOTDDIRD) ]; then \
		   rm -rf $(XROOTDDIRD); \
		fi; \
		cd $(XROOTDDIRS); \
		if [ ! -d xrootd ]; then \
		   echo "*** Extracting xrootd source ..."; \
		   gunzip -c $(XROOTDVERS).src.tgz | tar xf -; \
		   etag=`basename $(XROOTDETAG)` ; \
		   touch $$etag ; \
		fi)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(XROOTDDIRD)/pthreads-win32 ]; then \
		    cp $(XROOTDDIRD)/pthreads-win32/lib/*.dll "bin" ; \
		    cp $(XROOTDDIRD)/pthreads-win32/lib/*.lib "lib" ; \
		    cp $(XROOTDDIRD)/pthreads-win32/include/*.h "include" ; \
		  fi)
endif

ifneq ($(PLATFORM),win32)
$(XRDPLUGINS): $(XRDPLUGINSA)
		@(if [ -d $(XROOTDDIRL) ]; then \
		    lsplug=`find $(XROOTDDIRL) -name "libXrd*.$(XRDSOEXT)"` ; \
		    for i in $$lsplug ; do \
		       echo "Copying $$i ..." ; \
		       if [ "x$(ARCH)" = "xwin32gcc" ] ; then \
		          cp $$i bin ; \
		          lname=`basename $$i` ; \
		          ln -sf bin/$$lname $(LPATH)/$$lname ; \
		          ln -sf bin/$$lname "$(LPATH)/$$lname.a" ; \
		       else \
		          cp $$i $(LPATH)/ ; \
		       fi; \
		    done ; \
		  fi)
endif

ifneq ($(PLATFORM),win32)
$(XRDEXECS): $(XRDPLUGINSA)
		@(for i in $(XRDEXEC); do \
		     fopt="" ; \
		     if [ -f bin/$$i ] ; then \
		        fopt="-newer bin/$$i" ; \
		     fi ; \
		     bexe=`find $(XROOTDDIRD)/bin $$fopt -name $$i 2>/dev/null` ; \
		     if test "x$$bexe" != "x" ; then \
		        echo "Copying $$bexe executables ..." ; \
		        cp $$bexe bin/$$i ; \
		     fi ; \
		  done)
else
$(XRDEXECS): $(XRDPLUGINSA)
		@(echo "Copying xrootd executables ..." ; \
		cp $(XROOTDDIRD)/bin/*.exe "bin" ;)
endif

ifneq ($(PLATFORM),win32)
$(XRDPLUGINSA): $(XROOTDETAG)
		@(cd $(XROOTDDIRS); \
		RELE=`uname -r`; \
		CHIP=`uname -m | tr '[A-Z]' '[a-z]'`; \
		PROC=`uname -p`; \
		case "$(ARCH):$$RELE:$$CHIP:$$PROC" in \
		freebsd*:*)      xopt="--ccflavour=gcc";; \
		linuxicc:*)      xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxia64ecc:*)  xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxx8664gcc:*) xopt="--ccflavour=gccx8664 --use-xrd-strlcpy";; \
		linuxx8664icc:*) xopt="--ccflavour=iccx8664 --use-xrd-strlcpy";; \
		linuxppc64gcc:*) xopt="--ccflavour=gccppc64 --use-xrd-strlcpy";; \
		linux*:*)        xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		macos*:*)        xopt="--ccflavour=macos";; \
		solaris*:*:i86pc:x86*) xopt="--ccflavour=sunCCamd --use-xrd-strlcpy";; \
		solaris*:*:i86pc:*) xopt="--ccflavour=sunCCi86pc --use-xrd-strlcpy";; \
		solarisgcc:5.8)  xopt="--ccflavour=gcc";; \
		solaris*:5.8)    xopt="--ccflavour=sunCC";; \
		solarisgcc:5.9)  xopt="--ccflavour=gcc";; \
		solaris*:5.9)    xopt="--ccflavour=sunCC";; \
		solarisgcc:*)    xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		solaris*:*)      xopt="--ccflavour=sunCC --use-xrd-strlcpy";; \
		win32gcc:*)      xopt="win32gcc";; \
		*)               xopt="";; \
		esac; \
		if [ "x$(KRB5LIB)" = "x" ] ; then \
		   xopt="$$xopt --disable-krb5"; \
		fi; \
		if [ "x$(BUILDXRDGSI)" = "x" ] ; then \
		   xopt="$$xopt --disable-gsi"; \
		fi; \
		if [ ! "x$(SSLLIBDIR)" = "x" ] ; then \
		   xlib=`echo $(SSLLIBDIR) | cut -c3-`; \
		   xopt="$$xopt --with-ssl-libdir=$$xlib"; \
		fi; \
		if [ ! "x$(SSLINCDIR)" = "x" ] ; then \
		   xinc=`echo $(SSLINCDIR) | cut -c3-`; \
		   xopt="$$xopt --with-ssl-incdir=$$xinc"; \
		fi; \
		if [ ! "x$(SHADOWFLAGS)" = "x" ] ; then \
		   xopt="$$xopt --enable-shadowpw"; \
		fi; \
		if [ ! "x$(AFSLIB)" = "x" ] ; then \
		   xopt="$$xopt --enable-afs"; \
		fi; \
		if [ ! "x$(AFSLIBDIR)" = "x" ] ; then \
		   xlib=`echo $(AFSLIBDIR) | cut -c3-`; \
		   xopt="$$xopt --with-afs-libdir=$$xlib"; \
		fi; \
		if [ ! "x$(AFSINCDIR)" = "x" ] ; then \
		   xinc=`echo $(AFSINCDIR) | cut -c3-`; \
		   xopt="$$xopt --with-afs-incdir=$$xinc"; \
		fi; \
		xopt="$$xopt --disable-krb4 --enable-echo --no-arch-subdirs --disable-mon"; \
		cd xrootd; \
		echo "Options to Xrootd-configure: $$xopt $(XRDDBG)"; \
		GNUMAKE=$(MAKE) ./configure.classic $$xopt $(XRDDBG); \
		rc=$$? ; \
		if [ $$rc = "0" ] ; then \
		   echo "*** Building xrootd ..." ; \
		   $(MAKE); \
		else \
		   echo "*** Error condition reported by Xrootd-configure (rc = $$rc):" \
			" building only the client ... " ; \
		fi)
else
$(XRDPLUGINSA): $(XROOTDETAG)
		@(cd $(XROOTDDIRD); \
		echo "*** Building xrootd..."; \
		unset MAKEFLAGS; \
		nmake -f Makefile.msc CFG=$(XRDDBG))
endif

all-xrootd:   $(XRDPLUGINS) $(XRDEXECS)

clean-xrootd:
ifneq ($(PLATFORM),win32)
		-@(if [ -d $(XROOTDDIRD)/config ]; then \
			cd $(XROOTDDIRD); \
			$(MAKE) clean; \
		fi)
else
		@(if [ -d $(XROOTDDIRD) ]; then \
		   rm -rf $(XROOTDDIRD); \
		fi;)
endif

clean::         clean-xrootd

distclean-xrootd: clean-xrootd
		@rm -rf $(XRDEXECS) $(XROOTDDIRD) $(XROOTDETAG) \
			$(LPATH)/libXrd* bin/libXrd*

distclean::     distclean-xrootd
