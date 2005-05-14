# Module.mk for xrootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G Ganis, 27/7/2004

MODDIR     := xrootd
MODDIRS    := $(MODDIR)/src

XROOTDVERS := xrootd-20050328-0656
XROOTDDIR  := $(MODDIR)
XROOTDDIRS := $(MODDIRS)
XROOTDDIRD := $(MODDIRS)/xrootd
XROOTDDIRI := $(MODDIRS)/xrootd/src
XROOTDDIRL := $(MODDIRS)/xrootd/lib
XROOTDSRCS := $(MODDIRS)/$(XROOTDVERS).src.tgz
XROOTDETAG := $(MODDIRS)/headers.d

##### Xrootd config options #####
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
XRDDBG      = "--build=debug"
else
XRDDBG      =
endif
ifeq ($(PLATFORM),macosx)
XRDSOEXT    = so
else
XRDSOEXT    = $(SOEXT)
endif

##### Xrootd executables #####
XRDEXEC     = xrootd olbd
XRDEXECS   := $(patsubst %,bin/%,$(XRDEXEC))

##### Xrootd plugins #####
XRDPLUGINSA:= $(XROOTDDIRL)/libXrdSec.$(XRDSOEXT)
XRDPLUGINS := $(LPATH)/libXrdSec.$(XRDSOEXT)
ifeq ($(ARCH),win32gcc)
XRDPLUGINS := $(patsubst $(LPATH)/%.$(XRDSOEXT),bin/%.$(XRDSOEXT),$(XRDPLUGINS))
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
		   if [ "x`which gtar 2>/dev/null | awk '{if ($$1~/gtar/) print $$1;}'`" != "x" ]; then \
		      gtar zxf $(XROOTDVERS).src.tgz; \
		   else \
		      gunzip -c $(XROOTDVERS).src.tgz | tar xf -; \
		   fi; \
		   etag=`basename $(XROOTDETAG)` ; \
		   touch $$etag ; \
		fi)

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

$(XRDPLUGINSA): $(XROOTDETAG)
		@(cd $(XROOTDDIRS); \
		RELE=`uname -r`; \
		case "$(ARCH):$$RELE" in \
		freebsd*:*)      xopt="--ccflavour=gcc";; \
		linuxicc:*)      xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxia64ecc:*)  xopt="--ccflavour=icc --use-xrd-strlcpy";; \
		linuxx8664gcc:*) xopt="--ccflavour=gccx8664 --use-xrd-strlcpy";; \
		linuxx8664icc:*) xopt="--ccflavour=iccx8664 --use-xrd-strlcpy";; \
		linux*:*)        xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		macos*:*)        xopt="--ccflavour=macos";; \
		solarisgcc:5.8)  xopt="--ccflavour=gcc";; \
		solaris*:5.8)    xopt="--ccflavour=sunCC";; \
		solarisgcc:5.9)  xopt="--ccflavour=gcc";; \
		solaris*:5.9)    xopt="--ccflavour=sunCC";; \
		solarisgcc:*)    xopt="--ccflavour=gcc --use-xrd-strlcpy";; \
		solaris*:*)      xopt="--ccflavour=sunCC  --use-xrd-strlcpy";; \
		win32gcc:*)      xopt="win32gcc";; \
		*)               xopt="";; \
		esac; \
		if [ "x$(KRB5LIB)" = "x" ] ; then \
		   xopt="$$xopt --disable-krb4 --disable-krb5"; \
		fi; \
		xopt="$$xopt --enable-echo --no-arch-subdirs --disable-mon"; \
		cd xrootd; \
		echo "Options to Xrootd-configure: $$xopt $(XRDDBG)"; \
		GNUMAKE=$(MAKE) ./configure $$xopt $(XRDDBG); \
		rc=$$? ; \
		if [ $$rc = "0" ] ; then \
		   echo "*** Building xrootd ..." ; \
		   $(MAKE) -j1; \
		else \
		   echo "*** Error condition reported by Xrootd-configure (rc = $$rc):" \
			" building only the client ... " ; \
		fi)

all-xrootd:   $(XRDPLUGINS) $(XRDEXECS)

clean-xrootd:
		-@(if [ -d $(XROOTDDIRD)/config ]; then \
			cd $(XROOTDDIRD); \
			$(MAKE) clean; \
		fi)

clean::         clean-xrootd

distclean-xrootd: clean-xrootd
		@rm -rf $(XRDEXECS) $(XROOTDDIRD) $(XROOTDETAG) \
			$(LPATH)/libXrd* bin/libXrd*

distclean::     distclean-xrootd
