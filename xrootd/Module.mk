# Module.mk for xrootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G Ganis, 27/7/2004

MODDIR     := xrootd
MODDIRS    := $(MODDIR)/src

XROOTDVERS := xrootd-20041124-0752
XROOTDDIR  := $(MODDIR)
XROOTDDIRS := $(MODDIRS)
XROOTDDIRD := $(MODDIRS)/xrootd
XROOTDDIRI := $(MODDIRS)/xrootd/src
XROOTDSRCS := $(MODDIRS)/$(XROOTDVERS).src.tgz

##### Xrootd libs for use in netx #####
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
XRDDBG      = "--build=debug"
else	    
XRDDBG      =
endif	    
XRDLIBDIR   = $(XROOTDDIRD)/lib
XRDPLUGINS  = $(wildcard $(XRDLIBDIR)/libXrd*.$(SOEXT))
XRDSECLIB   = -Llib -lXrdSec

##### Xrootd executables #####
XOLBDA     := $(XROOTDDIRD)/bin/olbd
XROOTDA    := $(XROOTDDIRD)/bin/xrootd
XOLBD      := bin/olbd
XROOTD     := bin/xrootd

ALLEXECS   += $(XROOTD)

##### local rules #####
$(XROOTD): $(XROOTDA)
		cp $< $@
		cp $(XOLBDA) $(XOLBD)
		cp $(XRDPLUGINS) $(LPATH)/

$(XROOTDA): $(XROOTDSRCS)
		@(if [ -d $(XROOTDDIRD) ]; then \
			rm -rf $(XROOTDDIRD); \
		fi; \
		echo "*** Building $@..."; \
		cd $(XROOTDDIRS); \
		if [ ! -d xrootd ]; then \
			if [ "x`which gtar 2>/dev/null | awk '{if ($$1~/gtar/) print $$1;}'`" != "x" ]; then \
				gtar zxf $(XROOTDVERS).src.tgz; \
			else \
				gunzip -c $(XROOTDVERS).src.tgz | tar xf -; \
			fi; \
		fi; \
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
                *)               xopt="";; \
		esac; \
		if [ "x$(KRB5LIB)" = "x" ] ; then \
		   xopt="$$xopt --disable-krb4 --disable-krb5"; \
		fi; \
		xopt="$$xopt --enable-echo --no-arch-subdirs"; \
		cd xrootd; \
		echo "Options to Xrootd-configure:$(XRDDBG) $$xopt"; \
		GNUMAKE=$(MAKE) ./configure $(XRDDBG) $$xopt; \
		$(MAKE) -j1)

all-xrootd:   $(XROOTD)

clean-xrootd:
		-@(if [ -d $(XROOTDDIRD)/config ]; then \
			cd $(XROOTDDIRD); \
			$(MAKE) clean; \
		fi)

clean::         clean-xrootd

distclean-xrootd: clean-xrootd
		@rm -rf $(XROOTD) $(XOLBD) $(XROOTDDIRD) $(LPATH)/libXrd*

distclean::     distclean-xrootd
