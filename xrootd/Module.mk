# Module.mk for xrootd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G Ganis, 27/7/2004

MODDIR     := xrootd
MODDIRS    := $(MODDIR)/src

XROOTDVERS := xrootd-20040804-2326
XROOTDDIR  := $(MODDIR)
XROOTDDIRS := $(MODDIRS)
XROOTDDIRD := $(MODDIRS)/xrootd
XROOTDDIRI := $(MODDIRS)/xrootd/src
XROOTDSRCS := $(MODDIRS)/$(XROOTDVERS).tar.gz

##### Xrootd libs for use in netx #####
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
XRDOPT        = "--build=debug"
XRDARCHDIR    = arch_dbg
else
XRDOPT        =
XRDARCHDIR    = arch
endif
XRDLIBDIR     = $(XROOTDDIRD)/lib/$(XRDARCHDIR)
XRDPLUGINS    = $(wildcard $(XRDLIBDIR)/libXrd*.$(SOEXT))
XRDSECLIB     = -Llib -lXrdSec

##### Xrootd executables #####
XOLBDA     := $(XROOTDDIRD)/bin/$(XRDARCHDIR)/olbd
XROOTDA    := $(XROOTDDIRD)/bin/$(XRDARCHDIR)/xrootd
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
				gtar zxf $(XROOTDVERS).tar.gz; \
			else \
				gunzip -c $(XROOTDVERS).tar.gz | tar xf -; \
			fi; \
		fi; \
		cd xrootd; \
		GNUMAKE=$(MAKE) ./configure $(XRDOPT); \
		$(MAKE))

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
