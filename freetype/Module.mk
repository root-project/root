# Module.mk for freetype 2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 7/1/2003

MODDIR       := freetype
MODDIRS      := $(MODDIR)/src

FREETYPEVERS := freetype-2.1.3
FREETYPEDIR  := $(MODDIR)
FREETYPEDIRS := $(MODDIRS)
FREETYPEDIRI := $(MODDIRS)/$(FREETYPEVERS)/include

##### libfreetype #####
FREETYPELIBS := $(MODDIRS)/$(FREETYPEVERS).tar.gz
ifeq ($(PLATFORM),win32)
FREETYPELIBA := $(MODDIRS)/$(FREETYPEVERS)/objs/freetype213MT.lib
FREETYPELIB  := $(LPATH)/libfreetype.lib
else
FREETYPELIBA := $(MODDIRS)/$(FREETYPEVERS)/objs/.libs/libfreetype.a
FREETYPELIB  := $(LPATH)/libfreetype.a
endif

##### local rules #####
$(FREETYPELIB): $(FREETYPELIBA)
		cp $< $@

$(FREETYPELIBA):
ifeq ($(PLATFORM),win32)
		@(if [ ! -r $@ ]; then \
			echo "*** Building $@..."; \
			cd $(FREETYPEDIRS); \
			if [ ! -d $(FREETYPEVERS) ]; then \
				gunzip -c $(FREETYPEVERS).tar.gz | tar xf -; \
			fi; \
			cd $(FREETYPEVERS)/builds/win32/visualc; \
			cp ../../../../win/freetype.mak .; \
			cp ../../../../win/freetype.dep .; \
			unset MAKEFLAGS; \
			nmake -nologo -f freetype.mak \
			CFG="freetype - Win32 Release Multithreaded"; \
		fi)
else
		@(if [ ! -r $@ ]; then \
			echo "*** Building $@..."; \
			cd $(FREETYPEDIRS); \
			if [ ! -d $(FREETYPEVERS) ]; then \
				TAR=`which gtar 2>/dev/null`; \
				if [ $$? -eq 0 ]; then \
					gtar zxf $(FREETYPEVERS).tar.gz; \
				else \
					gunzip -c $(FREETYPEVERS).tar.gz | tar xf -; \
				fi; \
				if [ $(ARCH) = "macosx" ]; then \
					PATCH=$(FREETYPEVERS)/include/freetype/config/ftconfig.h; \
					sed -e "s/define FT_MACINTOSH 1/undef FT_MACINTOSH/" $$PATCH > ftconfig.hh; \
					mv ftconfig.hh $$PATCH; \
				fi; \
			fi; \
			cd $(FREETYPEVERS); \
			FREECC=""; \
			if [ $(ARCH) = "alphacxx6" ]; then \
				FREECC="CC=cc"; \
			fi; \
			GNUMAKE=$(MAKE) ./configure --with-pic $$FREECC CFLAGS=-O2; \
			$(MAKE); \
		fi)
endif

all-freetype:   $(FREETYPELIB)

clean-freetype:
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS)/builds/win32/visualc) ]; then \
			cd $(FREETYPEDIRS)/$(FREETYPEVERS)/builds/win32/visualc; \
			unset MAKEFLAGS; \
			nmake -nologo -f freetype.mak \
			CFG="freetype - Win32 Release Multithreaded" clean; \
		fi)
else
		-@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			cd $(FREETYPEDIRS)/$(FREETYPEVERS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-freetype

distclean-freetype: clean-freetype
		@rm -rf $(FREETYPELIB) $(FREETYPEDIRS)/$(FREETYPEVERS)

distclean::     distclean-freetype
