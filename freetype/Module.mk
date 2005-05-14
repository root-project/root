# Module.mk for freetype 2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 7/1/2003

ifneq ($(BUILTINFREETYPE), yes)
FREETYPELIBF    := $(shell freetype-config --libs)
FREETYPEINC     := $(shell freetype-config --cflags)
FREETYPELIB     := $(filter -l%,$(FREETYPELIBF))
FREETYPELDFLAGS := $(filter-out -l%,$(FREETYPELIBF))
FREETYPEDEP     :=
else

MODDIR       := freetype
MODDIRS      := $(MODDIR)/src

FREETYPEVERS := freetype-2.1.9
FREETYPEDIR  := $(MODDIR)
FREETYPEDIRS := $(MODDIRS)
FREETYPEDIRI := $(MODDIRS)/$(FREETYPEVERS)/include

##### libfreetype #####
FREETYPELIBS := $(MODDIRS)/$(FREETYPEVERS).tar.gz
ifeq ($(PLATFORM),win32)
FREETYPELIBA := $(MODDIRS)/$(FREETYPEVERS)/objs/freetype219MT.lib
FREETYPELIB  := $(LPATH)/libfreetype.lib
else
FREETYPELIBA := $(MODDIRS)/$(FREETYPEVERS)/objs/.libs/libfreetype.a
ifeq ($(PLATFORM),macosx)
FREETYPELIB  := $(LPATH)/libfreetype.dylib
else
FREETYPELIB  := $(LPATH)/libfreetype.a
endif
endif
FREETYPEINC  := $(FREETYPEDIRI:%=-I%)
FREETYPEDEP  := $(FREETYPELIB)
FREETYPELDFLAGS :=

##### local rules #####
$(FREETYPELIB): $(FREETYPELIBA)
ifeq ($(PLATFORM),macosx)
		$(MACOSXTARGET) $(CC) $(SOFLAGS)libfreetype.dylib -o $@ \
		   $(FREETYPEDIRS)/$(FREETYPEVERS)/objs/*.o
else
		cp $< $@
endif

$(FREETYPELIBA): $(FREETYPELIBS)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			rm -rf $(FREETYPEDIRS)/$(FREETYPEVERS); \
		fi; \
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
		CFG="freetype - Win32 Release Multithreaded")
else
		@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			rm -rf $(FREETYPEDIRS)/$(FREETYPEVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(FREETYPEDIRS); \
		if [ ! -d $(FREETYPEVERS) ]; then \
			if [ "x`which gtar 2>/dev/null | awk '{if ($$1~/gtar/) print $$1;}'`" != "x" ]; then \
				gtar zxf $(FREETYPEVERS).tar.gz; \
			else \
				gunzip -c $(FREETYPEVERS).tar.gz | tar xf -; \
			fi; \
		fi; \
		cd $(FREETYPEVERS); \
		FREECC=$(CC); \
		if [ $(ARCH) = "alphacxx6" ]; then \
			FREECC="cc"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			FREECC="gcc -m64"; \
		fi; \
		if [ $(ARCH) = "sgicc64" ]; then \
			FREECC="cc"; \
			ARCH_CFLAGS="-64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			FREECC="gcc -m64"; \
			ARCH_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			FREECC="cc"; \
			ARCH_CFLAGS="+DD64 -Ae"; \
		fi; \
		if [ $(ARCH) = "aix5" ]; then \
			FREEZLIB="--without-zlib"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure --with-pic $$FREEZLIB \
		CC=\"$$FREECC\" CFLAGS=\"$$ARCH_CFLAGS -O2\"; \
		$(MAKE))
endif

all-freetype:   $(FREETYPELIB)

clean-freetype:
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS)/builds/win32/visualc ]; then \
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
		@mv $(FREETYPELIBS) $(FREETYPEDIRS)/-$(FREETYPEVERS).tar.gz
		@rm -rf $(FREETYPELIB) $(FREETYPEDIRS)/freetype-*
		@mv $(FREETYPEDIRS)/-$(FREETYPEVERS).tar.gz $(FREETYPELIBS)

distclean::     distclean-freetype

endif
