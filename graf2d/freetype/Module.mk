# Module.mk for freetype 2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 7/1/2003

MODNAME      := freetype

ifneq ($(BUILTINFREETYPE),yes)

FREETYPELIBF    := $(shell freetype-config --libs)
FREETYPEINC     := $(shell freetype-config --cflags)
FREETYPELIB     := $(filter -l%,$(FREETYPELIBF))
FREETYPELDFLAGS := $(filter-out -l%,$(FREETYPELIBF))
FREETYPEDEP     :=

.PHONY:         distclean-$(MODNAME)
distclean-$(MODNAME):
		@rm -f $(LPATH)/libfreetype.lib $(LPATH)/libfreetype.a
distclean::     distclean-$(MODNAME)

else

MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src

FREETYPEVERS := freetype-2.3.12
FREETYPEDIR  := $(call stripsrc,$(MODDIR))
FREETYPEDIRS := $(call stripsrc,$(MODDIRS))
FREETYPEDIRI := $(FREETYPEDIRS)/$(FREETYPEVERS)/include

##### libfreetype #####
FREETYPELIBS := $(MODDIRS)/$(FREETYPEVERS).tar.gz
ifeq ($(PLATFORM),win32)
FREETYPELIB  := $(LPATH)/libfreetype.lib
ifeq (yes,$(WINRTDEBUG))
FREETYPELIBA := $(call stripsrc,$(MODDIRS)/$(FREETYPEVERS)/objs/freetype2312MT_D.lib)
FTNMCFG      := "freetype - Win32 Debug Multithreaded"
else
FREETYPELIBA := $(call stripsrc,$(MODDIRS)/$(FREETYPEVERS)/objs/freetype2312MT.lib)
FTNMCFG      := "freetype - Win32 Release Multithreaded"
endif
else
FREETYPELIBA := $(call stripsrc,$(MODDIRS)/$(FREETYPEVERS)/objs/.libs/libfreetype.a)
FREETYPELIB  := $(LPATH)/libfreetype.a
endif
FREETYPEINC  := $(FREETYPEDIRI:%=-I%)
FREETYPEDEP  := $(FREETYPELIB)
FREETYPELDFLAGS :=

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(FREETYPELIB): $(FREETYPELIBA)
ifeq ($(PLATFORM),aix5)
		ar rv $@ $(FREETYPEDIRS)/$(FREETYPEVERS)/objs/.libs/*.o
else
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)
endif

$(FREETYPELIBA): $(FREETYPELIBS)
		$(MAKEDIR)
ifeq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(RSYNC) --exclude '.svn' --exclude '*.lib' $(ROOT_SRCDIR)/$(FREETYPEDIRS)/win $(FREETYPEDIRS)
endif
		@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			rm -rf $(FREETYPEDIRS)/$(FREETYPEVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(FREETYPEDIRS); \
		if [ ! -d $(FREETYPEVERS) ]; then \
			gunzip -c $(FREETYPELIBS) | tar xf -; \
		fi; \
		cd $(FREETYPEVERS)/builds/win32/visualc; \
		cp ../../../../win/freetype.mak .; \
		cp ../../../../win/freetype.dep .; \
		unset MAKEFLAGS; \
		nmake -nologo -f freetype.mak \
		CFG=$(FTNMCFG) \
		NMAKECXXFLAGS="$(BLDCXXFLAGS) -D_CRT_SECURE_NO_DEPRECATE")
else
		@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			rm -rf $(FREETYPEDIRS)/$(FREETYPEVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(FREETYPEDIRS); \
		if [ ! -d $(FREETYPEVERS) ]; then \
			gunzip -c $(FREETYPELIBS) | tar xf -; \
		fi; \
		cd $(FREETYPEVERS); \
		FREECC=$(CC); \
		if [ "$(CC)" = "icc" ]; then \
			FREECC="icc -wd188 -wd181"; \
		fi; \
		if [ $(ARCH) = "alphacxx6" ]; then \
			FREECC="cc"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			FREECC="$$FREECC -m32"; \
			FREE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			FREECC="$$FREECC -m64"; \
			FREE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			FREECC="$$FREECC -m32"; \
			FREE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			FREECC="$$FREECC -m64"; \
			FREE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			FREECC="$$FREECC -m32"; \
			FREE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			FREECC="$$FREECC -m64"; \
			FREE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			FREECC="$$FREECC -arch i386"; \
			FREE_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			FREE_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			FREECC="$$FREECC -arch armv7"; \
			FREE_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			FREE_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			FREECC="$$FREECC -m64"; \
			FREE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "sgicc64" ]; then \
			FREECC="cc"; \
			FREE_CFLAGS="-64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			FREECC="$$FREECC -m64"; \
			FREE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			FREECC="cc"; \
			FREE_CFLAGS="+DD64 -Ae +W863"; \
		fi; \
		if [ $(ARCH) = "aix5" ]; then \
			FREEZLIB="--without-zlib"; \
		fi; \
		if [ $(ARCH) = "aixgcc" ]; then \
			FREEZLIB="--without-zlib"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure $$FREE_HOST --with-pic \
		--disable-shared $$FREEZLIB \
		CC=\'$$FREECC\' CFLAGS=\'$$FREE_CFLAGS -O\'; \
		$(MAKE))
endif

all-$(MODNAME): $(FREETYPELIB)

clean-$(MODNAME):
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS)/builds/win32/visualc ]; then \
			cd $(FREETYPEDIRS)/$(FREETYPEVERS)/builds/win32/visualc; \
			unset MAKEFLAGS; \
			nmake -nologo -f freetype.mak \
			CFG=$(FTNMCFG) clean; \
		fi)
else
		-@(if [ -d $(FREETYPEDIRS)/$(FREETYPEVERS) ]; then \
			cd $(FREETYPEDIRS)/$(FREETYPEVERS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
ifeq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@mv $(FREETYPELIBS) $(FREETYPEDIRS)/-$(FREETYPEVERS).tar.gz
endif
		@rm -rf $(FREETYPELIB) $(FREETYPEDIRS)/freetype-*
ifeq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@mv $(FREETYPEDIRS)/-$(FREETYPEVERS).tar.gz $(FREETYPELIBS)
endif
ifeq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(FREETYPEDIRS)/win
endif
endif

distclean::     distclean-$(MODNAME)

endif
