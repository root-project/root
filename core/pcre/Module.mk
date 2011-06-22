# Module.mk for pcre module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 28/11/2005

MODNAME      := pcre

ifneq ($(BUILTINPCRE), yes)

PCRELIBF     := $(shell pcre-config --libs)
PCREINC      := $(shell pcre-config --cflags)
PCRELIB      := $(filter -l%,$(PCRELIBF))
PCRELDFLAGS  := $(filter-out -l%,$(PCRELIBF))
PCREDEP      :=

.PHONY:         distclean-$(MODNAME)
distclean-$(MODNAME):
		@rm -f $(LPATH)/libpcre.lib $(LPATH)/libpcre.a
distclean::     distclean-$(MODNAME)

else

MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src

PCREVERS     := pcre-7.8
PCREDIR      := $(call stripsrc,$(MODDIR))
PCREDIRS     := $(call stripsrc,$(MODDIRS))
PCREDIRI     := $(PCREDIRS)/$(PCREVERS)

##### libpcre #####
PCRELIBS     := $(MODDIRS)/$(PCREVERS).tar.gz
ifeq ($(PLATFORM),win32)
PCRELIBA     := $(call stripsrc,$(MODDIRS)/win32/libpcre-7.8.lib)
PCRELIB      := $(LPATH)/libpcre.lib
ifeq (yes,$(WINRTDEBUG))
PCREBLD      := "libpcre - Win32 Debug"
else
PCREBLD      := "libpcre - Win32 Release"
endif
else
PCRELIBA     := $(call stripsrc,$(MODDIRS)/$(PCREVERS)/.libs/libpcre.a)
PCRELIB      := $(LPATH)/libpcre.a
endif
PCREINC      := $(PCREDIRI:%=-I%)
PCREDEP      := $(PCRELIB)
PCRELDFLAGS  :=

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(PCRELIB): $(PCRELIBA)
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

$(PCRELIBA): $(PCRELIBS)
		$(MAKEDIR)
ifeq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(RSYNC) --exclude '.svn' --exclude '*.lib' $(ROOT_SRCDIR)/$(PCREDIRS)/win32 $(PCREDIRS)
endif
		@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			rm -rf $(PCREDIRS)/$(PCREVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(PCREDIRS); \
		if [ ! -d $(PCREVERS) ]; then \
			gunzip -c $(PCRELIBS) | tar xf -; \
		fi; \
		cd win32; \
		unset MAKEFLAGS; \
		nmake -nologo -f Makefile.msc CFG=$(PCREBLD) \
		NMCXXFLAGS="$(BLDCXXFLAGS) -I../../../../include -FIw32pragma.h")
else
		@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			rm -rf $(PCREDIRS)/$(PCREVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(PCREDIRS); \
		if [ ! -d $(PCREVERS) ]; then \
			gunzip -c $(PCRELIBS) | tar xf -; \
		fi; \
		cd $(PCREVERS); \
		PCRECC="$(CC)"; \
		if [ $(ARCH) = "alphacxx6" ]; then \
			PCRECC="cc"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			PCRE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			PCRE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			PCRE_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			PCRE_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			PCRE_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			PCRE_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			PCRE_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "sgicc64" ]; then \
			PCRECC="cc"; \
			PCRE_CFLAGS="-64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			PCRECC="cc"; \
			PCRE_CFLAGS="+DD64 -Ae"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure $$PCRE_HOST --with-pic \
		--disable-shared CC=$$PCRECC CFLAGS="$$PCRE_CFLAGS -O"; \
		$(MAKE) libpcre.la)
endif

all-$(MODNAME): $(PCRELIB)

clean-pcre:
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(PCREDIRS)/win32 ]; then \
			cd $(PCREDIRS)/win32; \
			unset MAKEFLAGS; \
			nmake -nologo -f Makefile.msc clean; \
		fi)
else
		-@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			cd $(PCREDIRS)/$(PCREVERS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(PCREDIRS)/win32 ]; then \
			cd $(PCREDIRS)/win32; \
			unset MAKEFLAGS; \
			nmake -nologo -f Makefile.msc distclean; \
		fi)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(PCREDIRS)/win32
endif
endif
ifeq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@mv $(PCRELIBS) $(PCREDIRS)/-$(PCREVERS).tar.gz
endif
		@rm -rf $(PCRELIB) $(PCREDIRS)/pcre-*
ifeq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@mv $(PCREDIRS)/-$(PCREVERS).tar.gz $(PCRELIBS)
endif

distclean::     distclean-$(MODNAME)

endif
