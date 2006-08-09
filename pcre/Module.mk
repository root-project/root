# Module.mk for pcre module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 28/11/2005

ifneq ($(BUILTINPCRE), yes)
PCRELIBF     := $(shell pcre-config --libs)
PCREINC      := $(shell pcre-config --cflags)
PCRELIB      := $(filter -l%,$(PCRELIBF))
PCRELDFLAGS  := $(filter-out -l%,$(PCRELIBF))
PCREDEP      :=
else

MODDIR       := pcre
MODDIRS      := $(MODDIR)/src

PCREVERS     := pcre-6.4
PCREDIR      := $(MODDIR)
PCREDIRS     := $(MODDIRS)
PCREDIRI     := $(MODDIRS)/$(PCREVERS)

##### libpcre #####
PCRELIBS     := $(MODDIRS)/$(PCREVERS).tar.gz
ifeq ($(PLATFORM),win32)
PCRELIBA     := $(MODDIRS)/$(PCREVERS)/Win32/libpcre-6.4.lib
PCRELIB      := $(LPATH)/libpcre.lib
ifeq (yes,$(WINRTDEBUG))
PCREBLD      := "libpcre - Win32 Debug"
else
PCREBLD      := "libpcre - Win32 Release"
endif
else
PCRELIBA     := $(MODDIRS)/$(PCREVERS)/.libs/libpcre.a
PCRELIB      := $(LPATH)/libpcre.a
endif
PCREINC      := $(PCREDIRI:%=-I%)
PCREDEP      := $(PCRELIB)
PCRELDFLAGS  :=

##### local rules #####
$(PCRELIB): $(PCRELIBA)
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)

$(PCRELIBA): $(PCRELIBS)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			rm -rf $(PCREDIRS)/$(PCREVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(PCREDIRS); \
		if [ ! -d $(PCREVERS) ]; then \
			gunzip -c $(PCREVERS).tar.gz | tar xf -; \
		fi; \
		cd $(PCREVERS)/win32; \
		unset MAKEFLAGS; \
		nmake -nologo -f Makefile.msc CFG=$(PCREBLD) \
		NMCXXFLAGS="$(BLDCXXFLAGS) -I../../../../build/win -FIw32pragma.h")
else
		@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			rm -rf $(PCREDIRS)/$(PCREVERS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(PCREDIRS); \
		if [ ! -d $(PCREVERS) ]; then \
			gunzip -c $(PCREVERS).tar.gz | tar xf -; \
		fi; \
		cd $(PCREVERS); \
		PCRECC=$(CC); \
		if [ $(ARCH) = "alphacxx6" ]; then \
			PCRECC="cc"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			PCRECC="gcc"; \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			PCRECC="gcc"; \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "sgicc64" ]; then \
			PCRECC="cc"; \
			PCRE_CFLAGS="-64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			PCRECC="gcc"; \
			PCRE_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			PCRECC="cc"; \
			PCRE_CFLAGS="+DD64 -Ae"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure --with-pic --disable-shared \
		CC=$$PCRECC CFLAGS="$$PCRE_CFLAGS -O"; \
		$(MAKE) libpcre.la)
endif

all-pcre:       $(PCRELIB)

clean-pcre:
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(PCREDIRS)/$(PCREVERS)/win32 ]; then \
			cd $(PCREDIRS)/$(PCREVERS)/win32; \
			unset MAKEFLAGS; \
			nmake -nologo -f Makefile.msc clean; \
		fi)
else
		-@(if [ -d $(PCREDIRS)/$(PCREVERS) ]; then \
			cd $(PCREDIRS)/$(PCREVERS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-pcre

distclean-pcre: clean-pcre
		@mv $(PCRELIBS) $(PCREDIRS)/-$(PCREVERS).tar.gz
		@rm -rf $(PCRELIB) $(PCREDIRS)/pcre-*
		@mv $(PCREDIRS)/-$(PCREVERS).tar.gz $(PCRELIBS)

distclean::     distclean-pcre

endif
