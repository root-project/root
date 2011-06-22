# Module.mk for lzma module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 15/6/2011

MODNAME      := lzma
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

LZMADIR      := $(MODDIR)
LZMADIRS     := $(LZMADIR)/src
LZMADIRI     := $(LZMADIR)/inc

LZMAVERS     := xz-5.0.3
ifeq ($(BUILTINLZMA),yes)
LZMALIBDIRS  := $(call stripsrc,$(MODDIRS)/$(LZMAVERS))
LZMALIBDIRI  := -I$(LZMALIBDIRS)/src/liblzma/api
else
LZMALIBDIRS  :=
LZMALIBDIRI  := $(LZMAINCDIR:%=-I%)
endif

##### liblzma.a #####
ifeq ($(BUILTINLZMA),yes)
LZMALIBS     := $(MODDIRS)/$(LZMAVERS).tar.gz
ifeq ($(PLATFORM),win32)
LZMALIBA     := $(LZMALIBDIRS)/objs/liblzma.lib
LZMALIB      := $(LPATH)/liblzma.lib
ifeq (yes,$(WINRTDEBUG))
LZMACFG      := "liblzma - Win32 Debug Multithreaded"
else
LZMACFG      := "liblzma - Win32 Release Multithreaded"
endif
else
LZMALIBA     := $(LZMALIBDIRS)/src/liblzma/.libs/liblzma.a
LZMALIB      := $(LPATH)/liblzma.a
endif
LZMALIBDEP   := $(LZMALIB)
else
LZMALIBA     := $(LZMALIBDIR) $(LZMACLILIB)
LZMALIB      := $(LZMALIBDIR) $(LZMACLILIB)
LZMALIBDEP   :=
endif

##### ZipLZMA, part of libCore #####
LZMAH        := $(MODDIRI)/ZipLZMA.h
LZMAS        := $(MODDIRS)/ZipLZMA.c
LZMAO        := $(call stripsrc,$(LZMAS:.c=.o))

LZMADEP      := $(LZMAO:.o=.d)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(LZMAH))

# include all dependency files
INCLUDEFILES += $(LZMADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(LZMADIRI)/%.h
		cp $< $@

ifeq ($(BUILTINLZMA),yes)
$(LZMALIB):     $(LZMALIBA)
ifeq ($(PLATFORM),aix5)
		ar rv $@ $(LZMALIBDIRS)/src/liblzma/.libs/*.o
else
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)
endif

$(LZMALIBA): $(LZMALIBS)
		$(MAKEDIR)
ifeq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(RSYNC) --exclude '.svn' --exclude '*.lib' $(LZMADIRS)/win $(call stripsrc,$(LZMADIRS))
endif
		@(if [ -d $(LZMALIBDIRS) ]; then \
			rm -rf $(LZMALIBDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(call stripsrc,$(LZMADIRS)); \
		if [ ! -d $(LZMAVERS) ]; then \
			gunzip -c $(LZMALIBS) | tar xf -; \
		fi; \
		cd $(LZMAVERS)/builds/win32/visualc; \
		cp ../../../../win/lzma.mak .; \
		cp ../../../../win/lzma.dep .; \
		unset MAKEFLAGS; \
		nmake -nologo -f lzma.mak \
		CFG=$(LZMACFG) \
		NMAKECXXFLAGS="$(BLDCXXFLAGS) -D_CRT_SECURE_NO_DEPRECATE")
else
		@(if [ -d $(LZMALIBDIRS) ]; then \
			rm -rf $(LZMALIBDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(call stripsrc,$(LZMADIRS)); \
		if [ ! -d $(LZMAVERS) ]; then \
			gunzip -c $(LZMALIBS) | tar xf -; \
		fi; \
		cd $(LZMAVERS); \
		LZMACC="$(CC)"; \
		if [ "$(CC)" = "icc" ]; then \
			LZMACC="icc -wd188 -wd181"; \
		fi; \
		if [ $(ARCH) = "alphacxx6" ]; then \
			LZMACC="cc"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			LZMACC="$$LZMACC -arch i386"; \
			LZMA_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZMA_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			LZMACC="$$LZMACC -arch armv7"; \
			LZMA_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZMA_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "sgicc64" ]; then \
			LZMACC="cc"; \
			LZMA_CFLAGS="-64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			LZMACC="cc"; \
			LZMA_CFLAGS="+DD64 -Ae +W863"; \
		fi; \
		GNUMAKE=$(MAKE) CC=$$LZMACC CFLAGS="$$LZMA_CFLAGS -O" \
		./configure $$LZMA_HOST --with-pic --disable-shared; \
		cd src/liblzma; \
		$(MAKE))
endif
endif

all-$(MODNAME): $(LZMAO)

clean-$(MODNAME):
		@rm -f $(LZMAO)
ifeq ($(BUILTINLZMA),yes)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(LZMALIBDIRS)/builds/win32/visualc ]; then \
			cd $(LZMALIBDIRS)/builds/win32/visualc; \
			unset MAKEFLAGS; \
			nmake -nologo -f lzma.mak \
			CFG=$(LZMACFG) clean; \
		fi)
else
		-@(if [ -d $(LZMALIBDIRS) ]; then \
			cd $(LZMALIBDIRS); \
			$(MAKE) clean; \
		fi)
endif
endif

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(LZMADEP)
		@rm -rf $(call stripsrc,$(LZMADIRS)/$(LZMAVERS))
ifeq ($(BUILTINLZMA),yes)
		@rm -f $(LZMALIB)
endif
ifeq ($(PLATFORM),win32)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(call stripsrc,$(LZMADIRS)/win)
endif
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(LZMAO): $(LZMALIBDEP)
$(LZMAO): CFLAGS += $(LZMALIBDIRI)

