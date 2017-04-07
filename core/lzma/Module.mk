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

LZMAVERS     := xz-5.2.1
ifeq ($(BUILTINLZMA),yes)
LZMALIBDIRS  := $(call stripsrc,$(MODDIRS)/$(LZMAVERS))
LZMALIBDIRI  := -I$(LZMALIBDIRS)/src/liblzma/api
else
LZMALIBDIRS  :=
LZMALIBDIRI  := $(LZMAINCDIR:%=-I%)
endif

##### liblzma.a #####
ifeq ($(BUILTINLZMA),yes)
ifeq ($(PLATFORM),win32)
LZMALIBDIRI  := -I$(LZMALIBDIRS)/include
LZMALIBS     := $(MODDIRS)/$(LZMAVERS)-win32.tar.gz
LZMALIBA     := $(LZMALIBDIRS)/lib/liblzma.lib
LZMADLLA     := $(LZMALIBDIRS)/lib/liblzma.dll
LZMALIB      := $(LPATH)/liblzma.lib
else
LZMALIBS     := $(MODDIRS)/$(LZMAVERS).tar.gz
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

ifeq ($(BUILTINLZMA),yes)
ifeq ($(PLATFORM),win32)
LZMADLL      := bin/liblzma.dll
ALLLIBS += $(LZMADLL)
endif
endif

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
		ar rv $@ $(LZMALIBDIRS)/src/liblzma/*.o
else
		cp $< $@
		@(if [ $(PLATFORM) = "macosx" ]; then \
			ranlib $@; \
		fi)
endif

ifeq ($(PLATFORM),win32)
$(LZMADLL):      $(LZMALIBA)
		cp $(LZMADLLA) $@
endif

$(LZMALIBA): $(LZMALIBS)
		$(MAKEDIR)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(LZMALIBDIRS) ]; then \
			rm -rf $(LZMALIBDIRS); \
		fi; \
		echo "*** Extracting $@..."; \
		cd $(call stripsrc,$(LZMADIRS)); \
		if [ ! -d $(LZMAVERS) ]; then \
			gunzip -c $(LZMALIBS) | tar xf -; \
		fi; \
		touch $(LZMAVERS)/lib/liblzma.lib;)
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
			LZMACC="icc -wd188 -wd181 -wd1292 -wd10006 -wd10156 -wd2259 -wd981 -wd128"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxx32gcc" ]; then \
			LZMACC="$$LZMACC -mx32"; \
			LZMA_CFLAGS="-mx32"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxx8664k1omicc" ]; then \
			LZMACC="$$LZMACC -m64 $(MICFLAGS)"; \
			LZMA_CFLAGS="-m64 $(MICFLAGS)"; \
			LZMA_HOST="--host=x86_64-unknown-linux-gnu"; \
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
			LZMA_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZMA_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			LZMA_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZMA_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			LZMACC="$$LZMACC -m64"; \
			LZMA_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxppcgcc" ]; then \
			LZMACC="$$LZMACC -m32"; \
			LZMA_CFLAGS="-m32"; \
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
ifneq ($(PLATFORM),win32)
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
		@rm -f $(LPATH)/liblzma.*
ifeq ($(PLATFORM),win32)
		@rm -f $(LZMADLL)
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(LZMAO): $(LZMALIBDEP)
$(LZMAO): CFLAGS += $(LZMALIBDIRI)

