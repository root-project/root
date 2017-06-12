# Module.mk for lz4 module
# Copyright (c) 2017 Rene Brun and Fons Rademakers
#
# Author: Brian Bockelman, 29/5/2017

MODNAME      := lz4
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

LZ4DIR      := $(MODDIR)
LZ4DIRS     := $(LZ4DIR)/src
LZ4DIRI     := $(LZ4DIR)/inc

LZ4VERS     := 1.7.5
ifeq ($(BUILTINLZ4),yes)
LZ4LIBDIRS  := $(call stripsrc,$(MODDIRS)/lz4-$(LZ4VERS))
LZ4LIBDIRI  := -I$(LZ4LIBDIRS)/lib
else
LZ4LIBDIRS  :=
LZ4LIBDIRI  := $(LZ4INCDIR:%=-I%)
endif

##### liblz4.a #####
ifeq ($(BUILTINLZ4),yes)
ifeq ($(PLATFORM),win32)
LZ4LIBDIRI  := -I$(LZ4LIBDIRS)/include
LZ4LIBS     := $(MODDIRS)/$(LZ4VERS).tar.gz
LZ4LIBA     := $(LZ4LIBDIRS)/lib/liblz4.lib
LZ4DLLA     := $(LZ4LIBDIRS)/lib/liblz4.dll
LZ4LIB      := $(LPATH)/liblz4.lib
else
LZ4LIBS     := $(MODDIRS)/$(LZ4VERS).tar.gz
LZ4LIBA     := $(LZ4LIBDIRS)/lib/liblz4.a
LZ4LIB      := $(LPATH)/liblz4.a
endif
LZ4LIBDEP   := $(LZ4LIB)
else
LZ4LIBA     := $(LZ4LIBDIR) $(LZ4CLILIB)
LZ4LIB      := $(LZ4LIBDIR) $(LZ4CLILIB)
LZ4LIBDEP   :=
endif

##### ZipLZ4, part of libCore #####
LZ4H        := $(MODDIRI)/ZipLZ4.h
LZ4S        := $(MODDIRS)/ZipLZ4.c
LZ4O        := $(call stripsrc,$(LZ4S:.c=.o))

LZ4DEP      := $(LZ4O:.o=.d)

ifeq ($(BUILTINLZ4),yes)
ifeq ($(PLATFORM),win32)
LZ4DLL      := bin/liblz4.dll
ALLLIBS += $(LZ4DLL)
endif
endif

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(LZ4H))

# include all dependency files
INCLUDEFILES += $(LZ4DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(LZ4DIRI)/%.h
		cp $< $@

ifeq ($(BUILTINLZ4),yes)
$(LZ4LIB):     $(LZ4LIBA)
		cp $< $@
endif

ifeq ($(PLATFORM),win32)
$(LZ4DLL):      $(LZ4LIBA)
		cp $(LZ4DLLA) $@
endif

$(LZ4LIBA):
		$(MAKEDIR)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(LZ4LIBDIRS) ]; then \
			rm -rf $(LZ4LIBDIRS); \
		fi; \
                echo "*** Downloading http://github.com/lz4/v$(LZ4VERS).tar.gz..."; \
                curl -L https://github.com/lz4/lz4/archive/v$(LZ4VERS).tar.gz > $(LZ4LIBS) \
		echo "*** Extracting $@..."; \
		cd $(call stripsrc,$(LZ4DIRS)); \
		if [ ! -d lz4-$(LZ4VERS) ]; then \
			gunzip -c $(LZ4LIBS) | tar xf -; \
		fi; \
		touch $(LZ4VERS)/lib/liblz4.lib;)
else
		@(if [ -d $(LZ4LIBDIRS) ]; then \
			rm -rf $(LZ4LIBDIRS); \
		fi; \
                echo "*** Downloading http://github.com/lz4/v$(LZ4VERS).tar.gz..."; \
                curl -L https://github.com/lz4/lz4/archive/v$(LZ4VERS).tar.gz > $(LZ4LIBS); \
		echo "*** Building $@..."; \
		cd $(call stripsrc,$(LZ4DIRS)); \
		if [ ! -d lz4-$(LZ4VERS) ]; then \
			gunzip -c $(LZ4LIBS) | tar xf -; \
		fi; \
		cd lz4-$(LZ4VERS); \
		LZ4CC="$(CC)"; \
		if [ "$(CC)" = "icc" ]; then \
			LZ4CC="icc -wd188 -wd181 -wd1292 -wd10006 -wd10156 -wd2259 -wd981 -wd128"; \
		fi; \
		if [ $(ARCH) = "linux" ]; then \
			LZ4CC="$$LZ4CC -m32"; \
			LZ4_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664gcc" ]; then \
			LZ4CC="$$LZ4CC -m64"; \
			LZ4_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxx32gcc" ]; then \
			LZ4CC="$$LZ4CC -mx32"; \
			LZ4_CFLAGS="-mx32"; \
		fi; \
		if [ $(ARCH) = "linuxicc" ]; then \
			LZ4CC="$$LZ4CC -m32"; \
			LZ4_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "linuxx8664icc" ]; then \
			LZ4CC="$$LZ4CC -m64"; \
			LZ4_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxx8664k1omicc" ]; then \
			LZ4CC="$$LZ4CC -m64 $(MICFLAGS)"; \
			LZ4_CFLAGS="-m64 $(MICFLAGS)"; \
			LZ4_HOST="--host=x86_64-unknown-linux-gnu"; \
		fi; \
		if [ $(ARCH) = "macosx" ]; then \
			LZ4CC="$$LZ4CC -m32"; \
			LZ4_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "macosx64" ]; then \
			LZ4CC="$$LZ4CC -m64"; \
			LZ4_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "iossim" ]; then \
			LZ4_CFLAGS="-arch i386 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZ4_HOST="--host=i686-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "ios" ]; then \
			LZ4_CFLAGS="-arch armv7 -isysroot $(IOSSDK) -miphoneos-version-min=$(IOSVERS)"; \
			LZ4_HOST="--host=arm-apple-darwin10"; \
		fi; \
		if [ $(ARCH) = "solaris64CC5" ]; then \
			LZ4CC="$$LZ4CC -m64"; \
			LZ4_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxppc64gcc" ]; then \
			LZ4CC="$$LZ4CC -m64"; \
			LZ4_CFLAGS="-m64"; \
		fi; \
		if [ $(ARCH) = "linuxppcgcc" ]; then \
			LZ4CC="$$LZ4CC -m32"; \
			LZ4_CFLAGS="-m32"; \
		fi; \
		if [ $(ARCH) = "hpuxia64acc" ]; then \
			LZ4CC="cc"; \
			LZ4_CFLAGS="+DD64 -Ae +W863"; \
		fi; \
		GNUMAKE=$(MAKE) CC=$$LZ4CC CFLAGS="$$LZ4_CFLAGS -fPIC -O" \
		$(MAKE) lib)
endif

all-$(MODNAME): $(LZ4O)

clean-$(MODNAME):
		@rm -f $(LZ4O)
ifeq ($(BUILTINLZ4),yes)
ifneq ($(PLATFORM),win32)
		-@(if [ -d $(LZ4LIBDIRS) ]; then \
			cd $(LZ4LIBDIRS); \
			$(MAKE) clean; \
		fi)
endif
endif

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(LZ4DEP)
		@rm -rf $(call stripsrc,$(LZ4DIRS)/$(LZ4VERS))
		@rm -f $(LPATH)/liblz4.*
ifeq ($(PLATFORM),win32)
		@rm -f $(LZ4DLL)
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(LZ4O): $(LZ4LIBDEP)
$(LZ4O): CFLAGS += $(LZ4LIBDIRI)

