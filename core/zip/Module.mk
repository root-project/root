# Module.mk for zip module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := zip
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ZIPDIR       := $(MODDIR)
ZIPDIRS      := $(ZIPDIR)/src
ZIPDIRI      := $(ZIPDIR)/inc

##### libZip (part of libCore) #####
ZIPL         := $(MODDIRI)/LinkDef.h
ZIPDICTH     := $(MODDIRI)/Compression.h

ZIPOLDH      := $(MODDIRI)/RZip.h     \
                $(MODDIRI)/Compression.h

ZIPOLDS      := $(MODDIRS)/ZDeflate.c   \
                $(MODDIRS)/ZInflate.c

ZIPNEWH      := $(MODDIRI)/zlib.h \
                $(MODDIRI)/zconf.h

ZIPNEWS      := $(MODDIRS)/adler32.c    \
                $(MODDIRS)/compress.c   \
                $(MODDIRS)/crc32.c      \
                $(MODDIRS)/deflate.c    \
                $(MODDIRS)/gzclose.c    \
                $(MODDIRS)/gzlib.c      \
                $(MODDIRS)/gzread.c     \
                $(MODDIRS)/gzwrite.c    \
                $(MODDIRS)/infback.c    \
                $(MODDIRS)/inffast.c    \
                $(MODDIRS)/inflate.c    \
                $(MODDIRS)/inftrees.c   \
                $(MODDIRS)/trees.c      \
                $(MODDIRS)/uncompr.c    \
                $(MODDIRS)/zutil.c

ifeq ($(BUILTINZLIB),yes)
ZIPH         := $(ZIPOLDH) $(ZIPNEWH)
ZIPS         := $(ZIPOLDS) $(ZIPNEWS)
else
ZIPH         := $(ZIPOLDH)
ZIPS         := $(ZIPOLDS)
endif
ZIPS1        := $(MODDIRS)/RZip.cxx \
                $(MODDIRS)/Compression.cxx
ZIPO         := $(call stripsrc,$(ZIPS:.c=.o) $(ZIPS1:.cxx=.o))
ZIPDEP       := $(ZIPO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ZIPH))

# include all dependency files
INCLUDEFILES += $(ZIPDEP)

##### local rules #####
$(ZIPO) : CFLAGS += -I$(ZIPDIRS)
$(ZIPO) : CXXFLAGS += -I$(ZIPDIRS)

.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ZIPDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(ZIPO)

clean-$(MODNAME):
		@rm -f $(ZIPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ZIPDEP) $(ZIPDS) $(ZIPDH)

distclean::     distclean-$(MODNAME)
