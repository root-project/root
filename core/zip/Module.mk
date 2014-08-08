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
ZIPDS        := $(call stripsrc,$(MODDIRS)/G__Zip.cxx)
ZIPDO        := $(ZIPDS:.cxx=.o)
ZIPDH        := $(ZIPDS:.cxx=.h)
ZIPDICTH     := $(MODDIRI)/Compression.h

ZIPOLDH      := $(MODDIRI)/Compression.h \
                $(MODDIRI)/RZip.h

ZIPOLDS      := $(MODDIRS)/ZDeflate.c   \
                $(MODDIRS)/ZInflate.c

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
ZIPH         := $(ZIPOLDH)
ZIPS         := $(ZIPOLDS) $(ZIPNEWS)
else
ZIPH         := $(ZIPOLDH)
ZIPS         := $(ZIPOLDS)
endif

ZIPS1        := $(MODDIRS)/Compression.cxx \
                $(MODDIRS)/RZip.cxx

ZIPO         := $(call stripsrc,$(ZIPS:.c=.o) $(ZIPS1:.cxx=.o))
ZIPDEP       := $(ZIPO:.o=.d) $(ZIPDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ZIPH))

# include all dependency files
INCLUDEFILES += $(ZIPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(ZIPDIRI)/%.h
		cp $< $@

$(ZIPO) : CFLAGS += -I$(ZIPDIRI)
$(ZIPO) : CXXFLAGS += -I$(ZIPDIRI)

$(ZIPDS):      $(ZIPDICTH) $(ZIPL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(ZIPDICTH) $(ZIPL)

all-$(MODNAME): $(ZIPO)

clean-$(MODNAME):
		@rm -f $(ZIPO) $(ZIPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ZIPDEP) $(ZIPDS) $(ZIPDH)

distclean::     distclean-$(MODNAME)
 