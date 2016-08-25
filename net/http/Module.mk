# Module.mk for http module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/11/2002

MODNAME      := http
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HTTPDIR      := $(MODDIR)
HTTPDIRS     := $(HTTPDIR)/src
HTTPDIRI     := $(HTTPDIR)/inc
CIVETWEBDIR  := $(HTTPDIR)/civetweb

HTTPCLILIB   := $(OSTHREADLIB)
HTTPINCDIR   := $(CIVETWEBDIR) 

##### libRHTTP #####
HTTPL        := $(MODDIRI)/LinkDef.h
HTTPDS       := $(call stripsrc,$(MODDIRS)/G__HTTP.cxx)
HTTPDO       := $(HTTPDS:.cxx=.o)
HTTPDH       := $(HTTPDS:.cxx=.h)

HTTPH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HTTPS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HTTPO        := $(call stripsrc,$(HTTPS:.cxx=.o))

CIVETWEBS    := $(CIVETWEBDIR)/civetweb.c
CIVETWEBO    := $(call stripsrc,$(CIVETWEBS:.c=.o))

HTTPDEP      := $(HTTPO:.o=.d) $(HTTPDO:.o=.d)

HTTPLIB      := $(LPATH)/libRHTTP.$(SOEXT)
HTTPMAP      := $(HTTPLIB:.$(SOEXT)=.rootmap)

HTTPCXXFLAGS = $(HTTPINCDIR:%=-I%) $(FASTCGIINCDIR:%=-I%) $(FASTCGIFLAGS) -DUSE_WEBSOCKET

HTTPLIBEXTRA += $(ZLIBLIBDIR) $(ZLIBCLILIB)

ifeq ($(PLATFORM),linux)
HTTPLIBEXTRA += -lrt
endif

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HTTPH))
ALLLIBS     += $(HTTPLIB)
ALLMAPS     += $(HTTPMAP)

# include all dependency files
INCLUDEFILES += $(HTTPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HTTPDIRI)/%.h
		cp $< $@

$(HTTPLIB):     $(HTTPO) $(HTTPDO) $(CIVETWEBO) $(ORDER_) $(MAINLIBS) $(HTTPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRHTTP.$(SOEXT) $@ "$(HTTPO) $(HTTPDO) $(CIVETWEBO)" \
		   "$(HTTPLIBEXTRA) $(HTTPLIBDIR) $(HTTPCLILIB)"

$(HTTPDS):      $(HTTPH) $(HTTPL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HTTPH) $(HTTPL)

$(HTTPMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(HTTPL)
		$(RLIBMAP) -o $@ -l $(HTTPLIB) \
		   -d $(HTTPLIBDEPM) -c $(HTTPL)

all-$(MODNAME): $(HTTPLIB) $(HTTPMAP)

clean-$(MODNAME):
		@rm -f $(HTTPO) $(HTTPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HTTPDEP) $(HTTPDS) $(HTTPDH) $(HTTPLIB) $(HTTPMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(HTTPO) $(HTTPDO) : CXXFLAGS += $(HTTPCXXFLAGS)

$(CIVETWEBO) : CFLAGS += -DUSE_WEBSOCKET
