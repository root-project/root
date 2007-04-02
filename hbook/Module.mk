# Module.mk for hbook module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/2/2002

MODDIR       := hbook
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HBOOKDIR     := $(MODDIR)
HBOOKDIRS    := $(HBOOKDIR)/src
HBOOKDIRI    := $(HBOOKDIR)/inc

##### libHbook #####
HBOOKL       := $(MODDIRI)/LinkDef.h
HBOOKDS      := $(MODDIRS)/G__Hbook.cxx
HBOOKDO      := $(HBOOKDS:.cxx=.o)
HBOOKDH      := $(HBOOKDS:.cxx=.h)

HBOOKH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HBOOKS1      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HBOOKS2      := $(MODDIRS)/hntvar2.f
HBOOKO1      := $(HBOOKS1:.cxx=.o)
HBOOKO2      := $(HBOOKS2:.f=.o)
HBOOKO       := $(HBOOKO1) $(HBOOKO2)

HBOOKDEP     := $(HBOOKS1:.cxx=.d) $(HBOOKDO:.o=.d)

HBOOKLIB     := $(LPATH)/libHbook.$(SOEXT)
HBOOKMAP     := $(HBOOKLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HBOOKH))
ALLLIBS     += $(HBOOKLIB)
ALLMAPS     += $(HBOOKMAP)

# include all dependency files
INCLUDEFILES += $(HBOOKDEP)

##### local rules #####
include/%.h:    $(HBOOKDIRI)/%.h
		cp $< $@

$(HBOOKLIB):    $(HBOOKO) $(HBOOKDO) $(ORDER_) $(MAINLIBS) $(HBOOKLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHbook.$(SOEXT) $@ "$(HBOOKO1) $(HBOOKDO)" \
		   "$(HBOOKO2) $(CERNLIBDIR) $(CERNLIBS) \
		    $(SHIFTLIBDIR) $(SHIFTLIB) $(HBOOKLIBEXTRA) $(F77LIBS)"

$(HBOOKDS):     $(HBOOKH) $(HBOOKL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HBOOKH) $(HBOOKL)

$(HBOOKMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(HBOOKL)
		$(RLIBMAP) -o $(HBOOKMAP) -l $(HBOOKLIB) \
		   -d $(HBOOKLIBDEPM) -c $(HBOOKL)

all-hbook:      $(HBOOKLIB) $(HBOOKMAP)

clean-hbook:
		@rm -f $(HBOOKO) $(HBOOKDO)

clean::         clean-hbook

distclean-hbook: clean-hbook
		@rm -f $(HBOOKDEP) $(HBOOKDS) $(HBOOKDH) $(HBOOKLIB) $(HBOOKMAP)

distclean::     distclean-hbook
