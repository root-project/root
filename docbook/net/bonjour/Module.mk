# Module.mk for bonjour module
# Copyright (c) 2009 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/05/2009

MODNAME      := bonjour
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

BONJDIR      := $(MODDIR)
BONJDIRS     := $(BONJDIR)/src
BONJDIRI     := $(BONJDIR)/inc

##### libBonjour #####
BONJL        := $(MODDIRI)/LinkDef.h
BONJDS       := $(call stripsrc,$(MODDIRS)/G__BONJ.cxx)
BONJDO       := $(BONJDS:.cxx=.o)
BONJDH       := $(BONJDS:.cxx=.h)

BONJH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
BONJS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
BONJO        := $(call stripsrc,$(BONJS:.cxx=.o))

BONJDEP      := $(BONJO:.o=.d) $(BONJDO:.o=.d)

BONJLIB      := $(LPATH)/libBonjour.$(SOEXT)
BONJMAP      := $(BONJLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(BONJH))
ALLLIBS     += $(BONJLIB)
ALLMAPS     += $(BONJMAP)

# include all dependency files
INCLUDEFILES += $(BONJDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(BONJDIRI)/%.h
		cp $< $@

$(BONJLIB):     $(BONJO) $(BONJDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libBonjour.$(SOEXT) $@ "$(BONJO) $(BONJDO)" \
		   "$(BONJLIBEXTRA) $(DNSSDLIBDIR) $(DNSSDLIB)"

$(BONJDS):      $(BONJH) $(BONJL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(BONJH) $(BONJL)

$(BONJMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(BONJL)
		$(RLIBMAP) -o $@ -l $(BONJLIB) \
		   -d $(BONJLIBDEPM) -c $(BONJL)

all-$(MODNAME): $(BONJLIB) $(BONJMAP)

clean-$(MODNAME):
		@rm -f $(BONJO) $(BONJDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(BONJDEP) $(BONJDS) $(BONJDH) $(BONJLIB) $(BONJMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(BONJO): CXXFLAGS += $(BONJINCDIR:%=-I%)
