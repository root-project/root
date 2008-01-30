# Module.mk for glite module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/5/2002

MODDIR       := glite
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLITEDIR     := $(MODDIR)
GLITEDIRS    := $(GLITEDIR)/src
GLITEDIRI    := $(GLITEDIR)/inc

##### libRgLite #####
GLITEL       := $(MODDIRI)/LinkDef.h
GLITEDS      := $(MODDIRS)/G__gLite.cxx
GLITEDO      := $(GLITEDS:.cxx=.o)
GLITEDH      := $(GLITEDS:.cxx=.h)

GLITEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GLITES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GLITEO       := $(GLITES:.cxx=.o)

GLITEDEP     := $(GLITEO:.o=.d) $(GLITEDO:.o=.d)

GLITELIB     := $(LPATH)/libRgLite.$(SOEXT)
GLITEMAP     := $(GLITELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GLITEH))
ALLLIBS     += $(GLITELIB)
ALLMAPS     += $(GLITEMAP)

# include all dependency files
INCLUDEFILES += $(GLITEDEP)

##### local rules #####
include/%.h:    $(GLITEDIRI)/%.h
		cp $< $@

$(GLITELIB):    $(GLITEO) $(GLITEDO) $(ORDER_) $(MAINLIBS) $(GLITELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRgLite.$(SOEXT) $@ "$(GLITEO) $(GLITEDO)" \
		   "$(GLITELIBEXTRA) $(GLITELIBDIR) $(GAWLIB)"

$(GLITEDS):     $(GLITEH) $(GLITEL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GLITEH) $(GLITEL)

$(GLITEMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(GLITEL)
		$(RLIBMAP) -o $(GLITEMAP) -l $(GLITELIB) \
		   -d $(GLITELIBDEPM) -c $(GLITEL)

all-glite:      $(GLITELIB) $(GLITEMAP)

clean-glite:
		@rm -f $(GLITEO) $(GLITEDO)

clean::         clean-glite

distclean-glite: clean-glite
		@rm -f $(GLITEDEP) $(GLITEDS) $(GLITEDH) $(GLITELIB) $(GLITEMAP)

distclean::     distclean-glite

##### extra rules ######
$(GLITEO) $(GLITEDO): CXXFLAGS += $(GAW_CPPFLAGS)
