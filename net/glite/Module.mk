# Module.mk for glite module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/5/2002

MODNAME      := glite
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GLITEDIR     := $(MODDIR)
GLITEDIRS    := $(GLITEDIR)/src
GLITEDIRI    := $(GLITEDIR)/inc

##### libRgLite #####
GLITEL       := $(MODDIRI)/LinkDef.h
GLITEDS      := $(call stripsrc,$(MODDIRS)/G__gLite.cxx)
GLITEDO      := $(GLITEDS:.cxx=.o)
GLITEDH      := $(GLITEDS:.cxx=.h)

GLITEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GLITES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GLITEO       := $(call stripsrc,$(GLITES:.cxx=.o))

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
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GLITEDIRI)/%.h
		cp $< $@

$(GLITELIB):    $(GLITEO) $(GLITEDO) $(ORDER_) $(MAINLIBS) $(GLITELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRgLite.$(SOEXT) $@ "$(GLITEO) $(GLITEDO)" \
		   "$(GLITELIBEXTRA) $(GLITELIBDIR) $(GAWLIB)"

$(call pcmrule,GLITE)
	$(noop)

$(GLITEDS):     $(GLITEH) $(GLITEL) $(ROOTCLINGEXE) $(call pcmdep,GLITE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GLITE) -c $(GLITEH) $(GLITEL)

$(GLITEMAP):    $(GLITEH) $(GLITEL) $(ROOTCLINGEXE) $(call pcmdep,GLITE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GLITEDS) $(call dictModule,GLITE) -c $(GLITEH) $(GLITEL)

all-$(MODNAME): $(GLITELIB)

clean-$(MODNAME):
		@rm -f $(GLITEO) $(GLITEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GLITEDEP) $(GLITEDS) $(GLITEDH) $(GLITELIB) $(GLITEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GLITEO) $(GLITEDO): CXXFLAGS += $(GAW_CPPFLAGS)
