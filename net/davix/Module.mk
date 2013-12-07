# Module.mk for DAVIX module
#
# Author: Tigran Mkrtchyan <tigran.mkrtchyan@desy.de>

MODNAME      := davix
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

DAVIXDIR    := $(MODDIR)
DAVIXDIRS   := $(DAVIXDIR)/src
DAVIXDIRI   := $(DAVIXDIR)/inc

##### libRDAVIX #####
DAVIXL      := $(MODDIRI)/LinkDef.h
DAVIXDS     := $(call stripsrc,$(MODDIRS)/G__DAVIX.cxx)
DAVIXDO     := $(DAVIXDS:.cxx=.o)
DAVIXDH     := $(DAVIXDS:.cxx=.h)

DAVIXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
DAVIXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
DAVIXO      := $(call stripsrc,$(DAVIXS:.cxx=.o))

DAVIXDEP    := $(DAVIXO:.o=.d) $(DAVIXDO:.o=.d)

DAVIXLIB    := $(LPATH)/libRDAVIX.$(SOEXT)
DAVIXMAP    := $(DAVIXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(DAVIXH))
ALLLIBS     += $(DAVIXLIB)
ALLMAPS     += $(DAVIXMAP)

# include all dependency files
INCLUDEFILES += $(DAVIXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(DAVIXDIRI)/%.h
		cp $< $@

$(DAVIXLIB):    $(DAVIXO) $(DAVIXDO) $(ORDER_) $(MAINLIBS) $(DAVIXLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRDAVIX.$(SOEXT) $@ "$(DAVIXO) $(DAVIXDO)" \
		   "$(DAVIXLIBEXTRA) $(DAVIXLIBDIR) $(DAVIXCLILIB)"

$(DAVIXDS):     $(DAVIXH) $(DAVIXL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(DAVIXH) $(DAVIXL)

$(DAVIXMAP):    $(RLIBMAP) $(MAKEFILEDEP) $(DAVIXL)
		$(RLIBMAP) -o $@ -l $(DAVIXLIB) \
		   -d $(DAVIXLIBDEPM) -c $(DAVIXL)

all-$(MODNAME): $(DAVIXLIB) $(DAVIXMAP)

clean-$(MODNAME):
		@rm -f $(DAVIXO) $(DAVIXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(DAVIXDEP) $(DAVIXDS) $(DAVIXDH) $(DAVIXLIB) $(DAVIXMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(DAVIXO) $(DAVIXDO): CXXFLAGS += $(DAVIXINCDIR:%=-I%)
