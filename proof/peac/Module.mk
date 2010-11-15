# Module.mk for peac module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Maarten Ballintijn 18/10/2004

MODNAME      := peac
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PEACDIR      := $(MODDIR)
PEACDIRS     := $(PEACDIR)/src
PEACDIRI     := $(PEACDIR)/inc

##### libPeac #####
PEACL        := $(MODDIRI)/LinkDef.h
PEACDS       := $(call stripsrc,$(MODDIRS)/G__Peac.cxx)
PEACDO       := $(PEACDS:.cxx=.o)
PEACDH       := $(PEACDS:.cxx=.h)

PEACH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PEACS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PEACO        := $(call stripsrc,$(PEACS:.cxx=.o))

PEACDEP      := $(PEACO:.o=.d) $(PEACDO:.o=.d)

PEACLIB      := $(LPATH)/libPeac.$(SOEXT)
PEACMAP      := $(PEACLIB:.$(SOEXT)=.rootmap)

##### libPeacGui #####
PEACGUIL     := $(MODDIRI)/LinkDefGui.h
PEACGUIDS    := $(call stripsrc,$(MODDIRS)/G__PeacGui.cxx)
PEACGUIDO    := $(PEACGUIDS:.cxx=.o)
PEACGUIDH    := $(PEACGUIDS:.cxx=.h)

PEACGUIH     := $(MODDIRI)/TProofStartupDialog.h
PEACGUIS     := $(MODDIRS)/TProofStartupDialog.cxx
PEACGUIO     := $(call stripsrc,$(PEACGUIS:.cxx=.o))

PEACGUIDEP   := $(PEACGUIO:.o=.d) $(PEACGUIDO:.o=.d)

PEACGUILIB   := $(LPATH)/libPeacGui.$(SOEXT)
PEACGUIMAP   := $(PEACGUILIB:.$(SOEXT)=.rootmap)

# remove GUI files from PEAC files
PEACH        := $(filter-out $(PEACGUIH),$(PEACH))
PEACS        := $(filter-out $(PEACGUIS),$(PEACS))
PEACO        := $(filter-out $(PEACGUIO),$(PEACO))
PEACDEP      := $(filter-out $(PEACGUIDEP),$(PEACDEP))

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PEACH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PEACGUIH))
ALLLIBS     += $(PEACLIB) $(PEACGUILIB)
ALLMAPS     += $(PEACMAP) $(PEACGUIMAP)

# include all dependency files
INCLUDEFILES += $(PEACDEP) $(PEACGUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PEACDIRI)/%.h
		cp $< $@

$(PEACLIB):     $(PEACO) $(PEACDO) $(ORDER_) $(MAINLIBS) $(PEACLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPeac.$(SOEXT) $@ "$(PEACO) $(PEACDO)" \
		   "$(PEACLIBEXTRA)"

$(PEACDS):      $(PEACH) $(PEACL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PEACH) $(PEACL)

$(PEACMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(PEACL)
		$(RLIBMAP) -o $@ -l $(PEACLIB) \
		   -d $(PEACLIBDEPM) -c $(PEACL)

$(PEACGUILIB):  $(PEACGUIO) $(PEACGUIDO) $(ORDER_) $(MAINLIBS) $(PEACGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPeacGui.$(SOEXT) $@ \
		   "$(PEACGUIO) $(PEACGUIDO)" \
		   "$(PEACGUILIBEXTRA)"

$(PEACGUIDS):   $(PEACGUIH) $(PEACGUIL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PEACGUIH) $(PEACGUIL)

$(PEACGUIMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(PEACGUIL)
		$(RLIBMAP) -o $(PEACGUIMAP) -l $(PEACGUILIB) \
		   -d $(PEACGUILIBDEPM) -c $(PEACGUIL)

all-$(MODNAME): $(PEACLIB) $(PEACGUILIB) $(PEACMAP) $(PEACGUIMAP)

clean-$(MODNAME):
		@rm -f $(PEACO) $(PEACDO) $(PEACGUIO) $(PEACGUIDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PEACDEP) $(PEACDS) $(PEACDH) $(PEACLIB) \
		   $(PEACGUIDEP) $(PEACGUIDS) $(PEACGUIDH) $(PEACGUILIB) \
		   $(PEACMAP) $(PEACGUIMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PEACO):       CXXFLAGS += $(CLARENSINC)
