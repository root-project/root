# Module.mk for castor module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := castor
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CASTORDIR    := $(MODDIR)
CASTORDIRS   := $(CASTORDIR)/src
CASTORDIRI   := $(CASTORDIR)/inc

##### libRCastor #####
CASTORL      := $(MODDIRI)/LinkDef.h
CASTORDS     := $(call stripsrc,$(MODDIRS)/G__CASTOR.cxx)
CASTORDO     := $(CASTORDS:.cxx=.o)
CASTORDH     := $(CASTORDS:.cxx=.h)

CASTORH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CASTORS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CASTORO      := $(call stripsrc,$(CASTORS:.cxx=.o))

CASTORDEP    := $(CASTORO:.o=.d) $(CASTORDO:.o=.d)

CASTORLIB    := $(LPATH)/libRCastor.$(SOEXT)
CASTORMAP    := $(CASTORLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CASTORH))
ALLLIBS     += $(CASTORLIB)
ALLMAPS     += $(CASTORMAP)

# include all dependency files
INCLUDEFILES += $(CASTORDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CASTORDIRI)/%.h
		cp $< $@

$(CASTORLIB):   $(CASTORO) $(CASTORDO) $(ORDER_) $(MAINLIBS) $(CASTORLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRCastor.$(SOEXT) $@ \
		   "$(CASTORO) $(CASTORDO)" \
		   "$(CASTORLIBEXTRA) $(CASTORLIBDIR) $(CASTORCLILIB)"

$(call pcmrule,CASTOR)
	$(noop)

$(CASTORDS):    $(CASTORH) $(CASTORL) $(ROOTCLINGEXE) $(call pcmdep,CASTOR)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,CASTOR) -c $(CASTORH) $(CASTORL)

$(CASTORMAP):   $(CASTORH) $(CASTORL) $(ROOTCLINGEXE) $(call pcmdep,CASTOR)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(CASTORDS) $(call dictModule,CASTOR) -c $(CASTORH) $(CASTORL)

all-$(MODNAME): $(CASTORLIB)

clean-$(MODNAME):
		@rm -f $(CASTORO) $(CASTORDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CASTORDEP) $(CASTORDS) $(CASTORDH) $(CASTORLIB) $(CASTORMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(PLATFORM),win32)
$(CASTORO): CXXFLAGS += $(CASTORCFLAGS) $(CASTORINCDIR:%=-I%) -DNOGDI -D__INSIDE_CYGWIN__
ifneq (yes,$(WINRTDEBUG))
$(CASTORLIB): LDFLAGS += -nodefaultlib:msvcrtd.lib 
endif
else
$(CASTORO): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))
$(CASTORO): CXXFLAGS += $(CASTORCFLAGS) $(CASTORINCDIR:%=-I%)
endif
