# Module.mk for quartz module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Olivier Couet, 24/01/2012

QUARTZNDEBUG := -DNDEBUG
ifeq ($(ROOTBUILD),debug)
   QUARTZNDEBUG :=
endif

MODNAME      := quartz
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QUARTZDIR    := $(MODDIR)
QUARTZDIRS   := $(QUARTZDIR)/src
QUARTZDIRI   := $(QUARTZDIR)/inc

##### libGQuartz #####
QUARTZL      := $(MODDIRI)/LinkDef.h
QUARTZDS     := $(call stripsrc,$(MODDIRS)/G__Quartz.cxx)
QUARTZDO     := $(QUARTZDS:.cxx=.o)
QUARTZDH     := $(QUARTZDS:.cxx=.h)

QUARTZH1     := $(wildcard $(MODDIRI)/T*.h)
QUARTZH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QUARTZS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
QUARTZO      := $(call stripsrc,$(QUARTZS:.cxx=.o))
QUARTZOBJCPPS := $(wildcard $(MODDIRS)/*.mm)
QUARTZOBJCPPO := $(call stripsrc,$(QUARTZOBJCPPS:.mm=.o))

QUARTZDEP    := $(QUARTZO:.o=.d) $(QUARTZDO:.o=.d) $(QUARTZOBJCPPO:.o=.d)

QUARTZLIB    := $(LPATH)/libGQuartz.$(SOEXT)
QUARTZMAP    := $(QUARTZLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
QUARTZH_REL  := $(patsubst $(MODDIRI)/%.h,include/%.h,$(QUARTZH))
ALLHDRS      += $(QUARTZH_REL)
ALLLIBS      += $(QUARTZLIB)
ALLMAPS      += $(QUARTZMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(QUARTZH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Qt_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += requires objc \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(QUARTZLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif


# include all dependency files
INCLUDEFILES += $(QUARTZDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(QUARTZDIRI)/%.h
		cp $< $@

$(QUARTZLIB):   $(QUARTZO) $(QUARTZOBJCPPO) $(QUARTZDO) $(ORDER_) $(MAINLIBS) \
                $(QUARTZLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libGQuartz.$(SOEXT) $@ \
		   "$(QUARTZO) $(QUARTZOBJCPPO) $(QUARTZDO)" \
		   "$(QUARTZLIBEXTRA) -framework Cocoa"

$(call pcmrule,QUARTZ)
	$(noop)

$(QUARTZDS):    $(QUARTZH1) $(QUARTZL) $(ROOTCLINGEXE) $(call pcmdep,QUARTZ)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,QUARTZ) -c $(QUARTZH1) $(QUARTZL)

$(QUARTZMAP):   $(QUARTZH1) $(QUARTZL) $(ROOTCLINGEXE) $(call pcmdep,QUARTZ)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(QUARTZDS) $(call dictModule,QUARTZ) -c $(QUARTZH1) $(QUARTZL)

all-$(MODNAME): $(QUARTZLIB)

clean-$(MODNAME):
		@rm -f $(QUARTZO) $(QUARTZDO) $(QUARTZOBJCPPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(QUARTZDEP) $(QUARTZDS) $(QUARTZDH) $(QUARTZLIB) \
		   $(QUARTZMAP)

distclean::     distclean-$(MODNAME)

$(QUARTZOBJCPPO): OBJCXXFLAGS += $(CXXFLAGS) $(QUARTZNDEBUG)
