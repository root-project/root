# Module.mk for R interface module
# Copyright (c) 2013 Omar Andres Zapata Mesa
#
# Author: Omar Zapata, 30/5/2013
# updated Apr 29 2014

MODNAME      := r
MODDIR       := $(ROOT_SRCDIR)/bindings/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

RDIR  := $(MODDIR)
RDIRS := $(RDIR)/src
RDIRI := $(RDIR)/inc
RDIRP := $(RDIR)/pkg

##### libRInterface #####
RL           := $(MODDIRI)/LinkDef.h

RDS          := $(call stripsrc,$(MODDIRS)/G__RInterface.cxx)

RDO          := $(RDS:.cxx=.o)

#RDH          := $(RDS:.cxx=.h)

RDH          := $(MODDIRI)/RExports.h \
                $(MODDIRI)/TRInterface.h \
                $(MODDIRI)/TRInterface_Binding.h \
                $(MODDIRI)/TRObject.h \
                $(MODDIRI)/TRDataFrame.h \
                $(MODDIRI)/TRDataFrame__ctors.h \
                $(MODDIRI)/TRFunctionImport.h \
                $(MODDIRI)/TRFunctionImport__oprtr.h \
                $(MODDIRI)/TRFunctionExport.h \
                $(MODDIRI)/TRInternalFunction.h \
                $(MODDIRI)/TRInternalFunction__ctors.h 



RH    := $(RDH)  
RS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
RO    := $(call stripsrc,$(RS:.cxx=.o))


RDEP  := $(RO:.o=.d) $(RDO:.o=.d)

RLIB  := $(LPATH)/libRInterface.$(SOEXT)
RMAP  := $(RLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
RH_REL       := $(patsubst $(MODDIRI)/%.h,include/%.h,$(RH))
ALLHDRS      += $(RH_REL)
ALLLIBS      += $(RLIB)
ALLMAPS      += $(RMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(RH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module $(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(RLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(RDEP)


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/%.h:    $(RDIRI)/%.h
		cp $< $@

$(RLIB): $(RO) $(RDO) $(ORDER_) $(MAINLIBS) $(RLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libRInterface.$(SOEXT) $@     \
		   "$(RO) $(RDO)" \
		   "$(RLIBEXTRA) $(RLIBS)"

$(call pcmrule,R)
	$(noop)

$(RDS): $(RDH) $(RL)  $(ROOTCINTTMPDEP) $(call pcmdep,R)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@  $(call dictModule,R)  -c $(RDH) $(RFLAGS) $(RL)

$(RMAP): $(RL) $(RLINC) $(ROOTCINTTMPDEP) $(call pcmdep,R)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(RDS)  $(call dictModule,R) -c $(ROOT_SRCDIR:%=-I%) $(RFLAGS) $(RDH) $(RL)


all-$(MODNAME): $(RLIB)

clean-$(MODNAME):
		@rm -f $(RDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(RDEP) $(RDS) \
		   $(RLIB) $(RMAP)

distclean::     distclean-$(MODNAME)

#test-$(MODNAME): all-$(MODNAME)

##### extra rules ######
$(RO): CXXFLAGS += $(RFLAGS) -DUSE_ROOT_ERROR
$(RDO): CXXFLAGS += $(RFLAGS) -DUSE_ROOT_ERROR 
# add optimization to G__RInterface compilation
# Optimize dictionary with stl containers.
$(RDO) : NOOPT = $(OPT)
