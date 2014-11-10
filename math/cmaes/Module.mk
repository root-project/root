# Module.mk for cmaes module.

MODNAME       := cmaes
MODDIR        := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

CMAESDIR    := $(MODDIR)
CMAESDIRS   := $(CMAESDIR)/src
CMAESDIRI   := $(CMAESDIR)/inc
#CMAESDIRT   :=$(call stripsrc,$(CMAESDIR)/test)

CMAESBASEVERS := cmaes-1_0_0
CMAESBASESRCS := $(MODDIRS)/$(CMAESBASEVERS).tar.gz
CMAESBASEDIRS := $(MODDIRS)/$(CMAESBASEVERS)
CMAESBASEDIRI := -I$(MODDIRS)/$(CMAESBASEVERS)
CMAESBASEETAG := $(MODDIRS)/headers.d

##### liblcg_cmaes #####
ifeq ($(PLATFORM),win32)
CMAESBASELIBA      := $(CMAESBASEDIRS)/libcmaesbase.lib
CMAESBASELIB       := $(LPATH)/libcmaesbase.lib
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
CMAESBASEBLD        = "DEBUG=1"
else
CMAESBASEBLD        = ""
endif
else
CMAESBASELIBA      := $(CMAESBASEDIRS)/src/.libs/liblcg_cmaes.a
CMAESBASELIB       := $(LPATH)/libcmaesbase.a
endif
CMAESBASEDEP       := $(CMAESBASELIB)

##### libcmaes #####
CMAESL     := $(MODDIRI)/LinkDef.h
CMAESDS    := $(call stripsrc,$(MODDIRS)/G__cmaes.cxx)
CMAESDO    := $(CMAESDS:.cxx=.o)
CMAESDH    := $(CMAESDS:.cxx=.h)

CMAESAH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CMAESBH    := $(filter-out $(MODDIRI)/cmaes/LinkDef%,$(wildcard $(MODDIRI)/cmaes/*.h))
CMAESH     := $(CMAESAH) $(CMAESBH)
CMAESS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CMAESO     := $(call stripsrc,$(CMAESS:.cxx=.o))

CMAESDEP   := $(CMAESO:.o=.d) $(CMAESDO:.o=.d)

CMAESLIB   := $(LPATH)/libcmaes_root.$(SOEXT)
CMAESMAP   := $(CMAESLIB:.$(SOEXT)=.rootmap)

# use this compiler option if want to optimize object allocation in cmaes
# NOTE: using this option one loses the thread safety.
# It is worth to use only for minimization of cheap (non CPU intensive) functions
#CXXFLAGS += -DMN_USE_STACK_ALLOC

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CMAESH))
ALLLIBS      += $(CMAESLIB)
ALLMAPS      += $(CMAESMAP)

# include all dependency files
INCLUDEFILES += $(CMAESDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/cmaes/%.h: $(CMAESDIRI)/cmaes/%.h
		@(if [ ! -d "include/cmaes" ]; then     \
		   mkdir -p include/cmaes;              \
		fi)
		cp $< $@

include/%.h:    $(CMAESDIRI)/%.h
		cp $< $@

$(CMAESLIB):  $(CMAESO) $(CMAESDO) $(ORDER_) $(MAINLIBS) $(CMAESLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libcmaes_root.$(SOEXT) $@ \
		"$(CMAESO) $(CMAESDO)" "-lcmaes -lMathCore" "$(CMAESLIBEXTRA)"

$(call pcmrule,CMAES)
	$(noop)

$(CMAESDS):   $(CMAESH) $(CMAESL) $(ROOTCLINGEXE) $(call pcmdep,CMAES)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,CMAES)  -I$(CMAESINCDIR) -I/usr/include/eigen3 -c $(CMAESH) $(CMAESL)

$(CMAESMAP):  $(CMAESH) $(CMAESL) $(ROOTCLINGEXE) $(call pcmdep,CMAES)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(CMAESDS) $(call dictModule,CMAES) -c $(CMAESH) $(CMAESL)

all-$(MODNAME): $(CMAESLIB)

#test-$(MODNAME): all-$(MODNAME)
#ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
#		@$(INSTALL) $(CMAESDIR)/test $(CMAESDIRT)
#endif
		@cd $(CMAESDIRT) && $(MAKE) ROOTCONFIG=../../../bin/root-config

clean-$(MODNAME):
		@rm -f $(CMAESO) $(CMAESDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CMAESDEP) $(CMAESDS) $(CMAESDH) $(CMAESLIB) \
		   $(CMAESMAP)
		@rm -rf include/cmaes
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(CMAESDIRT)
else
		@cd $(CMAESDIRT) && $(MAKE) distclean ROOTCONFIG=../../../bin/root-config
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(CMAESO): CXXFLAGS += -DWARNINGMSG -DUSE_ROOT_ERROR
$(CMAESDO): CXXFLAGS += -DWARNINGMSG -DUSE_ROOT_ERROR
#for thread -safet
#$(CMAESO): CXXFLAGS += -DCMAES_THREAD_SAFE
# for openMP 
#ifneq ($(USE_PARALLEL_CMAES),)
#ifneq ($(USE_OPENMP),)
#$(CMAESO): CXXFLAGS += -DCMAES_THREAD_SAFE -DCMAES_PARALLEL_OPENMP
#math/cmaes/src/Numerical2PGradientCalculator.o: 
$(CMAESO):CXXFLAGS +=  -D_GLIBCXX_PARALLEL -fopenmp
$(CMAESDO):CXXFLAGS +=  -D_GLIBCXX_PARALLEL -fopenmp 
$(CMAESLIB):LDFLAGS += -fopenmp
#endif

$(CMAESO):CXXFLAGS += -I$(CMAESINCDIR) -I/usr/include/eigen3
$(CMAESDO):CXXFLAGS += -I$(CMAESINCDIR) -I/usr/include/eigen3
$(CMAESDS):CXXFLAGS += -I$(CMAESINCDIR) -I/usr/include/eigen3
$(CMAESLIB):LDFLAGS += $(CMAESLIBDIR)
# Optimize dictionary with stl containers.
$(CMAESDO): NOOPT = $(OPT)
