# Module.mk for minuit2 module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 07/05/2003

MODNAME       := minuit2
MODDIR        := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS       := $(MODDIR)/src
MODDIRI       := $(MODDIR)/inc

MINUIT2DIR    := $(MODDIR)
MINUIT2DIRS   := $(MINUIT2DIR)/src
MINUIT2DIRI   := $(MINUIT2DIR)/inc
MINUIT2DIRT   := $(call stripsrc,$(MINUIT2DIR)/test)

MINUITBASEVERS := Minuit-1_7_6
MINUITBASESRCS := $(MODDIRS)/$(MINUITBASEVERS).tar.gz
MINUITBASEDIRS := $(MODDIRS)/$(MINUITBASEVERS)
MINUITBASEDIRI := -I$(MODDIRS)/$(MINUITBASEVERS)
MINUITBASEETAG := $(MODDIRS)/headers.d

##### liblcg_Minuit #####
ifeq ($(PLATFORM),win32)
MINUITBASELIBA      := $(MINUITBASEDIRS)/libminuitbase.lib
MINUITBASELIB       := $(LPATH)/libminuitbase.lib
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
MINUITBASEBLD        = "DEBUG=1"
else
MINUITBASEBLD        = ""
endif
else
MINUITBASELIBA      := $(MINUITBASEDIRS)/src/.libs/liblcg_Minuit.a
MINUITBASELIB       := $(LPATH)/libminuitbase.a
endif
MINUITBASEDEP       := $(MINUITBASELIB)

##### libMinuit2 #####
MINUIT2L     := $(MODDIRI)/LinkDef.h
MINUIT2DS    := $(call stripsrc,$(MODDIRS)/G__Minuit2.cxx)
MINUIT2DO    := $(MINUIT2DS:.cxx=.o)
MINUIT2DH    := $(MINUIT2DS:.cxx=.h)

MINUIT2AH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MINUIT2BH    := $(filter-out $(MODDIRI)/Minuit2/LinkDef%,$(wildcard $(MODDIRI)/Minuit2/*.h))
MINUIT2H     := $(MINUIT2AH) $(MINUIT2BH)
MINUIT2S     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MINUIT2O     := $(call stripsrc,$(MINUIT2S:.cxx=.o))

MINUIT2DEP   := $(MINUIT2O:.o=.d) $(MINUIT2DO:.o=.d)

MINUIT2LIB   := $(LPATH)/libMinuit2.$(SOEXT)
MINUIT2MAP   := $(MINUIT2LIB:.$(SOEXT)=.rootmap)

# use this compiler option if want to optimize object allocation in Minuit2
# NOTE: using this option one loses the thread safety.
# It is worth to use only for minimization of cheap (non CPU intensive) functions
#CXXFLAGS += -DMN_USE_STACK_ALLOC

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MINUIT2H))
ALLLIBS      += $(MINUIT2LIB)
ALLMAPS      += $(MINUIT2MAP)

# include all dependency files
INCLUDEFILES += $(MINUIT2DEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/Minuit2/%.h: $(MINUIT2DIRI)/Minuit2/%.h
		@(if [ ! -d "include/Minuit2" ]; then     \
		   mkdir -p include/Minuit2;              \
		fi)
		cp $< $@

include/%.h:    $(MINUIT2DIRI)/%.h
		cp $< $@

$(MINUIT2LIB):  $(MINUIT2O) $(MINUIT2DO) $(ORDER_) $(MAINLIBS) $(MINUIT2LIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMinuit2.$(SOEXT) $@ \
		   "$(MINUIT2O) $(MINUIT2DO)" \
		   "$(MINUIT2LIBEXTRA)"

$(MINUIT2DS):   $(MINUIT2H) $(MINUIT2L) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MINUIT2H) $(MINUIT2L)

$(MINUIT2MAP):  $(RLIBMAP) $(MAKEFILEDEP) $(MINUIT2L)
		$(RLIBMAP) -o $@ -l $(MINUIT2LIB) \
		   -d $(MINUIT2LIBDEPM) -c $(MINUIT2L)

all-$(MODNAME):  $(MINUIT2LIB) $(MINUIT2MAP)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(MINUIT2DIR)/test $(MINUIT2DIRT)
endif
		@cd $(MINUIT2DIRT) && $(MAKE) ROOTCONFIG=../../../bin/root-config

clean-$(MODNAME):
		@rm -f $(MINUIT2O) $(MINUIT2DO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MINUIT2DEP) $(MINUIT2DS) $(MINUIT2DH) $(MINUIT2LIB) \
		   $(MINUIT2MAP)
		@rm -rf include/Minuit2
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(MINUIT2DIRT)
else
		@cd $(MINUIT2DIRT) && $(MAKE) distclean ROOTCONFIG=../../../bin/root-config
endif

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(MINUIT2O): CXXFLAGS += -DWARNINGMSG -DUSE_ROOT_ERROR
$(MINUIT2DO): CXXFLAGS += -DWARNINGMSG -DUSE_ROOT_ERROR
#for thread -safet
#$(MINUIT2O): CXXFLAGS += -DMINUIT2_THREAD_SAFE
# for openMP 
ifneq ($(USE_PARALLEL_MINUIT2),)
ifneq ($(USE_OPENMP),)
#$(MINUIT2O): CXXFLAGS += -DMINUIT2_THREAD_SAFE -DMINUIT2_PARALLEL_OPENMP
#math/minuit2/src/Numerical2PGradientCalculator.o: 
$(MINUIT2O):CXXFLAGS +=  -D_GLIBCXX_PARALLEL -fopenmp 
$(MINUIT2DO):CXXFLAGS +=  -D_GLIBCXX_PARALLEL -fopenmp 
$(MINUIT2LIB):LDFLAGS += -fopenmp
endif
ifneq ($(USE_MPI),)
$(MINUIT2O): CXX=mpic++ -DMPIPROC
$(MINUIT2DO): CXX=mpic++ 
$(MINUIT2LIB): LD=mpic++
endif
endif

# Optimize dictionary with stl containers.
$(MINUIT2DO): NOOPT = $(OPT)
