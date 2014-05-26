# Module.mk for Vc module
# Generated on Tue Apr  3 17:31:31 CEST 2012 by Vc/makeRootRelease.sh

MODNAME      := vc
#VCVERS       := vc-0.6.70-root

MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/include
VCBUILDDIR   := $(call stripsrc,$(MODDIRS))

ifeq ($(PLATFORM),win32)
VCLIBVC      := $(LPATH)/libVc.lib
else
VCLIBVC      := $(LPATH)/libVc.a
endif

VCH          := $(wildcard $(MODDIRI)/Vc/* $(MODDIRI)/Vc/*/*)

ALLHDRS      += $(patsubst $(MODDIRI)/%,include/%,$(VCH))
ALLLIBS      += $(VCLIBVC)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/Vc/%: $(MODDIRI)/Vc/%
	@(if [ ! -d "include/Vc" ]; then    \
	   mkdir -p include/Vc;             \
	fi)
	cp -R $< $@

escapeflag = $(subst ~,_,$(subst /,_,$(subst :,_,$(subst =,_,$(subst .,_,$(subst -,_,$(1)))))))

VCFLAGS0     := -DVC_COMPILE_LIB $(filter-out -Wall,$(filter-out -x%,$(filter-out -m%,$(filter-out /arch:%,$(OPT) $(CXXFLAGS)))))
VCFLAGS      := $(VCFLAGS0) $(VCFLAGS)
VCLIBVCOBJ   := const.cpp cpuid.cpp support.cpp  trigonometric.cpp \
	 $(foreach flag,$(call escapeflag,$(SIMDCXXFLAGS)),trigonometric_$(flag).cpp)
ifdef AVXCXXFLAG
VCLIBVCOBJ   += avx_sorthelper.cpp
endif
VCLIBVCOBJ   := $(addprefix $(VCBUILDDIR)/,$(addsuffix .o,$(VCLIBVCOBJ)))

$(VCLIBVC): $(VCLIBVCOBJ)
	$(MAKEDIR)
	@echo "Create static library $@"
	@rm -f $@
ifeq ($(PLATFORM),win32)
	link.exe -lib /out:$@ $(VCLIBVCOBJ)
else
	@ar r $@ $?
	@ranlib $@
endif

$(VCBUILDDIR)/avx_%.cpp.o: $(MODDIRS)/avx_%.cpp
	$(MAKEDIR)
	$(CXX) $(VCFLAGS) $(AVXCXXFLAG) -c $(CXXOUT)$@ $<

$(VCBUILDDIR)/trigonometric_%.cpp.o: $(MODDIRS)/trigonometric.cpp
	$(MAKEDIR)
	@for flag in $(SIMDCXXFLAGS); do \
		flag=`echo $$flag|tr '~' ' '`; \
		if test "$*" = "`echo "$$flag"|tr ' /:=.-' '______'`"; then \
			echo "$(CXX) $(VCFLAGS) $$flag -c $(CXXOUT)$@ $<"; \
			$(CXX) $(VCFLAGS) $$flag -c $(CXXOUT)$@ $<; \
			break; \
		fi; \
	done

$(VCBUILDDIR)/%.cpp.o: $(MODDIRS)/%.cpp
	$(MAKEDIR)
	$(CXX) $(VCFLAGS) -c $(CXXOUT)$@ $<


all-$(MODNAME): $(VCLIBVC)

clean-$(MODNAME):
	@rm -f $(VCLIBVC) $(VCLIBVCOBJ)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -rf include/Vc

distclean:: distclean-$(MODNAME)

