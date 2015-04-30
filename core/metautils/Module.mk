# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Philippe Canal 9/1/2004

MODNAME        := metautils
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

METAUTILSDIR   := $(MODDIR)
METAUTILSDIRS  := $(METAUTILSDIR)/src
METAUTILSDIRI  := $(METAUTILSDIR)/inc

##### $(METAUTILSO) #####
METAUTILSH     := $(filter-out $(MODDIRI)/TMetaUtils.%,\
  $(filter-out $(MODDIRI)/libcpp_string_view.h,\
  $(filter-out $(MODDIRI)/RWrap_libcpp_string_view.h,\
  $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h)))))
METAUTILSS     := $(filter-out $(MODDIRS)/TMetaUtils.%,\
  $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))

METAUTILSTH     += $(MODDIRI)/TMetaUtils.h
METAUTILSTS     += $(MODDIRS)/TMetaUtils.cxx
METAUTILSTH     += $(MODDIRI)/root_std_complex.h
METAUTILSTH     += $(MODDIRI)/libcpp_string_view.h
METAUTILSTH     += $(MODDIRI)/RWrap_libcpp_string_view.h

METAUTILSSLLVM := $(MODDIRS)/BaseSelectionRule.cxx \
                  $(MODDIRS)/ClassSelectionRule.cxx \
                  $(MODDIRS)/VariableSelectionRule.cxx \
                  $(MODDIRS)/RStl.cxx \
                  $(MODDIRS)/Scanner.cxx \
                  $(MODDIRS)/SelectionRules.cxx \
                  $(MODDIRS)/XMLReader.cxx

METAUTILSCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METAUTILSCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif

METAUTILSO     := $(call stripsrc,$(METAUTILSS:.cxx=.o))
METAUTILSOLLVM := $(call stripsrc,$(METAUTILSSLLVM:.cxx=.o))
METAUTILSO     := $(filter-out $(METAUTILSOLLVM),$(METAUTILSO))
METAUTILSTO    := $(call stripsrc,$(METAUTILSTS:.cxx=.o))

METAUTILSL     := $(MODDIRI)/LinkDef.h

METAUTILSDEP   := $(METAUTILSO:.o=.d) $(METAUTILSTO:.o=.d) $(METAUTILSOLLVM:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAUTILSH) $(METAUTILSTH))

# include all dependency files
INCLUDEFILES += $(METAUTILSDEP)

#### STL dictionary (replacement for cintdlls)

STLDICTS =
STLDICTS += lib/libvectorDict.$(SOEXT)
STLDICTS += lib/liblistDict.$(SOEXT)
STLDICTS += lib/libforward_listDict.$(SOEXT)
STLDICTS += lib/libdequeDict.$(SOEXT)
STLDICTS += lib/libmapDict.$(SOEXT)
STLDICTS += lib/libmap2Dict.$(SOEXT)
STLDICTS += lib/libsetDict.$(SOEXT)
STLDICTS += lib/libunordered_setDict.$(SOEXT)
STLDICTS += lib/libunordered_multisetDict.$(SOEXT)
STLDICTS += lib/libunordered_mapDict.$(SOEXT)
STLDICTS += lib/libunordered_multimapDict.$(SOEXT)
STLDICTS += lib/libmultimapDict.$(SOEXT)
STLDICTS += lib/libmultimap2Dict.$(SOEXT)
STLDICTS += lib/libmultisetDict.$(SOEXT)
STLDICTS += lib/libcomplexDict.$(SOEXT)
ifneq ($(PLATFORM),win32)
STLDICTS += lib/libvalarrayDict.$(SOEXT)
endif

STLDICTS_SRC := $(call stripsrc,$(patsubst lib/lib%Dict.$(SOEXT),$(METAUTILSDIRS)/G__std__%.cxx,$(STLDICTS)))
STLDICTS_HDR := $(call stripsrc,$(patsubst lib/lib%Dict.$(SOEXT),$(METAUTILSDIRS)/G__std__%.h,$(STLDICTS)))
STLDICTS_OBJ := $(patsubst %.cxx,%.o,$(STLDICTS_SRC))
STLDICTS_DEP := $(patsubst %.cxx,%.d,$(STLDICTS_SRC))

$(call stripsrc,$(METAUTILSDIRS)/G__std__%.cxx): $(METAUTILSDIRS)/%Linkdef.h $(ROOTCLINGSTAGE1DEP)
	$(ROOTCLINGSTAGE1) -f $@ -s lib/lib$*Dict.pcm -m lib/libCore.pcm $(call rootmapModule, lib/lib$*Dict.$(SOEXT)) -c $(subst multi,,${*:2=}) $(ROOT_SRCDIR)/core/metautils/src/$*Linkdef.h

$(STLDICTS): lib/lib%Dict.$(SOEXT): $(call stripsrc,$(METAUTILSDIRS)/G__std__%.o) $(ORDER_) $(MAINLIBS)
	@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ "$(filter-out $(MAINLIBS),$^)" ""

lib/lib%Dict.rootmap: $(METAUTILSDIRS)/%Linkdef.h $(ROOTCLINGSTAGE1DEP)
	$(ROOTCLINGSTAGE1) -r $(METAUTILSDIRS)/G__std__%.cxx -s lib/lib$*Dict.pcm -m lib/libCore.pcm $(call rootmapModule, lib/lib$*Dict.$(SOEXT))\
	   -c $(subst multi,,${*:2=}) \
	   $(ROOT_SRCDIR)/core/metautils/src/$*Linkdef.h

METAUTILSDEP   += $(STLDICTS_DEP)

STLDICTSMAPS = $(STLDICTS:.$(SOEXT)=.rootmap)

.PRECIOUS: $(STLDICTS_SRC)

# used in the main Makefile
ALLLIBS    += $(STLDICTS)
ALLMAPS    += $(STLDICTSMAPS)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METAUTILSDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(METAUTILSO) $(STLDICTS)

clean-$(MODNAME):
		@rm -f $(METAUTILSO) $(STLDICTS_OBJ) \
		   $(STLDICTS_DEP)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METAUTILSDEP) \
		   $(STLDICTS_OBJ) $(STLDICTS_DEP) $(STLDICTS_SRC) \
		   $(STLDICTS_HDR) $(STLDICTSMAPS)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(METAUTILSOLLVM): CXXFLAGS += $(METAUTILSCXXFLAGS)
$(METAUTILSOLLVM): $(LLVMDEP)
$(METAUTILSTO): CXXFLAGS += $(METAUTILSCXXFLAGS)
$(METAUTILSTO): $(LLVMDEP)
