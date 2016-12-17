# Module.mk for treeviewer module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := treeviewer
MODDIR       := $(ROOT_SRCDIR)/tree/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEVIEWERDIR  := $(MODDIR)
TREEVIEWERDIRS := $(TREEVIEWERDIR)/src
TREEVIEWERDIRI := $(TREEVIEWERDIR)/inc

##### libTreeViewer #####
TREEVIEWERL  := $(MODDIRI)/LinkDef.h
TREEVIEWERDS := $(call stripsrc,$(MODDIRS)/G__TreeViewer.cxx)
TREEVIEWERDO := $(TREEVIEWERDS:.cxx=.o)
TREEVIEWERDH := $(TREEVIEWERDS:.cxx=.h)

#TREEVIEWERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
#TREEVIEWERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ifeq ($(ARCH),win32old)
TREEVIEWERL  := $(MODDIRI)/LinkDefWin32.h
TREEVIEWERH  := TTreeViewerOld.h TPaveVar.h
TREEVIEWERS  := TTreeViewerOld.cxx TPaveVar.cxx
else
TREEVIEWERH  := TTreeViewer.h TTVSession.h TTVLVContainer.h HelpTextTV.h TSpider.h TSpiderEditor.h TParallelCoord.h \
                TParallelCoordVar.h TParallelCoordRange.h TParallelCoordEditor.h TGTreeTable.h TMemStatShow.h
TREEVIEWERS  := TTreeViewer.cxx TTVSession.cxx TTVLVContainer.cxx HelpTextTV.cxx TSpider.cxx TSpiderEditor.cxx \
                TParallelCoord.cxx TParallelCoordVar.cxx TParallelCoordRange.cxx TParallelCoordEditor.cxx \
		TGTreeTable.cxx TMemStatShow.cxx
endif
TREEVIEWERH  := $(patsubst %,$(MODDIRI)/%,$(TREEVIEWERH))
TREEVIEWERS  := $(patsubst %,$(MODDIRS)/%,$(TREEVIEWERS))

TREEVIEWERO  := $(call stripsrc,$(TREEVIEWERS:.cxx=.o))

TREEVIEWERDEP := $(TREEVIEWERO:.o=.d) $(TREEVIEWERDO:.o=.d)

TREEVIEWERLIB := $(LPATH)/libTreeViewer.$(SOEXT)
TREEVIEWERMAP := $(TREEVIEWERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
TREEVIEWERH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEVIEWERH))
ALLHDRS       += $(TREEVIEWERH_REL)
ALLLIBS       += $(TREEVIEWERLIB)
ALLMAPS       += $(TREEVIEWERMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(TREEVIEWERH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Tree_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(TREEVIEWERLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(TREEVIEWERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TREEVIEWERDIRI)/%.h
		cp $< $@

$(TREEVIEWERLIB): $(TREEVIEWERO) $(TREEVIEWERDO) $(ORDER_) $(MAINLIBS) \
                  $(TREEVIEWERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTreeViewer.$(SOEXT) $@ \
		   "$(TREEVIEWERO) $(TREEVIEWERDO)" \
		   "$(TREEVIEWERLIBEXTRA)"

$(call pcmrule,TREEVIEWER)
	$(noop)

$(TREEVIEWERDS): $(TREEVIEWERH) $(TREEVIEWERL) $(ROOTCLINGEXE) $(call pcmdep,TREEVIEWER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TREEVIEWER) -c -writeEmptyRootPCM $(TREEVIEWERH) $(TREEVIEWERL)

$(TREEVIEWERMAP): $(TREEVIEWERH) $(TREEVIEWERL) $(ROOTCLINGEXE) $(call pcmdep,TREEVIEWER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(TREEVIEWERDS) $(call dictModule,TREEVIEWER) -c $(TREEVIEWERH) $(TREEVIEWERL)

all-$(MODNAME): $(TREEVIEWERLIB)

clean-$(MODNAME):
		@rm -f $(TREEVIEWERO) $(TREEVIEWERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TREEVIEWERDEP) $(TREEVIEWERDS) $(TREEVIEWERDH) \
		   $(TREEVIEWERLIB) $(TREEVIEWERMAP)

distclean::     distclean-$(MODNAME)
