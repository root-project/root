# Module.mk for gui module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := gui
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GUIDIR       := $(MODDIR)
GUIDIRS      := $(GUIDIR)/src
GUIDIRI      := $(GUIDIR)/inc

##### libGui #####
GUIL1        := $(MODDIRI)/LinkDef1.h
GUIL2        := $(MODDIRI)/LinkDef2.h
GUIL3        := $(MODDIRI)/LinkDef3.h
GUIDS1       := $(MODDIRS)/G__Gui1.cxx
GUIDS2       := $(MODDIRS)/G__Gui2.cxx
GUIDS3       := $(MODDIRS)/G__Gui3.cxx
GUIDO1       := $(GUIDS1:.cxx=.o)
GUIDO2       := $(GUIDS2:.cxx=.o)
GUIDO3       := $(GUIDS3:.cxx=.o)
GUIDS        := $(GUIDS1) $(GUIDS2) $(GUIDS3)
GUIDO        := $(GUIDO1) $(GUIDO2) $(GUIDO3)
GUIDH        := $(GUIDS:.cxx=.h)

GUIH1        := TGObject.h TGClient.h TGWindow.h TGPicture.h TGDimension.h \
                TGFrame.h TGLayout.h TGString.h TGWidget.h TGIcon.h TGLabel.h \
                TGButton.h TGTextBuffer.h TGTextEntry.h TGMsgBox.h TGMenu.h \
                TGGC.h TGShutter.h TG3DLine.h TGProgressBar.h TGButtonGroup.h \
                TGNumberEntry.h TGTableLayout.h WidgetMessageTypes.h \
                TGIdleHandler.h

GUIH2        := TGObject.h TGScrollBar.h TGCanvas.h TGListBox.h TGComboBox.h \
                TGTab.h TGSlider.h TGPicture.h TGListView.h TGMimeTypes.h \
                TGFSContainer.h TGFileDialog.h TGStatusBar.h TGToolTip.h \
                TGToolBar.h TGListTree.h TGText.h TGView.h TGTextView.h \
                TGTextEdit.h TGTextEditDialogs.h TGDoubleSlider.h TGSplitter.h \
                TGFSComboBox.h TGImageMap.h TGApplication.h TGXYLayout.h \
                TGResourcePool.h TGFont.h
GUIH3        := TRootGuiFactory.h TRootApplication.h TRootCanvas.h \
                TRootBrowser.h TRootContextMenu.h TRootDialog.h \
                TRootControlBar.h TRootHelpDialog.h TRootEmbeddedCanvas.h \
                TGColorDialog.h TGColorSelect.h TGFontDialog.h \
                TGDockableFrame.h TGMdi.h TGMdiFrame.h TGMdiMainFrame.h \
                TGMdiDecorFrame.h TGMdiMenu.h TVirtualDragManager.h TGuiBuilder.h

GUIH4        := HelpText.h
GUIH1        := $(patsubst %,$(MODDIRI)/%,$(GUIH1))
GUIH2        := $(patsubst %,$(MODDIRI)/%,$(GUIH2))
GUIH3        := $(patsubst %,$(MODDIRI)/%,$(GUIH3))
GUIH4        := $(patsubst %,$(MODDIRI)/%,$(GUIH4))
GUIH         := $(GUIH1) $(GUIH2) $(GUIH3) $(GUIH4)
GUIS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GUIO         := $(GUIS:.cxx=.o)

GUIDEP       := $(GUIO:.o=.d) $(GUIDO:.o=.d)

GUILIB       := $(LPATH)/libGui.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GUIH))
ALLLIBS     += $(GUILIB)

# include all dependency files
INCLUDEFILES += $(GUIDEP)

##### local rules #####
include/%.h:    $(GUIDIRI)/%.h
		cp $< $@

$(GUILIB):      $(GUIO) $(GUIDO) $(MAINLIBS) $(GUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGui.$(SOEXT) $@ "$(GUIO) $(GUIDO)" \
		   "$(GUILIBEXTRA)"

$(GUIDS1):      $(GUIH1) $(GUIL1) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GUIH1) $(GUIL1)
$(GUIDS2):      $(GUIH2) $(GUIL2) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GUIH2) $(GUIL2)
$(GUIDS3):      $(GUIH3) $(GUIL3) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GUIH3) $(GUIL3)

$(GUIDO1):      $(GUIDS1)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<
$(GUIDO2):      $(GUIDS2)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<
$(GUIDO3):      $(GUIDS3)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-gui:        $(GUILIB)

map-gui:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GUILIB) \
		   -d $(GUILIBDEP) -c $(GUIL1) $(GUIL2) $(GUIL3)

map::           map-gui

clean-gui:
		@rm -f $(GUIO) $(GUIDO)

clean::         clean-gui

distclean-gui:  clean-gui
		@rm -f $(GUIDEP) $(GUIDS) $(GUIDH) $(GUILIB)

distclean::     distclean-gui
