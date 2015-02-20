# Module.mk for gui module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := gui
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GUIDIR       := $(MODDIR)
GUIDIRS      := $(GUIDIR)/src
GUIDIRI      := $(GUIDIR)/inc

##### libGui #####
GUIL0        := $(MODDIRI)/LinkDef.h
GUILS        := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h $(MODDIRI)/LinkDef3.h
GUIDS        := $(call stripsrc,$(MODDIRS)/G__Gui.cxx)
GUIDO        := $(GUIDS:.cxx=.o)
GUIDH        := $(GUIDS:.cxx=.h)

GUIH1        := TGObject.h TGClient.h TGWindow.h TGPicture.h TGDimension.h \
                TGFrame.h TGLayout.h TGString.h TGWidget.h TGIcon.h TGLabel.h \
                TGButton.h TGTextBuffer.h TGTextEntry.h TGMsgBox.h TGMenu.h \
                TGGC.h TGShutter.h TG3DLine.h TGProgressBar.h TGButtonGroup.h \
                TGNumberEntry.h TGTableLayout.h WidgetMessageTypes.h \
                TGIdleHandler.h TGInputDialog.h TGPack.h
GUIH2        := TGObject.h TGScrollBar.h TGCanvas.h TGListBox.h TGComboBox.h \
                TGTab.h TGSlider.h TGPicture.h TGListView.h TGMimeTypes.h \
                TGFSContainer.h TGFileDialog.h TGStatusBar.h TGToolTip.h \
                TGToolBar.h TGListTree.h TGText.h TGView.h TGTextView.h \
                TGTextEdit.h TGTextEditDialogs.h TGDoubleSlider.h TGSplitter.h \
                TGFSComboBox.h TGImageMap.h TGApplication.h TGXYLayout.h \
                TGResourcePool.h TGFont.h TGTripleSlider.h
GUIH3        := TRootGuiFactory.h TRootApplication.h TRootCanvas.h \
                TRootBrowserLite.h TRootContextMenu.h TRootDialog.h \
                TRootControlBar.h TRootHelpDialog.h TRootEmbeddedCanvas.h \
                TGColorDialog.h TGColorSelect.h TGFontDialog.h \
                TGDockableFrame.h TGMdi.h TGMdiFrame.h TGMdiMainFrame.h \
                TGMdiDecorFrame.h TGMdiMenu.h TVirtualDragManager.h \
                TGuiBuilder.h TGRedirectOutputGuard.h TGPasswdDialog.h \
                TGTextEditor.h TGSpeedo.h TGDNDManager.h TGTable.h \
                TGSimpleTableInterface.h TGSimpleTable.h TGTableCell.h \
		          TGTableHeader.h TGTableContainer.h \
		          TGCommandPlugin.h TGFileBrowser.h \
		          TRootBrowser.h TGSplitFrame.h TGShapedFrame.h TGEventHandler.h \
		          TGTextViewStream.h

GUIH4        := HelpText.h
GUIH1        := $(patsubst %,$(MODDIRI)/%,$(GUIH1))
GUIH2        := $(patsubst %,$(MODDIRI)/%,$(GUIH2))
GUIH3        := $(patsubst %,$(MODDIRI)/%,$(GUIH3))
GUIH4        := $(patsubst %,$(MODDIRI)/%,$(GUIH4))
GUIH         := $(GUIH1) $(GUIH2) $(GUIH3) $(GUIH4)
GUIS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GUIO         := $(call stripsrc,$(GUIS:.cxx=.o))

GUIDEP       := $(GUIO:.o=.d) $(GUIDO:.o=.d)

GUILIB       := $(LPATH)/libGui.$(SOEXT)
GUIMAP       := $(GUILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GUIH))
ALLLIBS     += $(GUILIB)
ALLMAPS     += $(GUIMAP)

# include all dependency files
INCLUDEFILES += $(GUIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GUIDIRI)/%.h
		cp $< $@

$(GUILIB):      $(GUIO) $(GUIDO) $(ORDER_) $(MAINLIBS) $(GUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGui.$(SOEXT) $@ "$(GUIO) $(GUIDO)" \
		   "$(GUILIBEXTRA)"

$(call pcmrule,GUI)
	$(noop)

$(GUIDS):       $(GUIH) $(GUIL0) $(GUILS) $(ROOTCLINGEXE) $(call pcmdep,GUI)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GUI) -c -writeEmptyRootPCM -I$(ROOT_SRCDIR) $(GUIH1) $(GUIH2) $(GUIH3) $(GUIL0)

$(GUIMAP):      $(GUIH) $(GUIL0) $(GUILS) $(ROOTCLINGEXE) $(call pcmdep,GUI)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GUIDS) $(call dictModule,GUI) -c -I$(ROOT_SRCDIR) $(GUIH1) $(GUIH2) $(GUIH3) $(GUIL0)

all-$(MODNAME): $(GUILIB)

clean-$(MODNAME):
		@rm -f $(GUIO) $(GUIDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GUIDEP) $(GUIDS) $(GUIDH) $(GUILIB) $(GUIMAP)

distclean::     distclean-$(MODNAME)
