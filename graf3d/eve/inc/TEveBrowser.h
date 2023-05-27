// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveBrowser
#define ROOT_TEveBrowser

#include "TEveElement.h"

#include "TContextMenu.h"
#include "TGListTree.h"
#include "TRootBrowser.h"
#include "TString.h"


class TGFileBrowser;
class TGSplitter;

class TEveGedEditor;


class TEveListTreeItem : public TGListTreeItem
{
private:
   TEveListTreeItem(const TEveListTreeItem&);             // not implemented
   TEveListTreeItem& operator=(const TEveListTreeItem&);  // not implemented

protected:
   TEveElement* fElement;

   void NotSupported(const char* func) const;

public:
   TEveListTreeItem(TEveElement* el) : TGListTreeItem(), fElement(el) {}
   ~TEveListTreeItem() override {}

   Bool_t          IsActive()       const override { return fElement->GetSelectedLevel() != 0; }
   Pixel_t         GetActiveColor() const override;
   void            SetActive(Bool_t) override      { NotSupported("SetActive"); }

   const char     *GetText()          const override { return fElement->GetElementName(); }
   Int_t           GetTextLength()    const override { return strlen(fElement->GetElementName()); }
   const char     *GetTipText()       const override { return fElement->GetElementTitle(); }
   Int_t           GetTipTextLength() const override { return strlen(fElement->GetElementTitle()); }
   void            SetText(const char *) override    { NotSupported("SetText"); }
   void            SetTipText(const char *) override { NotSupported("SetTipText"); }

   void            SetUserData(void *, Bool_t=kFALSE) override { NotSupported("SetUserData"); }
   void           *GetUserData() const override { return fElement; }

   const TGPicture*GetPicture()         const override { return fElement->GetListTreeIcon(fOpen); }
   const TGPicture*GetCheckBoxPicture() const override { return fElement->GetListTreeCheckBoxIcon(); }

   void            SetPictures(const TGPicture*, const TGPicture*) override { NotSupported("SetUserData"); }
   void            SetCheckBoxPictures(const TGPicture*, const TGPicture*) override { NotSupported("SetUserData"); }

   void            SetCheckBox(Bool_t=kTRUE) override { NotSupported("SetCheckBox"); }
   Bool_t          HasCheckBox()       const override { return kTRUE; }
   void            CheckItem(Bool_t=kTRUE) override   { printf("TEveListTreeItem::CheckItem - to be ignored ... all done via signal Checked().\n"); }
   void            Toggle() override;
   Bool_t          IsChecked()         const override { return fElement->GetRnrState(); }

   // Propagation of checked-state form children to parents. Not needed, ignore.

   // Item coloration (underline + minibox)
   Bool_t          HasColor()  const override { return fElement->HasMainColor(); }
   Color_t         GetColor()  const override { return fElement->GetMainColor(); }
   void            SetColor(Color_t) override { NotSupported("SetColor"); }
   void            ClearColor() override      { NotSupported("ClearColor"); }

   ClassDefOverride(TEveListTreeItem,0); // Special llist-tree-item for Eve.
};


class TEveGListTreeEditorFrame : public TGMainFrame
{
   TEveGListTreeEditorFrame(const TEveGListTreeEditorFrame&);            // Not implemented
   TEveGListTreeEditorFrame& operator=(const TEveGListTreeEditorFrame&); // Not implemented

   friend class TEveManager;

protected:
   TGCompositeFrame *fFrame;
   TGCompositeFrame *fLTFrame;

   TGCanvas         *fLTCanvas;
   TGListTree       *fListTree;
   TGSplitter       *fSplitter;
   TEveGedEditor    *fEditor;

   TContextMenu     *fCtxMenu;

   Bool_t            fSignalsConnected;

   static TString    fgEditorClass;

public:
   TEveGListTreeEditorFrame(const TGWindow *p = nullptr, Int_t width=250, Int_t height=700);
   ~TEveGListTreeEditorFrame() override;

   void ConnectSignals();
   void DisconnectSignals();

   void ReconfToHorizontal();
   void ReconfToVertical();

   TGListTree*    GetListTree() const { return fListTree; }
   TEveGedEditor* GetEditor()   const { return fEditor; }

   void ItemBelowMouse(TGListTreeItem *entry, UInt_t mask);
   void ItemClicked(TGListTreeItem *entry, Int_t btn, UInt_t mask, Int_t x, Int_t y);
   void ItemDblClicked(TGListTreeItem* item, Int_t btn);
   void ItemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);

   static void SetEditorClass(const char* edclass);

   ClassDefOverride(TEveGListTreeEditorFrame, 0); // Composite GUI frame for parallel display of a TGListTree and TEveGedEditor.
};

// ----------------------------------------------------------------

class TEveBrowser : public TRootBrowser
{
   TEveBrowser(const TEveBrowser&);            // Not implemented
   TEveBrowser& operator=(const TEveBrowser&); // Not implemented

protected:
   void SetupCintExport(TClass* cl);
   void CalculateReparentXY(TGObject* parent, Int_t& x, Int_t& y);

   TGFileBrowser    *fFileBrowser;
   TGPopupMenu      *fEvePopup;
   TGPopupMenu      *fSelPopup;
   TGPopupMenu      *fHilPopup;

public:
   TEveBrowser(UInt_t w, UInt_t h);
   ~TEveBrowser() override { CloseTabs(); }

   void ReallyDelete() override;
   void CloseTab(Int_t id) override;
   void CloseWindow() override;

   void InitPlugins(Option_t *opt="FI");

   TGFileBrowser* MakeFileBrowser(Bool_t make_default=kFALSE);
   TGFileBrowser* GetFileBrowser() const;
   void           SetFileBrowser(TGFileBrowser* b);

   void EveMenu(Int_t id);

   // Some getters missing in TRootBrowser
   TGMenuBar*         GetMenuBar()      const { return fMenuBar; }
   TGHorizontalFrame* GetTopMenuFrame() const { return fTopMenuFrame; }

   void HideBottomTab();

   void SanitizeTabCounts();

   ClassDefOverride(TEveBrowser, 0); // Specialization of TRootBrowser for Eve.
};

#endif
