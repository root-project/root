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

#include "TRootBrowser.h"
#include "TGListTree.h"

#include "TContextMenu.h"

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
   virtual ~TEveListTreeItem() {}

   virtual Bool_t          IsActive()       const { return fElement->GetSelectedLevel() != 0; }
   virtual Pixel_t         GetActiveColor() const;
   virtual void            SetActive(Bool_t)      { NotSupported("SetActive"); }

   virtual const char     *GetText()          const { return fElement->GetElementName(); }
   virtual Int_t           GetTextLength()    const { return strlen(fElement->GetElementName()); }
   virtual const char     *GetTipText()       const { return fElement->GetElementTitle(); }
   virtual Int_t           GetTipTextLength() const { return strlen(fElement->GetElementTitle()); }
   virtual void            SetText(const char *)    { NotSupported("SetText"); }
   virtual void            SetTipText(const char *) { NotSupported("SetTipText"); }

   virtual void            SetUserData(void *, Bool_t=kFALSE) { NotSupported("SetUserData"); }
   virtual void           *GetUserData() const { return fElement; }

   virtual const TGPicture*GetPicture()         const { return fElement->GetListTreeIcon(fOpen); }
   virtual const TGPicture*GetCheckBoxPicture() const { return fElement->GetListTreeCheckBoxIcon(); }

   virtual void            SetPictures(const TGPicture*, const TGPicture*) { NotSupported("SetUserData"); }
   virtual void            SetCheckBoxPictures(const TGPicture*, const TGPicture*) { NotSupported("SetUserData"); }

   virtual void            SetCheckBox(Bool_t=kTRUE) { NotSupported("SetCheckBox"); }
   virtual Bool_t          HasCheckBox()       const { return kTRUE; }
   virtual void            CheckItem(Bool_t=kTRUE)   { printf("TEveListTreeItem::CheckItem - to be ignored ... all done via signal Checked().\n"); }
   virtual void            Toggle();
   virtual Bool_t          IsChecked()         const { return fElement->GetRnrState(); }

   // Propagation of checked-state form children to parents. Not needed, ignore.

   // Item coloration (underline + minibox)
   virtual Bool_t          HasColor()  const { return fElement->HasMainColor(); }
   virtual Color_t         GetColor()  const { return fElement->GetMainColor(); }
   virtual void            SetColor(Color_t) { NotSupported("SetColor"); }
   virtual void            ClearColor()      { NotSupported("ClearColor"); }

   ClassDef(TEveListTreeItem,0); // Special llist-tree-item for Eve.
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
   TEveGListTreeEditorFrame(const TGWindow* p=0, Int_t width=250, Int_t height=700);
   virtual ~TEveGListTreeEditorFrame();

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

   ClassDef(TEveGListTreeEditorFrame, 0); // Composite GUI frame for parallel display of a TGListTree and TEveGedEditor.
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
   virtual ~TEveBrowser() {}

   virtual void ReallyDelete();
   virtual void CloseTab(Int_t id);
   virtual void CloseWindow();

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

   ClassDef(TEveBrowser, 0); // Specialization of TRootBrowser for Eve.
};

#endif
