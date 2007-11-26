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

#include "TRootBrowser.h"
#include "TGListTree.h"

#include "TContextMenu.h"

class TGFileBrowser;
class TGSplitter;

class TEveGedEditor;

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

  TGListTreeItem   *fNewSelected;

  void ResetSelectedTimer(TGListTreeItem* lti);

public:
  TEveGListTreeEditorFrame(const Text_t* name, Int_t width=250, Int_t height=700);
  virtual ~TEveGListTreeEditorFrame();

  void ReconfToHorizontal();
  void ReconfToVertical();

  TGListTree* GetListTree() { return fListTree; }

  void ItemChecked(TObject* obj, Bool_t state);
  void ItemClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);
  void ItemDblClicked(TGListTreeItem* item, Int_t btn);
  void ItemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);

  void ResetSelected();

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
  TGPopupMenu      *fRevePopup;

 public:
  TEveBrowser(UInt_t w, UInt_t h);
  virtual ~TEveBrowser() {}

  void InitPlugins();

  TGFileBrowser* MakeFileBrowser();
  TGFileBrowser* GetFileBrowser() const { return fFileBrowser; }

  void ReveMenu(Int_t id);

  ClassDef(TEveBrowser, 0); // Specialization of TRootBrowser for Reve.
};

#endif
