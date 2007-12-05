// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBrowser.h"

#include "TEveUtil.h"
#include "TEveElement.h"
#include "TEveManager.h"
#include "TEveGedEditor.h"

#include "TGFileBrowser.h"
#include "TBrowser.h"

#include <Riostream.h>

#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TRint.h"
#include "TVirtualX.h"
#include "TEnv.h"

#include "TApplication.h"
#include "TFile.h"
#include "TClassMenuItem.h"

#include "TColor.h"

#include "TGCanvas.h"
#include "TGSplitter.h"
#include "TGStatusBar.h"
#include "TGMenu.h"
#include "TGPicture.h"
#include "TGToolBar.h"
#include "TGLabel.h"
#include "TGXYLayout.h"
#include "TGNumberEntry.h"
#include <KeySymbols.h>

#include "TGLSAViewer.h"
#include "TGLSAFrame.h"
#include "TGTab.h"

#include "TGeoVolume.h"
#include "TGeoNode.h"

//______________________________________________________________________________
// TEveGListTreeEditorFrame
//
// Composite GUI frame for parallel display of a TGListTree and TEveGedEditor.
//

ClassImp(TEveGListTreeEditorFrame)

//______________________________________________________________________________
TEveGListTreeEditorFrame::TEveGListTreeEditorFrame(const Text_t* name, Int_t width, Int_t height) :
   TGMainFrame(gClient->GetRoot(), width, height),
   fCtxMenu     (0),
   fNewSelected (0)
{
   SetWindowName(name);
   SetCleanup(kNoCleanup);

   fFrame = new TGCompositeFrame(this, width, height, kVerticalFrame);

   // List-tree
   fLTFrame  = new TGCompositeFrame(fFrame, width, 3*height/7, kVerticalFrame);
   fLTCanvas = new TGCanvas(fLTFrame, 10, 10, kSunkenFrame | kDoubleBorder);
   fListTree = new TGListTree(fLTCanvas->GetViewPort(), 10, 10, kHorizontalFrame);
   fListTree->SetCanvas(fLTCanvas);
   fListTree->Associate(fFrame);
   fListTree->SetColorMode(TGListTree::EColorMarkupMode(TGListTree::kColorUnderline | TGListTree::kColorBox));
   fListTree->SetAutoCheckBoxPic(kFALSE);
   fLTCanvas->SetContainer(fListTree);
   fLTFrame->AddFrame(fLTCanvas, new TGLayoutHints
                      (kLHintsNormal | kLHintsExpandX | kLHintsExpandY, 1, 1, 1, 1));
   fFrame  ->AddFrame(fLTFrame, new TGLayoutHints
                      (kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   // Splitter
   fSplitter = new TGHSplitter(fFrame);
   fFrame->AddFrame(fSplitter, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 1,1,2,2));

   // Editor
   fFrame->SetEditDisabled(kEditEnable);
   fFrame->SetEditable();
   fEditor = new TEveGedEditor(0, width, 4*height/7);
   fEditor->SetGlobal(kFALSE);
   fEditor->ChangeOptions(fEditor->GetOptions() | kFixedHeight);
   fFrame->SetEditable(kEditDisable);
   fFrame->SetEditable(kFALSE);
   {
      TGFrameElement *el = 0;
      TIter next(fFrame->GetList());
      while ((el = (TGFrameElement *) next())) {
         if (el->fFrame == fEditor)
            if (el->fLayout) {
               el->fLayout->SetLayoutHints(kLHintsTop | kLHintsExpandX);
               el->fLayout->SetPadLeft(0); el->fLayout->SetPadRight(1);
               el->fLayout->SetPadTop(2);  el->fLayout->SetPadBottom(1);
               break;
            }
      }
   }
   fSplitter->SetFrame(fEditor, kFALSE);

   AddFrame(fFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   fCtxMenu = new TContextMenu("", "");

   fListTree->Connect("Checked(TObject*,Bool_t)", "TEveGListTreeEditorFrame",
                      this, "ItemChecked(TObject*, Bool_t)");
   fListTree->Connect("Clicked(TGListTreeItem*, Int_t, Int_t, Int_t)", "TEveGListTreeEditorFrame",
                      this, "ItemClicked(TGListTreeItem*, Int_t, Int_t, Int_t)");
   fListTree->Connect("DoubleClicked(TGListTreeItem*, Int_t)", "TEveGListTreeEditorFrame",
                      this, "ItemDblClicked(TGListTreeItem*, Int_t)");
   fListTree->Connect("KeyPressed(TGListTreeItem*, ULong_t, ULong_t)", "TEveGListTreeEditorFrame",
                      this, "ItemKeyPress(TGListTreeItem*, UInt_t, UInt_t)");

   Layout();
   MapSubwindows();
   MapWindow();
}

//______________________________________________________________________________
TEveGListTreeEditorFrame::~TEveGListTreeEditorFrame()
{
   delete fCtxMenu;

   // Should un-register editor, all items and list-tree from gEve ... eventually.

   delete fEditor;
   delete fSplitter;
   delete fListTree;
   delete fLTCanvas;
   delete fLTFrame;
   delete fFrame;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ReconfToHorizontal()
{
   UnmapWindow();

   fFrame->ChangeOptions(kHorizontalFrame);
   fLTFrame->ChangeOptions(kHorizontalFrame);
   fListTree->ChangeOptions(kVerticalFrame);

   TGFrameElement *el = 0;
   TIter next(fFrame->GetList());
   while ((el = (TGFrameElement *) next()))
   {
      if (el->fFrame == fSplitter)
      {
         delete fSplitter;
         el->fFrame = fSplitter = new TGVSplitter(fFrame);
         el->fLayout->SetLayoutHints(kLHintsLeft | kLHintsExpandY);
         el->fLayout->SetPadLeft(2); el->fLayout->SetPadRight (2);
         el->fLayout->SetPadTop (1); el->fLayout->SetPadBottom(1);
      }
      else if (el->fFrame == fEditor)
      {
         fEditor->ChangeOptions(fEditor->GetOptions() & (~kFixedHeight));
         fEditor->ChangeOptions(fEditor->GetOptions() |   kFixedWidth);
         el->fLayout->SetLayoutHints(kLHintsLeft | kLHintsExpandY);
      }
   }

   fEditor->Resize(fEditor->GetWidth() / 2 - 1, fEditor->GetHeight());
   fSplitter->SetFrame(fEditor, kFALSE);

   Layout();
   //fFrame->Layout();
   //fLTFrame->Layout();
   //fLTCanvas->Layout();
   //fListTree->ClearViewPort();
   MapSubwindows();
   MapWindow();
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ReconfToVertical()
{
   UnmapWindow();

   fFrame->ChangeOptions(kVerticalFrame);
   fLTFrame->ChangeOptions(kVerticalFrame);
   fListTree->ChangeOptions(kHorizontalFrame);

   TGFrameElement *el = 0;
   TIter next(fFrame->GetList());
   while ((el = (TGFrameElement *) next()))
   {
      if (el->fFrame == fSplitter)
      {
         delete fSplitter;
         el->fFrame = fSplitter = new TGHSplitter(fFrame);
         el->fLayout->SetLayoutHints(kLHintsTop | kLHintsExpandX);
         el->fLayout->SetPadLeft(2); el->fLayout->SetPadRight (2);
         el->fLayout->SetPadTop (1); el->fLayout->SetPadBottom(1);
      }
      else if (el->fFrame == fEditor)
      {
         fEditor->ChangeOptions(fEditor->GetOptions() & (~kFixedWidth));
         fEditor->ChangeOptions(fEditor->GetOptions() |   kFixedHeight);
         el->fLayout->SetLayoutHints(kLHintsTop | kLHintsExpandX);
      }
   }

   fEditor->Resize(fEditor->GetWidth(), fEditor->GetHeight() / 2 - 1);
   fSplitter->SetFrame(fEditor, kFALSE);

   Layout();
   //fFrame->Layout();
   //fLTFrame->Layout();
   //fLTCanvas->Layout();
   //fListTree->ClearViewPort();
   MapSubwindows();
   MapWindow();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ItemChecked(TObject* obj, Bool_t state)
{
   // Item's user-data is blindly casted into TObject.
   // We recast it blindly back into the render element.

   TEveElement* rnrEl = (TEveElement*) obj;
   gEve->ElementChecked(rnrEl, state);
   gEve->Redraw3D();
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ItemClicked(TGListTreeItem *item, Int_t btn, Int_t x, Int_t y)
{
   //printf("ItemClicked item %s List %d btn=%d, x=%d, y=%d\n",
   //  item->GetText(),fDisplayFrame->GetList()->GetEntries(), btn, x, y);

   TEveElement* re = (TEveElement*)item->GetUserData();
   if(re == 0) return;
   TObject* obj = re->GetObject();

   switch (btn)
   {
      case 1:
         gEve->ElementSelect(re);
         break;

      case 2:
         if (gEve->ElementPaste(re))
            gEve->Redraw3D();
         break;

      case 3:
         // If control pressed, show menu for render-element itself.
         // event->fState & kKeyControlMask
         // ??? how do i get current event?
         if (obj) fCtxMenu->Popup(x, y, obj);
         break;

      default:
         break;
   }
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ItemDblClicked(TGListTreeItem* item, Int_t btn)
{
   if (btn != 1) return;

   TEveElement* re = (TEveElement*) item->GetUserData();
   if (re == 0) return;

   re->ExpandIntoListTree(fListTree, item);

   TObject* obj = re->GetObject();
   if (obj)
   {
      // Browse geonodes.
      if (obj->IsA()->InheritsFrom(TGeoNode::Class()))
      {
         TGeoNode* n = dynamic_cast<TGeoNode*>(obj);
         if (item->GetFirstChild() == 0 && n->GetNdaughters())
         {
            fListTree->DeleteChildren(item);
            for (Int_t i=0; i< n->GetNdaughters(); i++)
            {
               TString title;
               title.Form("%d : %s[%d]", i,
                          n->GetDaughter(i)->GetVolume()->GetName(),
                          n->GetDaughter(i)->GetNdaughters());

               TGListTreeItem* child = fListTree->AddItem( item, title.Data());
               child->SetUserData(n->GetDaughter(i));
            }
         }
      }
   }
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ItemKeyPress(TGListTreeItem *entry, UInt_t keysym, UInt_t /*mask*/)
{
   static const TEveException eH("TEveGListTreeEditorFrame::ItemKeyPress ");

   // replace entry with selected!
   entry = fListTree->GetSelected();
   if (entry == 0) return;

   if (keysym == kKey_Delete)
   {
      TEveElement* rnr_el = dynamic_cast<TEveElement*>
         ((TEveElement*) entry->GetUserData());
      if (rnr_el == 0)
         return;

      if (entry->GetParent())
      {
         if (rnr_el->GetDenyDestroy() > 0 && rnr_el->GetNItems() == 1)
            throw(eH + "DestroyDenied set for this item.");

         TEveElement* parent_re = dynamic_cast<TEveElement*>
            ((TEveElement*) entry->GetParent()->GetUserData());

         if (parent_re)
         {
            ResetSelectedTimer(entry);
            gEve->RemoveElement(rnr_el, parent_re);
            gEve->Redraw3D();
         }
      }
      else
      {
         if (rnr_el->GetDenyDestroy() > 0)
            throw(eH + "DestroyDenied set for this top-level item.");
         ResetSelectedTimer(entry);
         gEve->RemoveFromListTree(rnr_el, fListTree, entry);
         gEve->Redraw3D();
      }
   }
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ResetSelectedTimer(TGListTreeItem* lti)
{
   fNewSelected = lti->GetPrevSibling();
   if (! fNewSelected) {
      fNewSelected = lti->GetNextSibling();
      if (! fNewSelected)
         fNewSelected = lti->GetParent();
   }

   TTimer::SingleShot(0, IsA()->GetName(), this, "ResetSelected()");
}

//______________________________________________________________________________
void TEveGListTreeEditorFrame::ResetSelected()
{
   fListTree->HighlightItem(fNewSelected);
   fListTree->SetSelected(fNewSelected);
   fNewSelected = 0;
}


//______________________________________________________________________________
// TEveBrowser
//
// Specialization of TRootBrowser for Reve.

ClassImp(TEveBrowser)

//______________________________________________________________________________
void TEveBrowser::SetupCintExport(TClass* cl)
{
   TList* l = cl->GetMenuList();
   TClassMenuItem* n = new TClassMenuItem(TClassMenuItem::kPopupUserFunction, cl,
                                          "Export to CINT", "ExportToCINT", this, "const char*,TObject*", 1);

   l->AddFirst(n);
}

//______________________________________________________________________________
void TEveBrowser::CalculateReparentXY(TGObject* parent, Int_t& x, Int_t& y)
{
   UInt_t   w, h;
   Window_t childdum;
   gVirtualX->GetWindowSize(parent->GetId(), x, y, w, h);
   gVirtualX->TranslateCoordinates(parent->GetId(),
                                   gClient->GetDefaultRoot()->GetId(),
                                   0, 0, x, y, childdum);
}

/******************************************************************************/

namespace
{
enum EReveMenu_e {
   kNewViewer,  kNewScene,  kNewProjector,
   kNewBrowser, kNewCanvas, kNewCanvasExt, kNewTextEditor, kNewHtmlBrowser,
   kVerticalBrowser
};
}

//______________________________________________________________________________
TEveBrowser::TEveBrowser(UInt_t w, UInt_t h) :
   TRootBrowser(0, "Eve Main Window", w, h, "", kFALSE),
   fFileBrowser(0)
{
   // Construct Eve menu

   fRevePopup = new TGPopupMenu(gClient->GetRoot());
   fRevePopup->AddEntry("New &Viewer",      kNewViewer);
   fRevePopup->AddEntry("New &Scene",       kNewScene);
   fRevePopup->AddEntry("New &Projector",   kNewProjector);
   fRevePopup->AddSeparator();
   fRevePopup->AddEntry("New &Browser",     kNewBrowser);
   fRevePopup->AddEntry("New &Canvas",      kNewCanvas);
   fRevePopup->AddEntry("New Canvas Ext",   kNewCanvasExt);
   fRevePopup->AddEntry("New Text Editor",  kNewTextEditor);
   // fRevePopup->AddEntry("New HTML Browser", kNewHtmlBrowser);
   fRevePopup->AddSeparator();
   fRevePopup->AddEntry("Vertical browser", kVerticalBrowser);
   fRevePopup->CheckEntry(kVerticalBrowser);

   fRevePopup->Connect("Activated(Int_t)", "TEveBrowser",
                       this, "ReveMenu(Int_t)");

   fMenuBar->AddPopup("&Eve", fRevePopup, new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));

   fPreMenuFrame->ChangeOptions(fPreMenuFrame->GetOptions() | kRaisedFrame);
   fTopMenuFrame->Layout();
   fTopMenuFrame->MapSubwindows();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBrowser::ReveMenu(Int_t id)
{
   switch (id)
   {
      case kNewViewer:
         gEve->SpawnNewViewer("Viewer Pepe");
         break;

      case kNewScene:
         gEve->SpawnNewScene("Scena Mica");
         break;

      case kNewProjector: {
         TEveElement* pr = (TEveElement*) (gROOT->GetClass("TEveProjectionManager")->New());
         pr->SetRnrElNameTitle("Projector", "User-created projector.");
         gEve->AddToListTree(pr, kTRUE);
         break;
      }
      case kNewBrowser:
         gROOT->ProcessLineFast("new TBrowser");
         break;

      case kNewCanvas:
         StartEmbedding(1);
         gROOT->ProcessLineFast("new TCanvas");
         StopEmbedding();
         SetTabTitle("Canvas", 1);
         break;

      case kNewCanvasExt:
         gROOT->ProcessLineFast("new TCanvas");
         break;

      case kNewTextEditor:
         StartEmbedding(1);
         gROOT->ProcessLineFast(Form("new TGTextEditor((const char *)0, (const TGWindow *)0x%lx)", gClient->GetRoot()));
         StopEmbedding();
         SetTabTitle("Editor", 1);
         break;

      case kNewHtmlBrowser:
         gSystem->Load("libGuiHtml");
         if (gSystem->Load("libRHtml") >= 0)
         {
            StartEmbedding(1);
            gROOT->ProcessLine(Form("new TGHtmlBrowser(\"http://root.cern.ch/root/html/ClassIndex.html\", \
                              (const TGWindow *)0x%lx)", gClient->GetRoot()));
            StopEmbedding();
            SetTabTitle("HTML", 1);
         }
         break;

      case kVerticalBrowser:
         if (fRevePopup->IsEntryChecked(kVerticalBrowser)) {
            gEve->GetLTEFrame()->ReconfToHorizontal();
            fRevePopup->UnCheckEntry(kVerticalBrowser);
         } else {
            gEve->GetLTEFrame()->ReconfToVertical();
            fRevePopup->CheckEntry(kVerticalBrowser);
         }
         break;

      default:
         break;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveBrowser::InitPlugins()
{
   // File browser plugin...
   StartEmbedding(0);
   //gROOT->ProcessLine(Form("new TGFileBrowser((const TGWindow *)0x%lx, 200, 500)",
   //                   gClient->GetRoot()));
   {
      TGFileBrowser *fb = MakeFileBrowser();
      fb->BrowseObj(gROOT);
      fb->AddFSDirectory("/");
      fb->Show();

      fFileBrowser = fb;
   }
   StopEmbedding();
   SetTabTitle("Files", 0);

   // Class browser plugin
   /*
     StartEmbedding(0);
     gROOT->ProcessLine(Form("new TGClassBrowser((const TGWindow *)0x%lx, 200, 500)",
     gClient->GetRoot()));
     StopEmbedding();
     SetTabTitle("Classes", 0, 1);
   */

   // --- main frame

   // Canvas plugin...
   /* Now in menu
      StartEmbedding(1);
      gROOT->ProcessLineFast("new TCanvas");
      StopEmbedding();
      SetTabTitle("Canvas", 1);
   */

   // Editor plugin...
   /* Now in menu
      StartEmbedding(1);
      gROOT->ProcessLineFast(Form("new TGTextEditor((const char *)0, (const TGWindow *)0x%lx)",
      gClient->GetRoot()));
      StopEmbedding();
      SetTabTitle("Editor", 1);
   */

   // --- bottom area

   // Command plugin...
   StartEmbedding(2);
   gROOT->ProcessLineFast(Form("new TGCommandPlugin((const TGWindow *)0x%lx, 700, 300)",
                               gClient->GetRoot()));
   StopEmbedding();
   SetTabTitle("Command", 2);

   // --- Select first tab everywhere
   SetTab(0, 0);
   SetTab(1, 0);
   SetTab(2, 0);
}

//______________________________________________________________________________
TGFileBrowser* TEveBrowser::MakeFileBrowser()
{
   // Create a file-browser. Caller should provide
   // Start/StopEmbedding() calls and populate the new browser.

   TBrowserImp    imp;
   TBrowser      *tb = new TBrowser("Pipi", "Strel", &imp);
   TGFileBrowser *fb = new TGFileBrowser(gClient->GetRoot(), tb, 200, 500);
   tb->SetBrowserImp((TBrowserImp *)fb);
   fb->SetBrowser(tb);
   fb->SetNewBrowser(this);
   return fb;
}
