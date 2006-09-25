// @(#)root/ged:$Name:  $:$Id: TGedEditor.cxx,v 1.26 2006/03/20 21:43:41 pcanal Exp $
// Author: Marek Biskup, Ilka Antcheva 02/08/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedEditor                                                           //
//                                                                      //
// Editor is a composite frame that contains TGedToolBox and            //
// TGedAttFrames. It is connected to a Canvas and listens for           //
// selected objects                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedEditor.h" 
#include "TCanvas.h"
#include "TGCanvas.h"
#include "TGTab.h"
#include "TGedFrame.h"
#include "TGLabel.h"
#include "TGFrame.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TSystem.h"

class TGedTabInfo : public TObject {
   // Helper class for managing visibility and order of created tabs.
public:
   TGTabElement      *fElement;
   TGCompositeFrame  *fContainer;

   TGedTabInfo(TGTabElement* el, TGCompositeFrame* f) : 
      fElement(el), fContainer(f) {}
};


ClassImp(TGedEditor)
//______________________________________________________________________________
TGedEditor::TGedEditor(TCanvas* canvas) :
   TGMainFrame(gClient->GetRoot(), 175, 20),
   fCan          (0),
   fTab          (0),
   fTabContainer (0),
   fModel        (0),
   fPad          (0),
   fCanvas       (0),
   fClass        (0),
   fGlobal       (kTRUE)
{
   // Constructor of graphics editor.

   fCan = new TGCanvas(this, 170, 10, kFixedWidth); 
   AddFrame(fCan, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));

   fTab = new TGTab(fCan->GetViewPort(), 10, 10);
   fTab->Associate(fCan);
   fTab->SetCleanup(kDeepCleanup);
   fCan->SetContainer(fTab);

   fTabContainer = GetEditorTab("Style");

   gROOT->GetListOfCleanups()->Add(this);

   SetCanvas(canvas);
   if (fCanvas) {
      UInt_t ch = fCanvas->GetWindowHeight();
      if (ch)
         Resize(GetWidth(), ch > 700 ? 700 : ch);
      else
         Resize(GetWidth(), fCanvas->GetWh()<450 ? 450 : fCanvas->GetWh() + 4);
   } else {
      Resize(GetDefaultSize());
   }

   MapSubwindows();
   MapWindow();
}

//______________________________________________________________________________
TGedEditor::~TGedEditor()
{
   // Editor destructor.

   Hide();

   if(fGlobal){
      TQObject::Disconnect("TCanvas", "Selected(TVirtualPad *, TObject *, Int_t)");
      TQObject::Disconnect("TCanvas", "Closed()"); 
   }

   // delete class editors
   TIter next(fFrameMap.GetTable());
   TPair* pair;
   while ((pair = (TPair*) next())) {
      if (pair->Value() != 0) {
         TGedFrame* frame  = (TGedFrame*) pair->Value();
         delete frame;
      }
   }

   TGedTabInfo* ti;
   TIter it1(&fCreatedTabs);
   while ((ti = (TGedTabInfo*) it1())) {
      fTab->AddFrame(ti->fElement,0);
      fTab->AddFrame(ti->fContainer,0);
   }

   delete fTab;
   delete ((TGFrameElement*)fList->First())->fLayout;
   delete fCan;
}

//______________________________________________________________________________
void TGedEditor::Update(TGedFrame* /*frame*/)
{
   // Virtual method that is called on any change in the dependent frames.
   // This implementation simply calls fPad Modified()/Update().

   if (fPad) {
      fPad->Modified();
      fPad->Update();
   }
}

//______________________________________________________________________________
TGCompositeFrame* TGedEditor::GetEditorTab(const Text_t* name)
{
   // Find or create tab with name.
   return GetEditorTabInfo(name)->fContainer;
}

//______________________________________________________________________________
TGedTabInfo* TGedEditor::GetEditorTabInfo(const Text_t* name)
{
   // Find or create tab with name.

   // look in list of created tabs
   if ( ! fCreatedTabs.IsEmpty()) {
      TIter next(&fCreatedTabs);
      TGedTabInfo* ti;
      while ((ti = (TGedTabInfo *) next())) {
         if (*ti->fElement->GetText() == name)
            return ti;
      }
   }

   // create tab
   TGCompositeFrame* tc = fTab->AddTab(new TGString(name));

   // remove created frame end tab element from the fTab frame
   TGTabElement *te = fTab->GetTabTab(fTab->GetNumberOfTabs() - 1);
   fTab->RemoveFrame(tc);
   fTab->RemoveFrame(te);

   // create a title frame for each tab
   TGedNameFrame* nf = new TGedNameFrame(tc);
   nf->SetGedEditor(this);
   nf->SetModelClass(0);
   tc->AddFrame(nf, nf->GetLayoutHints());

   // add to list of created tabs
   TGedTabInfo* ti = new TGedTabInfo(te, tc);
   fCreatedTabs.Add(ti);

   return ti;
}

//______________________________________________________________________________
TGCompositeFrame* TGedEditor::CreateEditorTabSubFrame(const Text_t* name,
                                                      TGedFrame* owner)
{
   // Create a vertical frame to be used by 'owner' in extra tab 'name'.
   // The new frame is registered into the sub-frame listo of 'owner'.

   TGCompositeFrame* tabcont  = GetEditorTab(name);

   TGCompositeFrame* newframe = new TGVerticalFrame(tabcont);
   owner->AddExtraTab(new TGedFrame::TGedSubFrame(TString(name), newframe));
   return newframe;
}

//______________________________________________________________________________
void TGedEditor::CloseWindow()
{
   // Called when closed via WM close button. Calls Hide().

   Hide();
}

//______________________________________________________________________________
void TGedEditor::ReinitWorkspace()
{  
   // Clears windows in editor tab.
   // Moves all visible GedFrames to the map of available frames and hide them. 

   TIter it(fTabContainer->GetList());
   it(); // skip name-frame

   // Unmap and withdraw currently shown frames and thus prepare for
   // construction of a new class layout or destruction.
   TIter next(&fVisibleTabs);
   TGedTabInfo* ti;
   while ((ti = (TGedTabInfo*)next())) {
      TGTabElement     *te = ti->fElement;
      TGCompositeFrame *tc = ti->fContainer;

      fTab->RemoveFrame(te);
      fTab->RemoveFrame(tc);

      // printf("ReinitWorkspace remove %d from %s \n", tc->GetList()->GetSize(),tc->GetName());
      TIter frames(tc->GetList());
      frames(); // skip name-frame
      TGFrameElement* fr;
      while ((fr = (TGFrameElement *) frames()) != 0) {
         TGFrame *f = fr->fFrame;
         {
            tc->RemoveFrame(f);
            f->UnmapWindow();
         }
         te->UnmapWindow();
         tc->UnmapWindow();

         fVisibleTabs.Remove(ti);
      }
   }
}

//______________________________________________________________________________
void TGedEditor::SetGlobal(Bool_t global)
{
   // Set editor global.

   fGlobal = global;
   if (fGlobal) {
      TQObject::Connect("TCanvas", "Selected(TVirtualPad *, TObject *, Int_t)",
                        "TGedEditor", this, "GlobalSetModel(TVirtualPad *, TObject *, Int_t)"); 
       
      TQObject::Connect("TCanvas", "Closed()",
                        "TGedEditor", this, "GlobalClosed()");
   }
}

//______________________________________________________________________________
void TGedEditor::GlobalClosed()
{
   // Delete global editor if no canvas exists.

   if (gROOT->GetListOfCanvases()->IsEmpty())
      TVirtualPadEditor::Terminate();
}

//______________________________________________________________________________
void TGedEditor::GlobalSetModel(TVirtualPad *pad, TObject */*obj*/, Int_t ev)
{
   // Set canvas to global editor.

   if (ev != kButton1Down) return;

   TCanvas* can = pad->GetCanvas();
   // Do nothing if canvas is the same as before or
   // local editor of the canvas is active.
   if (can == fCanvas || can->GetShowEditor())
      return;

   Show();
}

//______________________________________________________________________________
void TGedEditor::ConnectToCanvas(TCanvas *c)
{
   // Connect this editor to the Selected signal of canvas 'c'.

   c->Connect("Selected(TVirtualPad*,TObject*,Int_t)", "TGedEditor",
              this, "SetModel(TVirtualPad*,TObject*,Int_t)");
}

//______________________________________________________________________________
void TGedEditor::DisconnectFromCanvas()
{
   // Disconnect this editor from the Selected signal of fCanvas.

   if (fCanvas)
      Disconnect(fCanvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");
}

//______________________________________________________________________________
void TGedEditor::SetCanvas(TCanvas *newcan)
{
   // Change connection to another canvas.

   if (!newcan || (fCanvas == newcan)) return;

   DisconnectFromCanvas();
   fCanvas = newcan;

   SetWindowName(Form("%s_Editor", fCanvas->GetName()));
   fPad = fCanvas->GetSelectedPad();
   if (fPad == 0) fPad = fCanvas;
   ConnectToCanvas(fCanvas);
}

//______________________________________________________________________________
void TGedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   // Activate object editors according to the selected object.

   if (event != kButton1Down) return;

   if (gPad) gPad->GetVirtCanvas()->SetCursor(kWatch);
   gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kWatch));
 
   fPad = pad;
   if (obj == 0) obj = fPad;

   // keep selected by name
   TGTabElement* seltab = fTab->GetCurrentTab();

   Bool_t mapTabs = kFALSE;
   if (fModel != obj) {
      fModel = obj;
      if (fModel == 0 || fModel->IsA() != fClass) {
         ReinitWorkspace();
         mapTabs = kTRUE;
         // add Sytle tab to list of visible tabs
         fVisibleTabs.Add(fCreatedTabs.First());
         if (fModel) {
            fClass = fModel->IsA();
            // build a list of editors
            ActivateEditor(fClass, kTRUE);
         } else {
            fClass = 0;
         }

         // add class editors to fTabContainer
         TGedFrame* gfr;
         TIter ngf(&fGedFrames);
         while ((gfr = (TGedFrame*) ngf()))
            fTabContainer->AddFrame(gfr, gfr->GetLayoutHints());

         fExclMap.Clear();
         fGedFrames.Clear();

         // add visible tabs in fTab
         TIter next(&fVisibleTabs);
         TGedTabInfo* ti;
         while ((ti = (TGedTabInfo *) next())) {
            fTab->AddFrame(ti->fElement,0);
            fTab->AddFrame(ti->fContainer,0);
         }  
      }
   } // end fModel != obj

   ConfigureGedFrames();

   if (mapTabs) { // selected object is different class
      TGedTabInfo* ti;
      TIter next(&fVisibleTabs);
      while ((ti = (TGedTabInfo *) next())) {
         ti->fElement->MapWindow();
         ti->fContainer->MapWindow();
      }
      if (seltab == 0 || fTab->SetTab(seltab->GetString(), kFALSE) == kFALSE)
         fTab->SetTab(0, kFALSE);
   } 

   if (fGlobal)
      Layout();
   else
      ((TGMainFrame*)GetMainFrame())->Layout();

   if (gPad) gPad->GetVirtCanvas()->SetCursor(kPointer);
   gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kPointer));
}

//______________________________________________________________________________
void TGedEditor::Show()
{
   // Show editor.

   // gPad is setup properly in calling code for global and canvas editor.
   SetCanvas(gPad->GetCanvas());

   if (fGlobal) {
      SetModel(fCanvas->GetClickSelectedPad(), fCanvas->GetClickSelected(), kButton1Down);

      if (fCanvas->GetShowEditor())
         fCanvas->ToggleEditor();

      UInt_t dw = fClient->GetDisplayWidth();
      UInt_t cw = fCanvas->GetWindowWidth();
      UInt_t ch = fCanvas->GetWindowHeight();
      UInt_t cx = (UInt_t)fCanvas->GetWindowTopX();
      UInt_t cy = (UInt_t)fCanvas->GetWindowTopY();
      if (!ch) 
         cy = cy + 20;      // embeded canvas protection

      Int_t gedx = 0, gedy = 0;

      if (cw + GetWidth() > dw) {
         gedx = cx + cw - GetWidth();
         gedy = ch - GetHeight();
      } else {
         if (cx > GetWidth())
            gedx = cx - GetWidth() - 20;
         else
            gedx = cx + cw + 10;
         gedy = cy - 20;
      }
      MoveResize(gedx, gedy, GetWidth(), ch > 700 ? 700 : ch);
      SetWMPosition(gedx, gedy);
   } else {
      SetModel(fCanvas, fCanvas, kButton1Down);
   }
   MapWindow();
   gVirtualX->RaiseWindow(GetId());

   if (!gROOT->GetListOfCleanups()->FindObject(this))
      gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
void TGedEditor::Hide()
{
   // Hide editor. The editor is put into non-active state.

   UnmapWindow();
   ReinitWorkspace();
   fModel = 0; fClass = 0;
   DisconnectFromCanvas();
   fCanvas = 0; fPad = 0;
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TGedEditor::RecursiveRemove(TObject* obj)
{
   // Remove references to fModel in case the fModel is being deleted.
   // Deactivate attribute frames if they point to obj.
  
   if (obj == fPad) {
      // printf("TGedEditor::RecursiveRemove: %s - pad deleted.\n", locglob);
      SetModel(fCanvas, fCanvas, kButton1Down);
      return;
   }

   if (obj == fModel) {
      // printf("TGedEditor::RecursiveRemove: %s - model deleted.\n", locglob);
      SetModel(fPad, fPad, kButton1Down);
      return;
   }
}

//______________________________________________________________________________
void TGedEditor::ActivateEditor(TClass* cl, Bool_t recurse)
{
   // Searches for GedFrames for given class. In recursive mode look for class 
   // editor in its list of bases.

   TPair     *pair = (TPair*) fFrameMap.FindObject(cl);
   TClass    *edClass = 0;
   TGedFrame *frame = 0;

   if (pair == 0) {
      edClass = gROOT->GetClass(Form("%sEditor", cl->GetName()));

      if (edClass && edClass->InheritsFrom(TGedFrame::Class())) {
         frame = reinterpret_cast<TGedFrame*>(edClass->New());
         frame->ReparentWindow(fTabContainer);
         frame->SetModelClass(cl);
         frame->SetGedEditor(this);
      }
      fFrameMap.Add(cl, frame);
   } else {
      frame =  (TGedFrame*)pair->Value();
   }

   Bool_t exclfr    = kFALSE;
   Bool_t exclbases = kFALSE;

   if (frame) {
      TPair* exclpair = (TPair*) fExclMap.FindObject(cl);
      if (exclpair) {
         exclfr = kTRUE;
         exclbases = (exclpair->Value() != 0);
      }

      if (!exclfr && frame->AcceptModel(fModel)){
         // handle extra tabs in the gedframe
         if (frame->GetExtraTabs()) {
            TIter next(frame->GetExtraTabs());
            TGedFrame::TGedSubFrame* subf;
            while ((subf = (TGedFrame::TGedSubFrame*)next())) {
               // locate the composite frame on created tabs
               TGedTabInfo* ti = GetEditorTabInfo(subf->fName);
               ti->fContainer->AddFrame(subf->fFrame);
               if(! pair)
                  subf->fFrame->ReparentWindow(ti->fContainer);
               fVisibleTabs.Add(ti);
            }
         }
         InsertGedFrame(frame);
      }
   }    

   if (recurse && !exclbases) {
      if (frame)
         frame->ActivateBaseClassEditors(cl);
      else
         ActivateEditors(cl->GetListOfBases(), recurse);
   }
}

//______________________________________________________________________________
void TGedEditor::ActivateEditors(TList* bcl, Bool_t recurse)
{
   // Searches GedFrames for classes in the given list.

   TBaseClass *base;
   TIter next(bcl);

   while ((base = (TBaseClass*) next())) {
      ActivateEditor(base->GetClassPointer(), recurse);
   }
}

//______________________________________________________________________________
void  TGedEditor::ExcludeClassEditor(TClass* cl, Bool_t recurse)
{
   // Exclude editor for class cl from current construction.
   // If recurse is true the base-class editors of cl are also excluded.

   TPair* pair = (TPair*) fExclMap.FindObject(cl);
   if (pair) {
      if (recurse && pair->Value() == 0)
         pair->SetValue((TObject*)1); // hack, reuse TObject as Bool_t
   } else {
      fExclMap.Add(cl, (TObject*)(recurse ? 1 : 0));
   }
}

//______________________________________________________________________________
void TGedEditor::InsertGedFrame(TGedFrame* f)
{
   // Insert GedFrame in fGedFrames list according to priorities.

   // printf("%s %s  insert gedframe %s \n", fModel->GetName(), fModel->IsA()->GetName(),f->GetModelClass()->GetName());
   TObjLink* lnk = fGedFrames.FirstLink();
   if (lnk == 0) {
      fGedFrames.Add(f);
      return;
   }
   TGedFrame* cf;
   while (lnk) {
      cf = (TGedFrame*) lnk->GetObject();
      if (f->GetPriority() < cf->GetPriority()) {
         fGedFrames.AddBefore(lnk, f);
         return;
      }
      lnk = lnk->Next();
   }
   fGedFrames.Add(f);
}

//______________________________________________________________________________
void TGedEditor::ConfigureGedFrames()
{
   // Call SetModel in class editors.

   TGFrameElement *el;

   // Call SetModel on TGedNameFrames (first in the container list)
   // and map extra-tabs.
   TIter vistabs(&fVisibleTabs);
   vistabs(); // skip Style tab
   TGedTabInfo* ti;
   while ((ti = (TGedTabInfo *) vistabs())) {
      TIter fr(ti->fContainer->GetList());
      el = (TGFrameElement*) fr();
      ((TGedFrame*) el->fFrame)->SetModel(fModel);
      do {
         el->fFrame->MapSubwindows();
         el->fFrame->Layout();
         el->fFrame->MapWindow();
      } while((el = (TGFrameElement *) fr()));
      ti->fContainer->Layout();
   }

   TIter next(fTabContainer->GetList());
   while ((el = (TGFrameElement *) next())) {
      if ((el->fFrame)->InheritsFrom(TGedFrame::Class())) {
         el->fFrame->MapSubwindows();
         ((TGedFrame *)(el->fFrame))->SetModel(fModel);
         el->fFrame->Layout();
         el->fFrame->MapWindow();
      }
   }
   fTabContainer->Layout();
}

//______________________________________________________________________________
void TGedEditor::PrintFrameStat()
{
   // Print contents of fFrameMap.

   printf("TGedEditor::PrintFrameStat()\n");
   Int_t sum = 0;
   TIter next(fFrameMap.GetTable());
   TPair* pair;
   while ((pair = (TPair*) next())) {
      if (pair->Value() != 0) {
         TClass* cl  = (TClass*) pair->Key();
         printf("TGedFrame created for %s \n", cl->GetName());
         sum ++;
      }
   }
   printf("SUMMARY: %d editors stored in the local map.\n", sum);
}
