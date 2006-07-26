// @(#)root/ged:$Name:  $:$Id: TGedEditor.cxx,v 1.29 2006/06/23 15:19:22 antcheva Exp $
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


ClassImp(TGedEditor)

//______________________________________________________________________________
TGedEditor::TGedEditor(TCanvas* canvas) :
   TGMainFrame(gClient->GetRoot(), 175, 20)
{
   // Constructor of graphics editor.

   fCan = new TGCanvas(this, 170, 10, kFixedWidth);
   fTab = new TGTab(fCan->GetViewPort(), 10, 10);
   fCan->SetContainer(fTab);
   AddFrame(fCan, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   fTab->Associate(fCan);
   fTabContainer = fTab->AddTab("Style");
   fStyle = new TGCompositeFrame(fTabContainer, 110, 30, kVerticalFrame);
   fStyle->AddFrame(new TGedNameFrame(fStyle, 1),\
                    new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 2, 2));
   fTabContainer->AddFrame(fStyle, new TGLayoutHints(kLHintsTop | kLHintsExpandX,\
                                                     5, 0, 2, 2));
   fWid = GetCounter();
   fGlobal = kTRUE;

   if (canvas) {
      if (!canvas->GetSelected())
         canvas->SetSelected(canvas);
      if (!canvas->GetSelectedPad())
         canvas->SetSelectedPad(canvas);
      fModel  = canvas->GetSelected();
      fPad    = canvas->GetSelectedPad();
      fCanvas = canvas;
      fClass  = fModel->IsA();
      GetEditors();
      SetWindowName(Form("%s_Editor", canvas->GetName()));
   } else {
      fModel  = 0;
      fPad    = 0;
      fCanvas = 0;
      fClass  = 0;
      if (gPad) SetCanvas(gPad->GetCanvas());
      SetWindowName("Global Editor");
   }
   MapSubwindows();
   if (canvas) {
      UInt_t ch = fCanvas->GetWindowHeight();
      if (ch)
         Resize(GetWidth(), ch > 700 ? 700 : ch);
      else
         Resize(GetWidth(), canvas->GetWh()<450 ? 450 : canvas->GetWh() + 4);
                                                       // canvas borders=4pix
   } else {
      Resize(GetDefaultSize());
   }
   MapWindow();

   gROOT->GetListOfCleanups()->Add(this);
   if (fCanvas) ConnectToCanvas(fCanvas);

}

//______________________________________________________________________________
TGedEditor::~TGedEditor()
{
   // Editor destructor.

   gROOT->GetListOfCleanups()->Remove(this);

   fStyle->Cleanup();
   //Cleanup() cannot be used because of TH1/2Editors
   delete fTab;       //delete tab widget and its containers
   delete fCan;       //delete TGCanvas
}

//______________________________________________________________________________
void TGedEditor::CloseWindow()
{
   // When closed via WM close button, just unmap (i.e. hide) editor
   // for later use.

   UnmapWindow();
   Disconnect(fCanvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TGedEditor::GetEditors()
{
   // Get existing editors of selected object

   // Look in TClass::GetEditorList() for any object deriving from TGedFrame,
   Bool_t found = kFALSE;
   TGedElement *ge;
   TList *list = fModel->IsA()->GetEditorList();
   if (list->First() != 0) {

      TIter next1(list);
      while ((ge = (TGedElement *) next1())) {
         // check if the editor ge->fGedframe is already in the list of fStyle
         TList *l = fStyle->GetList();
         if (l->First() != 0) {
            TGFrameElement *fr;
            TIter next(l);
            while ((fr = (TGFrameElement *) next())) {
               TGedFrame *f = ge->fGedFrame;
               found = (fr->fFrame->InheritsFrom(f->ClassName()) && (ge->fCanvas == fCanvas));
               if (found) break;
               else {
                  GetClassEditor(fModel->IsA());
                  TList *blist = fModel->IsA()->GetListOfBases();
                  if (blist->First() != 0)
                     GetBaseClassEditor(fModel->IsA());
               }
            }
         }
      }
   } else {

      // scan list of base classes
      list = fModel->IsA()->GetListOfBases();
      if (list->First() != 0)
         GetBaseClassEditor(fModel->IsA());

      //search for a class editor = classname + 'Editor'
      GetClassEditor(fModel->IsA());
   }

   fStyle->Layout();
   fStyle->MapSubwindows();
}

//______________________________________________________________________________
void TGedEditor::GetBaseClassEditor(TClass *cl)
{
   // Scan the base classes of cl and add attribute editors to the list.

   TList *list = cl->GetListOfBases();
   if (list->First() == 0) return;

   TBaseClass *base;
   TIter next(list);

   while ((base = (TBaseClass *)next())) {

      TClass *c1;
      if ((c1 = base->GetClassPointer())) GetClassEditor(c1);

      if (c1->GetListOfBases()->First() == 0) continue;
      else GetBaseClassEditor(c1);
   }
}

//______________________________________________________________________________
void TGedEditor::GetClassEditor(TClass *cl)
{
   // Add attribute editor of class cl to the list of fStyle frame.

   TClass *class2, *class3;
   Bool_t found = kFALSE;
   class2 = gROOT->GetClass(Form("%sEditor",cl->GetName()));
   if (class2 && class2->InheritsFrom(TGedFrame::Class())) {
      TList *list = fStyle->GetList();
      if (list->First() != 0) {
         TGFrameElement *fr;
         TIter next(list);
         while ((fr = (TGFrameElement *) next())) {;
            found = fr->fFrame->InheritsFrom(class2);
            if (found) break;
         }
      }
      if (found == kFALSE) {
         gROOT->ProcessLine(Form("((TGCompositeFrame *)0x%lx)->AddFrame(new %s((TGWindow *)0x%lx, %d),\
                                 new TGLayoutHints(kLHintsTop | kLHintsExpandX,0, 0, 2, 2))",\
                                 (Long_t)fStyle, class2->GetName(), (Long_t)fStyle, fWid));
         fWid++;
         class3 = (TClass*)gROOT->GetListOfClasses()->FindObject(cl->GetName());
         TGedElement *ge;
         TIter next3(class3->GetEditorList());
         while ((ge = (TGedElement *)next3())) {
            if (!strcmp(ge->fGedFrame->ClassName(), class2->GetName()) && (ge->fCanvas == 0)) {
               ge->fCanvas = fCanvas;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGedEditor::ConnectToCanvas(TCanvas *c)
{
   // Connect this editor to the selected object in the canvas 'c'.

   c->Connect("Selected(TVirtualPad*,TObject*,Int_t)", "TGedEditor",\
               this, "SetModel(TVirtualPad*,TObject*,Int_t)");
   if (!c->GetSelected())
      c->SetSelected(c);
   if (!c->GetSelectedPad())
      c->SetSelectedPad(c);
   c->Selected(c->GetSelectedPad(), c->GetSelected(), kButton1Down);
}

//______________________________________________________________________________
void TGedEditor::SetCanvas(TCanvas *newcan)
{
   // Change connection to another canvas.

   if (!newcan || !fCanvas || (fCanvas == newcan)) return;

   if (fCanvas && (fCanvas != newcan))
      DisconnectEditors(fCanvas);
   fCanvas = newcan;

   SetWindowName(Form("%s_Editor", fCanvas->GetName()));
   if (!fCanvas->GetSelected())
      fCanvas->SetSelected(newcan);
   if (!fCanvas->GetSelectedPad())
      fCanvas->SetSelectedPad(newcan);
   fModel  = fCanvas->GetSelected();
   fPad    = fCanvas->GetSelectedPad();
   fClass  = fModel->IsA();
   GetEditors();
   ConnectToCanvas(fCanvas);
   SetModel(fPad, fModel, kButton1Down);
}

//______________________________________________________________________________
void TGedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   // Activate object editors according to the selected object.

   if (!fGlobal && (event != kButton1Down)) return;

   TCanvas *c = (TCanvas *) gTQSender;

   if (!fGlobal && (c != fCanvas)) return;

   fModel = obj;
   fPad   = pad;

   if ((obj != 0) && (obj->IsA() != fClass) && !obj->IsA()->InheritsFrom(fClass)) {
      fClass = obj->IsA();
      GetEditors();
   } else if ((obj == 0) && fPad) {
      TCanvas *canvas = fPad->GetCanvas();
      if (canvas) {
         fPad->SetSelected(fPad);
         canvas->Selected(fPad, fPad, 0);
         return;
      } else {
         DeleteEditors();
         DeleteWindow();
         return;
      }
   }
   if (gPad) gPad->GetVirtCanvas()->SetCursor(kWatch);
   gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kWatch));

   TGFrameElement *el;
   TIter next(fStyle->GetList());
   while ((el = (TGFrameElement *) next())) {
      if ((el->fFrame)->InheritsFrom(TGedFrame::Class()))
         ((TGedFrame *)(el->fFrame))->SetModel(fPad, fModel, event);
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

   if (gPad && (gPad->GetCanvas() != fCanvas))
      SetCanvas(gPad->GetCanvas());
   else
      ConnectToCanvas(fCanvas);

   if (fCanvas->GetShowEditor())
      fCanvas->ToggleEditor();

   if (fGlobal) {
      UInt_t dw = fClient->GetDisplayWidth();
      UInt_t cw = fCanvas->GetWindowWidth();
      UInt_t ch = fCanvas->GetWindowHeight();
      UInt_t cx = (UInt_t)fCanvas->GetWindowTopX();
      UInt_t cy = (UInt_t)fCanvas->GetWindowTopY();
      if (!ch)
         cy = cy + 20;      // embedded canvas protection

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
   }
   MapWindow();
   if (!gROOT->GetListOfCleanups()->FindObject(this))
      gROOT->GetListOfCleanups()->Add(this);
}

//______________________________________________________________________________
void TGedEditor::Hide()
{
   // Hide editor.

   if (gPad->GetCanvas() == fCanvas) {
      UnmapWindow();
      Disconnect(fCanvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");
      gROOT->GetListOfCleanups()->Remove(this);
   }
}

//______________________________________________________________________________
void TGedEditor::DisconnectEditors(TCanvas *canvas)
{
   // Disconnect GUI editors connected to canvas.

   if (!canvas) return;

   Disconnect(canvas, "Selected(TVirtualPad*,TObject*,Int_t)", this, "SetModel(TVirtualPad*,TObject*,Int_t)");

   TClass * cl;
   TIter next(gROOT->GetListOfClasses());
   while((cl = (TClass *)next())) {
      if (cl->GetEditorList()->First() != 0) {
         TList *editors = cl->GetEditorList();
         TIter next1(editors);
         TGedElement *ge;
         while ((ge = (TGedElement *)next1())) {
            if (ge->fCanvas == canvas) {
               ge->fCanvas = 0;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGedEditor::DeleteEditors()
{
   // Delete GUI editors connected to the canvas fCanvas.

   DisconnectEditors(fCanvas);
   Bool_t del = kTRUE;

   TClass * cl;
   TIter next(gROOT->GetListOfClasses());

   while((cl = (TClass *)next())) {
      if (cl->GetEditorList()->First() != 0) {
         TList *editors = cl->GetEditorList();
         TIter next1(editors);
         TGedElement *ge;
         while ((ge = (TGedElement *)next1())) {
            if (ge->fCanvas != 0) {
               del = kFALSE;
            }
         }
      }
   }

   if (del) {
      TIter next(gROOT->GetListOfClasses());
      while((cl = (TClass *)next())) {
         if (cl->GetEditorList()->First() != 0) {
            TList *editors = cl->GetEditorList();
            TIter next1(editors);
            TGedElement *ge;
            while ((ge = (TGedElement *)next1())) {
               editors->Remove(ge);
               delete ge;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGedEditor::RecursiveRemove(TObject* obj)
{
   // Remove references to fModel in case the fModel is being deleted.
   // Deactivate attribute frames if they point to obj.


   if ((fModel != obj) || (obj == fCanvas)) return;
   if (obj == fPad)
      SetModel(fCanvas, fCanvas, kButton1Down);
   else
      SetModel(fPad, fPad, kButton1Down);
}
