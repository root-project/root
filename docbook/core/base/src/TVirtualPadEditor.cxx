// @(#)root/base:$Id: TVirtualPadEditor.cxx,v 1.0 2003/11/25
// Author: Ilka Antcheva   25/11/03
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPadEditor                                                    //
//                                                                      //
// Abstract base class used by ROOT graphics editor                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TVirtualPadEditor.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "TVirtualPad.h"

TVirtualPadEditor *TVirtualPadEditor::fgPadEditor  = 0;
TString            TVirtualPadEditor::fgEditorName = "";

ClassImp(TVirtualPadEditor)

//______________________________________________________________________________
TVirtualPadEditor::TVirtualPadEditor()
{
   // Virtual editor ctor.

}

//______________________________________________________________________________
TVirtualPadEditor::~TVirtualPadEditor()
{
   // Virtual editor dtor.

}

//______________________________________________________________________________
TVirtualPadEditor *TVirtualPadEditor::LoadEditor()
{
   // Static function returning a pointer to a new pad editor.
   // This pointer can be adopted by a TCanvas (i.e. TRootCanvas)
   // when it embeds the editor.

   TPluginHandler *h;
   if (fgEditorName.Length() == 0)
      fgEditorName = gEnv->GetValue("Root.PadEditor","Ged");
   h = gROOT->GetPluginManager()->FindHandler("TVirtualPadEditor",
                                              fgEditorName);
   if (h) {
      if (h->LoadPlugin() == -1)
         return 0;
      return (TVirtualPadEditor*) h->ExecPlugin(1, gPad ? gPad->GetCanvas() : 0);
   }

   return 0;
}

//______________________________________________________________________________
const char *TVirtualPadEditor::GetEditorName()
{
   // Returns the type of the default pad editor. Static method.

   return fgEditorName;
}

//______________________________________________________________________________
TVirtualPadEditor *TVirtualPadEditor::GetPadEditor(Bool_t load)
{
   // Returns the pad editor dialog. Static method.

   if (!fgPadEditor && load)
      fgPadEditor = LoadEditor();

   return fgPadEditor;
}

//______________________________________________________________________________
void TVirtualPadEditor::SetPadEditorName(const char *name)
{
   // Set type of default pad editor. Static method.

   if (fgEditorName == name) return;
   delete fgPadEditor;
   fgPadEditor = 0;
   fgEditorName = name;
}

//______________________________________________________________________________
void TVirtualPadEditor::ShowEditor()
{
   // Show the global pad editor. Static method.

   if (!fgPadEditor) {
      GetPadEditor();
      if (!fgPadEditor) return;
      fgPadEditor->SetGlobal(kTRUE);
   }
   fgPadEditor->Show();
}

//______________________________________________________________________________
void TVirtualPadEditor::HideEditor()
{
   //  Hide the pad editor. Static method.

   if (fgPadEditor)
      fgPadEditor->Hide();
}

//______________________________________________________________________________
void TVirtualPadEditor::Terminate()
{
   // Close the global pad editor. Static method.

   if (!fgPadEditor) return;

   delete fgPadEditor;
   fgPadEditor = 0;
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateFillAttributes(Int_t color, Int_t style)
{
   // Update fill attributes via the pad editor

   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->FillAttributes(color, style);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateTextAttributes(Int_t align, Float_t angle,
                                             Int_t col, Int_t font, Float_t tsize)
{
   // Update text attributes via the pad editor

   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->TextAttributes(align, angle, col, font, tsize);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateLineAttributes(Int_t color, Int_t style,
                                             Int_t width)
{
   // Update line attributes via the pad editor

   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->LineAttributes(color, style, width);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateMarkerAttributes(Int_t color, Int_t style,
                                               Float_t msize)
{
   // Update marker attributes via the pad editor

   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->MarkerAttributes(color, style, msize);
}
