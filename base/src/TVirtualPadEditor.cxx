// @(#)root/base:$Name:  $:$Id: TVirtualPadEditor.cxx,v 1.0 2003/11/25
// Author: Ilka Antcheva   25/11/03
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
   // Ctor of ABC  

}

//______________________________________________________________________________
TVirtualPadEditor::~TVirtualPadEditor()
{
   // Cleanup virtual editor

   fgPadEditor = 0;
}

//______________________________________________________________________________
TVirtualPadEditor *TVirtualPadEditor::LoadEditor()
{
   // Static function returning a pointer to the current pad editor.
   // If the new pad edittor does not exist, the old one is set.

      TPluginHandler *h;
      if (fgEditorName.Length() == 0) 
         fgEditorName = gEnv->GetValue("Root.PadEditor","Ged");
         h = gROOT->GetPluginManager()->FindHandler("TVirtualPadEditor",
                                                     fgEditorName.Data());
      if (h) {
         if (h->LoadPlugin() == -1)
            return 0;
         fgPadEditor = (TVirtualPadEditor*) h->ExecPlugin(1, gPad);
      }

   return fgPadEditor;
}

//______________________________________________________________________________
const char *TVirtualPadEditor::GetEditorName()
{
   // static: return the name of the default pad editor
   
   return fgEditorName.Data();
}

//______________________________________________________________________________
TVirtualPadEditor *TVirtualPadEditor::GetPadEditor()
{
   // static: return the current pad editor

   if (!fgPadEditor) LoadEditor();
   
   return fgPadEditor;
}

//______________________________________________________________________________
void TVirtualPadEditor::SetPadEditorName(const char *name)
{
   // static: set name of default pad editor
   
   if (fgEditorName == name) return;
   delete fgPadEditor;
   fgPadEditor = 0;
   fgEditorName = name;
}
  
//______________________________________________________________________________
void TVirtualPadEditor::SetPadEditor(TVirtualPadEditor *editor)
{
   // static: set the pad editor

   fgPadEditor = editor;
}

//______________________________________________________________________________
void TVirtualPadEditor::ShowEditor()
{
   // static: show the pad editor

   if (!fgPadEditor) GetPadEditor();
   
   fgPadEditor->Show();
}

//______________________________________________________________________________
void TVirtualPadEditor::HideEditor()
{
   // static: hide the pad editor

   fgPadEditor->Hide();
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateFillAttributes(Int_t color, Int_t style)
{
   // Update fill attributes via the pad editor

   if (!fgPadEditor) GetPadEditor();
   
   fgPadEditor->FillAttributes(color, style);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateTextAttributes(Int_t align, Float_t angle,
                                             Int_t col, Int_t font, Float_t tsize)
{
   // Update fill attributes via the pad editor

   if (!fgPadEditor) GetPadEditor();
   
   fgPadEditor->TextAttributes(align, angle, col, font, tsize);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateLineAttributes(Int_t color, Int_t style, 
                                             Int_t width)
{
   // Update fill attributes via the pad editor

   if (!fgPadEditor) GetPadEditor();
   
   fgPadEditor->LineAttributes(color, style, width);
}

//______________________________________________________________________________
void TVirtualPadEditor::UpdateMarkerAttributes(Int_t color, Int_t style, 
                                               Float_t msize)
{
   // Update fill attributes via the pad editor

   if (!fgPadEditor) GetPadEditor();
   
   fgPadEditor->MarkerAttributes(color, style, msize);
}
