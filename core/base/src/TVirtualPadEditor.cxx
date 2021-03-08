// @(#)root/base:$Id: TVirtualPadEditor.cxx,v 1.0 2003/11/25
// Author: Ilka Antcheva   25/11/03
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualPadEditor
\ingroup Base

Abstract base class used by ROOT graphics editor
*/

#include "TROOT.h"
#include "TVirtualPadEditor.h"
#include "TPluginManager.h"
#include "TEnv.h"
#include "TVirtualPad.h"

TVirtualPadEditor *TVirtualPadEditor::fgPadEditor  = nullptr;
TString            TVirtualPadEditor::fgEditorName = "";

ClassImp(TVirtualPadEditor);

////////////////////////////////////////////////////////////////////////////////
/// Virtual editor ctor.

TVirtualPadEditor::TVirtualPadEditor()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual editor dtor.

TVirtualPadEditor::~TVirtualPadEditor()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to a new pad editor.
/// This pointer can be adopted by a TCanvas (i.e. TRootCanvas)
/// when it embeds the editor.

TVirtualPadEditor *TVirtualPadEditor::LoadEditor()
{
   TPluginHandler *h;
   if (fgEditorName.Length() == 0)
      fgEditorName = gEnv->GetValue("Root.PadEditor","Ged");
   h = gROOT->GetPluginManager()->FindHandler("TVirtualPadEditor",
                                              fgEditorName);
   if (h) {
      if (h->LoadPlugin() == -1)
         return nullptr;
      return (TVirtualPadEditor*) h->ExecPlugin(1, gPad ? gPad->GetCanvas() : nullptr);
   }

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the type of the default pad editor. Static method.

const char *TVirtualPadEditor::GetEditorName()
{
   return fgEditorName;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the pad editor dialog. Static method.

TVirtualPadEditor *TVirtualPadEditor::GetPadEditor(Bool_t load)
{
   if (!fgPadEditor && load)
      fgPadEditor = LoadEditor();

   return fgPadEditor;
}

////////////////////////////////////////////////////////////////////////////////
/// Set type of default pad editor. Static method.

void TVirtualPadEditor::SetPadEditorName(const char *name)
{
   if (fgEditorName == name) return;
   delete fgPadEditor;
   fgPadEditor = nullptr;
   fgEditorName = name;
}

////////////////////////////////////////////////////////////////////////////////
/// Show the global pad editor. Static method.

void TVirtualPadEditor::ShowEditor()
{
   if (!fgPadEditor) {
      GetPadEditor();
      if (!fgPadEditor) return;
      fgPadEditor->SetGlobal(kTRUE);
   }
   fgPadEditor->Show();
}

////////////////////////////////////////////////////////////////////////////////
///  Hide the pad editor. Static method.

void TVirtualPadEditor::HideEditor()
{
   if (fgPadEditor)
      fgPadEditor->Hide();
}

////////////////////////////////////////////////////////////////////////////////
/// Close the global pad editor. Static method.

void TVirtualPadEditor::Terminate()
{
   if (!fgPadEditor) return;

   delete fgPadEditor;
   fgPadEditor = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Update fill attributes via the pad editor

void TVirtualPadEditor::UpdateFillAttributes(Int_t color, Int_t style)
{
   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->FillAttributes(color, style);
}

////////////////////////////////////////////////////////////////////////////////
/// Update text attributes via the pad editor

void TVirtualPadEditor::UpdateTextAttributes(Int_t align, Float_t angle,
                                             Int_t col, Int_t font, Float_t tsize)
{
   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->TextAttributes(align, angle, col, font, tsize);
}

////////////////////////////////////////////////////////////////////////////////
/// Update line attributes via the pad editor

void TVirtualPadEditor::UpdateLineAttributes(Int_t color, Int_t style,
                                             Int_t width)
{
   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->LineAttributes(color, style, width);
}

////////////////////////////////////////////////////////////////////////////////
/// Update marker attributes via the pad editor

void TVirtualPadEditor::UpdateMarkerAttributes(Int_t color, Int_t style,
                                               Float_t msize)
{
   ShowEditor();

   if (fgPadEditor)
      fgPadEditor->MarkerAttributes(color, style, msize);
}
