// @(#)root/base:$Name:  $:$Id: TVirtualPadEditor.h,v 1.0 2003/11/25
// Author: Rene Brun   25/11/03

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualPadEditor
#define ROOT_TVirtualPadEditor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPadEditor                                                    //
//                                                                      //
// Abstract base class for pad editing                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif

class TVirtualPadEditor {

protected:
   static TVirtualPadEditor *fgPadEditor;    // singleton editor dialog
   static TString            fgEditorName;   // name of the default pad editor "Ged"

public:
   TVirtualPadEditor();
   virtual ~TVirtualPadEditor();

   // methods related to the old editor interface
   virtual void     DrawLine(Double_t, Double_t , Double_t , Double_t ) { }
   virtual void     DrawLineNDC(Double_t , Double_t , Double_t , Double_t ) { }
   virtual void     DrawText(Double_t , Double_t , const char *) { }
   virtual void     DrawTextNDC(Double_t , Double_t , const char *) { }
   virtual void     FillAttributes(Int_t , Int_t ) { }
   virtual void     LineAttributes(Int_t , Int_t , Int_t ) { }
   virtual void     MarkerAttributes(Int_t , Int_t , Float_t ) { }
   virtual void     TextAttributes(Int_t ,Float_t ,Int_t ,Int_t ,Float_t ) { }

   virtual void     Build() { }
   virtual void     Hide() { }
   virtual Bool_t   IsGlobal() const = 0;
   virtual void     DeleteEditors() { }
   virtual void     SetGlobal(Bool_t) { }
   virtual void     Show() { }

   // methods related to the new editor interface

   //static methods for both interfaces
   static const char        *GetEditorName();
   static TVirtualPadEditor *GetPadEditor(Bool_t load = kTRUE);
   static TVirtualPadEditor *LoadEditor();
   static void      HideEditor();
   static void      ShowEditor();
   static void      SetPadEditorName(const char *name);
   static void      Terminate();
   static void      UpdateFillAttributes(Int_t col, Int_t sty);
   static void      UpdateLineAttributes(Int_t col, Int_t sty, Int_t width);
   static void      UpdateMarkerAttributes(Int_t col, Int_t sty, Float_t msiz);
   static void      UpdateTextAttributes(Int_t align,Float_t angle,Int_t col,Int_t font,Float_t tsize);

   ClassDef(TVirtualPadEditor,0)  //Abstract interface for graphics pad editor
};

#endif
