// @(#)root/gpad:$Name:  $:$Id: TPadEditorOld.cxx,v 1.0 2003/11/26
// Author: Ilka Antcheva   26/11/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TPadEditorOld
#define ROOT_TPadEditorOld


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPadEditorOld                                                        //
//                                                                      //
// Class providing the old pad editor interface                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualPadEditor
#include "TVirtualPadEditor.h"
#endif

class TCanvas;
class TControlBar;

class TPadEditorOld : public TVirtualPadEditor {
   
protected:
   TControlBar   *fControlBar;  // control bar
   
public:
   TPadEditorOld(TCanvas* canvas = 0);
   virtual      ~TPadEditorOld();

   void          Build();
   TControlBar  *GetEditorBar() const { return fControlBar; }

   virtual void  DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   virtual void  DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2);
   virtual void  DrawText(Double_t x, Double_t y, const char *text);
   virtual void  DrawTextNDC(Double_t u, Double_t v, const char *text);
   virtual void  FillAttributes(Int_t col, Int_t sty);
   virtual void  LineAttributes(Int_t col, Int_t sty, Int_t width);
   virtual void  MarkerAttributes(Int_t col, Int_t sty, Float_t msiz);
   virtual void  TextAttributes(Int_t align,Float_t angle,Int_t col,Int_t font,Float_t tsize);
   
   ClassDef(TPadEditorOld,0)  //Old editor  
};

#endif
