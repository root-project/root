// @(#)root/gui:$Name:  $:$Id: TGTextView.h,v 1.8 2000/10/22 19:28:58 rdm Exp $
// Author: Fons Rademakers   1/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextView
#define ROOT_TGTextView


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextView                                                           //
//                                                                      //
// A TGTextView is a text viewer widget. It is a specialization of      //
// TGView. It uses the TGText class (which contains all text            //
// manipulation code, i.e. loading a file in memory, changing,          //
// removing lines, etc.). Use a TGTextView to view non-editable text.   //
// For supported messages see TGView.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGView
#include "TGView.h"
#endif
#ifndef ROOT_TGText
#include "TGText.h"
#endif


class TGTextView : public TGView {

friend class TGClient;

protected:
   TGText         *fText;         // text buffer
   TGText         *fClipText;     // clipboard text buffer
   FontStruct_t    fFont;         // text font
   Int_t           fMaxAscent;    // maximum ascent in font
   Int_t           fMaxDescent;   // maximum descent in font
   Int_t           fMaxWidth;     // maximum width of character in font
   GContext_t      fNormGC;       // graphics context for drawing text
   GContext_t      fSelGC;        // graphics context for drawing marked text
   GContext_t      fSelbackGC;    // graphics context for drawing marked background
   Bool_t          fMarkedFromX;  // true if text is marked from x
   Bool_t          fMarkedFromY;  // true if text is marker from y
   Bool_t          fDeleteGC;     // delete widget specific contexts

   static TGGC          fgDefaultGC, fgDefaultSelectedGC,
                        fgDefaultSelectedBackgroundGC;
   static FontStruct_t  fgDefaultFontStruct;

   void Init(ULong_t bg);
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void Mark(Long_t xPos, Long_t yPos);
   virtual void UnMark();
   virtual void Copy(TObject &) { MayNotUse("Copy(TObject &)"); }

public:
   TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, Int_t id = -1,
              UInt_t sboptions = 0, ULong_t back = GetWhitePixel());
   TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
              Int_t id = -1, UInt_t sboptions = 0, ULong_t back = GetWhitePixel());
   TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, const char *string,
              Int_t id = -1, UInt_t sboptions = 0, ULong_t back = GetWhitePixel());

   virtual ~TGTextView();

   virtual Bool_t IsSaved() { fIsSaved = fText->IsSaved(); return fIsSaved;}
   virtual Long_t ToObjXCoord(Long_t xCoord, Long_t line);
   virtual Long_t ToObjYCoord(Long_t yCoord);
   virtual Long_t ToScrXCoord(Long_t xCoord, Long_t line);
   virtual Long_t ToScrYCoord(Long_t yCoord);
   virtual void   AdjustWidth();
   virtual Bool_t LoadFile(const char *fname, long startpos = 0, long length = -1);
   virtual Bool_t LoadBuffer(const char *txtbuf);
   virtual void   Clear(Option_t * = "");
   virtual Bool_t Copy();
   virtual Bool_t SelectAll();
   virtual Bool_t Search(const char *string, Bool_t direction, Bool_t caseSensitive);
   virtual void   SetFont(FontStruct_t font);
   virtual Long_t ReturnHeighestColHeight() { return fText->RowCount()*fScrollVal.fY; }
   virtual Long_t ReturnLongestLineWidth();
   virtual Long_t ReturnLineLength(Long_t line) { return fText->GetLineLength(line); }
   virtual Long_t ReturnLongestLine() { return fText->GetLongestLine(); }
   virtual Long_t ReturnLineCount() { return fText->RowCount(); }
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleSelectionRequest(Event_t *event);

   virtual void SetText(TGText *text);
   virtual void AddText(TGText *text);
   virtual void AddLine(const char *string);
   TGText      *GetText() const { return fText; }

   virtual void DataChanged() { Emit("DataChanged()"); }  //*SIGNAL*

   ClassDef(TGTextView,0)  // Editable text widget base class (links TGText to TGEditView)
};

#endif
