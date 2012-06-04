// @(#)root/gui:$Id$
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

class TViewTimer;

class TGTextView : public TGView {

protected:
   TGText         *fText;         // text buffer
   TGText         *fClipText;     // clipboard text buffer
   FontStruct_t    fFont;         // text font
   Int_t           fMaxAscent;    // maximum ascent in font
   Int_t           fMaxDescent;   // maximum descent in font
   Int_t           fMaxWidth;     // maximum width of character in font
   TGGC            fNormGC;       // graphics context for drawing text
   TGGC            fSelGC;        // graphics context for drawing marked text
   TGGC            fSelbackGC;    // graphics context for drawing marked background
   Bool_t          fMarkedFromX;  // true if text is marked from x
   Bool_t          fMarkedFromY;  // true if text is marker from y
   Bool_t          fIsMarked;     // true if text is marked/selected
   Bool_t          fIsMarking;    // true if in marking mode
   Bool_t          fIsSaved;      // true is content is saved
   Bool_t          fReadOnly;     // text cannot be editted
   TGLongPosition  fMarkedStart;  // start position of marked text
   TGLongPosition  fMarkedEnd;    // end position of marked text
   TViewTimer     *fScrollTimer;  // scrollbar timer
   Atom_t         *fDNDTypeList;  // handles DND types

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;
   static TGGC         *fgDefaultSelectedGC;
   static const TGGC   *fgDefaultSelectedBackgroundGC;

   void Init(Pixel_t bg);
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);
   virtual void Mark(Long_t xPos, Long_t yPos);
   virtual void UnMark();
   virtual void Copy(TObject &) const { MayNotUse("Copy(TObject &)"); }
   virtual void HLayout();
   virtual void VLayout();

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();
   static const TGGC   &GetDefaultSelectedGC();
   static const TGGC   &GetDefaultSelectedBackgroundGC();

private:
   TGTextView(const TGTextView&);
   TGTextView& operator=(const TGTextView&);

public:
   TGTextView(const TGWindow *parent = 0, UInt_t w = 1, UInt_t h = 1, Int_t id = -1,
              UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());
   TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
              Int_t id = -1, UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());
   TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, const char *string,
              Int_t id = -1, UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());

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

   virtual void   SetSBRange(Int_t direction);
   virtual void   SetHsbPosition(Long_t newPos);
   virtual void   SetVsbPosition(Long_t newPos);
   virtual void   ShowBottom();
   virtual void   ShowTop();

   virtual void   SavePrimitive(std::ostream &out, Option_t * = "");
   virtual void   SetText(TGText *text);
   virtual void   AddText(TGText *text);
   virtual void   AddLine(const char *string);
   virtual void   AddLineFast(const char *string);
   virtual void   Update();
   virtual void   Layout();

   virtual void   SetBackground(Pixel_t p);
   virtual void   SetSelectBack(Pixel_t p);
   virtual void   SetSelectFore(Pixel_t p);
   virtual void   SetForegroundColor(Pixel_t);

   TGText        *GetText() const { return fText; }

   virtual void   SetReadOnly(Bool_t on = kTRUE) { fReadOnly = on; } //*TOGGLE* *GETTER=IsReadOnly
   Bool_t IsReadOnly() const { return fReadOnly; }
   Bool_t IsMarked() const { return fIsMarked; }

   virtual Bool_t HandleDNDDrop(TDNDData *data);
   virtual Atom_t HandleDNDPosition(Int_t x, Int_t y, Atom_t action,
                                    Int_t xroot, Int_t yroot);
   virtual Atom_t HandleDNDEnter(Atom_t * typelist);
   virtual Bool_t HandleDNDLeave();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleSelectionClear(Event_t *event);
   virtual Bool_t HandleSelectionRequest(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleTimer(TTimer *t);
   virtual Bool_t HandleCrossing(Event_t *event);

   virtual void DataChanged() { Emit("DataChanged()"); }  //*SIGNAL*
   virtual void DataDropped(const char *fname) { Emit("DataDropped(char *)", fname); }  //*SIGNAL*
   virtual void Marked(Bool_t mark) { Emit("Marked(Bool_t)", mark); } // *SIGNAL*
   virtual void Clicked(const char *word) { Emit("Clicked(char *)", word); }  //*SIGNAL*
   virtual void DoubleClicked(const char *word) { Emit("DoubleClicked(char *)", word); }  //*SIGNAL*

   ClassDef(TGTextView,0)  // Non-editable text viewer widget
};


class TViewTimer : public TTimer {
private:
   TGView   *fView;

   TViewTimer(const TViewTimer&);             // not implemented
   TViewTimer& operator=(const TViewTimer&);  // not implemented

public:
   TViewTimer(TGView *t, Long_t ms) : TTimer(ms, kTRUE), fView(t) { }
   Bool_t Notify();
};


#endif
