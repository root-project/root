// @(#)root/gui:$Name:  $:$Id: TGTextEdit.h,v 1.4 2000/07/11 09:29:10 rdm Exp $
// Author: Fons Rademakers   1/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEdit
#define ROOT_TGTextEdit


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEdit                                                           //
//                                                                      //
// A TGTextEdit is a specialization of TGTextView. It provides the      //
// text edit functionality to the static text viewing widget.           //
// For the messages supported by this widget see the TGView class.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGTextView
#include "TGTextView.h"
#endif

class TGPopupMenu;
class TGSearchType;


class TGTextEdit : public TGTextView {

friend class TGClient;

public:
   enum EInsertMode { kInsert, kReplace };
   enum {
      kM_FILE_NEW, kM_FILE_OPEN, kM_FILE_CLOSE, kM_FILE_SAVE, kM_FILE_SAVEAS,
      kM_FILE_PRINT, kM_EDIT_CUT, kM_EDIT_COPY, kM_EDIT_PASTE, kM_EDIT_SELECTALL,
      kM_SEARCH_FIND, kM_SEARCH_FINDAGAIN, kM_SEARCH_GOTO
   };

protected:
   GContext_t       fCursor0GC;     // graphics context for erasing cursor
   GContext_t       fCursor1GC;     // graphics context for drawing cursor
   Int_t            fCursorState;   // cursor state (1=drawn, 2=erased)
   TViewTimer      *fCurBlink;      // cursor blink timer
   TGPopupMenu     *fMenu;          // popup menu with editor actions
   TGSearchType    *fSearch;        // structure used by search dialog
   TGLongPosition   fCurrent;       // current cursor position
   EInsertMode      fInsertMode;    // text insertion mode (kInsert (default), kReplace)

   static Cursor_t  fgDefaultCursor;

   void Init();

   virtual void SetMenuState();
   virtual void CursorOn();
   virtual void CursorOff();
   virtual void DrawCursor(Int_t mode);
   virtual void SetCurrent(TGLongPosition new_coord);
   virtual void AdjustPos();

   virtual void InsChar(char character);
   virtual void DelChar();
   virtual void BreakLine();
   virtual void PrevChar();
   virtual void NextChar();
   virtual void LineUp();
   virtual void LineDown();
   virtual void ScreenUp();
   virtual void ScreenDown();
   virtual void Home();
   virtual void End();
   virtual void Copy(TObject &) { MayNotUse("Copy(TObject &)"); }

public:
   TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, Int_t id = -1,
              UInt_t sboptions = 0, ULong_t back = fgWhitePixel);
   TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
              Int_t id = -1, UInt_t sboptions = 0, ULong_t back = fgWhitePixel);
   TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, const char *string,
              Int_t id = -1, UInt_t sboptions = 0, ULong_t back = fgWhitePixel);

   virtual ~TGTextEdit();

   virtual Bool_t SaveFile(const char *fname, Bool_t saveas = kFALSE);
   virtual void   Clear(Option_t * = "");
   virtual Bool_t Copy();
   virtual Bool_t Cut();
   virtual Bool_t Paste();
   virtual void   Print(Option_t * = "");
   virtual void   Delete(Option_t * = "");
   virtual Bool_t Search(const char *string, Bool_t direction, Bool_t caseSensitive);
   virtual Bool_t Replace(TGLongPosition pos, const char *oldText, const char *newText,
                          Bool_t direction, Bool_t caseSensitive);
   virtual Bool_t Goto(Long_t line);
   virtual void   SetInsertMode(EInsertMode mode = kInsert);
   EInsertMode    GetInsertMode() const { return fInsertMode; }
   TGPopupMenu   *GetMenu() const { return fMenu; }

   virtual void   DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual void   ScrollCanvas(Int_t newTop, Int_t direction);

   TGLongPosition GetCurrentPos() const { return fCurrent; }
   virtual Long_t ReturnLongestLineWidth();

   virtual Bool_t HandleTimer(TTimer *t);
   virtual Bool_t HandleSelection (Event_t *event);
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGTextEdit,0)  // Text edit widget
};

#endif
