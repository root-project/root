// @(#)root/gui:$Id$
// Author: Fons Rademakers   1/7/2000

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEdit
#define ROOT_TGTextEdit


#include "TGTextView.h"

class TGPopupMenu;
class TGSearchType;
class TGTextEditHist;

class TGTextEdit : public TGTextView {

private:
   TGTextEdit(const TGTextEdit&) = delete;
   TGTextEdit& operator=(const TGTextEdit&) = delete;

public:
   enum EInsertMode { kInsert, kReplace };
   enum {
      kM_FILE_NEW, kM_FILE_OPEN, kM_FILE_CLOSE, kM_FILE_SAVE, kM_FILE_SAVEAS,
      kM_FILE_PRINT, kM_EDIT_CUT, kM_EDIT_COPY, kM_EDIT_PASTE, kM_EDIT_SELECTALL,
      kM_SEARCH_FIND, kM_SEARCH_FINDAGAIN, kM_SEARCH_GOTO
   };

protected:
   GContext_t       fCursor0GC;     ///< graphics context for erasing cursor
   GContext_t       fCursor1GC;     ///< graphics context for drawing cursor
   Int_t            fCursorState;   ///< cursor state (1=drawn, 2=erased)
   TViewTimer      *fCurBlink;      ///< cursor blink timer
   TGPopupMenu     *fMenu;          ///< popup menu with editor actions
   TGSearchType    *fSearch;        ///< structure used by search dialog
   TGLongPosition   fCurrent;       ///< current cursor position
   EInsertMode      fInsertMode;    ///< *OPTION={GetMethod="GetInsertMode";SetMethod="SetInsertMode";Items=(kInsert="&Insert",kReplace="&Replace")}*
   Bool_t           fEnableMenu;    ///< enable context menu with editor actions
   TGTextEditHist  *fHistory;       ///< undo manager
   Bool_t           fEnableCursorWithoutFocus; ///< enable cursor visibility when focus went out from
                                               ///< text editor window (default is kTRUE)

   static TGGC     *fgCursor0GC;
   static TGGC     *fgCursor1GC;

   void Init();

   virtual void SetMenuState();
   virtual void CursorOn();
   virtual void CursorOff();
   virtual void DrawCursor(Int_t mode);
   virtual void AdjustPos();
   void Copy(TObject &) const override { MayNotUse("Copy(TObject &)"); }

   static const TGGC &GetCursor0GC();
   static const TGGC &GetCursor1GC();

public:
   TGTextEdit(const TGWindow *parent = nullptr, UInt_t w = 1, UInt_t h = 1, Int_t id = -1,
              UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());
   TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
              Int_t id = -1, UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());
   TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, const char *string,
              Int_t id = -1, UInt_t sboptions = 0, Pixel_t back = GetWhitePixel());

   virtual ~TGTextEdit();

   virtual Bool_t SaveFile(const char *fname, Bool_t saveas = kFALSE);
           void   Clear(Option_t * = "") override;
           Bool_t Copy() override;
   virtual Bool_t Cut();
   virtual Bool_t Paste();
   virtual void   InsChar(char character);
   virtual void   DelChar();
   virtual void   BreakLine();
   virtual void   PrevChar();
   virtual void   NextChar();
   virtual void   LineUp();
   virtual void   LineDown();
   virtual void   ScreenUp();
   virtual void   ScreenDown();
   virtual void   Home();
   virtual void   End();
           void   Print(Option_t * = "") const override;
           void   Delete(Option_t * = "") override;
           Bool_t Search(const char *string, Bool_t direction = kTRUE, Bool_t caseSensitive = kFALSE) override;
   virtual void   Search(Bool_t close);
   virtual Bool_t Replace(TGLongPosition pos, const char *oldText, const char *newText,
                          Bool_t direction, Bool_t caseSensitive);
   virtual Bool_t Goto(Long_t line, Long_t column = 0);
   virtual void   SetInsertMode(EInsertMode mode = kInsert); //*SUBMENU*
   EInsertMode    GetInsertMode() const { return fInsertMode; }
   TGPopupMenu   *GetMenu() const { return fMenu; }
   virtual void   EnableMenu(Bool_t on = kTRUE) { fEnableMenu = on; } //*TOGGLE* *GETTER=IsMenuEnabled
   virtual Bool_t IsMenuEnabled() const { return fEnableMenu; }
   TList         *GetHistory() const { return (TList *)fHistory; }
   virtual void   EnableCursorWithoutFocus(Bool_t on = kTRUE) { fEnableCursorWithoutFocus = on; }
   virtual Bool_t IsCursorEnabledithoutFocus() const { return fEnableCursorWithoutFocus; }

           void   DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height) override;
           void   ScrollCanvas(Int_t newTop, Int_t direction) override;
   virtual void   SetFocus() { RequestFocus(); }

   virtual void   SetCurrent(TGLongPosition new_coord);
   TGLongPosition GetCurrentPos() const { return fCurrent; }
           Long_t ReturnLongestLineWidth() override;

           Bool_t HandleTimer(TTimer *t) override;
           Bool_t HandleSelection (Event_t *event) override;
           Bool_t HandleButton(Event_t *event) override;
           Bool_t HandleKey(Event_t *event) override;
           Bool_t HandleMotion(Event_t *event) override;
           Bool_t HandleCrossing(Event_t *event) override;
           Bool_t HandleFocusChange(Event_t *event) override;
           Bool_t HandleDoubleClick(Event_t *event) override;
           Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   virtual void   FindAgain() { Emit("FindAgain()"); }  //*SIGNAL*
   virtual void   Closed() { Emit("Closed()"); }        //*SIGNAL*
   virtual void   Opened() { Emit("Opened()"); }        //*SIGNAL*
   virtual void   Saved() { Emit("Saved()"); }          //*SIGNAL*
   virtual void   SavedAs() { Emit("SavedAs()"); }      //*SIGNAL*

   void           SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGTextEdit,0)  // Text edit widget
};

#endif
