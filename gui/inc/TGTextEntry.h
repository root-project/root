// @(#)root/gui:$Name:  $:$Id: TGTextEntry.h,v 1.7 2000/10/20 15:51:02 rdm Exp $
// Author: Fons Rademakers   08/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEntry
#define ROOT_TGTextEntry


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEntry                                                          //
//                                                                      //
// A TGTextEntry is a one line text input widget.                       //
//                                                                      //
// Changing text in the text entry widget will generate the event:      //
// kC_TEXTENTRY, kTE_TEXTCHANGED, widget id, 0.                         //
// Hitting the enter key will generate:                                 //
// kC_TEXTENTRY, kTE_ENTER, widget id, 0.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif
#ifndef ROOT_TGTextBuffer
#include "TGTextBuffer.h"
#endif

class TBlinkTimer;
class TGToolTip;


class TGTextEntry : public TGFrame, public TGWidget {

friend class TGClient;

public:
   enum   EEchoMode { kNormal, kNoEcho, kPassword };
   enum   EInsertMode { kInsert, kReplace };

protected:
   TGTextBuffer     *fText;              // text buffer
   Int_t             fStartX;            // selection begin in pixels
   Int_t             fEndX;              // selection end in pixels
   Int_t             fStartIX;           // selection begin in characters
   Int_t             fEndIX;             // selection end in characters
   Bool_t            fSelectionOn;       // selection status (on/off)
   Int_t             fOffset;            // start position of text (in pixels)
   Int_t             fCursorX;           // cursor position in pixels
   Int_t             fCursorIX;          // cursor position in characters
   Bool_t            fCursorOn;          // cursor status (on/off)
   FontStruct_t      fFontStruct;        // text font
   GContext_t        fNormGC;            // normal drawing context
   GContext_t        fSelGC, fSelbackGC; // selection mode drawing contexts
   Atom_t            fClipboard;         // clipboard property
   TBlinkTimer      *fCurBlink;          // cursor blink timer
   TGToolTip        *fTip;               // associated tooltip
   Int_t             fMaxLen;            // maximum length of text
   Bool_t            fDeleteGC;          // if kTRUE delete the fNormGC and fSelGC
   Bool_t            fEdited;            // kFALSE, if the line edit's contents have not been changed since the construction
   Bool_t            fFrameDrawn;        // kTRUE draw itself inside a two-pixel frame, kFALSE draw without any frame
   EEchoMode         fEchoMode;          // echo mode (kNormal(default), kNoEcho, kPassword)
   EInsertMode       fInsertMode;        // text insertion mode (kInsert(default) , kReplace)
   ETextJustification fAlignment;        // alignment mode available (kTextLeft(default), kTextRight, kTextCenterX defined in TGWidget.h)

            void        CopyText() const;
   virtual  void        DoRedraw();
            Int_t       GetCharacterIndex(Int_t xcoord);
   virtual  void        Init();
   virtual  Bool_t      IsCursorOutOfFrame();
            void        Paste();
   virtual  void        PastePrimary(Window_t wid, Atom_t property, Bool_t del);
   virtual  void        ScrollByChar();
   virtual  void        UpdateOffset();

   static TString      *fgClipboardText; // application clipboard text
   static Atom_t        fgClipboard;
   static Cursor_t      fgDefaultCursor;
   static FontStruct_t  fgDefaultFontStruct;
   static TGGC          fgDefaultSelectedGC;
   static TGGC          fgDefaultSelectedBackgroundGC;
   static TGGC          fgDefaultGC;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGTextEntry(const TGWindow *p, TGTextBuffer *text, Int_t id = -1,
               GContext_t norm = GetDefaultGC()(),
               FontStruct_t font = GetDefaultFontStruct(),
               UInt_t option = kSunkenFrame | kDoubleBorder,
               ULong_t back = GetWhitePixel());

   TGTextEntry(const TGWindow *parent, const char *text,  Int_t id = -1);
   TGTextEntry(const TString &contents, const TGWindow *parent,  Int_t id = -1);

   virtual ~TGTextEntry();

   virtual  void        AppendText(const char *text);
            void        Backspace();
            void        Clear(Option_t *option="");
            void        CursorLeft(Bool_t mark = kFALSE , Int_t steps = 1);
            void        CursorRight(Bool_t mark = kFALSE , Int_t steps = 1);
            void        CursorWordForward(Bool_t mark = kFALSE);
            void        CursorWordBackward(Bool_t mark = kFALSE);
            void        Cut();
            void        Del();
            void        Deselect();
   virtual  void        DrawBorder();
            void        End(Bool_t mark = kFALSE);
   ETextJustification   GetAlignment() const       { return fAlignment; }
       TGTextBuffer    *GetBuffer() const { return fText; }
            Int_t       GetCursorPosition() const  { return fCursorIX; }
            TString     GetDisplayText() const;
       EEchoMode        GetEchoMode() const        { return fEchoMode; }
       EInsertMode      GetInsertMode() const      { return fInsertMode; }
            TString     GetMarkedText() const;
            Int_t       GetMaxLength() const    { return fMaxLen; }
   const    char       *GetText() const { return fText->GetString(); }
            Bool_t      HasMarkedText() const  { return fSelectionOn && (fStartIX != fEndIX); }
            void        Home(Bool_t mark = kFALSE);
   virtual  void        Insert(const char *);
   virtual  void        InsertText(const char *text, Int_t pos);
            Bool_t      IsFrameDrawn() const       { return fFrameDrawn; }
            Bool_t      IsEdited() const           { return fEdited; }
            void        MarkWord(Int_t pos);
            Int_t       MaxMark() const { return fStartIX > fEndIX ? fStartIX : fEndIX; }
            Int_t       MinMark() const { return fStartIX < fEndIX ? fStartIX : fEndIX; }
            void        NewMark(Int_t pos);
            void        Remove();
   virtual  void        RemoveText(Int_t start, Int_t end);
            void        SelectAll();
   virtual  void        SetAlignment(ETextJustification mode = kTextLeft);
   virtual  void        SetCursorPosition(Int_t pos);
   virtual  void        SetEchoMode(EEchoMode mode = kNormal);
            void        SetEdited(Bool_t flag = kTRUE) { fEdited = flag; }
            void        SetEnabled(Bool_t flag = kTRUE) { SetState( flag ); }
   virtual  void        SetFocus();
   virtual  void        SetFont(FontStruct_t font);
            void        SetFont(const char* fontName);
   virtual  void        SetFrameDrawn(Bool_t flag = kTRUE);
   virtual  void        SetInsertMode(EInsertMode mode = kInsert);
   virtual  void        SetMaxLength(Int_t maxlen);
   virtual  void        SetState(Bool_t state);
   virtual  void        SetText(const char *text);
   virtual  void        SetToolTipText(const char *text, Long_t delayms = 1000);

   virtual  Bool_t      HandleButton(Event_t *event);
   virtual  Bool_t      HandleDoubleClick(Event_t *event);
   virtual  Bool_t      HandleCrossing(Event_t *event);
   virtual  Bool_t      HandleMotion(Event_t *event);
   virtual  Bool_t      HandleKey(Event_t *event);
   virtual  Bool_t      HandleFocusChange(Event_t *event);
   virtual  Bool_t      HandleSelection(Event_t *event);
   virtual  Bool_t      HandleTimer(TTimer *t);
   virtual  Bool_t      HandleConfigureNotify(Event_t *event);

   virtual  void        TextChanged(const char *text = 0);      //*SIGNAL*
   virtual  void        ReturnPressed();                        //*SIGNAL*
   virtual  void        CursorOutLeft();                        //*SIGNAL*
   virtual  void        CursorOutRight();                       //*SIGNAL*
   virtual  void        CursorOutUp();                          //*SIGNAL*
   virtual  void        CursorOutDown();                        //*SIGNAL*
   virtual  void        DoubleClicked();                        //*SIGNAL*

   ClassDef(TGTextEntry,0) // The TGTextEntry widget is a simple line editor for inputting text
};

#endif
