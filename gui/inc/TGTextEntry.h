// @(#)root/gui:$Name:  $:$Id: TGTextEntry.h,v 1.3 2000/07/11 09:29:10 rdm Exp $
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
   Int_t             fMaxLen;            // maximum length of text
   Bool_t            fDeleteGC;          // if kTRUE delete the fNormGC and fSelGC
   Bool_t            fEdited;            // kFALSE, if the line edit's contents have not been changed since the construction
   Bool_t            fFrameDrawn;        // kTRUE draw itself inside a two-pixel frame, kFALSE draw without any frame
   EEchoMode         fEchoMode;          // echo mode (kNormal(default), kNoEcho, kPassword)
   EInsertMode       fInsertMode;        // text insertion mode (kInsert(default) , kReplace)
   ETextJustification fAlignment;        // alignment mode available (kTextLeft(default), kTextRight, kTextCenterX defined in TGWidget.h)

   static TString      *fgClipboardText; // application clipboard text
   static Cursor_t      fgDefaultCursor;
   static TGGC          fgDefaultGC;
   static TGGC          fgDefaultSelectedGC;
   static TGGC          fgDefaultSelectedBackgroundGC;
   static FontStruct_t  fgDefaultFontStruct;
   static Atom_t        fgClipboard;

            Int_t       GetCharacterIndex(Int_t xcoord);
   virtual  void        DoRedraw();
   virtual  Bool_t      IsCursorOutOfFrame();
   virtual  void        PastePrimary(Window_t wid, Atom_t property, Bool_t del);

            void        MarkWord(Int_t pos);
            Int_t       MinMark() const { return fStartIX < fEndIX ? fStartIX : fEndIX; }
            Int_t       MaxMark() const { return fStartIX > fEndIX ? fStartIX : fEndIX; }

   virtual  void        UpdateOffset();
   virtual  void        ScrollByChar();
   virtual  void        Init();
   virtual  void        SetCursorPosition(Int_t pos);
            void        NewMark(Int_t pos);
            TString     GetDisplayText() const;

            void        CursorLeft(Bool_t mark = kFALSE , Int_t steps = 1);
            void        CursorRight(Bool_t mark = kFALSE , Int_t steps = 1);
            void        CursorWordForward(Bool_t mark = kFALSE);
            void        CursorWordBackward(Bool_t mark = kFALSE);
            void        Backspace();
            void        Del();
            void        Remove();
            void        Home(Bool_t mark = kFALSE);
            void        End(Bool_t mark = kFALSE);
            void        Cut();
            void        CopyText() const;
            void        Paste();
   virtual  void        Insert(const char *);

public:
   TGTextEntry(const TGWindow *p, TGTextBuffer *text, Int_t id = -1,
               GContext_t norm = fgDefaultGC(),
               FontStruct_t font = fgDefaultFontStruct,
               UInt_t option = kSunkenFrame | kDoubleBorder,
               ULong_t back = fgWhitePixel);

   TGTextEntry(const TGWindow *parent, const char *text,  Int_t id = -1);
   TGTextEntry(const TString &contents, const TGWindow *parent,  Int_t id = -1);

   virtual ~TGTextEntry();

   virtual  Bool_t      HandleButton(Event_t *event);
   virtual  Bool_t      HandleDoubleClick(Event_t *event);
   virtual  Bool_t      HandleMotion(Event_t *event);
   virtual  Bool_t      HandleKey(Event_t *event);
   virtual  Bool_t      HandleFocusChange(Event_t *event);
   virtual  Bool_t      HandleSelection(Event_t *event);
   virtual  Bool_t      HandleTimer(TTimer *t);
   virtual  Bool_t      HandleConfigureNotify(Event_t *event);
   virtual  void        DrawBorder();
   virtual  void        SetState(Bool_t state);
   virtual  void        SetFont(FontStruct_t  font);
            void        SetFont(const char* fontName);
       TGTextBuffer    *GetBuffer() const { return fText; }
   const    char       *GetText() const { return fText->GetString(); }
   virtual  void        SetText(const char *text);
   virtual  void        SetFrameDrawn(Bool_t flag = kTRUE);
   virtual  void        AppendText(const char *text);
   virtual  void        InsertText(const char *text, Int_t pos);
   virtual  void        RemoveText(Int_t start, Int_t end);
            Bool_t      IsFrameDrawn() const       { return fFrameDrawn; }
            Bool_t      IsEdited() const           { return fEdited; }
   virtual  void        SetEchoMode(EEchoMode mode = kNormal);
       EEchoMode        GetEchoMode() const        { return fEchoMode; }
   virtual  void        SetInsertMode(EInsertMode mode = kInsert);
       EInsertMode      GetInsertMode() const      { return fInsertMode; }
   virtual  void        SetAlignment(ETextJustification mode = kTextLeft);
   ETextJustification   GetAlignment() const       { return fAlignment; }
            void        SetEnabled(Bool_t flag = kTRUE) { SetState( flag ); }
            Int_t       GetCursorPosition() const  { return fCursorIX; }
            Bool_t      HasMarkedText() const  { return fSelectionOn && (fStartIX != fEndIX); }
            TString     GetMarkedText() const;
            void        SetEdited(Bool_t flag = kTRUE) { fEdited = flag; }
            Int_t       GetMaxLength() const    { return fMaxLen; }
   virtual  void        SetMaxLength(Int_t maxlen);
            void        SelectAll();
            void        Deselect();
            void        Clear(Option_t *option="");
   virtual  void        SetFocus();
//            Bool_t      HasFocus();
   virtual  void        TextChanged(const char* text = 0);      //*SIGNAL*
   virtual  void        ReturnPressed();                        //*SIGNAL*

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGTextEntry,0) // The TGTextEntry widget is a simple line editor for inputting text
};

#endif
