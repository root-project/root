// @(#)root/gui:$Name:  $:$Id: TGTextEntry.cxx,v 1.12 2001/08/31 17:49:44 rdm Exp $
// Author: Fons Rademakers   08/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

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
// Hitting the tab key will generate:                                   //
// kC_TEXTENTRY, kTE_TAB, widget id, 0.                                 //
//                                                                      //
// This widget has the behaviour e.g. of the "Location" field in        //
// netscape. That includes handling Control/Shift key modifiers and     //
// scrolling the text.                                                  //
//
//
// enum TGTextEntry::EEchoMode
//
// This enum type describes the ways in which TGTextEntry can display
// its contents. The currently defined values are:
//Begin_Html
/*
<ul>
<li>  kNormal - display characters as they are entered. This is the default.
<li>  kNoEcho - do not display anything.
<li>  kPassword - display asterisks instead of the characters actually entered.
</ul>
*/
//End_Html
// See also SetEchoMode(), GetEchoMode().
//
// enum TGTextEntry::EInsertMode
//
// This enum type describes the way how typed characters are
// inserted in the text entry. This mode is switched by "Insert" key.
//Begin_Html
/*
<ul>
<li>  kInsert - typed character are inserted (cursor has shape of short line).
<li>  kReplace - typed characters substitute already typed ones
                 (cursor has the shape of filled rectangle).
</ul>
*/
//End_Html
//
// enum TGWidget::ETextJustification
//
// This enum type (defined in TGWidget.h) describes the text alignment modes.
// These modes are valid untill text fits the frame width
//Begin_Html
/*
<ul>
<li>  kTextLeft    - left-side text alignment
<li>  kTextRight   - right-side text alignment
<li>  kTextCenterX - center text alignment
</ul>
*/
//End_Html
//
//
// The key press event handler converts a key press to some line editor action.
// Here are the default key bindings:
//Begin_Html
/*
<ul>
<li><i> Left Arrow </i>
        Move the cursor one character leftwards.
        Scroll the text when cursor is out of frame.
<li><i> Right Arrow </i>
        Move the cursor one character rightwards
        Scroll the text when cursor is out of frame.
<li><i> Backspace </i>
        Deletes the character on the left side of the text cursor and moves the
        cursor one position to the left. If a text has been marked by the user
        (e.g. by clicking and dragging) the cursor will be put at the beginning
        of the marked text and the marked text will be removed.
<li><i> Home </i>
        Moves the text cursor to the left end of the line. If mark is TRUE text
        will be marked towards the first position, if not any marked text will
        be unmarked if the cursor is moved.
<li><i> End </i>
        Moves the text cursor to the right end of the line. If mark is TRUE text
        will be marked towards the last position, if not any marked text will
        be unmarked if the cursor is moved.
<li><i> Delete </i>
        Deletes the character on the right side of the text cursor. If a text
        has been marked by the user (e.g. by clicking and dragging) the cursor
        will be put at the beginning of the marked text and the marked text will
        be removed.
<li><i> Insert </i>
        Switches character insert mode.
<li><i> Shift - Left Arrow </i>
        Mark text one character leftwards
<li><i> Shift - Right Arrow </i>
        Mark text one character rightwards
<li><i> Control - Left Arrow </i>
        Move the cursor one word leftwards
<li><i> Control - Right Arrow </i>
        Move the cursor one word rightwards.
<li><i> Control - Shift - Left Arrow </i>
        Mark text one word leftwards
<li><i> Control - Shift - Right Arrow </i>
        Mark text one word rightwards
<li><i> Control-A </i>
        Move the cursor to the beginning of the line
<li><i> Control-B </i>
        Move the cursor one character leftwards
<li><i> Control-C </i>
        Copy the marked text to the clipboard.
<li><i> Control-D </i>
        Delete the character to the right of the cursor
<li><i> Control-E </i>
        Move the cursor to the end of the line
<li><i> Control-F </i>
        Move the cursor one character rightwards
<li><i> Control-H </i>
        Delete the character to the left of the cursor
<li><i> Control-K </i>
        Delete marked text if any or delete all
        characters to the right of the cursor
<li><i> Control-U </i>
        Delete all characters on the line
<li><i> Control-V </i>
        Paste the clipboard text into line edit.
<li><i> Control-X </i>
        Cut the marked text, copy to clipboard.
<li><i> Control-Y </i>
        Paste the clipboard text into line edit.
</ul>
All other keys with valid ASCII codes insert themselves into the line.
*/
//End_Html
//
////////////////////////////////////////////////////////////////////////////////

//******************* TGTextEntry signals *************************************
//______________________________________________________________________________
// TGTextEntry::ReturnPressed()
//
//    This signal is emitted when the return or enter key is pressed.
//
//______________________________________________________________________________
// TGTextEntry::TabPressed()
//
//    This signal is emitted when the <TAB> key is pressed.
//    Use for changing focus.
//
//______________________________________________________________________________
// TGTextEntry::TextChanged(const char *text)
//
//    This signal is emitted every time the text has changed.
//    The argument is the new text.
//
//______________________________________________________________________________
// TGTextEntry::CursorOutLeft()
//
// This signal is emitted when cursor is going out of left side.
//
//______________________________________________________________________________
// TGTextEntry::CursorOutRight()
//
// This signal is emitted when cursor is going out of right side.
//
//______________________________________________________________________________
// TGTextEntry::CursorOutUp()
//
// This signal is emitted when cursor is going out of upper side.
//
//______________________________________________________________________________
// TGTextEntry::CursorOutDown()
//
// This signal is emitted when cursor is going out of bottom side.
//
//______________________________________________________________________________
// TGTextEntry::DoubleClicked()
//
// This signal is emitted when widget is double clicked.


#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTimer.h"
#include "KeySymbols.h"



//______________________________________________________________________________
class TBlinkTimer : public TTimer {
private:
   TGTextEntry   *fTextEntry;
public:
   TBlinkTimer(TGTextEntry *t, Long_t ms) : TTimer(ms, kTRUE) { fTextEntry = t; }
   Bool_t Notify();
};

//______________________________________________________________________________
Bool_t TBlinkTimer::Notify()
{
   fTextEntry->HandleTimer(0);
   Reset();
   return kFALSE;
}


ClassImp(TGTextEntry)

TString *TGTextEntry::fgClipboardText = 0;   // application clipboard prototype

//______________________________________________________________________________
TGTextEntry::TGTextEntry(const TGWindow *p, TGTextBuffer *text, Int_t id,
                         GContext_t norm, FontStruct_t font, UInt_t options,
                         ULong_t back) :
   TGFrame(p, 1, 1, options | kOwnBackground, back)
{
   // Create a text entry widget. It will adopt the TGTextBuffer object
   // (i.e. the text buffer will be deleted by the text entry widget).

   fWidgetId      = id;
   fMsgWindow     = p;
   fNormGC        = norm;
   fFontStruct    = font;
   fText          = text;

   Init();
}

//______________________________________________________________________________
TGTextEntry::TGTextEntry(const TGWindow *parent, const char *text, Int_t id) :
   TGFrame(parent, 1, 1, kSunkenFrame | kDoubleBorder | kOwnBackground, fgWhitePixel)
{
   // Simple text entry constructor.

   fWidgetId      = id;
   fMsgWindow     = parent;
   fNormGC        = fgDefaultGC();
   fFontStruct    = fgDefaultFontStruct;
   fText          = new TGTextBuffer();
   fText->AddText(0, text);

   Init();                             // default initialization
}

//______________________________________________________________________________
TGTextEntry::TGTextEntry(const TString &contents, const TGWindow *parent, Int_t id) :
   TGFrame(parent, 1, 1, kSunkenFrame | kDoubleBorder | kOwnBackground, fgWhitePixel)
{
   // Simple test entry constructor. Notice TString argument comes before the
   // parent argument (to make this ctor different from the first one taking a
   // const char*).

   fWidgetId      = id;
   fMsgWindow     = parent;
   fNormGC        = fgDefaultGC();
   fFontStruct    = fgDefaultFontStruct;
   fText          = new TGTextBuffer();
   fText->AddText(0, contents.Data());

   Init();                             // default initialization
}

//______________________________________________________________________________
TGTextEntry::~TGTextEntry()
{
   // Delete a text entry widget.

   if (fDeleteGC) {
      gVirtualX->DeleteGC(fNormGC);
      gVirtualX->DeleteGC(fSelGC);
   }
   delete fText;
   delete fCurBlink;
   delete fTip;
}

//______________________________________________________________________________
void TGTextEntry::Init()
{
   // Do default initialization.

   fWidgetFlags = kWidgetWantFocus | kWidgetIsEnabled;
   fSelGC       = fgDefaultSelectedGC();
   fSelbackGC   = fgDefaultSelectedBackgroundGC();
   fDeleteGC    = kFALSE;

   fOffset = 0;
   fMaxLen = 255;             // TString::Length() can not exceed 255 characters
   fFrameDrawn = kTRUE;
   fEdited = kFALSE;
   fEchoMode = kNormal;
   fAlignment= kTextLeft;
   fInsertMode = kInsert;

   int tw, max_ascent, max_descent;
   tw = gVirtualX->TextWidth(fFontStruct, GetText(), fText->GetTextLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   Resize(tw + 8, max_ascent + max_descent + 7);

   Int_t offset = IsFrameDrawn() ? 4 : 0;
   fCursorX     = offset ;
   fCursorIX    = fStartIX = fEndIX = fOffset = 0;
   fSelectionOn = fCursorOn = kFALSE;
   fCurBlink    = 0;
   fTip         = 0;
   fClipboard   = fgClipboard;

   gVirtualX->SetCursor(fId, fgDefaultCursor);

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kButtonMotionMask, kNone, kNone);

   AddInput(kKeyPressMask | kFocusChangeMask |
            kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
void TGTextEntry::ReturnPressed()
{
   // This signal is emitted when the return or enter key is pressed.

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTENTRY, kTE_ENTER), fWidgetId, 0);
   fClient->ProcessLine(fCommand, MK_MSG(kC_TEXTENTRY, kTE_ENTER),fWidgetId, 0);

   Emit("ReturnPressed()");
}

//______________________________________________________________________________
void TGTextEntry::TabPressed()
{
   // This signal is emitted when the <TAB> key is pressed.

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTENTRY, kTE_TAB), fWidgetId, 0);
   fClient->ProcessLine(fCommand, MK_MSG(kC_TEXTENTRY, kTE_TAB), fWidgetId, 0);

   Emit("TabPressed()");
}

//______________________________________________________________________________
void TGTextEntry::TextChanged(const char *)
{
   // This signal is emitted every time the text has changed.

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTENTRY, kTE_TEXTCHANGED),fWidgetId, 0);
   fClient->ProcessLine(fCommand, MK_MSG(kC_TEXTENTRY, kTE_TEXTCHANGED),fWidgetId, 0);

   Emit("TextChanged(char*)", GetText());  // The argument is the new text.
}

//______________________________________________________________________________
void TGTextEntry::CursorOutLeft()
{
   // This signal is emitted when cursor is going out of left side.

   Emit("CursorOutLeft()");
}

//______________________________________________________________________________
void TGTextEntry::CursorOutRight()
{
   // This signal is emitted when cursor is going out of right side.

   Emit("CursorOutRight()");
}

//______________________________________________________________________________
void TGTextEntry::CursorOutUp()
{
   // This signal is emitted when cursor is going out of upper side.

   Emit("CursorOutUp()");
}

//______________________________________________________________________________
void TGTextEntry::CursorOutDown()
{
   // This signal is emitted when cursor is going out of bottom side.

   Emit("CursorOutDown()");
}

//______________________________________________________________________________
void TGTextEntry::DoubleClicked()
{
   // This signal is emitted when widget is double clicked.

   Emit("DoubleClicked()");
}

//______________________________________________________________________________
TString TGTextEntry::GetDisplayText() const
{
   // Returns the text that's currently displayed.  This is normally
   // the same as GetText(), but can be e.g.
   // "*****" if EEchoMode is kPassword or
   // ""      if it is kNoEcho.

   TString res;

   switch (GetEchoMode()) {
   case kNormal:
         res = GetText();
         break;
   case kNoEcho:
         res = "";
         break;
   case kPassword:
         res.Prepend('*', fText->GetTextLength());  // fill with '*'
         break;
    }
    return res;
}

//______________________________________________________________________________
void TGTextEntry::SetFont(FontStruct_t font)
{
   // Changes text entry font.

   if (font != fFontStruct) {
      GCValues_t gval;
      fFontStruct = font;

      // get unique copies of the norm GC and the sel GC (if not already copied)
      if (!fDeleteGC) {
         GContext_t norm = gVirtualX->CreateGC(fId, 0);
         gVirtualX->CopyGC(fNormGC, norm, 0);
         fNormGC = norm;
         GContext_t sel  = gVirtualX->CreateGC(fId, 0);
         gVirtualX->CopyGC(fSelGC, sel, 0);
         fSelGC = sel;
      }
      gval.fMask = kGCFont;
      gval.fFont = gVirtualX->GetFontHandle(fFontStruct);
      gVirtualX->ChangeGC(fNormGC,&gval);
      gVirtualX->ChangeGC(fSelGC,&gval);
      fClient->NeedRedraw(this);
      fDeleteGC = kTRUE;
   }
}

//______________________________________________________________________________
void TGTextEntry::SetFont(const char *fontName)
{
   // Changes text entry font specified by name.

   FontStruct_t tfont = gClient->GetFontByName(fontName);
   if (!tfont) return;
   SetFont(tfont);
}

//______________________________________________________________________________
void TGTextEntry::SetState(Bool_t state)
{
   // Set state of widget. If kTRUE=enabled, kFALSE=disabled.

   if (state) {
      SetFlags(kWidgetIsEnabled);
      SetBackgroundColor(fgWhitePixel);
   } else {
      ClearFlags(kWidgetIsEnabled);
      SetBackgroundColor(fgDefaultFrameBackground);
      fCursorOn = kFALSE;   // remove the cursor when disabling the widget
      if (fCurBlink) fCurBlink->Remove();
   }
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Int_t TGTextEntry::GetCharacterIndex(Int_t xcoord)
{
   // Returns the index of the character to whose left edge goalx is closest.

   int tw, ix, up, down, len;

   // check for out of boundaries first...
   TString dt = GetDisplayText();
   len = dt.Length();
   tw = gVirtualX->TextWidth(fFontStruct, dt.Data(), len);
   if (xcoord < 0) return 0;
   if (xcoord > tw) return len; // len-1

   // do a binary approximation
   up = len; //-1
   down = 0;
   while (up-down > 1) {
      ix = (up+down) >> 1;
      tw = gVirtualX->TextWidth(fFontStruct, fText->GetString(), ix);
      if (tw > xcoord)
         up = ix;
      else
         down = ix;
      if (tw == xcoord) break;
   }
   ix = down;

   // safety check...
   ix = TMath::Max(ix, 0);
   ix = TMath::Min(ix, len); // len-1

   return ix;
}

//______________________________________________________________________________
void TGTextEntry::SetFrameDrawn(Bool_t enable)
{
   // Sets the text entry to draw itself inside a two-pixel frame if
   // enable is kTRUE, and to draw itself without any frame if enable is
   // kFALSE. The default is kTRUE.

   if (fFrameDrawn == enable) return;

   fFrameDrawn = enable;
   fClient->NeedRedraw(this);
   // ChangedBy("SetFrameDrawn");  // emit signal ChangedBy
}

//______________________________________________________________________________
void TGTextEntry::SetAlignment(ETextJustification mode)
{
   // Sets the alignment of the text entry.
   // Possible values are kTextLeft(default), kTextRight, kTextCenterX.
   // See also GetAlignment().

   if ((mode == kTextRight ||
        mode == kTextCenterX ||
        mode == kTextLeft)) {

      fAlignment = mode;
      UpdateOffset();
      fClient->NeedRedraw(this);
      // ChangedBy("SetAlignment");  // emit signal ChangedBy
   }
}

//______________________________________________________________________________
void TGTextEntry::SetInsertMode(EInsertMode mode)
{
   // Sets the mode how characters are entered to the text entry.

   if (fInsertMode == mode) return;

   fInsertMode = mode;
   fClient->NeedRedraw(this);
   // ChangedBy("SetInsertMode");  // emit signal ChangedBy
}

//______________________________________________________________________________
void TGTextEntry::SetText(const char *text)
{
   // Sets text entry to text, clears the selection and moves
   // the cursor to the end of the line.
   // If necessary the text is truncated to fit MaxLength().
   // See also  GetText().

   TString oldText(GetText());

   fText->Clear();
   fText->AddText(0, text); // new text

   Int_t dif = fText->GetTextLength() - fMaxLen;
   if (dif > 0) fText->RemoveText(fMaxLen, dif);       // truncate

   End(kFALSE);
   if (oldText != GetText()) {
      TextChanged();            // emit signal
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
void TGTextEntry::SetMaxLength(Int_t maxlen)
{
   // Set the maximum length of the text in the editor.  If the text is
   // currently too long, it is chopped off at the limit. Any marked text will
   // be unmarked.  The cursor position is set to 0 and the first part of the
   // string is shown.   The range  of maxlen  is (0,255)
   // See  also GetMaxLength().

   fMaxLen = maxlen<0 ? 0 : TMath::Min(255,maxlen) ; // safety check for maxlen<0 and maxlen>255

   Int_t dif = fText->GetTextLength() - fMaxLen;
   if (dif > 0) fText->RemoveText(fMaxLen, dif);    // truncate

   SetCursorPosition(0);
   Deselect();

   // ChangedBy("SetMaxLength");  // emit signal ChangedBy
}

//______________________________________________________________________________
void TGTextEntry::SetEchoMode(EEchoMode mode)
{
   // The echo modes available are:
   //Begin_Html
   // <ul>
   // <li> kNormal   - display characters as they are entered.  This is the default.
   // <li> kNoEcho   - do not display anything.
   // <li> kPassword - display asterisks instead of the characters actually entered.
   // </ul>
   //End_Html
   // It is always possible to cut and paste any marked text;  only the widget's own
   // display is affected.
   // See also GetEchoMode(), GetDisplayText().

   if (fEchoMode == mode) return;

   Int_t offset = IsFrameDrawn() ? 4 : 0;
   fEchoMode = mode;
   if (GetEchoMode() == kNoEcho) { fCursorX = offset; }
   UpdateOffset();
   fClient->NeedRedraw(this);
   // ChangedBy("SetEchoMode");  // emit signal ChangedBy
}

//______________________________________________________________________________
TString TGTextEntry::GetMarkedText() const
{
   // Returns the text marked by the user (e.g. by clicking and
   // dragging), or zero if no text is marked.
   // See also HasMarkedText().

   Int_t minP = MinMark();
   Int_t len = MaxMark() - minP;
   TString res(GetText()+minP,len);
   return res;
}

//______________________________________________________________________________
void TGTextEntry::NewMark(Int_t newPos)
{
   // New character mark at position pos.
   // See also SetCursorPosition().

   TString dt = GetDisplayText();
   Int_t offset = IsFrameDrawn() ? 4 : 0;
   Int_t x = fOffset + offset;
   Int_t len = dt.Length();

   Int_t pos = newPos < len ? newPos : len;
   fEndIX = pos < 0 ? 0 : pos;

   fSelectionOn = fSelectionOn && (fEndIX != fStartIX) && (GetEchoMode() != kNoEcho) ;
   SetCursorPosition(pos);

   if (fSelectionOn) {
      fEndX =  x + gVirtualX->TextWidth(fFontStruct, dt.Data() , fEndIX);
      fStartX = x + gVirtualX->TextWidth(fFontStruct, dt.Data() , fStartIX);
   }
}

//______________________________________________________________________________
void TGTextEntry::SetCursorPosition(Int_t newPos)
{
   // Set the cursor position to newPos.
   // See also NewMark().

   Int_t offset = IsFrameDrawn() ? 4 : 0;
   if (GetEchoMode() == kNoEcho) { fCursorX = offset; return; }

   UpdateOffset();
   TString dt = GetDisplayText();

   Int_t x = fOffset + offset;
   Int_t len = dt.Length();

   Int_t pos;

   if (newPos < len)
      pos = newPos;
   else {
      pos = len;
      if (newPos > len) CursorOutRight();
   }

   if (pos < 0) {
      fCursorIX = 0;
      CursorOutLeft();
   } else
      fCursorIX = pos;

   fCursorX = x + gVirtualX->TextWidth(fFontStruct, dt.Data(), fCursorIX);

   if (!fSelectionOn){
      fStartIX = fCursorIX;
      fStartX  = fCursorX;
   }
}

//______________________________________________________________________________
void TGTextEntry::MarkWord(Int_t pos)
{
   // Marks the word nearest to cursor position.
   // See also HandleDoubleClick().

   Int_t i = pos - 1;
   while (i >= 0 && isprint(GetText()[i]) && !isspace(GetText()[i])) i--;
   i++;
   Int_t newStartIX = i;

   i = pos;
   while (isprint(GetText()[i]) && !isspace(GetText()[i])) i++;
   while(isspace(GetText()[i])) i++;

   fSelectionOn = kTRUE;
   fStartIX = newStartIX;
   fEndIX = i;
   NewMark(i);
}

//______________________________________________________________________________
void TGTextEntry::Insert(const char *newText)
{
   // Removes any currently selected text, inserts newText,
   // sets it as the new contents of the text entry.

   TString old(GetText());
   TString t(newText);

   if (t.IsNull()) return;

   for (int i=0; i<t.Length(); i++) {
      if (t[i] < ' ') t[i] = ' '; // unprintable/linefeed becomes space
   }

   Int_t minP = MinMark();
   Int_t maxP = MaxMark();
   Int_t cp = fCursorIX;

   if (HasMarkedText()) {
      fText->RemoveText(minP, maxP-minP);
      cp = minP;
   }

   if (fInsertMode == kReplace) fText->RemoveText(cp,t.Length());
   Int_t ncp = TMath::Min(cp+t.Length(), GetMaxLength());
   fText->AddText(cp, t.Data());
   Int_t dlen = fText->GetTextLength()-GetMaxLength();
   if (dlen>0) fText->RemoveText(GetMaxLength(),dlen); // truncate

   SetCursorPosition(ncp);
   if (old != GetText()) TextChanged();
}

//______________________________________________________________________________
void TGTextEntry::CursorRight(Bool_t mark, Int_t steps)
{
   // Moves the cursor rightwards one or more characters.
   // See also CursorLeft().

   Int_t cp = fCursorIX + steps;

   if (cp == fCursorIX)  {
      if (!mark) {
         fSelectionOn = kFALSE;
         fEndIX = fStartIX = fCursorIX;
      }
   } else if (mark) {
      fSelectionOn = kTRUE;
      NewMark(cp);
   } else {
      fSelectionOn = kFALSE;
      SetCursorPosition(cp);
   }
}

//______________________________________________________________________________
void TGTextEntry::CursorLeft(Bool_t mark, Int_t steps)
{
   // Moves the cursor leftwards one or more characters.
   // See also CursorRight().

   CursorRight(mark, -steps);
}

//______________________________________________________________________________
void TGTextEntry::CursorWordForward(Bool_t mark)
{
   // Moves the cursor one word to the right.  If mark is kTRUE, the text
   // is marked.
   // See also CursorWordBackward().

   Int_t i = fCursorIX;
   while (i < (Int_t)fText->GetTextLength() && !isspace(GetText()[i])) ++i;
   while (i < (Int_t)fText->GetTextLength() && isspace(GetText()[i])) ++i;
   CursorRight(mark, i - fCursorIX);
}

//______________________________________________________________________________
void TGTextEntry::CursorWordBackward(Bool_t mark)
{
   // Moves the cursor one word to the left.  If mark is kTRUE, the text
   // is marked.
   // See also CursorWordForward().

   Int_t i = fCursorIX;
   while (i > 0 && isspace(GetText()[i-1])) --i;
   while (i > 0 && !isspace(GetText()[i-1])) --i;
   CursorLeft(mark,  fCursorIX - i);
}

//______________________________________________________________________________
void TGTextEntry::Backspace()
{
   // Deletes the character on the left side of the text cursor and moves the
   // cursor one position to the left. If a text has been marked by the user
   // (e.g. by clicking and dragging) the cursor will be put at the beginning
   // of the marked text and the marked text will be removed.
   // See also  Del().

   if (HasMarkedText())  {
      Del();
   } else if (fCursorIX > 0) {
      CursorLeft(kFALSE);
      Del();
   }
}

//______________________________________________________________________________
void TGTextEntry::Del()
{
   // Deletes the character on the right side of the text cursor. If a text
   // has been marked by the user (e.g. by clicking and dragging) the cursor
   // will be put at the beginning of the marked text and the marked text will
   // be removed.
   // See also Backspace().

   Int_t minP = MinMark();
   Int_t maxP = MaxMark();

   if (HasMarkedText())  {
      fText->RemoveText(minP, maxP-minP);
      fSelectionOn = kFALSE;
      SetCursorPosition(minP);
   }  else if (fCursorIX != (Int_t)fText->GetTextLength()) {
      fSelectionOn = kFALSE;
      fText->RemoveText(fCursorIX , 1);
      SetCursorPosition(fCursorIX);
   }
   TextChanged();
}

//______________________________________________________________________________
void TGTextEntry::Remove()
{
   // Deletes all characters on the right side of the cursor.
   // See also Del() Backspace().

   if (fCursorIX < (Int_t)fText->GetTextLength()) {
      fText->RemoveText(fCursorIX , fText->GetTextLength() - fCursorIX);
      SetCursorPosition(fCursorIX);
      TextChanged();                      // emit signal
   }
}

//______________________________________________________________________________
void TGTextEntry::CopyText() const
{
   // Copies the marked text to the clipboard, if there is any and
   // GetEchoMode() is kNormal.
   // See also  Cut() Paste().

   if (HasMarkedText() && GetEchoMode() == kNormal) {
      if (!fgClipboardText) fgClipboardText = new TString();
      *fgClipboardText = GetMarkedText();  // assign
      gVirtualX->SetPrimarySelectionOwner(fId);
   }
}

//______________________________________________________________________________
void TGTextEntry::Paste()
{
   // Inserts text at the cursor position, deleting any
   // previous marked text.
   // See also CopyText() Cut().

   if (gVirtualX->GetPrimarySelectionOwner() == kNone) {
      // No primary selection, so use the buffer
      if (fgClipboardText) Insert(fgClipboardText->Data());
   } else {
      gVirtualX->ConvertPrimarySelection(fId, fClipboard, 0);
   }
}

//______________________________________________________________________________
void TGTextEntry::Cut()
{
   // Copies the marked text to the clipboard and deletes it, if there is any.
   // See also CopyText() Paste().

   if (HasMarkedText()) {
      CopyText();
      Del();
   }
}

//______________________________________________________________________________
void TGTextEntry::Clear(Option_t *)
{
   // Clears up the text entry.

   SetText("");
}

//______________________________________________________________________________
void TGTextEntry::Home(Bool_t mark)
{
   // Moves the text cursor to the left end of the line. If mark is kTRUE text
   // will be marked towards the first position, if not any marked text will
   // be unmarked if the cursor is moved.
   // See also End().

   fOffset = 0;
   if (mark){
      fSelectionOn = kTRUE;
      fStartIX = fCursorIX;
      NewMark(0);
   } else {
      fSelectionOn = kFALSE;
      SetCursorPosition(0);
   }
}

//______________________________________________________________________________
void TGTextEntry::End(Bool_t mark)
{
   // Moves the text cursor to the right end of the line. If mark is kTRUE text
   // will be marked towards the last position, if not any marked text will
   // be unmarked if the cursor is moved.
   // See also Home().

   TString dt = GetDisplayText();
   Int_t len  = dt.Length();

   fOffset = (Int_t)GetWidth() - gVirtualX->TextWidth(fFontStruct, dt.Data(), len);
   if (fOffset > 0) fOffset = 0;

   if (mark){
      fSelectionOn = kTRUE;
      fStartIX = fCursorIX;
      NewMark(len);
   } else {
      fSelectionOn = kFALSE;
      SetCursorPosition(len);
   }
}

//______________________________________________________________________________
void TGTextEntry::SelectAll()
{
   // Selects all text (i.e. marks it) and moves the cursor to the
   // end. Useful when a default value has been inserted. If the user
   // types before clicking on the widget the selected text will be
   // erased.

   fSelectionOn = kTRUE;
   fStartIX = 0;
   NewMark(fText->GetTextLength());
   DoRedraw();
}

//______________________________________________________________________________
void TGTextEntry::Deselect()
{
   // Deselects all text (i.e. removes marking) and leaves the cursor at the
   // current position.

   fSelectionOn = kFALSE;
   fEndIX = fStartIX = fCursorIX;
   DoRedraw();
}

//______________________________________________________________________________
void TGTextEntry::DrawBorder()
{
   // Draw the border of the text entry widget.

   switch (fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder)) {
      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, fgShadowGC(), 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgShadowGC(), 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgBlackGC(), 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgBlackGC(), 1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgHilightGC(), 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgHilightGC(), fWidth-1, fHeight-1, fWidth-1, 0);
         gVirtualX->DrawLine(fId, fgBckgndGC(),  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgBckgndGC(),  fWidth-2, 1, fWidth-2, fHeight-2);
         break;

      default:
         TGFrame::DrawBorder();
         break;
   }
}

//______________________________________________________________________________
void TGTextEntry::DoRedraw()
{
   // Draw the text entry widget.

   Int_t x, y, max_ascent, max_descent;
   Int_t offset = IsFrameDrawn() ? 4 : 0;
   TString dt  = GetDisplayText();               // text to be displayed
   Int_t len   = dt.Length();                    // length of displayed text

   // TGFrame::DoRedraw() == drawing border twice
   Int_t border = IsFrameDrawn() ? fBorderWidth : 0;

   gVirtualX->ClearArea(fId, border,  border,
            fWidth - (border << 1), fHeight - (border << 1));

   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);

   y = (GetHeight() - (max_ascent + max_descent))/2 ;     // center y
   x = fOffset + offset;

   if (fEchoMode == kNoEcho) {
      fSelectionOn = kFALSE;
      fCursorX = offset;
   }

   if ((GetInsertMode() == kInsert) || (fEchoMode == kNoEcho)) {
      // line cursor
      if (fCursorOn) {
         gVirtualX->DrawLine(fId, fgBlackGC(), fCursorX, 3,
                     fCursorX, max_ascent + max_descent + 3);
      }
      gVirtualX->DrawString(fId, fNormGC, x , y + max_ascent, dt.Data(), len);

   } else {
      // filled rectangle (block) cursor
      gVirtualX->DrawString(fId, fNormGC, x , y + max_ascent, dt.Data(), len);

      if (fCursorOn) {
         Int_t ind       = fCursorIX < len-1 ? fCursorIX : len-1;
         Int_t charWidth = ind < 0 ||  fCursorIX > len -1 ? 4 :
                           gVirtualX->TextWidth(fFontStruct, &dt[ind],1);

         Int_t before = gVirtualX->TextWidth(fFontStruct, dt, fCursorIX) + x;

         gVirtualX->FillRectangle(fId, fSelbackGC , before, 3 ,
                                  charWidth , max_ascent + max_descent + 1);

         if (fCursorIX < len)
            gVirtualX->DrawString(fId, fSelGC, before , y + max_ascent, &dt[ind], 1);
      }
   }

  if (fSelectionOn) {
    int xs, ws, ixs, iws;

      xs  = TMath::Min(fStartX, fEndX);
      ws  = TMath::Abs(fEndX - fStartX);
      ixs = TMath::Min(fStartIX, fEndIX);
      iws = TMath::Abs(fEndIX - fStartIX);

      gVirtualX->FillRectangle(fId, fSelbackGC, xs , 3, ws,
                               max_ascent + max_descent + 1);

      gVirtualX->DrawString(fId, fSelGC, xs, y + max_ascent,
                            dt.Data()+ixs, iws);
   }
   if (IsFrameDrawn()) DrawBorder();
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleKey(Event_t* event)
{
   //  The key press event handler converts a key press to some line editor action.
   //  Here are the default key bindings:
   //Begin_Html
   //  <ul>
   //  <li><i> Left Arrow </i>
   //          Move the cursor one character leftwards.
   //          Scroll the text when  cursor is out of frame.
   //  <li><i> Right Arrow </i>
   //          Move the cursor one character rightwards
   //          Scroll the text when  cursor is out of frame.
   //  <li><i> Backspace </i>
   //          Deletes the character on the left side of the text cursor and moves the
   //          cursor one position to the left. If a text has been marked by the user
   //          (e.g. by clicking and dragging) the cursor will be put at the beginning
   //          of the marked text and the marked text will be removed.
   //  <li><i> Home </i>
   //          Moves the text cursor to the left end of the line. If mark is TRUE text
   //          will be marked towards the first position, if not any marked text will
   //          be unmarked if the cursor is moved.
   //  <li><i> End </i>
   //          Moves the text cursor to the right end of the line. If mark is TRUE text
   //          will be marked towards the last position, if not any marked text will
   //          be unmarked if the cursor is moved.
   //  <li><i> Delete </i>
   //          Deletes the character on the right side of the text cursor. If a text
   //          has been marked by the user (e.g. by clicking and dragging) the cursor
   //          will be put at the beginning of the marked text and the marked text will
   //          be removed.
   //  <li><i> Insert </i>
   //          Switches character insert mode.
   //  <li><i> Shift - Left Arrow </i>
   //          Mark text one character leftwards
   //  <li><i> Shift - Right Arrow </i>
   //          Mark text one character rightwards
   //  <li><i> Control - Left Arrow </i>
   //          Move the cursor one word leftwards
   //  <li><i> Control - Right Arrow </i>
   //          Move the cursor one word rightwards.
   //  <li><i> Control - Shift - Left Arrow </i>
   //          Mark text one word leftwards
   //  <li><i> Control - Shift - Right Arrow </i>
   //          Mark text one word rightwards
   //  <li><i> Control-A </i>
   //          Move the cursor to the beginning of the line
   //  <li><i> Control-B </i>
   //          Move the cursor one character leftwards
   //  <li><i> Control-C </i>
   //          Copy the marked text to the clipboard.
   //  <li><i> Control-D </i>
   //          Delete the character to the right of the cursor
   //  <li><i> Control-E </i>
   //          Move the cursor to the end of the line
   //  <li><i> Control-F </i>
   //          Move the cursor one character rightwards
   //  <li><i> Control-H </i>
   //          Delete the character to the left of the cursor
   //  <li><i> Control-K </i>
   //          Delete marked text if any or delete all
   //          characters to the right of the cursor
   //  <li><i> Control-U </i>
   //          Delete all characters on the line
   //  <li><i> Control-V </i>
   //          Paste the clipboard text into line edit.
   //  <li><i> Control-X </i>
   //          Cut the marked text, copy to clipboard.
   //  <li><i> Control-Y </i>
   //          Paste the clipboard text into line edit.
   //  </ul>
   //End_Html
   //  All other keys with valid ASCII codes insert themselves into the line.

   Int_t  n;
   char   tmp[10];
   UInt_t keysym;

   if (fTip && event->fType == kGKeyPress) fTip->Hide();

   if (!IsEnabled()) return kTRUE;

   gVirtualX->LookupString(event, tmp, sizeof(tmp), keysym);
   n = strlen(tmp);
   Int_t unknown = 0;

   if ((EKeySym)keysym  == kKey_Enter || (EKeySym)keysym  == kKey_Return) {

      ReturnPressed();                                      // emit signal
      if (!TestBit(kNotDeleted)) return kTRUE;
      fSelectionOn = kFALSE;

   } else if ((EKeySym)keysym  == kKey_Tab) {

      TabPressed();                                         // emit signal
      fSelectionOn = kFALSE;

   } else if (event->fState & kKeyControlMask) {  // Cntrl key modifier pressed
      switch ((EKeySym)keysym & ~0x20) {     // treat upper and lower the same
      case kKey_A:
         Home(event->fState & kKeyShiftMask);
         break;
      case kKey_B:
         CursorLeft(event->fState & kKeyShiftMask);
         break;
      case kKey_C:
         CopyText();
         break;
      case kKey_D:
         Del();
         break;
      case kKey_E:
         End(event->fState & kKeyShiftMask);
         break;
      case kKey_F:
         CursorRight(event->fState & kKeyShiftMask);
         break;
      case kKey_H:
         Backspace();
         break;
      case kKey_K:
         HasMarkedText() ? Del() : Remove();
         break;
      case kKey_U:
         Home();
         Remove();
         break;
      case kKey_V:
         Paste();
         break;
      case kKey_X:
         Cut();
         break;
      case kKey_Y:
         Paste();
         break;
      case kKey_Right:
         CursorWordForward(event->fState & kKeyShiftMask);
         break;
      case kKey_Left:
         CursorWordBackward(event->fState & kKeyShiftMask);
         break;
      default:
         unknown++;
      }
   } else if (n && keysym <127 && keysym >=32  &&     // printable keys
               (EKeySym)keysym  != kKey_Delete &&
               (EKeySym)keysym  != kKey_Backspace) {

      Insert(tmp);
      fSelectionOn = kFALSE;

   } else {
      switch ((EKeySym)keysym) {
      case kKey_Down:
         CursorOutDown();
         break;
      case kKey_Up:
         CursorOutUp();
         break;
      case kKey_Left:
         CursorLeft(event->fState & kKeyShiftMask);
         break;
      case kKey_Right:
         CursorRight(event->fState & kKeyShiftMask);
         break;
      case kKey_Backspace:
         Backspace();
         break;
      case kKey_Home:
         Home(event->fState & kKeyShiftMask);
         break;
      case kKey_End:
         End(event->fState & kKeyShiftMask);
         break;
      case kKey_Delete:
         Del();
         break;
      case kKey_Insert:                     // switch on/off insert mode
         SetInsertMode(GetInsertMode() == kInsert ? kReplace : kInsert);
         break;
      default:
         unknown++;
      }
   }

   UpdateOffset();
   fClient->NeedRedraw(this);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleButton(Event_t *event)
{
   // Handle mouse button event in text entry widget.

   if (fTip) fTip->Hide();

   if (!IsEnabled()) return kTRUE;

   if (event->fType == kButtonPress) {
      SetFocus();
      if (fEchoMode == kNoEcho) return kTRUE;

      if (event->fCode == kButton1) {
         Int_t offset =  IsFrameDrawn() ? 4 : 0;
         Int_t x = fOffset + offset;
         Int_t position     = GetCharacterIndex(event->fX - x);
         fSelectionOn = kFALSE;
         SetCursorPosition(position);
         DoRedraw();
      } else if (event->fCode == kButton2) {
         if (gVirtualX->GetPrimarySelectionOwner() == kNone) {
            // No primary selection, so use the cut buffer
            PastePrimary(fClient->GetRoot()->GetId(), kCutBuffer, kFALSE);
         } else {
            gVirtualX->ConvertPrimarySelection(fId, fClipboard, event->fTime);
         }
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kEnterNotify)
         fTip->Reset();
      else
         fTip->Hide();
   }

   // only turn off/on cursor if widget does not have the focus
   if (gVirtualX->GetInputFocus() != fId) {
      if (event->fType == kEnterNotify) {
         fCursorOn = kTRUE;
         if (!fCurBlink) fCurBlink = new TBlinkTimer(this, 500);
         fCurBlink->Reset();
         gSystem->AddTimer(fCurBlink);
      } else {
         fCursorOn = kFALSE;
         // fSelectionOn = kFALSE;        // "netscape location behavior"
         if (fCurBlink) fCurBlink->Remove();
      }
      fClient->NeedRedraw(this);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in the text entry widget.

   if (!IsEnabled() || (GetEchoMode() == kNoEcho)) return kTRUE;

   Int_t offset =  IsFrameDrawn() ? 4 : 0;
   Int_t x = fOffset + offset;
   Int_t position = GetCharacterIndex(event->fX - x); // + 1;
   fSelectionOn = kTRUE;
   NewMark(position);
   UpdateOffset();
   DoRedraw();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleDoubleClick(Event_t *event)
{
   // Handle mouse double click event in the text entry widget.

   if (!IsEnabled()) return kTRUE;

   Int_t offset = IsFrameDrawn() ? 4 : 0;
   Int_t x = fOffset + offset ;

   DoubleClicked();
   SetFocus();
   if (fEchoMode == kNoEcho) return kTRUE;

   Int_t position = GetCharacterIndex(event->fX  - x);
   MarkWord(position);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleConfigureNotify(Event_t* event)
{
   // Handles resize events for this widget.

   TGFrame::HandleConfigureNotify(event);
   UpdateOffset();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleFocusChange(Event_t *event)
{
   // Handle focus change event in text entry widget.

   if (!IsEnabled()) return kTRUE;

   // check this when porting to Win32
   if ((event->fCode == kNotifyNormal) && (event->fState != kNotifyPointer)) {
      if (event->fType == kFocusIn) {
         fCursorOn = kTRUE;
         if (!fCurBlink) fCurBlink = new TBlinkTimer(this, 500);
         fCurBlink->Reset();
         gSystem->AddTimer(fCurBlink);
      } else {
          fCursorOn = kFALSE;
          // fSelectionOn = kFALSE;        // "netscape location behavior"
          if (fCurBlink) fCurBlink->Remove();
      }
      fClient->NeedRedraw(this);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleSelection(Event_t *event)
{
   // Handle text selection event.

   PastePrimary((Window_t)event->fUser[0], (Atom_t)event->fUser[3], kTRUE);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleSelectionRequest(Event_t *event)
{
   // Handle request to send current clipboard contents to requestor window.

   Event_t reply;
   char   *buffer;
   Long_t  len;

   reply.fType    = kSelectionNotify;
   reply.fTime    = event->fTime;
   reply.fUser[0] = event->fUser[0];     // requestor
   reply.fUser[1] = event->fUser[1];     // selection
   reply.fUser[2] = event->fUser[2];     // target
   reply.fUser[3] = event->fUser[3];     // property

   len = 0;
   if (fgClipboardText) len = fgClipboardText->Length();
   buffer = new char[len+1];
   if (fgClipboardText) strcpy (buffer, fgClipboardText->Data());
   buffer[len] = '\0';

   gVirtualX->ChangeProperty((Window_t) event->fUser[0], (Atom_t) event->fUser[3],
                             (Atom_t) event->fUser[2], (UChar_t*) buffer,
                             (Int_t) len);
   delete [] buffer;

   gVirtualX->SendEvent((Window_t)event->fUser[0], &reply);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextEntry::PastePrimary(Window_t wid, Atom_t property, Bool_t del)
{
   // Paste text from selection (either primary or cut buffer) into
   // text entry widget.

   TString data;
   Int_t   nchar;

   if (!IsEnabled()) return;

   gVirtualX->GetPasteBuffer(wid, property, data, nchar, del);

   if (nchar) Insert(data.Data());
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Bool_t TGTextEntry::HandleTimer(TTimer *)
{
   // Handle cursor blink timer.

   fCursorOn = !fCursorOn;
   DoRedraw();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEntry::IsCursorOutOfFrame()
{
   // Returns kTRUE if cursor is out of frame.

   //   fCursorX = fOffset + 4 + gVirtualX->TextWidth(fFontStruct,
   //                                  GetDisplayText(), fCursorIX);

   Int_t offset =  IsFrameDrawn() ? 4 : 0;
   Int_t w = GetWidth();
   return ((fCursorX < offset) || (fCursorX > w-offset));
}

//______________________________________________________________________________
void TGTextEntry::ScrollByChar()
{
   // Shift position of cursor by one character.

   if (GetEchoMode() == kNoEcho) return;

   TString dt = GetDisplayText();
   Int_t len = dt.Length();
   Int_t ind = fCursorIX < len-1 ? fCursorIX : len-1;
   Int_t charWidth = gVirtualX->TextWidth(fFontStruct, &dt[ind],1);
   Int_t w = GetWidth();
   Int_t d;
   Int_t offset =  IsFrameDrawn() ? 4 : 0;

   if (fCursorX < offset) {
      fOffset += charWidth;
      fCursorX += charWidth;
      d = fCursorX;

      if (d < offset){          // correction
         d -= offset;
         fOffset -= d;
         fCursorX -= d;
         charWidth += d;
      }
   } else if (fCursorX > w-offset) {
      fOffset -= charWidth;
      fCursorX -= charWidth;
      d = w - fCursorX;

      if (d < offset) {        // correction
         d -= offset;
         fOffset += d;
         fCursorX += d;
         charWidth += d;
      }
   }
}

//______________________________________________________________________________
void TGTextEntry::UpdateOffset()
{
   // Updates start text offset according GetAlignment() mode,
   // if cursor is out of frame => scroll the text.
   // See also SetAlignment() and ScrollByChar().

   TString dt = GetDisplayText();
   Int_t textWidth = gVirtualX->TextWidth(fFontStruct, dt.Data() , dt.Length());
   Int_t offset = IsFrameDrawn() ? 4 : 0;
   Int_t w = GetWidth() - 2 * offset;   // subtract border twice

   if (textWidth > w) {                          // may need to scroll.
      if (IsCursorOutOfFrame()) ScrollByChar();
   }
   else if (fAlignment == kTextRight)   fOffset = w - textWidth - 1;
   else if (fAlignment == kTextCenterX) fOffset = (w - textWidth)/2;
   else if (fAlignment == kTextLeft)    fOffset = 0;
}

//______________________________________________________________________________
void TGTextEntry::SetFocus()
{
   // Gives the keyboard input focus to this text entry widget.

   gVirtualX->SetInputFocus(fId);
}

//______________________________________________________________________________
void TGTextEntry::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tool tip text associated with this text entry. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // text = 0.

   if (fTip) {
      delete fTip;
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = new TGToolTip(fClient->GetRoot(), this, text, delayms);
}

//_____________________________________________________________________
void TGTextEntry::InsertText(const char *text, Int_t pos)
{
   // Inserts text at position pos, clears the selection and moves
   // the cursor to the end of the line.
   // If necessary the text is truncated to fit MaxLength().
   // See also GetText(), SetText(), AppendText(), RemoveText().

   Int_t position = TMath::Min((Int_t)fText->GetTextLength(), pos);
   TString newText(GetText());
   newText.Insert(position, text);
   SetText(newText.Data());
}

//_____________________________________________________________________
void TGTextEntry::AppendText(const char *text)
{
   // Appends text to the end of text entry, clears the selection
   // and moves the cursor to the end of the line.
   // If necessary the text is truncated to fit MaxLength().
   // See also GetText(), InsertText(), SetText(), RemoveText().

   InsertText(text, fText->GetTextLength());
}

//_____________________________________________________________________
void TGTextEntry::RemoveText(Int_t start, Int_t end)
{
   // Removes text at the range, clears the selection and moves
   // the cursor to the end of the line.
   // See also GetText(), InsertText(), SetText(), AppendText().

   Int_t pos = TMath::Min(start, end);
   Int_t len = TMath::Abs(end-start);
   TString newText(GetText());
   newText.Remove(pos, len);
   SetText(newText.Data());
}

//______________________________________________________________________________
FontStruct_t TGTextEntry::GetDefaultFontStruct()
{ return fgDefaultFontStruct; }

//______________________________________________________________________________
const TGGC &TGTextEntry::GetDefaultGC()
{ return fgDefaultGC; }
