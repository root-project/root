// @(#)root/gui:$Name:  $:$Id: TGButton.h,v 1.39 2006/05/14 10:23:26 brun Exp $
// Author: Fons Rademakers   06/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGButton
#define ROOT_TGButton


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGButton, TGTextButton, TGPictureButton, TGCheckButton and           //
// TGRadioButton                                                        //
//                                                                      //
// This header defines all GUI button widgets.                          //
//                                                                      //
// TGButton is a button abstract base class. It defines general button  //
// behaviour.                                                           //
//                                                                      //
// Selecting a text or picture button will generate the event:          //
// kC_COMMAND, kCM_BUTTON, button id, user data.                        //
//                                                                      //
// Selecting a check button will generate the event:                    //
// kC_COMMAND, kCM_CHECKBUTTON, button id, user data.                   //
//                                                                      //
// Selecting a radio button will generate the event:                    //
// kC_COMMAND, kCM_RADIOBUTTON, button id, user data.                   //
//                                                                      //
// If a command string has been specified (via SetCommand()) then this  //
// command string will be executed via the interpreter whenever a       //
// button is selected. A command string can contain the macros:         //
// $MSG   -- kC_COMMAND, kCM[CHECK|RADIO]BUTTON packed message          //
//           (use GET_MSG() and GET_SUBMSG() to unpack)                 //
// $PARM1 -- button id                                                  //
// $PARM2 -- user data pointer                                          //
// Before executing these macros are expanded into the respective       //
// Long_t's                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif

//--- Button states

enum EButtonState {
   kButtonUp,
   kButtonDown,
   kButtonEngaged,
   kButtonDisabled
};


class TGWidget;
class TGHotString;
class TGPicture;
class TGToolTip;
class TGButtonGroup;


class TGButton : public TGFrame, public TGWidget {

friend class TGButtonGroup;

private:
   TGButton& operator=(const TGButton&);

protected:
   UInt_t         fTWidth;      // button width
   UInt_t         fTHeight;     // button height
   EButtonState   fState;       // button state
   Bool_t         fStayDown;    // true if button has to stay down
   GContext_t     fNormGC;      // graphics context used for drawing button
   void          *fUserData;    // pointer to user data structure
   TGToolTip     *fTip;         // tool tip associated with button
   TGButtonGroup *fGroup;       // button group this button belongs to

   virtual void   SetToggleButton(Bool_t) { }
   virtual void   EmitSignals(Bool_t wasUp);

   static const TGGC *fgDefaultGC;
   static const TGGC *fgHibckgndGC;

   static Window_t fgReleaseBtn; // the last released button

public:
   static const TGGC   &GetDefaultGC();
   static const TGGC   &GetHibckgndGC();

   TGButton(const TGWindow *p = 0, Int_t id = -1, GContext_t norm = GetDefaultGC()(),
            UInt_t option = kRaisedFrame | kDoubleBorder);
   TGButton(const TGButton& tgb): TGFrame(tgb), TGWidget(tgb), fTWidth(tgb.fTWidth),
      fTHeight(tgb.fTHeight), fState(tgb.fState), fStayDown(tgb.fStayDown),
      fNormGC(tgb.fNormGC), fUserData(tgb.fUserData), fTip(tgb.fTip), fGroup(tgb.fGroup) { }
   virtual ~TGButton();

   virtual Bool_t       HandleButton(Event_t *event);
   virtual Bool_t       HandleCrossing(Event_t *event);
   virtual void         SetUserData(void *userData) { fUserData = userData; }
   virtual void        *GetUserData() const { return fUserData; }
   virtual void         SetToolTipText(const char *text, Long_t delayms = 400);  //*MENU*
   virtual TGToolTip   *GetToolTip() const { return fTip; }
   virtual void         SetState(EButtonState state, Bool_t emit = kFALSE);
   virtual EButtonState GetState() const { return fState; }
   virtual void         AllowStayDown(Bool_t a) { fStayDown = a; }
   virtual void         SetGroup(TGButtonGroup *gr);
   TGButtonGroup       *GetGroup() const { return fGroup; }

   virtual Bool_t       IsDown() const { return !(fOptions & kRaisedFrame); }
   virtual void         SetDown(Bool_t on = kTRUE, Bool_t emit = kFALSE);  //*MENU*
   virtual Bool_t       IsOn() const { return IsDown(); }
   virtual void         SetOn(Bool_t on = kTRUE,  Bool_t emit = kFALSE) { SetDown(on, emit); }
   virtual Bool_t       IsToggleButton() const { return kFALSE; }
   virtual Bool_t       IsExclusiveToggle() const { return kFALSE; }
   virtual void         Toggle(Bool_t emit = kFALSE) { SetDown(IsDown() ? kFALSE : kTRUE, emit); }
   virtual void         SetEnabled(Bool_t e = kTRUE) { SetState(e ? kButtonUp : kButtonDisabled); } //*MENU*
   virtual void         SavePrimitive(ofstream &out, Option_t *option);

   virtual void Pressed()  { Emit("Pressed()"); }   // *SIGNAL*
   virtual void Released() { Emit("Released()"); }  // *SIGNAL*
   virtual void Clicked()  { Emit("Clicked()"); }   // *SIGNAL*
   virtual void Toggled(Bool_t on) { Emit("Toggled(Bool_t)", on); }  // *SIGNAL*

   ClassDef(TGButton,0)  // Button widget abstract base class
};


class TGTextButton : public TGButton {

private:
   TGTextButton& operator=(const TGTextButton&);

protected:
   TGHotString   *fLabel;         // button text
   Int_t          fTMode;         // text drawing mode (ETextJustification)
   Int_t          fHKeycode;      // hotkey
   FontStruct_t   fFontStruct;    // font to draw text
   Bool_t         fHasOwnFont;    // kTRUE - font defined locally,  kFALSE - globally

   static const TGFont *fgDefaultFont;

   void Init();
   virtual void DoRedraw();

public:
   static FontStruct_t GetDefaultFontStruct();

   TGTextButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                GContext_t norm = GetDefaultGC()(),
                FontStruct_t font = GetDefaultFontStruct(),
                UInt_t option = kRaisedFrame | kDoubleBorder);
   TGTextButton(const TGWindow *p = 0, const char *s = 0, Int_t id = -1,
                GContext_t norm = GetDefaultGC()(),
                FontStruct_t font = GetDefaultFontStruct(),
                UInt_t option = kRaisedFrame | kDoubleBorder);
   TGTextButton(const TGWindow *p, const char *s, const char *cmd,
                Int_t id = -1, GContext_t norm = GetDefaultGC()(),
                FontStruct_t font = GetDefaultFontStruct(),
                UInt_t option = kRaisedFrame | kDoubleBorder);
   TGTextButton(const TGTextButton& tgt): TGButton(tgt), fLabel(tgt.fLabel),
      fTMode(tgt.fTMode), fHKeycode(tgt.fHKeycode), fFontStruct(tgt.fFontStruct),
      fHasOwnFont(tgt.fHasOwnFont) { }
   virtual ~TGTextButton();

   virtual TGDimension GetDefaultSize() const
                        { return TGDimension(fTWidth+8, fTHeight+7); }

   virtual Bool_t     HandleKey(Event_t *event);
   const TGHotString *GetText() const { return fLabel; }
   virtual const char *GetTitle() const { return fLabel->Data(); }
   TString            GetString() const { return TString(fLabel->GetString()); }
   void               SetTextJustify(Int_t tmode) { fTMode = tmode; }
   virtual void       SetText(TGHotString *new_label);
   virtual void       SetText(const TString &new_label);
   virtual void       SetTitle(const char *label) { SetText(label); }
           void       Rename(const char *title)  { SetTitle(title); } //*MENU*
   virtual void       SetFont(FontStruct_t font, Bool_t global = kFALSE);
   virtual void       SetFont(const char *fontName, Bool_t global = kFALSE);  //*MENU*
   virtual void       SetTextColor(Pixel_t color, Bool_t global = kFALSE);
   virtual void       SetForegroundColor(Pixel_t fore) { SetTextColor(fore, kFALSE); }

   Bool_t             HasOwnFont() const;

   virtual void       SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGTextButton,0)  // A text button widget
};


class TGPictureButton : public TGButton {

private:
   TGPictureButton& operator=(const TGPictureButton&);

protected:
   const TGPicture   *fPic;            // picture to be put in button
   const TGPicture   *fPicD;           // picture shown when button disabled
   Bool_t             fOwnDisabledPic; // kTRUE if disabled picture was autogenerated

   virtual void DoRedraw();
   virtual void CreateDisabledPicture();

public:
   TGPictureButton(const TGWindow *p, const TGPicture *pic, Int_t id = -1,
                   GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGWindow *p, const TGPicture *pic, const char *cmd,
                   Int_t id = -1, GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGWindow *p = 0, const char* pic = 0, Int_t id = -1,
                   GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGPictureButton& tgp): TGButton(tgp), fPic(tgp.fPic),
      fPicD(tgp.fPicD), fOwnDisabledPic(tgp.fOwnDisabledPic) { }
   virtual ~TGPictureButton();

   virtual void     SetPicture(const TGPicture *new_pic);
   virtual void     SetDisabledPicture(const TGPicture *pic);
   const TGPicture *GetPicture() const { return fPic; };
   virtual void     SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGPictureButton,0)  // A picture button widget
};


class TGCheckButton : public TGTextButton {

protected:
   EButtonState    fPrevState;     // previous check button state

   void Init();
   void PSetState(EButtonState state, Bool_t emit);
   virtual void DoRedraw();
   virtual void EmitSignals(Bool_t wasUp = kTRUE);

   static const TGFont *fgDefaultFont;
   static const TGGC   *fgDefaultGC;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGCheckButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   TGCheckButton(const TGWindow *p = 0, const char *s = 0, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   TGCheckButton(const TGWindow *p, const char *s, const char *cmd, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   virtual ~TGCheckButton();

   virtual TGDimension GetDefaultSize() const
                  { return TGDimension(fTWidth+22, fTHeight+2); }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual Bool_t IsOn() const { return fState == kButtonDown; }
   virtual Bool_t IsDown() const { return fState == kButtonDown; }
   virtual void   SetState(EButtonState state, Bool_t emit = kFALSE);
   virtual void   SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGCheckButton,0)  // A check button widget
};


class TGRadioButton : public TGTextButton {

private:
   TGRadioButton(const TGRadioButton&);
   TGRadioButton& operator=(const TGRadioButton&);

protected:
   EButtonState       fPrevState;   // radio button state
   const TGPicture   *fOn;          // button ON picture
   const TGPicture   *fOff;         // button OFF picture

   void Init();
   void PSetState(EButtonState state, Bool_t emit);
   virtual void DoRedraw();
   virtual void EmitSignals(Bool_t wasUp = kTRUE);

   static const TGFont *fgDefaultFont;
   static const TGGC   *fgDefaultGC;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGRadioButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   TGRadioButton(const TGWindow *p = 0, const char *s = 0, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   TGRadioButton(const TGWindow *p, const char *s, const char *cmd, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t option = 0);
   virtual ~TGRadioButton();

   virtual TGDimension GetDefaultSize() const
                  { return TGDimension(fTWidth+22, fTHeight+2); }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual void   SetState(EButtonState state, Bool_t emit = kFALSE);
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual Bool_t IsExclusiveToggle() const { return kTRUE; }
   virtual Bool_t IsOn() const { return fState == kButtonDown; }
   virtual Bool_t IsDown() const { return fState == kButtonDown; }
   virtual void   SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGRadioButton,0)  // A radio button widget
};

#endif
