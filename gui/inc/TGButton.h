// @(#)root/gui:$Name:  $:$Id: TGButton.h,v 1.5 2000/10/22 19:28:57 rdm Exp $
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


class TGButton : public TGFrame, public TGWidget {

friend class TGClient;
friend class TGButtonGroup;

protected:
   UInt_t         fTWidth;      // button width
   UInt_t         fTHeight;     // button height
   EButtonState   fState;       // button state
   Bool_t         fStayDown;    // true if button has to stay down
   GContext_t     fNormGC;      // graphics context used for drawing button
   void          *fUserData;    // pointer to user data structure
   TGToolTip     *fTip;         // tool tip associated with button
   TGButtonGroup *fGroup;       // button group this button belongs to

   virtual void   SetGroup(TGButtonGroup *group);
   virtual void   SetToggleButton(Bool_t) { }

   static TGGC fgHibckgndGC;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC fgDefaultGC;

public:
   TGButton(const TGWindow *p, Int_t id, GContext_t norm = fgDefaultGC(),
            UInt_t option = kRaisedFrame | kDoubleBorder);
   virtual ~TGButton();

   virtual Bool_t       HandleButton(Event_t *event);
   virtual Bool_t       HandleCrossing(Event_t *event);
   virtual void         SetUserData(void *userData) { fUserData = userData; }
   virtual void        *GetUserData() const { return fUserData; }
   virtual void         SetToolTipText(const char *text, Long_t delayms = 1000);
   virtual void         SetState(EButtonState state);
   virtual EButtonState GetState() const { return fState; }
   virtual void         AllowStayDown(Bool_t a) { fStayDown = a; }
   TGButtonGroup       *GetGroup() const { return fGroup; }

   virtual Bool_t       IsDown() const { return !(fOptions & kRaisedFrame); }
   virtual void         SetDown(Bool_t on = kTRUE)
                           { on ? SetState(kButtonDown) : SetState(kButtonUp); }
   virtual Bool_t       IsOn() const { return IsDown(); }
   virtual void         SetOn(Bool_t on = kTRUE) { SetDown(on); }
   virtual Bool_t       IsToggleButton() const { return kFALSE; }
   virtual Bool_t       IsExclusiveToggle() const { return kFALSE; }
   virtual void         Toggle() { SetDown(IsDown() ? kFALSE : kTRUE); }

   void Pressed()  { Emit("Pressed()"); }   //*SIGNAL*
   void Released() { Emit("Released()"); }  //*SIGNAL*
   void Clicked()  { Emit("Clicked()"); }   //*SIGNAL*
   void Toggled(Bool_t on) { Emit("Toggled(Bool_t)", on); }  //*SIGNAL*

   static const TGGC   &GetDefaultGC();
   static const TGGC   &GetHibckgndGC();

   ClassDef(TGButton,0)  // Button widget abstract base class
};


class TGTextButton : public TGButton {

friend class TGClient;

protected:
   TGHotString   *fLabel;         // button text
   Int_t          fTMode;         // text drawing mode (ETextJustification)
   Int_t          fHKeycode;      // hotkey
   FontStruct_t   fFontStruct;    // font to draw text

   static FontStruct_t  fgDefaultFontStruct;

   void Init();
   virtual void DoRedraw();

public:
   TGTextButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                GContext_t norm = fgDefaultGC(),
                FontStruct_t font = fgDefaultFontStruct,
                UInt_t option = kRaisedFrame | kDoubleBorder);
   TGTextButton(const TGWindow *p, const char *s, Int_t id = -1,
                GContext_t norm = fgDefaultGC(),
                FontStruct_t font = fgDefaultFontStruct,
                UInt_t option = kRaisedFrame | kDoubleBorder);
   TGTextButton(const TGWindow *p, const char *s, const char *cmd,
                Int_t id = -1, GContext_t norm = fgDefaultGC(),
                FontStruct_t font = fgDefaultFontStruct,
                UInt_t option = kRaisedFrame | kDoubleBorder);
   virtual ~TGTextButton();

   virtual TGDimension GetDefaultSize() const
                { return TGDimension(fTWidth+8, fTHeight+7); }

   virtual Bool_t     HandleKey(Event_t *event);
   const TGHotString *GetText() const { return fLabel; }
   TString            GetString() const { return TString(fLabel->GetString()); }
   void               SetTextJustify(Int_t tmode) { fTMode = tmode; }
   void               SetText(TGHotString *new_label);
   void               SetText(const TString &new_label);

   static FontStruct_t GetDefaultFontStruct();

   ClassDef(TGTextButton,0)  // A text button widget
};


class TGPictureButton : public TGButton {

protected:
   const TGPicture   *fPic;   // picture to be put in button

   virtual void DoRedraw();

public:
   TGPictureButton(const TGWindow *p, const TGPicture *pic, Int_t id = -1,
                   GContext_t norm = fgDefaultGC(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGWindow *p, const TGPicture *pic, const char *cmd,
                   Int_t id = -1, GContext_t norm = fgDefaultGC(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);

   void SetPicture(const TGPicture *new_pic);

   ClassDef(TGPictureButton,0)  // A picture button widget
};


class TGCheckButton : public TGButton {

friend class TGClient;

protected:
   TGHotString    *fLabel;         // check button label
   Int_t           fHKeycode;      // hotkey
   FontStruct_t    fFontStruct;    // font to draw label
   EButtonState    fPrevState;     // previous check button state

   void Init();
   void PSetState(EButtonState state);
   virtual void DoRedraw();

   static FontStruct_t  fgDefaultFontStruct;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC          fgDefaultGC;

public:
   TGCheckButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   TGCheckButton(const TGWindow *p, const char *s, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   TGCheckButton(const TGWindow *p, const char *s, const char *cmd, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   virtual ~TGCheckButton();

   virtual TGDimension GetDefaultSize() const
                  { return TGDimension(fTWidth+22, fTHeight+2); }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual void   SetState(EButtonState state) { PSetState(fPrevState = state); }

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGCheckButton,0)  // A check button widget
};


class TGRadioButton : public TGButton {

friend class TGClient;

protected:
   TGHotString       *fLabel;       // radio button label
   Int_t              fHKeycode;    // hotkey
   EButtonState       fPrevState;   // radio button state
   const TGPicture   *fOn;          // button ON picture
   const TGPicture   *fOff;         // button OFF picture
   FontStruct_t       fFontStruct;  // font to draw label

   void Init();
   void PSetState(EButtonState state);
   virtual void DoRedraw();

   static Pixmap_t      fgR1, fgR2, fgR3, fgR4, fgR5, fgR6;
   static FontStruct_t  fgDefaultFontStruct;
#ifdef R__SUNCCBUG
public:
#endif
   static TGGC          fgDefaultGC;

public:
   TGRadioButton(const TGWindow *p, TGHotString *s, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   TGRadioButton(const TGWindow *p, const char *s, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   TGRadioButton(const TGWindow *p, const char *s, const char *cmd, Int_t id = -1,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 UInt_t option = 0);
   virtual ~TGRadioButton();

   virtual TGDimension GetDefaultSize() const
                  { return TGDimension(fTWidth+22, fTHeight+2); }

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual void SetState(EButtonState state) { PSetState(fPrevState = state); }
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual Bool_t IsExclusiveToggle() const { return kTRUE; }

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   ClassDef(TGRadioButton,0)  // A radio button widget
};

#endif
