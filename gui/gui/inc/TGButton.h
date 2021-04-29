// @(#)root/gui:$Id$
// Author: Fons Rademakers   06/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGButton
#define ROOT_TGButton


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGButton, TGTextButton, TGPictureButton, TGCheckButton TGRadioButton //
// and TGSplitButton                                                    //
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

#include "TGFrame.h"
#include "TGWidget.h"

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
class TGPopupMenu;
class TGTextLayout;

class TGButton : public TGFrame, public TGWidget {

friend class TGButtonGroup;

protected:
   UInt_t         fTWidth;      ///< button width
   UInt_t         fTHeight;     ///< button height
   EButtonState   fState;       ///< button state
   Bool_t         fStayDown;    ///< true if button has to stay down
   GContext_t     fNormGC;      ///< graphics context used for drawing button
   void          *fUserData;    ///< pointer to user data structure
   TGToolTip     *fTip;         ///< tool tip associated with button
   TGButtonGroup *fGroup;       ///< button group this button belongs to
   Pixel_t        fBgndColor;   ///< actual background color
   Pixel_t        fHighColor;   ///< highlight color
   UInt_t         fStyle;       ///< button style (modern or classic)

   virtual void   SetToggleButton(Bool_t) { }
   virtual void   EmitSignals(Bool_t wasUp);

   static const TGGC *fgDefaultGC;
   static const TGGC *fgHibckgndGC;

   static Window_t fgReleaseBtn; // the last released button

private:
   TGButton(const TGButton&) = delete;
   TGButton& operator=(const TGButton&) = delete;

public:
   static const TGGC   &GetDefaultGC();
   static const TGGC   &GetHibckgndGC();

   TGButton(const TGWindow *p = nullptr, Int_t id = -1, GContext_t norm = GetDefaultGC()(),
            UInt_t option = kRaisedFrame | kDoubleBorder);
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

   virtual Bool_t       IsDown() const;// { return !(fOptions & kRaisedFrame); }
   virtual void         SetDown(Bool_t on = kTRUE, Bool_t emit = kFALSE);
   virtual Bool_t       IsOn() const { return IsDown(); }
   virtual void         SetOn(Bool_t on = kTRUE,  Bool_t emit = kFALSE) { SetDown(on, emit); }
   virtual Bool_t       IsToggleButton() const { return kFALSE; }
   virtual Bool_t       IsExclusiveToggle() const { return kFALSE; }
   virtual void         Toggle(Bool_t emit = kFALSE) { SetDown(IsDown() ? kFALSE : kTRUE, emit); }
   virtual void         SetEnabled(Bool_t e = kTRUE); //*TOGGLE* *GETTER=IsEnabled
   virtual UInt_t       GetStyle() const { return fStyle; }
   virtual void         SetStyle(UInt_t newstyle);
   virtual void         SetStyle(const char *style);

   virtual void         SavePrimitive(std::ostream &out, Option_t *option = "");

   GContext_t GetNormGC() const { return fNormGC; }

   virtual void Pressed()  { Emit("Pressed()"); }   // *SIGNAL*
   virtual void Released() { Emit("Released()"); }  // *SIGNAL*
   virtual void Clicked()  { Emit("Clicked()"); }   // *SIGNAL*
   virtual void Toggled(Bool_t on) { Emit("Toggled(Bool_t)", on); }  // *SIGNAL*

   ClassDef(TGButton,0)  // Button widget abstract base class
};


class TGTextButton : public TGButton {

protected:
   TGHotString   *fLabel;         // button text
   Int_t          fMLeft;         // margin left
   Int_t          fMRight;        // margin right
   Int_t          fMTop;          // margin top
   Int_t          fMBottom;       // margin bottom
   Int_t          fTMode;         // text justify mode
   Int_t          fWrapLength;    // wrap length
   Int_t          fHKeycode;      // hotkey
   TGTextLayout  *fTLayout;       // text layout
   FontStruct_t   fFontStruct;    // font to draw text
   Bool_t         fHasOwnFont;    // kTRUE - font defined locally,  kFALSE - globally
   Bool_t         fStateOn;       // bit to save the state across disable/enable
   Bool_t         fPrevStateOn;   // bit to save previos state On/Off

   static const TGFont *fgDefaultFont;

   void Init();
   virtual void DoRedraw();

private:
   TGTextButton(const TGTextButton&) = delete;
   TGTextButton& operator=(const TGTextButton&) = delete;

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

   virtual ~TGTextButton();

   virtual TGDimension GetDefaultSize() const;

   virtual Bool_t     HandleKey(Event_t *event);
   const TGHotString *GetText() const { return fLabel; }
   virtual const char *GetTitle() const { return fLabel->Data(); }
   TString            GetString() const { return TString(fLabel->GetString()); }
   virtual void       SetTextJustify(Int_t tmode);
   Int_t GetTextJustify() const { return fTMode; }
   virtual void       SetText(TGHotString *new_label);
   virtual void       SetText(const TString &new_label);
   virtual void       SetTitle(const char *label) { SetText(label); }
   virtual void       SetFont(FontStruct_t font, Bool_t global = kFALSE);
   virtual void       SetFont(const char *fontName, Bool_t global = kFALSE);
   virtual void       SetTextColor(Pixel_t color, Bool_t global = kFALSE);
   virtual void       SetForegroundColor(Pixel_t fore) { SetTextColor(fore); }
   Bool_t             HasOwnFont() const;
   void               SetWrapLength(Int_t wl) { fWrapLength = wl; Layout(); }
   Int_t              GetWrapLength() const { return fWrapLength; }
   void               SetMargins(Int_t left=0, Int_t right=0, Int_t top=0, Int_t bottom=0)
                        { fMLeft = left; fMRight = right; fMTop = top; fMBottom = bottom; }

   virtual void       SetLeftMargin(Int_t val)   { fMLeft = val; }
   virtual void       SetRightMargin(Int_t val)  { fMRight = val; }
   virtual void       SetTopMargin(Int_t val)    { fMTop = val; }
   virtual void       SetBottomMargin(Int_t val) { fMBottom = val; }

   Int_t              GetLeftMargin() const { return fMLeft; }
   Int_t              GetRightMargin() const { return fMRight; }
   Int_t              GetTopMargin() const { return fMTop; }
   Int_t              GetBottomMargin() const { return fMBottom; }

   void               ChangeText(const char *title)  { SetTitle(title); } //*MENU*icon=bld_rename.png*

   FontStruct_t GetFontStruct() const { return fFontStruct; }

   virtual void       Layout();
   virtual void       SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGTextButton,0)  // A text button widget
};


class TGPictureButton : public TGButton {

protected:
   const TGPicture   *fPic;            ///< picture to be put in button
   const TGPicture   *fPicD;           ///< picture shown when button disabled
   Bool_t             fOwnDisabledPic; ///< kTRUE if disabled picture was autogenerated

   virtual void DoRedraw();
   virtual void CreateDisabledPicture();

private:
   TGPictureButton(const TGPictureButton&) = delete;
   TGPictureButton& operator=(const TGPictureButton&) = delete;

public:
   TGPictureButton(const TGWindow *p, const TGPicture *pic, Int_t id = -1,
                   GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGWindow *p, const TGPicture *pic, const char *cmd,
                   Int_t id = -1, GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   TGPictureButton(const TGWindow *p = nullptr, const char* pic = 0, Int_t id = -1,
                   GContext_t norm = GetDefaultGC()(),
                   UInt_t option = kRaisedFrame | kDoubleBorder);
   virtual ~TGPictureButton();

   virtual void     SetPicture(const TGPicture *new_pic);
   virtual void     SetDisabledPicture(const TGPicture *pic);
   const TGPicture *GetPicture() const { return fPic; };
   const TGPicture *GetDisabledPicture() const { return fPicD; };
   virtual void     SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGPictureButton,0)  // A picture button widget
};


class TGCheckButton : public TGTextButton {

private:

   TGCheckButton(const TGCheckButton&) = delete;
   TGCheckButton& operator=(const TGCheckButton&) = delete;

protected:
   EButtonState       fPrevState;     ///< previous check button state
   const TGPicture   *fOn;            ///< button ON picture
   const TGPicture   *fOff;           ///< button OFF picture
   const TGPicture   *fDisOn;         ///< button disabled and was ON picture
   const TGPicture   *fDisOff;        ///< button disabled and was OFF picture


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

   virtual TGDimension GetDefaultSize() const;

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual Bool_t IsOn() const { return fState == kButtonDown; }
   virtual Bool_t IsDown() const { return fState == kButtonDown; }
   virtual Bool_t IsDisabledAndSelected() const { return ((fState == kButtonDisabled) && fStateOn); }
   virtual void   SetDisabledAndSelected(Bool_t);
   virtual void   SetState(EButtonState state, Bool_t emit = kFALSE);
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGCheckButton,0)  // A check button widget
};


class TGRadioButton : public TGTextButton {

private:
   TGRadioButton(const TGRadioButton&) = delete;
   TGRadioButton& operator=(const TGRadioButton&) = delete;

protected:
   EButtonState       fPrevState;   ///< previous radio button state
   const TGPicture   *fOn;          ///< button ON picture
   const TGPicture   *fOff;         ///< button OFF picture
   const TGPicture   *fDisOn;       ///< button disabled and was ON picture
   const TGPicture   *fDisOff;      ///< button disabled and was OFF picture

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

   virtual TGDimension GetDefaultSize() const;

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual void   SetState(EButtonState state, Bool_t emit = kFALSE);
   virtual void   SetDisabledAndSelected(Bool_t);
   virtual Bool_t IsToggleButton() const { return kTRUE; }
   virtual Bool_t IsExclusiveToggle() const { return kTRUE; }
   virtual Bool_t IsOn() const { return fStateOn; }
   virtual Bool_t IsDown() const { return fStateOn; }
   virtual Bool_t IsDisabledAndSelected() const { return ((fState == kButtonDisabled) && fStateOn); }
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGRadioButton,0)  // A radio button widget
};


class TGSplitButton : public TGTextButton {

friend class TGPopupMenu;

private:
   TGSplitButton(const TGSplitButton&) = delete;
   TGSplitButton& operator=(const TGSplitButton&) = delete;

   void CalcSize();
   void DrawTriangle(const GContext_t gc, Int_t x, Int_t y);
   Bool_t HandleSButton(Event_t *event);
   Bool_t HandleSCrossing(Event_t *event);
   Bool_t HandleSKey(Event_t *event);
   void SetMenuState(Bool_t state) ;

protected:
///@{
/// @name Data members for menu functionality
   Bool_t       fSplit;         ///< kTRUE if menu is in split mode
   EButtonState fMBState;       ///< state for menu button in split mode
   UInt_t       fTBWidth;       ///< width of text button
   UInt_t       fMBWidth;       ///< width of menu button
   TGPopupMenu *fPopMenu;       ///< pointer to popup menu
   Int_t        fEntryId;       ///< Id of the currently active menu entry
   TGHotString *fMenuLabel;     ///< Label of the menu;
   Cursor_t     fDefaultCursor; ///< Default Cursor
   Bool_t       fKeyNavigate;   ///< kTRUE if key navigation is being used
   TGString     fWidestLabel;   ///< longest label that can be on the button
   TGString     fHeighestLabel; ///< highest label that can be on the button
///@]

   virtual void DoRedraw();
   void Init();
   void BindKeys(Bool_t on = kTRUE);
   void BindMenuKeys(Bool_t on = kTRUE);

public:
   TGSplitButton(const TGWindow *p, TGHotString *menulabel,
                TGPopupMenu *popmenu, Bool_t split = kTRUE,
                Int_t id = -1, GContext_t norm = GetDefaultGC()(),
                FontStruct_t fontstruct = GetDefaultFontStruct(),
                UInt_t option = kRaisedFrame | kDoubleBorder);

   virtual ~TGSplitButton();

   virtual TGDimension GetDefaultSize() const ;

   virtual void   SetText(TGHotString *new_label);
   virtual void   SetText(const TString &new_label);
   virtual void   SetFont(FontStruct_t font, Bool_t global = kFALSE);
   virtual void   SetFont(const char *fontName, Bool_t global = kFALSE);
   virtual void   SetMBState(EButtonState state);
   virtual void   SetSplit(Bool_t split);
   Bool_t         IsSplit() { return fSplit; }
   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual void   Layout();

   virtual void MBPressed()  { Emit("MBPressed()"); }   // *SIGNAL*
   virtual void MBReleased() { Emit("MBReleased()"); }  // *SIGNAL*
   virtual void MBClicked()  { Emit("MBClicked()"); }   // *SIGNAL*
   virtual void ItemClicked(Int_t id) { Emit("ItemClicked(Int_t)", id); } // *SIGNAL*

   // Slots
   void HandleMenu(Int_t id) ;

   ClassDef(TGSplitButton, 0) //a split button widget
};

#endif
