// @(#)root/gui:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   16/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGButtonGroup
#define ROOT_TGButtonGroup


#include "TGFrame.h"

class TGButton;
class TMap;

class TGButtonGroup : public TGGroupFrame {

friend class TGButton;

private:
   TGButtonGroup(const TGButtonGroup&) = delete;
   TGButtonGroup& operator=(const TGButtonGroup&) = delete;

protected:
   Bool_t  fState;           ///< kTRUE if group is enabled
   Bool_t  fExclGroup;       ///< kTRUE if group is exclusive
   Bool_t  fRadioExcl;       ///< kTRUE if radio buttons are exclusive
   Bool_t  fDrawBorder;      ///< kTRUE if border and title are drawn
   TMap   *fMapOfButtons;    ///< map of button/id pairs in this group

   void Init();
   void DoRedraw() override;

public:
   TGButtonGroup(const TGWindow *parent = nullptr,
                 const TString &title = "",
                 UInt_t options = kChildFrame | kVerticalFrame,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 Pixel_t back = GetDefaultFrameBackground());

   TGButtonGroup(const TGWindow *parent,
                 UInt_t r, UInt_t c, Int_t s = 0, Int_t h = 0 ,
                 const TString &title = "",
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 Pixel_t back = GetDefaultFrameBackground());

   virtual ~TGButtonGroup();

   virtual void Pressed(Int_t id)  { Emit("Pressed(Int_t)",id); }   //*SIGNAL*
   virtual void Released(Int_t id) { Emit("Released(Int_t)",id);}   //*SIGNAL*
   virtual void Clicked(Int_t id)  { Emit("Clicked(Int_t)",id); }   //*SIGNAL*

   virtual void ButtonPressed();
   virtual void ButtonReleased();
   virtual void ButtonClicked();
   virtual void ReleaseButtons();

   Bool_t IsEnabled() const { return fState; }
   Bool_t IsExclusive() const { return fExclGroup; }
   Bool_t IsRadioButtonExclusive() const  { return fRadioExcl; }
   Bool_t IsBorderDrawn() const { return fDrawBorder; }
   Int_t  GetCount() const;
   Int_t  GetId(TGButton *button) const;

   virtual void SetExclusive(Bool_t flag = kTRUE);
   virtual void SetRadioButtonExclusive(Bool_t flag = kTRUE);
   virtual void SetState(Bool_t state = kTRUE);
   virtual void SetBorderDrawn(Bool_t enable = kTRUE);
   virtual void SetButton(Int_t id, Bool_t down = kTRUE);
   void         SetTitle(TGString *title) override;
   void         SetTitle(const char *title) override;

   virtual Int_t     Insert(TGButton *button, int id = -1);
   virtual void      Remove(TGButton *button);
   virtual TGButton *Find(Int_t id) const;
   virtual TGButton *GetButton(Int_t id) const { return Find(id); }
   virtual void      Show();
   virtual void      Hide();
   void              DrawBorder() override;
   virtual void      SetLayoutHints(TGLayoutHints *l, TGButton *button = nullptr);
   void              SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGButtonGroup,0)  // Organizes TGButtons in a group
};


class TGVButtonGroup : public TGButtonGroup {

public:
   TGVButtonGroup(const TGWindow *parent,
                  const TString &title = "",
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  Pixel_t back = GetDefaultFrameBackground()) :
      TGButtonGroup(parent, title, kChildFrame | kVerticalFrame,
                    norm, font, back) {}

   virtual ~TGVButtonGroup() {}
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGVButtonGroup,0)  // A button group with one vertical column
};


class TGHButtonGroup : public TGButtonGroup {

public:
   TGHButtonGroup(const TGWindow *parent,
                  const TString &title = "",
                  GContext_t norm = GetDefaultGC()(),
                  FontStruct_t font = GetDefaultFontStruct(),
                  Pixel_t back = GetDefaultFrameBackground()) :
      TGButtonGroup(parent, title, kChildFrame | kHorizontalFrame,
                    norm, font, back) { }

   virtual ~TGHButtonGroup() {}
   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGHButtonGroup,0)  // A button group with one horizontal row
};

#endif
