// @(#)root/gui:$Name:  $:$Id: TGButtonGroup.h,v 1.1 2000/10/17 12:30:14 rdm Exp $
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGButtonGroup, TGVButtonGroup and TGHButtonGroup                     //
//                                                                      //
// This header defines button group frames.                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TMap
#include "TMap.h"
#endif


class TGButton;



class TGButtonGroup : public TGGroupFrame, public TQObject {

friend class TGButton;

protected:
   Bool_t  fExclGroup;       // kTRUE if group is exclusive
   Bool_t  fRadioExcl;       // kTRUE if radio buttons are exclusive
   Bool_t  fDrawBorder;      // kTRUE if border and title are drawn
   TMap   *fMapOfButtons;    // map of button/id pairs in this group

   void Init();

public:
   TGButtonGroup(TGWindow *parent = 0,
                 const TString &title = "",
                 UInt_t options = kChildFrame | kVerticalFrame,
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 ULong_t back = fgDefaultFrameBackground);

   TGButtonGroup(TGWindow *parent,
                 UInt_t r, UInt_t c, Int_t s = 0, Int_t h = 0 ,
                 const TString &title = "",
                 GContext_t norm = fgDefaultGC(),
                 FontStruct_t font = fgDefaultFontStruct,
                 ULong_t back = fgDefaultFrameBackground);

   virtual ~TGButtonGroup();

   void Pressed(Int_t id)  { Emit("Pressed(Int_t)",id); }     //*SIGNAL*
   void Released(Int_t id) { Emit("Released(Int_t)",id);}     //*SIGNAL*
   void Clicked(Int_t id)  { Emit("Clicked(Int_t)",id); }     //*SIGNAL*

   void ButtonPressed();
   void ButtonReleased();
   void ButtonClicked();
   void ReleaseButtons();

   Bool_t IsExclusive() const { return fExclGroup; }
   Bool_t IsRadioButtonExclusive() const  { return fRadioExcl; }
   Bool_t IsBorderDrawn() const { return fDrawBorder; }

   void SetExclusive(Bool_t flag = kTRUE);
   void SetRadioButtonExclusive(Bool_t flag = kTRUE);
   void SetBorderDrawn(Bool_t enable = kTRUE);
   void SetButton(Int_t id, Bool_t down = kTRUE);
   void SetTitle(const char *title = "");

   Int_t     Insert(TGButton *button, int id = -1);
   void      Remove(TGButton *button);
   TGButton *Find(Int_t id) const;
   TGButton *GetButton(Int_t id) const { return Find(id); }
   Int_t     GetCount() const { return fMapOfButtons->GetSize(); }
   Int_t     GetId(TGButton *button) const;
   void      Show();
   void      Hide();
   void      DrawBorder() { if (fDrawBorder) TGGroupFrame::DrawBorder(); }
   void      SetLayoutHints(TGLayoutHints *l, TGButton *button = 0);

   ClassDef(TGButtonGroup,0)  // Organizes TGButtons in a group
};


class TGVButtonGroup : public TGButtonGroup {

public:
   TGVButtonGroup(TGWindow *parent = 0,
                  const TString &title = "",
                  GContext_t norm = fgDefaultGC(),
                  FontStruct_t font = fgDefaultFontStruct,
                  ULong_t back = fgDefaultFrameBackground) :
      TGButtonGroup(parent, title, kChildFrame | kVerticalFrame,
                    norm, font, back) { }

   virtual ~TGVButtonGroup() { }

   ClassDef(TGVButtonGroup,0)  // A button group with one vertical column
};


class TGHButtonGroup : public TGButtonGroup {

public:
   TGHButtonGroup(TGWindow *parent = 0,
                  const TString &title = "",
                  GContext_t norm = fgDefaultGC(),
                  FontStruct_t font = fgDefaultFontStruct,
                  ULong_t back = fgDefaultFrameBackground) :
      TGButtonGroup(parent, title, kChildFrame | kHorizontalFrame,
                    norm, font, back) { }

   virtual ~TGHButtonGroup() { }

   ClassDef(TGHButtonGroup,0)  // A button group with one horizontal row
};

#endif
