// @(#)root/gui:$Id$
// Author: Fons Rademakers   25/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGToolBar
#define ROOT_TGToolBar


#include "TGFrame.h"

class TGButton;
class TGPictureButton;
class TList;
class TMap;

struct ToolBarData_t {
   const char *fPixmap;
   const char *fTipText;
   Bool_t      fStayDown;
   Int_t       fId;
   TGButton   *fButton;
};



class TGToolBar : public TGCompositeFrame {

protected:
   TList   *fPictures;      ///< list of pictures that should be freed
   TList   *fTrash;         ///< list of buttons and layout hints to be deleted
   TMap    *fMapOfButtons;  ///< map of button/id pairs in this group

private:
   TGToolBar(const TGToolBar&) = delete;
   TGToolBar& operator=(const TGToolBar&) = delete;

public:
   TGToolBar(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
             UInt_t options = kHorizontalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGToolBar();

   virtual TGButton *AddButton(const TGWindow *w, ToolBarData_t *button, Int_t spacing = 0);
   virtual TGButton *AddButton(const TGWindow *w, TGPictureButton *button, Int_t spacing = 0);

   virtual void ChangeIcon(ToolBarData_t *button, const char *new_icon);
   virtual void Cleanup();
   virtual TGButton *GetButton(Int_t id) const;
   virtual Longptr_t GetId(TGButton *button) const;
   virtual void      SetId(TGButton *button, Longptr_t id);

   virtual void ButtonPressed();
   virtual void ButtonReleased();
   virtual void ButtonClicked();

   virtual void Pressed(Int_t id)  { Emit("Pressed(Int_t)",id); }   //*SIGNAL*
   virtual void Released(Int_t id) { Emit("Released(Int_t)",id);}   //*SIGNAL*
   virtual void Clicked(Int_t id)  { Emit("Clicked(Int_t)",id); }   //*SIGNAL*

   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGToolBar,0)  //A bar containing picture buttons
};

#endif
