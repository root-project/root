// @(#)root/gui:$Name:  $:$Id: TGToolBar.h,v 1.10 2004/09/08 08:13:11 brun Exp $
// Author: Fons Rademakers   25/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGToolBar
#define ROOT_TGToolBar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGToolBar                                                            //
//                                                                      //
// A toolbar is a composite frame that contains TGPictureButtons.       //
// Often used in combination with a TGHorizontal3DLine.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGButton;
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

private:
   TList   *fPictures;    // list of pictures that should be freed
   TList   *fTrash;       // list of buttons and layout hints to be deleted
   TMap   *fMapOfButtons;    // map of button/id pairs in this group

public:
   TGToolBar(const TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
             UInt_t options = kHorizontalFrame,
             Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGToolBar();

   virtual TGButton *AddButton(const TGWindow *w, ToolBarData_t *button, Int_t spacing = 0);

   virtual void ChangeIcon(ToolBarData_t *button, const char *new_icon);
   virtual void Cleanup();

   virtual void ButtonPressed();
   virtual void ButtonReleased();
   virtual void ButtonClicked();

   virtual void Pressed(Int_t id)  { Emit("Pressed(Int_t)",id); }   //*SIGNAL*
   virtual void Released(Int_t id) { Emit("Released(Int_t)",id);}   //*SIGNAL*
   virtual void Clicked(Int_t id)  { Emit("Clicked(Int_t)",id); }   //*SIGNAL*

   virtual void   SavePrimitive(ofstream &out, Option_t *option);

   ClassDef(TGToolBar,0)  //A bar containing picture buttons
};

#endif
