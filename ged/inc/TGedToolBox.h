// @(#)root/gui:$Name:  $:$Id: TGedToolBox.h,v 1.2 2004/02/22 11:50:29 brun Exp $
// Author: Marek Biskup, Ilka Antcheva   21/07/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGedToolBox
#define ROOT_TGedToolBox

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedToolBox                                                          //
//                                                                      //
// A toolbox is a composite frame that contains TGPictureButtons.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGToolBar
#include "TGToolBar.h"
#endif

struct ToolBarData_t;

class TGedToolBox : public TGToolBar {

private:
   void CreateButtons(ToolBarData_t* buttons);  //adds buttons by *buttons

public:
   TGedToolBox(const TGWindow *p, UInt_t w, UInt_t h,
               UInt_t options = kHorizontalFrame,
               Pixel_t back = GetDefaultFrameBackground());
   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);

   virtual void   SavePrimitive(ostream &out, Option_t *option = "");
        
   ClassDef(TGedToolBox,0)  //a bar with picture buttons
};

#endif
