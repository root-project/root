// @(#)root/gpad:$Name$:$Id$
// Author: Nenad Buncic   20/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TControlBar
#define ROOT_TControlBar


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TControlBar                                                         //
//                                                                     //
//   ControlBar is fully user configurable tool which provides fast    //
// access to frequently used operations. User can choose between       //
// buttons and drawnbuttons (let's say icons) and assign to them his   //
// own actions (let's say ROOT or C++ commands).                       //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TControlBarButton
#include "TControlBarButton.h"
#endif

#ifndef ROOT_TControlBarImp
#include "TControlBarImp.h"
#endif

class TList;

class TControlBar : public TControlBarButton {

friend class  TControlBarImp;

private:
   virtual void   Create();
   void           Initialize(Int_t x, Int_t y);

protected:

   TControlBarImp *fControlBarImp;  //system specific implementation
   Int_t           fOrientation;    //orientation
   TList          *fButtons;        //list of buttons
   Int_t           fNoroc;          //number of rows or columns

public:
   enum { kVertical = 1, kHorizontal = 2 };

   TControlBar();
   TControlBar(const char *orientation, const char *title="");
   TControlBar(const char *orientation, const char *title, Int_t x, Int_t y);
   virtual ~TControlBar();

   void           AddButton(TControlBarButton *button);
   void           AddButton(TControlBarButton &button);
   void           AddButton(const char *label, const char *action, const char *hint="", const char *type="button");
   void           AddControlBar(TControlBar *controlBar);
   void           AddControlBar(TControlBar &controlBar);
   void           AddSeparator();
   TControlBarImp *GetControlBarImp() { return fControlBarImp; }
   TList          *GetListOfButtons() { return fButtons; }
   Int_t          GetNumberOfColumns() { return fNoroc; }
   Int_t          GetNumberOfRows() { return fNoroc; }
   Int_t          GetOrientation() { return fOrientation; }
   void           Hide();
   void           SetNumberOfColumns(Int_t n) { fNoroc = n; }
   void           SetNumberOfRows(Int_t n) { fNoroc = n; }
   void           SetOrientation(const char *o);
   void           SetOrientation(Int_t o);
   void           Show();

   ClassDef(TControlBar,0) //Control bar
};

#endif
