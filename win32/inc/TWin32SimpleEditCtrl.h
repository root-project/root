// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   01/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32SimpleEditCtrl
#define ROOT_TWin32SimpleEditCtrl

////////////////////////////////////////////////////////
//                                                    //
// TWin32SimpleEditCtrl                               //
//                                                    //
// class to use WIN32 common controls                 //
//                 for TGWin32WindowsObject objects   //
//                                                    //
//                                                    //
////////////////////////////////////////////////////////

#include "TWin32CommCtrl.h"

class TWin32SimpleEditCtrl : public TWin32CommCtrl {

private:
   Int_t  fEditBufferLength;  // the length of the input buffer

protected:

public:
   TWin32SimpleEditCtrl(){fhwndWindow=0;} //default ctor
   TWin32SimpleEditCtrl(TGWin32WindowsObject *winobj,const char *title, Int_t lTitle, Float_t x,Float_t y, Float_t width, Float_t  height, const Text_t *type="EDIT", UInt_t style= WS_VISIBLE | ES_LEFT);
   virtual ~TWin32SimpleEditCtrl();  // default dtor
   virtual void             Add(TObject *obj, const char *name){; }
   virtual void             CreateAccessories(){ }                      // Create some extra things depends of the type of the control.
   virtual void             MoveControl();                               // Set the control to the new position
   virtual LRESULT APIENTRY OnNotify(LPARAM lParam){return 0;}           // CallBack function to manage the notify messages
   virtual LRESULT APIENTRY OnSubClassCtrl(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

   virtual UInt_t           GetItemObject(RECT *rpc){return 0;}          // returns the selected item lParam value and its position
   virtual Char_t          *GetText(Text_t *receiveBuffer);

//   ClassDef(TWin32SimpleEditCtrl,0)   // Basic Windows WIN32 common control class
};

#endif
