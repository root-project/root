// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   22/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TWin32ContextMenuImp                                                       //
//                                                                            //
// This class provides an interface to  context sensitive popup menus.        //
// These menus pop up when the user hits  the right mouse button,  and        //
// are destroyed when the menu pops downs.                                    //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TWin32ContextMenuImp
#define ROOT_TWin32ContextMenuImp

#include "TContextMenuImp.h"

#include "TWin32Menu.h"

#ifdef ROOT_TWin32MenuItem
#include "TWin32MenuItem.h"
#endif

//*-*
//*-* Context Menu is derived from TWin32Menu (since it is special type of PopUp menu
//*-*   with
//*-*
//*-*    TWin32MenuItem  fTitle
//*-*    TWin32MenuItem  fProperties
//*-*    TGWin32WindowsObject fWindowsObj
//*-*
//*-*   where
//*-*
//*-*     fTitle      is the first item of the menu
//*-*     fProperties is the last one
//*-*     fWindowsObj is a pointer to the parent Windows object
//*-*     ("normal" menu has no direct relation with any Windows objects)
//*-*

class TGWin32WindowsObject;
class TWin32Dialog;
class TGWin32Command;
class TWin32SendClass;

class TWin32ContextMenuImp : protected TWin32HookViaThread, public TContextMenuImp, public TWin32Menu {

 private:

   TGWin32WindowsObject   *fWindowObj;
   TWin32Dialog           *fDialog;
   TWin32MenuItem          fTitle;
   TWin32MenuItem         *fProperties;
   int                     fPopupCreated;

   void  ClearProperties();
   void  CreatePopup  ();

   void  UpdateProperties();

 protected:

   void ExecThreadCB(TWin32SendClass *code);


 public:

    TWin32ContextMenuImp(TContextMenu *c=0);
    virtual ~TWin32ContextMenuImp();
    void       CreatePopup  ( TObject *object );
    void       Dialog       ( TObject *object, TMethod *method );
    void       DisplayPopup ( Int_t x, Int_t y);

    TGWin32WindowsObject   *GetWinObject(){ return fWindowObj;}
    void                    SetWinObject(TGWin32WindowsObject *winobj){fWindowObj=winobj;}

    // ClassDef(TWin32ContextMenuImp,0) //Context sensitive popup menu implementation
};
#endif


