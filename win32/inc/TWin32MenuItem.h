// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   03/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TWin32MenuItem
#define ROOT_TWin32MenuItem


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32MenuItem                                                       //
//                                                                      //
// This class defines the menu items.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "TVirtualMenuItem.h"

class TWin32SendClass;

class TWin32MenuItem : public TVirtualMenuItem {

protected:

   virtual void Free(TWin32Menu *menu);          // It is like Remove(TWin32Menu) but without feed back, so it is faster
   virtual void ExecuteEventCB(TWin32Canvas *c){;} //User click this item
   virtual void ExecThreadCB(TWin32SendClass *command);


public:

   TWin32MenuItem();
   TWin32MenuItem(EMenuItemType custom);
   TWin32MenuItem(char *name,const char *title,UINT type=MF_STRING,UINT state=MF_ENABLED,Int_t id=-1);
   TWin32MenuItem(char *name,const char *title,Win32CanvasCB callback,UINT type=MF_STRING,UINT state=MF_ENABLED,Int_t id=-1);
   TWin32MenuItem(char *name,const char *title,Win32BrowserImpCB callback,UINT type=MF_STRING,UINT state=MF_ENABLED,Int_t id=-1);


   virtual const char *ClassName();

   virtual void ExecuteEvent(TWin32Canvas *c);            //User click this item


   // ClassDef(TWin32MenuItem,0);
};

#endif





