// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   23/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TWin32Menu
#define ROOT_TWin32Menu

#include "Windows4Root.h"

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#ifndef ROOT_TWin32MenuItem
#include "TWin32MenuItem.h"
#endif

typedef enum {kMenu, kPopUpMenu} EMenuType;

class TWin32Menu : public TNamed {

protected:

   friend  class      TVirtualMenuItem;

   HMENU         fMenu;          // The present Menu handle
   TList         fItemsList;     // List of the items
   int           fItemNumber;
   int           fBaseItemNumber;
//   TWin32CallBackList fMenuAction;

   void  DeleteTheItem(TVirtualMenuItem *item);
   void  DetachItems();

public:

   TWin32Menu();
   TWin32Menu(char *name, const char *title="Win32Menu", EMenuType issubmenu = kMenu);
   virtual ~TWin32Menu();

   void  Add(TVirtualMenuItem *item);
   void  Add(EMenuItemType custom);
   Int_t DefineItemPosition(TVirtualMenuItem *item);
   HMENU GetMenuHandle();
   void  InsertItemAtPoistion(TVirtualMenuItem *item, Int_t position);
   void  Modify(TVirtualMenuItem *item, UINT type);
   void  RemoveTheItem(TVirtualMenuItem *item);
   void  SetMethod();
   void  DisplayRootMenu();

   // ClassDef(TWin32Menu,0)
};

#endif
