// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   18/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef CodeROOT_TVirtualMenuItem
#define CodeROOT_TVirtualMenuItem



//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualMenuItem                                                     //
//                                                                      //
// This ABC class defines the menu items.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "Rtypes.h"
#include "Windows4Root.h"
#include "TNamed.h"

#ifndef ROOT_TWin32HookViaThread
#include "TWin32HookViaThread.h"
#endif

//*-*  One and the same Item may belong to separate menus.
//*-*  We can delete item of there is no menu refereing it anymore

typedef enum { kSeparator=MFT_SEPARATOR, kMenuBreak=MFT_MENUBREAK,
               kMenuBarBreak=MFT_MENUBARBREAK, kSubMenu=MF_POPUP,
               kRightJustify=MFT_RIGHTJUSTIFY, kString=MFT_STRING
             }   EMenuItemType;

typedef enum {kMenuModified, kMenuEnabled, kMenuChecked } EMenuModified;

class TWin32Canvas;
class TWin32BrowserImp;
class TWin32Menu;
class TWin32MenuItem;
class TVirtualMenuItem;

typedef void  (*Win32CanvasCB)(TWin32Canvas *c, TVirtualMenuItem *item);
typedef void  (*Win32BrowserImpCB)(TWin32BrowserImp *c, TVirtualMenuItem *item);

typedef  union {
           Int_t          fItemID;        // command Id of the simple item
           TWin32Menu    *fItemHandle;    // the pointer to the menu for submenu item
   } ItemId_t;

class TVirtualMenuItem : protected TWin32HookViaThread, public TNamed {

protected:

   friend  class TWin32Menu;
   TList    fMenuList;             // List of the parent menu.

   UINT           fType;           // Type of this item
   Win32CanvasCB  fCanvasCB;       // Pointer to the Canvas CallBack function
   Win32BrowserImpCB  fBrowserCB;  // Pointer to the BrowserImp CallBack function
   ItemId_t       fItem;           // Item description

   MENUITEMINFO   fMenuItem;       // WIN32 structure to define a meny item

   Bool_t fInstantiated;
   Bool_t fActivated;

   virtual void Join(TWin32Menu *menu);
   virtual void Free(TWin32Menu *menu) = 0;        // It is like Remove(TWin32Menu) but without feed back, so it is faster
   SetItem(Int_t iFlag);
   virtual void SetItemStatus(EMenuModified state=kMenuModified);
   virtual void ExecuteEventCB(TWin32Canvas *c) = 0;              //User click this item


public:

   TVirtualMenuItem();
   TVirtualMenuItem(EMenuItemType custom);
#if 0
   TVirtualMenuItem(char *name,const char *title,UINT type=kString,UINT state=MF_ENABLED,Int_t id=-1);
   TVirtualMenuItem(char *name,const char *title,Win32CanvasCB callback,UINT type=kString,UINT state=MF_ENABLED,Int_t id=-1);
   TVirtualMenuItem(char *name,const char *title,Win32BrowserImpCB callback,UINT type=kString,UINT state=MF_ENABLED,Int_t id=-1);
#else
   TVirtualMenuItem(char *name,const char *title,UINT type=kString,UINT state=MFS_ENABLED,Int_t id=-1);
   TVirtualMenuItem(char *name,const char *title,Win32CanvasCB callback,UINT type=kString,UINT state=MFS_ENABLED,Int_t id=-1);
   TVirtualMenuItem(char *name,const char *title,Win32BrowserImpCB callback,UINT type=kString,UINT state=MFS_ENABLED,Int_t id=-1);
#endif

   virtual ~TVirtualMenuItem();

   virtual const char *ClassName();


   virtual void Checked();

   virtual void Disable();
   virtual void Enable();
   virtual void ExecuteEvent(TWin32Canvas *c) = 0;          //User click this item

   UINT GetState(){ return fMenuItem.fState; }
   UINT GetType(){ return fType; }
   UINT GetCommandId();
   LPCTSTR GetItem(){ return GetTitle(); }
   virtual TWin32Menu *GetPopUpItem();

   virtual void Grayed();
   void ModifyTitle(char const *title);

   void Remove();
   void Remove(TWin32Menu *menu);
   UINT SetCommandId(UINT id);
   void SetLabel(const char *str);


   void SetPosition(int position, TWin32Menu *menu); //Set position of this item with menu Menu

   virtual void UnChecked();

//   ClassDef(TVirtualMenuItem,0)
};

#endif
