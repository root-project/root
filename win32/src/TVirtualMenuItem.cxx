// @(#)root/win32:$Name:  $:$Id: TVirtualMenuItem.cxx,v 1.1.1.1 2000/05/16 17:00:46 rdm Exp $
// Author: Valery Fine   18/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualMenuItem                                                     //
//                                                                      //
// This class defines the menu items.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualMenuItem.h"

#include "TWin32Menu.h"
#include "TCollection.h"

// ClassImp(TVirtualMenuItem)

//______________________________________________________________________________
TVirtualMenuItem::TVirtualMenuItem() {
    fItem.fItemID = -1;
    fMenuItem.wID = -1;
}

//______________________________________________________________________________
TVirtualMenuItem::TVirtualMenuItem(EMenuItemType custom){

//*-* special item to draw separators and break lines
      fMenuItem.cbSize = sizeof(MENUITEMINFO);

      fMenuItem.fType  = custom;
      fMenuItem.fState = MFS_ENABLED;
      fMenuItem.wID = 0;
      fMenuItem.hbmpChecked   = NULL;
      fMenuItem.hbmpUnchecked = NULL;

      fMenuItem.dwTypeData = (char *)GetTitle();
      if (fMenuItem.dwTypeData) fMenuItem.cch   = strlen(fMenuItem.dwTypeData);

      fMenuItem.hSubMenu = NULL;

      fType         = custom;
      fItem.fItemID = 0;
      fCanvasCB     = NULL;
      fBrowserCB    = NULL;

}
//______________________________________________________________________________
TVirtualMenuItem::TVirtualMenuItem(char *name, const char *label, UINT type, UINT state, Int_t id) : TNamed(name,label)
{
//*-*   Keep the type of the current Item:
//*-*    MF_STRING
//*-*    MF_BITMAP
//*-*    MF_OWNERDRAW
//*-*    MF_POPUP
   fType         = type;
   if (fType == kSubMenu)
       fMenuItem.fType = kString;
   else
       fMenuItem.fType = type;

//*-*   Keep the state of the current Item:
//*-*    MF_GRAYED
//*-*    MF_DISABLEd
//*-*    MF_ENABLED
//*-*    MF_CHECKED
//*-*    MF_UNCHECKED
   fMenuItem.cbSize = sizeof(MENUITEMINFO);
   fMenuItem.fState = state;
   fMenuItem.hbmpChecked   = NULL;
   fMenuItem.hbmpUnchecked = NULL;

   fMenuItem.dwTypeData = (char *)GetTitle();
   if (fMenuItem.dwTypeData) fMenuItem.cch   = strlen(fMenuItem.dwTypeData);

   fCanvasCB     = NULL;
   fBrowserCB    = NULL;

   fMenuItem.hSubMenu = NULL;

   if (type == kSubMenu) {
     fItem.fItemHandle = new TWin32Menu(name,label,kPopUpMenu);  // create SubMenu as the current item
     fMenuItem.hSubMenu = fItem.fItemHandle->GetMenuHandle();
   }
   else {
      fItem.fItemID       = id;
      fMenuItem.wID       = id;
   }
}

//______________________________________________________________________________
TVirtualMenuItem::TVirtualMenuItem(char *name, const char *label, Win32CanvasCB callback, UINT type, UINT state, Int_t id) : TNamed(name,label)
{
//*-*   Keep the type of the current Item:
//*-*    MF_STRING     = 0x0000L  !!!  be carefull  !!!
//*-*    MF_BITMAP     = 0x0004L
//*-*    MF_OWNERDRAW  = 0x0100L
//*-*    MF_POPUP      = 0x0010L
   fMenuItem.cbSize = sizeof(MENUITEMINFO);
   fType           = type;
   fMenuItem.fType = type;

//*-*   Keep the state of the current Item:
//*-*    MF_GRAYED     = 0x0001L
//*-*    MF_DISABLED   = 0x0002L
//*-*    MF_ENABLED    = 0x0000L  !!!  be carefull  !!!
//*-*    MF_CHECKED    = 0x0008L
//*-*    MF_UNCHECKED  = 0x0000L  !!!  be carefull  !!!

   fMenuItem.fState = state;

   fMenuItem.hbmpChecked   = NULL;
   fMenuItem.hbmpUnchecked = NULL;

   fMenuItem.dwTypeData = (char *)GetTitle();
   if (fMenuItem.dwTypeData) fMenuItem.cch   = strlen(fMenuItem.dwTypeData);

   fItem.fItemID       = id;
   fMenuItem.wID       = id;

   fMenuItem.hSubMenu = NULL;

   fCanvasCB = callback;
   fBrowserCB     = NULL;
}

//______________________________________________________________________________
TVirtualMenuItem::TVirtualMenuItem(char *name, const char *label, Win32BrowserImpCB callback, UINT type, UINT state, Int_t id) : TNamed(name,label)
{
//*-*   Keep the type of the current Item:
//*-*    MF_STRING     = 0x0000L  !!!  be carefull  !!!
//*-*    MF_BITMAP     = 0x0004L
//*-*    MF_OWNERDRAW  = 0x0100L
//*-*    MF_POPUP      = 0x0010L
    fMenuItem.cbSize = sizeof(MENUITEMINFO);

    fType         = type;
    fMenuItem.fType = type;

//*-*   Keep the state of the current Item:
//*-*    MF_GRAYED     = 0x0001L
//*-*    MF_DISABLED   = 0x0002L
//*-*    MF_ENABLED    = 0x0000L  !!!  be carefull  !!!
//*-*    MF_CHECKED    = 0x0008L
//*-*    MF_UNCHECKED  = 0x0000L  !!!  be carefull  !!!
   fMenuItem.fState = state;

   fMenuItem.hbmpChecked   = NULL;
   fMenuItem.hbmpUnchecked = NULL;

   fMenuItem.dwTypeData = (char *)GetTitle();
   if (fMenuItem.dwTypeData) fMenuItem.cch   = strlen(fMenuItem.dwTypeData);

   fItem.fItemID       = id;
   fMenuItem.wID       = id;

   fMenuItem.hSubMenu = NULL;

   fCanvasCB  = NULL;
   fBrowserCB = callback;
}

//______________________________________________________________________________
TVirtualMenuItem::~TVirtualMenuItem()
{

//*-* Remove this item from all menus and Command lists

   TIter next(&fMenuList);
   {
    TWin32Menu *menu;
    while (menu = (TWin32Menu *)next() )
       menu->DeleteTheItem(this);
   }

   if (fType ==  kSubMenu && fItem.fItemHandle) { //Delete PopUp Item
       delete fItem.fItemHandle;
       fItem.fItemHandle = 0;
       fMenuItem.hSubMenu = NULL;
   }
//*-*  We don't check here whether this item has been removed from the command list !!!
}

//______________________________________________________________________________
void TVirtualMenuItem::Checked(){
//*-*  Places a check mark next to the menu item
    fMenuItem.fState = fMenuItem.fState & ~(MFS_UNCHECKED) |  MFS_CHECKED ;
    SetItemStatus(kMenuChecked);
}

//______________________________________________________________________________
const char *TVirtualMenuItem::ClassName()
{
   return "TVirtualMenuItem";
}
//______________________________________________________________________________
void TVirtualMenuItem::Disable(){
//*-*   Disables the menu item so that it cannot be selected, but does not gray it.
    fMenuItem.fState = fMenuItem.fState & ~(MFS_ENABLED | MFS_GRAYED) |  MFS_DISABLED ;
    SetItemStatus(kMenuEnabled);
}
//______________________________________________________________________________
void TVirtualMenuItem::Enable(){
//*-*  Enables the menu item so that it can be selected
//*-*  and restores it from its grayed state.
    fMenuItem.fState = fMenuItem.fState & ~(MFS_DISABLED | MFS_GRAYED) | MFS_ENABLED;
    SetItemStatus(kMenuEnabled);
}

//______________________________________________________________________________
UINT TVirtualMenuItem::GetCommandId(){
    if (fMenuItem.hSubMenu)
      return (UINT) (fItem.fItemHandle->GetMenuHandle());
    else
      return fMenuItem.wID; // idfItem.fItemID;
}

//______________________________________________________________________________
TWin32Menu *TVirtualMenuItem::GetPopUpItem(){
    if (fMenuItem.hSubMenu)
        return fItem.fItemHandle;
    else
        return 0;
}
//______________________________________________________________________________
void TVirtualMenuItem::Join(TWin32Menu *menu)
{
    // Register the menu where this item is inserted in
    if(menu) {
        fMenuList.Add   ((TObject *)menu);
        fMenuItem.fMask = MIIM_DATA | MIIM_STATE | MIIM_TYPE | MIIM_SUBMENU | MIIM_ID ;
        if (InsertMenuItem(menu->GetMenuHandle(),99999,TRUE,&fMenuItem)==FALSE)
            printf(" Insert Menu Item Error: %d \n", GetLastError());
    }
}
//______________________________________________________________________________
void TVirtualMenuItem::ModifyTitle(char const *title){
    SetTitle(title);
    fMenuItem.dwTypeData = (char *)GetTitle();
    if (fMenuItem.dwTypeData)
        fMenuItem.cch   = 4;
//        fMenuItem.cch   = strlen(fMenuItem.dwTypeData);
    SetItemStatus(kMenuModified);
}

//______________________________________________________________________________
void TVirtualMenuItem::Grayed(){
//*-*  Disables the menu item and grays it so it cannot be selected.
   fMenuItem.fState = fMenuItem.fState & ~(MFS_DISABLED | MFS_ENABLED) | MFS_GRAYED;
   SetItemStatus(kMenuEnabled);
}

//______________________________________________________________________________
void TVirtualMenuItem::Remove()
{
   // I have no idea how this should work, other than to unmanage the widget
}

//______________________________________________________________________________
void TVirtualMenuItem::Remove(TWin32Menu *menu) {

  if (fMenuList.FindObject((TObject *)menu)) {
       menu->DeleteTheItem(this);
       fMenuList.Remove((TObject *)menu);
  }
}

//______________________________________________________________________________
UINT TVirtualMenuItem::SetCommandId(UINT id){
//*-*  Check whether the Command ID for this item was assigned
//*-*  If this is for the first time just assign the new value
//*-*  return the ID for this Item

// Check type of item.
// It is possible to set Id for the simple item only

//    cout << "TVirtualMenuItem::SetCommandId(" << id << ")" << endl;
    if (!fMenuItem.hSubMenu)
    {
        if (fItem.fItemID == -1)
        {
            fItem.fItemID = id;
            fMenuItem.wID = id;
        }
        else
            Error("TWin32MenuItem", "Impossible to set the Item ID twice");
    }

    return GetCommandId();
}


//______________________________________________________________________________
void TVirtualMenuItem::SetItemStatus(EMenuModified state){
//   if (fItem.fItemID == -1) Error("TWin23Menu", "There is no ID for the present item");
   if (fMenuItem.wID == -1) Error("TWin23Menu", "There is no ID for the present item");

   TIter next(&fMenuList);
   TWin32Menu *menu;

//*- change the appearance of this item in the all menu where this is presented in

  while (menu = (TWin32Menu *) next())
  {
      HMENU hMenu = menu->GetMenuHandle();
      fMenuItem.fMask = MIIM_STATE;
      if (state == kMenuModified) fMenuItem.fMask= MIIM_TYPE ;
      SetMenuItemInfo(hMenu,fMenuItem.wID,FALSE,&fMenuItem);
  }
}

//______________________________________________________________________________
void TVirtualMenuItem::SetLabel(const char *str){
}

//______________________________________________________________________________
void TVirtualMenuItem::SetPosition(int position, TWin32Menu *menu)
{
  if (!fMenuList.FindObject(menu)) fMenuList.Add((TObject *)menu);
#ifndef WIN32
  menu->InsertItemAtPosition(this,position);
#endif
}


//______________________________________________________________________________
void TVirtualMenuItem::UnChecked(){
//*-*  Does not place a check mark next to the menu item (default)
    fMenuItem.fState = fMenuItem.fState & ~(MFS_CHECKED) |  MFS_UNCHECKED ;
    SetItemStatus(kMenuChecked);
}
