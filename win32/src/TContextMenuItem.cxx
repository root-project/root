// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   13/03/96


#include "TContextMenuItem.h"

#ifndef ROOT_TWin32ContextMenuImp
#include "TWin32ContextMenuImp.h"
#endif

#ifndef ROOT_TContextMenu
#include "TContextMenu.h"
#endif

#ifndef ROOT_TWin32Canvas
#include "TWin32Canvas.h"
#endif

#ifndef ROOT_TCanvas
#include "TCanvas.h"
#endif


// ClassImp(TContextMenuItem)

//______________________________________________________________________________
TContextMenuItem::TContextMenuItem(){;} // default ctor

//______________________________________________________________________________
TContextMenuItem::TContextMenuItem(TContextMenu *contextmenu, TObject *obj, TMethod *method,
                                   char *name, const char *title, UINT type,
                                   UINT state, Int_t id) :
                  TVirtualMenuItem(name,title,type, state, id)
{
    fObject = obj;
    fMethod = method;
    fContextMenu = contextmenu;

//*-* I have to implemet this because fCanvas if TContextMenu will be make ZERO on destructions

}

//______________________________________________________________________________
const char *TContextMenuItem::ClassName()
{
   return "ContextMenuItem";
}

//______________________________________________________________________________
void TContextMenuItem::ExecuteEvent(TWin32Canvas *winobj)
{
//*-*  ContextMenu item has no special call back function "by item".

  TWin32ContextMenuImp *contextMenu = (TWin32ContextMenuImp *)fMenuList.First();
  if ( fContextMenu )
        fContextMenu->Action(fObject, fMethod );
}

//______________________________________________________________________________
void TContextMenuItem::Free(TWin32Menu *menu)
{
// Unlike TWin32MenuItem this Free just deletes itself
  //  cout << "TContextMenuItem::Free(TWin32Menu *menu)" << endl;

  if(menu) {
    fMenuList.Remove((TObject *)menu);
    if (fContextMenu)
    {
       TGWin32WindowsObject *winobj = ((TWin32ContextMenuImp *)(fContextMenu->GetContextMenuImp()))->GetWinObject();
      if (winobj) {
        if (fType != kSubMenu && fItem.fItemID > 0)
              winobj->UnRegisterMenuItem(fItem.fItemID);
        else
          Error("Free","Wrong CanvasImp pointer");

        if (fItem.fItemID > 0)
        {
            fItem.fItemID = 0 ;
            delete this;
        }
      }
      else
        Error("TContextMenuItem::Free","wrong TCanvas pointer");
    }
    else
      Error("TContextMenuItem::Free","Wrond Menu item");
  }
}
