// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   03/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32MenuItem                                                       //
//                                                                      //
// This class defines the menu items.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TWin32MenuItem
#include "TWin32MenuItem.h"
#endif

#ifndef ROOT_TWin32Menu
#include "TWin32Menu.h"
#endif

#ifndef ROOT_TGWin32Command
#include "TGWin32Command.h"
#endif

#ifndef ROOT_TWin32BrowserImp
#include "TWin32BrowserImp.h"
#endif

// ClassImp(TWin32MenuItem)

//______________________________________________________________________________
TWin32MenuItem::TWin32MenuItem() : TVirtualMenuItem(){ ; }
//______________________________________________________________________________
TWin32MenuItem::TWin32MenuItem(EMenuItemType custom) : TVirtualMenuItem(custom){ ; }
//______________________________________________________________________________
TWin32MenuItem::TWin32MenuItem(char *name,const char *title,UINT type,UINT state,Int_t id) :
             TVirtualMenuItem(name,title,type,state,id) {;}
//______________________________________________________________________________
TWin32MenuItem::TWin32MenuItem(char *name,const char *title,Win32CanvasCB callback,UINT type,UINT state,Int_t id) :
             TVirtualMenuItem(name,title,callback,type,state,id) {;}

//______________________________________________________________________________
TWin32MenuItem::TWin32MenuItem(char *name,const char *title,Win32BrowserImpCB callback,UINT type,UINT state,Int_t id) :
             TVirtualMenuItem(name,title,callback,type,state,id) {;}

//______________________________________________________________________________
const char *TWin32MenuItem::ClassName()
{
   return "MenuItem";
}

//______________________________________________________________________________
void TWin32MenuItem::ExecuteEvent(TWin32Canvas *canvas)
{
//  We have to pass this pointer to another thread to synch threads.
//  But we have to check object since it could be destroyed by mean time !!!
// fCanvasCB(canvas,this);====? check this

   TWin32SendClass *code = new TWin32SendClass(this,(UInt_t)canvas,0,0,0);
   ExecCommandThread(code);
}

//______________________________________________________________________________
void TWin32MenuItem::ExecThreadCB(TWin32SendClass *command)
{
//  This function must be called from the "Command tread only"
    if (fCanvasCB){
         TWin32Canvas *canvas = (TWin32Canvas *)(command->GetData(0));
         fCanvasCB(canvas,(TVirtualMenuItem *)this);
    }
    else if (fBrowserCB) {
        TWin32BrowserImp *browser = (TWin32BrowserImp *)(command->GetData(0));
        fBrowserCB(browser,(TVirtualMenuItem *)this);
    }
    delete command;
}

//______________________________________________________________________________
void TWin32MenuItem::Free(TWin32Menu *menu)
{
//*-*  It is like Remove(TWin32Menu) but without feed back, so it is faster
    if(menu) {
        fMenuList.Remove((TObject *)menu);
    }
}
