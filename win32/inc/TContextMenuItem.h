// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   13/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TContextMenuItem
#define ROOT_TContextMenuItem

#include "TVirtualMenuItem.h"

class TMethod;
class TContextMenu;
class TCanvas;
class TGWin32WindowsObject;

class TContextMenuItem : public TVirtualMenuItem {

private:

  TMethod *fMethod;
  TObject *fObject;
  TContextMenu *fContextMenu;
  TCanvas  *fCanvas;

protected:
  virtual void Free(TWin32Menu *menu);  // Unlike TWin32MenuItem Free just deletes itself
   void ExecuteEventCB(TWin32Canvas *c){;}          //User click this item
   void ExecThreadCB(TWin32SendClass *command){;}   // It is empty for this class

public:
   TContextMenuItem();
   TContextMenuItem(TContextMenu *contextmenu,TObject *obj,TMethod *method,char *name="ContextMenuItem",
                    const char *title="MenuItem",UINT type=MF_STRING,UINT state=MF_ENABLED,Int_t id=-1);

   const char *ClassName();
   void ExecuteEvent(TWin32Canvas *c);          //User click this item
   void ExecCommandThread(TWin32SendClass *command){
     printf("TContextMenuItem must not call ExecCommandThread function !!! \n");}   // It is empty for this class

  // ClassDef(TContextMenuItem,0)
};

#endif
