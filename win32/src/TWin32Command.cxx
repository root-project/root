// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   04/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// The class to control WM_COMMAND Win32 messages

#include "TWin32Command.h"
#include "TWin32CommCtrl.h"

//______________________________________________________________________________
TWin32Command::TWin32Command(Int_t s, Int_t lowerBound) : TObjArray(s,lowerBound){
//*-*   This contrsuctor leaves the index "0" for sprecial things to customize menu like:
//*-*   Separators, MenuBreaks, MenuBarBreaks and so on
}

//______________________________________________________________________________
void TWin32Command::ExecuteEvent(Int_t Id, TWin32Canvas *c) {
    ((TVirtualMenuItem *)(At(Id)))->ExecuteEvent(c);
}

//______________________________________________________________________________
void TWin32Command::JoinMenuItem(TVirtualMenuItem *item) {

//*-*  Add the present menu item to the command list

//*-*  Id = 0 means the item is a special one (like a separator)

  if (item->GetType() != kSubMenu) {
    UINT id = item->GetCommandId();
    if (id == -1 )            //  Booking the free array slot
      item->SetCommandId(AddAtFree((TObject *)item));
    else if (id)              // Occupy id's slot
      AddAtAndExpand((TObject *)item,id);
  }
}

//______________________________________________________________________________
void TWin32Command::JoinControlItem(TWin32CommCtrl *ctrl) {

//*-*  Add the present control to the command list

    UINT id = ctrl->GetCommandId();
    if (id == -1 )            //  Booking the free array slot
      ctrl->SetCommandId(AddAtFree((TObject *)ctrl));
    else if (id)              // Occupy id's slot
      AddAtAndExpand((TObject *)ctrl,id);
}


