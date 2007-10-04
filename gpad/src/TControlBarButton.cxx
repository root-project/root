// @(#)root/gpad:$Id$
// Author: Nenad Buncic   20/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TControlBarButton                                                   //
//                                                                     //
// TControlBarButtons are created by the TControlBar. Not for general  //
// consumption.                                                        //
//                                                                     //
/////////////////////////////////////////////////////////////////////////

#include "TControlBarButton.h"
#include "TCanvas.h"
#include "TError.h"
#include "TApplication.h"

#include <ctype.h>

const char *kBStr = "BUTTON";
const char *kDStr = "DRAWNBUTTON";
const char *kSStr = "SEPARATOR";


ClassImp(TControlBarButton)

//_______________________________________________________________________
TControlBarButton::TControlBarButton() : TNamed()
{
   // Default controlbar button ctor.

   fType   = 0;
}

//_______________________________________________________________________
TControlBarButton::TControlBarButton(const char *label, const char *action,
                                     const char *hint, const char *type)
   : TNamed(label, hint)
{
   // Create controlbar button.

   SetType(type);
   SetAction(action);
}

//_______________________________________________________________________
void TControlBarButton::Action()
{
   // Execute controlbar button command.

   if (!fAction.IsNull()) {

      gApplication->ProcessLine(fAction.Data());

      if (gPad) gPad->Update();
   }
}

//_______________________________________________________________________
void TControlBarButton::SetAction(const char *action)
{
   // Set action to be executed by this button.

   if (action) {
      char *s = Strip(action);
      fAction = s;
      delete [] s;
   } else
      Error("SetAction", "action missing");
}


//_______________________________________________________________________
void TControlBarButton::SetType(const char *type)
{
   // Set button type. Type can be either "button", "drawnbutton" or
   // "separator". String is case insensitive. Default is "button".

   fType = kButton;

   if (type && *type) {
      if (!strcasecmp(type, kBStr))
         fType = kButton;
      else if (!strcasecmp(type, kDStr))
         fType = kDrawnButton;
      else if (!strcasecmp(type, kSStr))
         fType = kSeparator;
      else
         Error("SetType", "unknown type '%s' !\n\t(choice of: %s, %s, %s)",
               type, kBStr, kDStr, kSStr);
   }
}

//_______________________________________________________________________
void TControlBarButton::SetType(Int_t type)
{
   // Set button type. Type can be either kButton, kDrawnButton or kSeparator.
   // Default is kButton.

   switch (type) {

      case kButton:
      case kDrawnButton:
      case kSeparator:
         fType = type;
         break;

      default:
         fType = kButton;
         Error("SetType", "unknown type: %d !\n\t(choice of: %d, %d, %d)",
               type, kButton, kDrawnButton, kSeparator);
   }
}
