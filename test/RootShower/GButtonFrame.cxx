// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <TGButton.h>
#include "GButtonFrame.h"

//______________________________________________________________________________
// GButtonFrame
//
// A GButtonFrame is a frame containing the RootShower buttons. 
//______________________________________________________________________________


//______________________________________________________________________________
GButtonFrame::GButtonFrame(const TGWindow* p, TGWindow* buttonHandler, 
                           Int_t nextEventId, Int_t showTrackId, 
                           Int_t interruptSimId) : 
                           TGCompositeFrame(p, 100, 100, kVerticalFrame)
{
   // Create GButtonFrame object, with TGWindow parent *p.
   //
   // buttonHandler = pointer to button handler TGWindow
   // nextEventId = id of NextEvent button

   // Create Layout hints
   fButtonLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 5, 2, 2, 2);

   // Create Event Buttons
   fNextEventButton = new TGTextButton(this, "Start &New Event", nextEventId);
   fNextEventButton->Associate(buttonHandler);
   fNextEventButton->SetToolTipText("Start new simulation event");
   AddFrame(fNextEventButton, fButtonLayout);
   fStopSimButton = new TGTextButton(this, "&Interrupt Simulation", interruptSimId);
   fStopSimButton->Associate(buttonHandler);
   fStopSimButton->SetToolTipText("Interrupts the current simulation");
   AddFrame(fStopSimButton, fButtonLayout);
   fShowTrackButton = new TGTextButton(this, "&Show Selection", showTrackId);
   fShowTrackButton->Associate(buttonHandler);
   fShowTrackButton->SetToolTipText("Shows the selected track");
   AddFrame(fShowTrackButton, fButtonLayout);

   fNextEventButton->Resize(150,GetDefaultHeight());
   fStopSimButton->Resize(150,GetDefaultHeight());
   fShowTrackButton->Resize(150,GetDefaultHeight());

   SetState(kAllActive);
   fShowTrackButton->SetState(kButtonDisabled);
   fStopSimButton->SetState(kButtonDisabled);
}

//______________________________________________________________________________
GButtonFrame::~GButtonFrame()
{
   // Destroy GButtonFrame object. Delete all created widgets

   delete fNextEventButton;
   delete fButtonLayout;
}

//______________________________________________________________________________
void GButtonFrame::SetState(EState state)
{
   // Set the state of the GButtonFrame. This sets the state of
   // the TGButton's in this frame.

   switch (state) {
      case kAllActive:
         fNextEventButton->SetState(kButtonUp);
         fShowTrackButton->SetState(kButtonUp);
         fStopSimButton->SetState(kButtonDisabled);
         break;

      case kNoneActive:
         fNextEventButton->SetState(kButtonDisabled);
         fShowTrackButton->SetState(kButtonDisabled);
         fStopSimButton->SetState(kButtonUp);
         break;

   } // switch state 
   // make sure window gets updated...
   gClient->HandleInput();
}

