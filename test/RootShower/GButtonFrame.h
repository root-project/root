// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GButtonFrame                                                         //
//                                                                      //
// This File contains the declaration of the GButtonFrame-class for     //
// the RootShower application                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef GBUTTONFRAME_H
#define GBUTTONFRAME_H

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGButton;


class GButtonFrame: public TGCompositeFrame {

private:
   TGLayoutHints  *fButtonLayout;      // Buttons layout
   TGButton       *fNextEventButton;   // "Start New Event" button
   TGButton       *fShowTrackButton;   // "Show Selection" button
   TGButton       *fStopSimButton;     // "Interrupt Simulation" button

public:
   // enum
   enum EState {
      kAllActive,
      kNoneActive
   };

   // Constructor & destructor
   GButtonFrame(const TGWindow* p, TGWindow* buttonHandler, Int_t nextEventId,
                Int_t showTrackId, Int_t interruptSimId);
   virtual ~GButtonFrame();

   void SetState(EState state);
};

#endif // GBUTTONFRAME_H
