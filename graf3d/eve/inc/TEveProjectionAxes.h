// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveProjectionAxes
#define ROOT_TEveProjectionAxes

#include "TEveText.h"

class TEveProjectionManager;

class TEveProjectionAxes : public TEveText
{
public:
   enum EMode      { kPosition, kValue };

private:
   TEveProjectionAxes(const TEveProjectionAxes&);            // Not implemented
   TEveProjectionAxes& operator=(const TEveProjectionAxes&); // Not implemented

protected:
   TEveProjectionManager*  fManager;  // model object

   Bool_t          fDrawCenter;  // draw center of distortion
   Bool_t          fDrawOrigin;  // draw origin

   EMode           fStepMode;       // tick-mark positioning
   Int_t           fNumTickMarks;  // number of tick-mark on axis

public:
   TEveProjectionAxes(TEveProjectionManager* m);
   virtual ~TEveProjectionAxes();

   TEveProjectionManager* GetManager(){ return fManager; }

   void            SetStepMode(EMode x)     { fStepMode = x;        }
   EMode           GetStepMode()   const    { return fStepMode;     }
   void            SetNumTickMarks(Int_t x) { fNumTickMarks = x;    }
   Int_t           GetNumTickMarks()  const { return fNumTickMarks; }

   void            SetDrawCenter(Bool_t x){ fDrawCenter = x;    }
   Bool_t          GetDrawCenter() const  { return fDrawCenter; }
   void            SetDrawOrigin(Bool_t x){ fDrawOrigin = x;    }
   Bool_t          GetDrawOrigin() const  { return fDrawOrigin; }

   virtual void    ComputeBBox();

   virtual const TGPicture* GetListTreeIcon(Bool_t open=kFALSE);

   ClassDef(TEveProjectionAxes, 1); // Short description.
};

#endif
