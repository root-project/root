// @(#)root/gpad:$Id$
// Author: Richard Maunder  10/3/2005

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewer3DPad
#define ROOT_TViewer3DPad

#include "TVirtualViewer3D.h"

class TVirtualPad;

class TViewer3DPad : public TVirtualViewer3D {
private:
   TVirtualPad &fPad;       ///< the pad we paint into.
   Bool_t       fBuilding;  ///< is scene being built?

   // Non-copyable
   TViewer3DPad(const TViewer3DPad &) = delete;
   TViewer3DPad & operator = (const TViewer3DPad &) = delete;

public:
   TViewer3DPad(TVirtualPad & pad) : fPad(pad), fBuilding(kFALSE) {};
   ~TViewer3DPad() {};

   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fBuilding; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t *addChildren = nullptr);
   virtual Int_t  AddObject(UInt_t placedID, const TBuffer3D &buffer, Bool_t *addChildren = nullptr);

   // Composite shapes not supported on this viewer currently - ignore.
   // Will result in a set of individual component shapes
   virtual Bool_t OpenComposite(const TBuffer3D & buffer, Bool_t *addChildren = nullptr);
   virtual void   CloseComposite();
   virtual void   AddCompositeOp(UInt_t operation);

   ClassDef(TViewer3DPad,0)  //A 3D Viewer painter for TPads
};

#endif
