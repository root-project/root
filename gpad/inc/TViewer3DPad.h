/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TViewer3DPad
#define ROOT_TViewer3DPad

#ifndef ROOT_TVirtualViewer3D
#include "TVirtualViewer3D.h"
#endif

class TVirtualPad;

class TViewer3DPad : public TVirtualViewer3D
{
private:
   TVirtualPad &  fPad;
   Bool_t         fBuilding;
   
   // Non-copyable
   TViewer3DPad(const TViewer3DPad &);
   TViewer3DPad & operator = (const TViewer3DPad &);
      
public:
   TViewer3DPad(TVirtualPad & pad) : fPad(pad), fBuilding(kFALSE) {};
   ~TViewer3DPad() {};
  
   virtual Bool_t PreferLocalFrame() const;
   virtual void   BeginScene();
   virtual Bool_t BuildingScene() const { return fBuilding; }
   virtual void   EndScene();
   virtual Int_t  AddObject(const TBuffer3D & buffer, Bool_t * addChildren = 0);               
   virtual Int_t  AddObject(UInt_t placedID, const TBuffer3D & buffer, Bool_t * addChildren = 0);
};

#endif
