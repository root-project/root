// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   18/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoFinder
#define ROOT_TGeoFinder

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoVolume;
class TGeoMatrix;
class TGeoNode;

/*************************************************************************
 * TGeoFinder - virtual base class for tracking inside a volume. 
 *  
 *************************************************************************/

class TGeoFinder : public TObject
{
   enum EGeoPattern {
      kPatternX         = BIT(15),
      kPatternY         = BIT(16),
      kPatternZ         = BIT(17),
      kPatternCylR      = BIT(18),
      kPatternCylPhi    = BIT(19),
      kPatternSphR      = BIT(20),
      kPatternSphPhi    = BIT(21),
      kPatternSphThe    = BIT(22),
      kPatternHoneycomb = BIT(23)
   };
protected:
   TGeoVolume         *fVolume;         // volume to which this finder apply

public :
   TGeoFinder();
   TGeoFinder(TGeoVolume *vol);
   virtual ~TGeoFinder();

   virtual void        cd(Int_t idiv) = 0;
   virtual Int_t       GetByteCount() const {return 4;}
   virtual TGeoMatrix *GetMatrix() = 0;
   virtual void        SetBasicVolume(TGeoVolume *vol) = 0;
   virtual void        SetVolume(TGeoVolume *vol)  {fVolume = vol;}
   virtual TGeoVolume *GetVolume() const     {return fVolume;}

   virtual TGeoNode   *FindNode(Double_t *point) = 0; 

  ClassDef(TGeoFinder, 0)       // class for tracking inside volumes 
};

#endif

