// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoAtt
#define ROOT_TGeoAtt

#ifndef ROOT_TObject
#include "TObject.h"
#endif

/*************************************************************************
 * TGeoAtt - visualization and tracking attributes for volumes and nodes
 *
 *
 *
 *************************************************************************/

class TGeoAtt
{
public:
  enum {
      kBitMask          = 0x00ffffff        // bit mask
   };

   enum EGeoVisibilityAtt {
      kVisOverride      = BIT(0),           // volume's vis. attributes are overidden
      kVisNone          = BIT(1),           // the volume/node is invisible, as well as daughters
      kVisThis          = BIT(2),           // this volume/node is visible
      kVisDaughters     = BIT(3),           // all leaves are visible
      kVisOneLevel      = BIT(4),           // first level daughters are visible
      kVisStreamed      = BIT(5),           // true if attributes have been streamed
      kVisTouched       = BIT(6)            // true if attributes are changed after closing geom
   };                          // visibility attributes

   enum EGeoActivityAtt   {
      kActOverride      = BIT(8),           // volume's activity attributes are overidden
      kActNone          = BIT(9),           // the volume/node is ignored by tracking, as well as daughters
      kActThis          = BIT(10),           // this volume/node is active for tracking
      kActDaughters     = BIT(11)            // all leaves are active
   };                          // activity flags

   enum EGeoOptimizationAtt {
      kUseBoundingBox   = BIT(16),           // use bounding box for tracking
      kUseVoxels        = BIT(17),           // compute and use voxels
      kUseGsord         = BIT(18)            // use slicing in G3 style     
   };
                          // tracking optimization attributes
protected :
// data members
   UInt_t                fGeoAtt;            // option flags
public:
   // constructors
   TGeoAtt();
   TGeoAtt(Option_t *vis_opt, Option_t *activity_opt="", Option_t *optimization_opt="");
   // destructor
   virtual ~TGeoAtt();
   // methods
   void                SetBit(UInt_t f)   {fGeoAtt |= f & kBitMask;}
   void                ResetBit(UInt_t f) {fGeoAtt &= ~(f & kBitMask);}
   Bool_t              TestBit(UInt_t f) const {return (Bool_t)((fGeoAtt & f) != 0);}

   virtual void        SetVisibility(Bool_t vis=kTRUE);
   void                SetVisDaughters(Bool_t vis=kTRUE);
   void                SetVisStreamed(Bool_t vis=kTRUE);
   void                SetVisTouched(Bool_t vis=kTRUE);
   virtual void        SetActivity(Option_t *option);
   virtual void        SetOptimization(Option_t *option);

   virtual Bool_t      IsVisible() const {return TestBit(kVisThis);}
   Bool_t              IsVisDaughters() const {return TestBit(kVisDaughters);}
   Bool_t              IsVisStreamed() const {return TestBit(kVisStreamed);}
   Bool_t              IsVisTouched() const {return TestBit(kVisTouched);}
//   EGeoVisibilityAtt   GetVisAttributes();
//   EGeoActivityAtt     GetActivityAtt();
//   EGeoOptimizationAtt GetOptimizationAtt();

  ClassDef(TGeoAtt, 1)         // class for visibility, activity and optimization attributes for volumes/nodes
};

#endif

