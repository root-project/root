// @(#)root/base:$Name:  $:$Id: TBuffer3D.cxx,v 1.00
// Author: Olivier Couet   05/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

ClassImp(TBuffer3D)

//______________________________________________________________________________
TBuffer3D::TBuffer3D(Int_t type,
                     UInt_t reqPnts, UInt_t reqPntsCapacity,
                     UInt_t reqSegs, UInt_t reqSegsCapacity, 
                     UInt_t reqPols, UInt_t reqPolsCapacity) :
      fType(type)
{
	Init();
   SetRawSizes(reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity);
}


//______________________________________________________________________________
TBuffer3D::~TBuffer3D()
{
   if (fPnts) delete [] fPnts;
   if (fSegs) delete [] fSegs;
   if (fPols) delete [] fPols;
}

//______________________________________________________________________________
void TBuffer3D::Init()
{
   fID            = 0;
   fColor         = 0;
   fTransparency  = 0;
   fLocalFrame	   = kFALSE;
   fReflection    = kFALSE;
   SetLocalMasterIdentity();

   // Reset bounding box
   for (UInt_t i=0; i<3; i++) {
      fBBLowVertex[i] = 0.0;
      fBBHighVertex[i] = 0.0;
   }

   fPnts          = 0;
   fSegs          = 0;
   fPols          = 0;

   fNbPnts        = 0;           
   fNbSegs        = 0;           
   fNbPols        = 0;        
   fPntsCapacity  = 0;  
   fSegsCapacity  = 0;  
   fPolsCapacity  = 0;  

   ClearSectionsValid();
}

void TBuffer3D::ClearSectionsValid()
{ 
   fSections = 0U; 
   SetRawSizes(0, 0, 0, 0, 0, 0);
}

void TBuffer3D::SetLocalMasterIdentity()
{
   for (UInt_t i=0; i<16; i++) {
      if (i%5) {
         fLocalMaster[i] = 0.0;
      }
      else {
         fLocalMaster[i] = 1.0;
      }
   }
}

 //______________________________________________________________________________
Bool_t TBuffer3D::SetRawSizes(UInt_t reqPnts, UInt_t reqPntsCapacity,
                              UInt_t reqSegs, UInt_t reqSegsCapacity, 
                              UInt_t reqPols, UInt_t reqPolsCapacity)
{
   Bool_t allocateOK = kTRUE;

   fNbPnts = reqPnts;
   fNbSegs = reqSegs;
   fNbPols = reqPols;
   
   if (reqPntsCapacity > fPntsCapacity) {
      delete [] fPnts;
      fPnts = new Double_t[reqPntsCapacity];
      if (fPnts) {
         fPntsCapacity = reqPntsCapacity;
      } else {
         fPntsCapacity = fNbPnts = 0;
         allocateOK = kFALSE;
      }
   }
   if (reqSegsCapacity > fSegsCapacity) {
      delete [] fSegs;
      fSegs = new Int_t[reqSegsCapacity];
      if (fSegs) {
         fSegsCapacity = reqSegsCapacity;
      } else {
         fSegsCapacity = fNbSegs = 0;
         allocateOK = kFALSE;
      }
   }
   if (reqPolsCapacity > fPolsCapacity) {
      delete [] fPols;
      fPols = new Int_t[reqPolsCapacity];
      if (fPols) {
         fPolsCapacity = reqPolsCapacity;
      } else {
         fPolsCapacity = fNbPols = 0;
         allocateOK = kFALSE;
      }
   }

   return allocateOK; 
}

//______________________________________________________________________________
TBuffer3DSphere::TBuffer3DSphere(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                 UInt_t reqSegs, UInt_t reqSegsCapacity, 
                                 UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(TBuffer3DTypes::kSphere, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(0.0),
   fThetaMin(0.0), fThetaMax(180.0),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

//______________________________________________________________________________
Bool_t TBuffer3DSphere::IsSolidUncut() const
{
   // TODO: Rounding errors?
   if (fRadiusInner   != 0.0   ||
       fThetaMin      != 0.0   ||
       fThetaMax      != 180.0 ||
       fPhiMin        != 0.0   || 
       fPhiMax        != 360.0 ) {
          return kFALSE;
       } else {
          return kTRUE;
       }
}

//______________________________________________________________________________
TBuffer3DTube::TBuffer3DTube(UInt_t reqPnts, UInt_t reqPntsCapacity,
                             UInt_t reqSegs, UInt_t reqSegsCapacity, 
                             UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(TBuffer3DTypes::kTube, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(1.0), fHalfLength(1.0)   
{
}

//______________________________________________________________________________
TBuffer3DTube::TBuffer3DTube(Int_t type,
                             UInt_t reqPnts, UInt_t reqPntsCapacity,
                             UInt_t reqSegs, UInt_t reqSegsCapacity, 
                             UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3D(type, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fRadiusInner(0.0), fRadiusOuter(1.0), fHalfLength(1.0)
{
}

//______________________________________________________________________________
TBuffer3DTubeSeg::TBuffer3DTubeSeg(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity, 
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTube(TBuffer3DTypes::kTubeSeg, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

//______________________________________________________________________________
TBuffer3DTubeSeg::TBuffer3DTubeSeg(Int_t type,
                                   UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity, 
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTube(type, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity),
   fPhiMin(0.0), fPhiMax(360.0)
{
}

//______________________________________________________________________________
TBuffer3DCutTube::TBuffer3DCutTube(UInt_t reqPnts, UInt_t reqPntsCapacity,
                                   UInt_t reqSegs, UInt_t reqSegsCapacity, 
                                   UInt_t reqPols, UInt_t reqPolsCapacity) :
   TBuffer3DTubeSeg(TBuffer3DTypes::kCutTube, reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity)
{
   fLowPlaneNorm[0] = 0.0; fLowPlaneNorm[0] = 0.0; fLowPlaneNorm[0] = -1.0;
   fHighPlaneNorm[0] = 0.0; fHighPlaneNorm[0] = 0.0; fHighPlaneNorm[0] = 1.0;
}

//CS specific
UInt_t TBuffer3D::fCSLevel = 0;

//______________________________________________________________________________
UInt_t TBuffer3D::GetCSLevel()
{
   return fCSLevel;
}

//______________________________________________________________________________
void TBuffer3D::IncCSLevel()
{
   ++fCSLevel;
}

//______________________________________________________________________________
UInt_t TBuffer3D::DecCSLevel()
{
   return --fCSLevel;
}
