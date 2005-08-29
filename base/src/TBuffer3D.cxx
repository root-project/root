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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer3D                                                            //
//                                                                      //
// Generic 3D primitive description class - see TBuffer3DTypes for      //
// producer classes                                                     //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer3D                                                            //
//                                                                      //
// Generic 3D primitive description class - see TBuffer3DTypes for      //
// producer classes                                                     //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
   // Construct from supplied shape type and raw sizes
//                                                                      //
// TBuffer3D                                                            //
//                                                                      //
// Generic 3D primitive description class - see TBuffer3DTypes for      //
// producer classes                                                     //
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
   // Destructor
   // Construct from supplied shape type and raw sizes
//                                                                      //
// TBuffer3D                                                            //
//                                                                      //
// Generic 3D primitive description class - see TBuffer3DTypes for      //
// producer classes                                                     //
//////////////////////////////////////////////////////////////////////////

   // Initialise buffer
ClassImp(TBuffer3D)
   // Destructor
   // Construct from supplied shape type and raw sizes

//______________________________________________________________________________
TBuffer3D::TBuffer3D(Int_t type,
                     UInt_t reqPnts, UInt_t reqPntsCapacity,
                     UInt_t reqSegs, UInt_t reqSegsCapacity, 
                     UInt_t reqPols, UInt_t reqPolsCapacity) :
      fType(type)
   // Initialise buffer
{
   // Destructor
   // Construct from supplied shape type and raw sizes
	Init();
   SetRawSizes(reqPnts, reqPntsCapacity, reqSegs, reqSegsCapacity, reqPols, reqPolsCapacity);
}


//______________________________________________________________________________
TBuffer3D::~TBuffer3D()
   // Initialise buffer
{
   // Destructor
   if (fPnts) delete [] fPnts;
   if (fSegs) delete [] fSegs;
   if (fPols) delete [] fPols;
//______________________________________________________________________________
}

//______________________________________________________________________________
void TBuffer3D::Init()
{
   // Initialise buffer
   fID            = 0;
   fColor         = 0;
   // Set fLocalMaster in section kCore to identity
   fTransparency  = 0;
   fLocalFrame	   = kFALSE;
   fReflection    = kFALSE;
   SetLocalMasterIdentity();

   // Reset bounding box
   for (UInt_t v=0; v<8; v++) {
      for (UInt_t i=0; i<3; i++) {
         fBBVertex[v][i] = 0.0;
      }
   }
   // Set fLocalMaster in section kCore to identity

   // Set kRaw tesselation section of buffer with supplied sizes
   fPnts          = 0;
   fSegs          = 0;
   fPols          = 0;

   fNbPnts        = 0;           
   fNbSegs        = 0;           
   fNbPols        = 0;        
   fPntsCapacity  = 0;  
   fSegsCapacity  = 0;  
   fPolsCapacity  = 0;  
   // Set fLocalMaster in section kCore to identity

   // Set kRaw tesselation section of buffer with supplied sizes
   ClearSectionsValid();
}

//______________________________________________________________________________
void TBuffer3D::ClearSectionsValid()
{
   // Clear any sections marked valid
   fSections = 0U; 
   SetRawSizes(0, 0, 0, 0, 0, 0);
}

//______________________________________________________________________________
void TBuffer3D::SetLocalMasterIdentity()
{
   // Set kRaw tesselation section of buffer with supplied sizes
   // Set fLocalMaster in section kCore to identity
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
void TBuffer3D::SetAABoundingBox(const Double_t origin[3], const Double_t halfLengths[3])
{
   // Set fBBVertex in kBoundingBox section to a axis aligned (local) BB
   // using supplied origin and box half lengths
   //
   //   7-------6
   //  /|      /|
   // 3-------2 |
   // | 4-----|-5
   // |/      |/
   // 0-------1 
   //

   // Vertex 0
   fBBVertex[0][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[0][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[0][2] = origin[2] - halfLengths[2];   // z
   // Vertex 1
   fBBVertex[1][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[1][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[1][2] = origin[2] - halfLengths[2];   // z
   // Vertex 2
   fBBVertex[2][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[2][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[2][2] = origin[2] - halfLengths[2];   // z
   // Vertex 3
   fBBVertex[3][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[3][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[3][2] = origin[2] - halfLengths[2];   // z
   // Vertex 4
   fBBVertex[4][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[4][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[4][2] = origin[2] + halfLengths[2];   // z
   // Vertex 5
   fBBVertex[5][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[5][1] = origin[1] - halfLengths[1];   // y
   fBBVertex[5][2] = origin[2] + halfLengths[2];   // z
   // Vertex 6
   fBBVertex[6][0] = origin[0] + halfLengths[0];   // x
   fBBVertex[6][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[6][2] = origin[2] + halfLengths[2];   // z
   // Vertex 7
   fBBVertex[7][0] = origin[0] - halfLengths[0];   // x
   fBBVertex[7][1] = origin[1] + halfLengths[1];   // y
   fBBVertex[7][2] = origin[2] + halfLengths[2];   // z
}

//______________________________________________________________________________
Bool_t TBuffer3D::SetRawSizes(UInt_t reqPnts, UInt_t reqPntsCapacity,
                              UInt_t reqSegs, UInt_t reqSegsCapacity, 
                              UInt_t reqPols, UInt_t reqPolsCapacity)
{
   // Set kRaw tesselation section of buffer with supplied sizes
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
   // Test if buffer represents a solid uncut sphere
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
UInt_t TBuffer3D::fgCSLevel = 0;

//______________________________________________________________________________
UInt_t TBuffer3D::GetCSLevel()
{
   return fgCSLevel;
}

//______________________________________________________________________________
void TBuffer3D::IncCSLevel()
{
   ++fgCSLevel;
}

//______________________________________________________________________________
UInt_t TBuffer3D::DecCSLevel()
{
   return --fgCSLevel;
}
