// @(#)root/base:$Name:  $:$Id: TBuffer3D.h,v 1.00 
// Author: Olivier Couet   05/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBuffer3D
#define ROOT_TBuffer3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBuffer3D                                                            //
//                                                                      //
// 3D primitives description                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TBuffer3D : public TObject {

private:

public:
   enum EBuffer3DType {kBRIK,   kPGON, kPCON, kSPHE,   kTUBE,   kTUBS,
                       kTORUS,  kXTRU, kLINE, kCSHAPE, kPARA,
                       kM3DBOX, kMARKER };

   enum EBuffer3DOption {kPAD, kRANGE, kSIZE, kX3D, kOGL};

   TBuffer3D();
   TBuffer3D(Int_t n1, Int_t n2, Int_t n3);
   virtual  ~TBuffer3D();

   void ReAllocate(Int_t n1, Int_t n2, Int_t n3);
   void Paint(Option_t *option);

   TObject  *fId;       // Pointer to he original object
   Int_t     fOption;   // Option (see EBuffer3DOption)
   Int_t     fType;     // Primitive type (see EBuffer3DType)
   Int_t     fNbPnts;   // Number of points describing the shape
   Int_t     fNbSegs;   // Number of segments describing the shape
   Int_t     fNbPols;   // Number of polygons describing the shape
   Int_t    *fSegs;     // c0, p0, q0, c1, p1, q1, ..... ..... ....  
   Int_t    *fPols;     // c0, n0, s0, s1, ... sn, c1, n1, s0, ... sn
   Int_t     fPntsSize; // Current size of fPnts
   Int_t     fSegsSize; // Current size of fSegs
   Int_t     fPolsSize; // Current size of fSegs
   Double_t *fPnts;     // x0, y0, z0, x1, y1, z1, ..... ..... ....

   ClassDef(TBuffer3D,0) // 3D primitives description
};

#endif
