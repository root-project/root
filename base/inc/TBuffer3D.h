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

class TBuffer3D : public TObject
{
private:
   void Init();   

   // Non-copyable
   TBuffer3D(const TBuffer3D &);
   const TBuffer3D & operator=(const TBuffer3D &);

   const Int_t fType;        // Primitive type - predefined ones in TBuffer3DTypes.h
                                
   UInt_t    fNbPnts;        // Number of points describing the shape
   UInt_t    fNbSegs;        // Number of segments describing the shape
   UInt_t    fNbPols;        // Number of polygons describing the shape
   
   UInt_t    fPntsCapacity;  // Current capacity of fPnts space
   UInt_t    fSegsCapacity;  // Current capacity of fSegs space
   UInt_t    fPolsCapacity;  // Current capacity of fSegs space
      
   UInt_t    fSections;      // Section validity flags

public:

   enum ESection { kNone            = BIT(0),
                   kCore            = BIT(1),
                   kBoundingBox     = BIT(2),
                   kShapeSpecific   = BIT(3),
                   kRawSizes        = BIT(4), 
                   kRaw             = BIT(5),
                   kAll             = kCore|kBoundingBox|kShapeSpecific|kRawSizes|kRaw 
   };
   
   TBuffer3D(Int_t type);
   TBuffer3D(Int_t type, 
             UInt_t reqPnts, UInt_t reqPntsCapacity,
             UInt_t reqSegs, UInt_t reqSegsCapacity, 
             UInt_t reqPols, UInt_t reqPolsCapacity); 
   virtual  ~TBuffer3D();

   // Section validity flags
   void   SetSectionsValid(UInt_t mask)     { fSections |= mask & kAll; }
   void   ClearSectionsValid();
   Bool_t SectionsValid(UInt_t mask) const   { return (Bool_t) (GetSections(mask) == mask); }
   UInt_t GetSections(UInt_t mask)   const   { return (UInt_t) (fSections & mask); }
      
   // SECTION: kRawSize get/set
   Bool_t SetRawSizes(UInt_t reqPnts, UInt_t reqPntsCapacity,
                      UInt_t reqSegs, UInt_t reqSegsCapacity, 
                      UInt_t reqPols, UInt_t reqPolsCapacity);
   
   UInt_t NbPnts() const { return fNbPnts; }   
   UInt_t NbSegs() const { return fNbSegs; }      
   UInt_t NbPols() const { return fNbPols; }  
   
   // SECTION: kCore
   Int_t  Type() const { return fType; }
   void   SetLocalMasterIdentity();

   TObject    *fID;              // ID/object generating buffer - see TVirtualViewer3D for setting
   Int_t       fColor;           // Color index
   Short_t     fTransparency;    // Percentage transparency [0,100]
   Bool_t      fLocalFrame;      // True = Local, False = Master reference frame
   Bool_t      fReflection;      // Matrix is reflection - TODO: REMOVE when OGL viewer rewokred to local frame
   Double_t    fLocalMaster[16]; // Local->Master Matrix - identity if master frame
   
   // SECTION: kBoundingBox
   Double_t    fBBLowVertex[3];  // Low x,y,z vertex
   Double_t    fBBHighVertex[3]; // High x,y,z vertex
   
   // SECTION: kShapeSpecific - none for base class
      
   // SECTION: kRaw
   Double_t *fPnts;              // x0, y0, z0, x1, y1, z1, ..... ..... ....
   Int_t    *fSegs;              // c0, p0, q0, c1, p1, q1, ..... ..... ....  
   Int_t    *fPols;              // c0, n0, s0, s1, ... sn, c1, n1, s0, ... sn

   ClassDef(TBuffer3D,0)     // 3D primitives description
};

class TBuffer3DSphere : public TBuffer3D
{
private:
   void Init();

   // Non-copyable class
   TBuffer3DSphere(const TBuffer3DSphere &);
   const TBuffer3DSphere & operator=(const TBuffer3DSphere &);
   
public:
   TBuffer3DSphere();
   TBuffer3DSphere(UInt_t reqPnts, UInt_t reqPntsCapacity,
                   UInt_t reqSegs, UInt_t reqSegsCapacity, 
                   UInt_t reqPols, UInt_t reqPolsCapacity);

   Bool_t IsSolidUncut() const;

   // SECTION: kShapeSpecific
   Double_t fRadiusInner;  
   Double_t fRadiusOuter;  
   Double_t fThetaMin;     // Lower theta limit (orientation?)
   Double_t fThetaMax;     // Higher theta limit (orientation?)
   Double_t fPhiMin;       // Lower phi limit (orientation?)
   Double_t fPhiMax;       // Higher phi limit (orientation?)
};

class TBuffer3DTube : public TBuffer3D
{
private:
   void Init();

   // Non-copyable class
   TBuffer3DTube(const TBuffer3DTube &);
   const TBuffer3DTube & operator=(const TBuffer3DTube &);
   
protected:
   TBuffer3DTube(Int_t type);
   TBuffer3DTube(Int_t type,
                 UInt_t reqPnts, UInt_t reqPntsCapacity,
                 UInt_t reqSegs, UInt_t reqSegsCapacity, 
                 UInt_t reqPols, UInt_t reqPolsCapacity);
public:
   TBuffer3DTube();
   TBuffer3DTube(UInt_t reqPnts, UInt_t reqPntsCapacity,
                 UInt_t reqSegs, UInt_t reqSegsCapacity, 
                 UInt_t reqPols, UInt_t reqPolsCapacity);

   // SECTION: kShapeSpecific
   Double_t fRadiusInner;  // Inner radius
   Double_t fRadiusOuter;  // Outer radius
   Double_t fHalfLength;   // Half length (dz)           
};

class TBuffer3DTubeSeg : public TBuffer3DTube
{
private:
   void Init();

   // Non-copyable class
   TBuffer3DTubeSeg(const TBuffer3DTubeSeg &);
   const TBuffer3DTubeSeg & operator=(const TBuffer3DTubeSeg &);
   
public:
   TBuffer3DTubeSeg();
   TBuffer3DTubeSeg(UInt_t reqPnts, UInt_t reqPntsCapacity,
                    UInt_t reqSegs, UInt_t reqSegsCapacity, 
                    UInt_t reqPols, UInt_t reqPolsCapacity);

   // SECTION: kShapeSpecific
   Double_t fPhiMin;       // Lower phi limit
   Double_t fPhiMax;       // Higher phi limit
};
#endif
