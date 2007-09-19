// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   24/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoints3D
#define ROOT_TPoints3D

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPoints3D                                                            //
//                                                                      //
// A 3-D PolyLine.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPoints3DABC
#include "TPoints3DABC.h"
#endif


class TPoints3D : public TPoints3DABC {

protected:
   enum EOwnerBits {
       kIsOwner         = BIT(23)
   };

   TPoints3DABC *fPoints;

   Bool_t IsOwner() const {return TestBit(kIsOwner);}
   Bool_t DoOwner(Bool_t done=kTRUE);

public:
   TPoints3D(TPoints3DABC *points=0);
   TPoints3D(Int_t n, Option_t *option="");
   TPoints3D(Int_t n, Float_t *p, Option_t *option="");
   TPoints3D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option="");
   TPoints3D(const TPoints3D &points);
   virtual ~TPoints3D();

   virtual void      Copy(TObject &points) const;
   virtual void      Delete(Option_t *);
   virtual void      Delete();
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py);
   virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Int_t     GetLastPosition()       const;
   virtual Int_t     GetN() const;
   virtual Float_t  *GetP() const;
   virtual Float_t   GetX(Int_t idx)  const;
   virtual Float_t   GetY(Int_t idx)  const;
   virtual Float_t   GetZ(Int_t idx)  const;
   virtual Float_t  *GetXYZ(Float_t *xyz,Int_t idx,Int_t num=1)  const;
   virtual const Float_t  *GetXYZ(Int_t idx);
   virtual Option_t *GetOption() const ;
   virtual void      ls(Option_t *option="") const;
   virtual void      PaintPoints(Int_t, Float_t *,Option_t *){;}
   virtual void      Print(Option_t *option="") const;
   virtual Int_t     SetLastPosition(Int_t idx);
   virtual void      SetOption(Option_t *option="");
   virtual Int_t     SetPoint(Int_t point, Float_t x, Float_t y, Float_t z); // *MENU*
   virtual Int_t     SetPoints(Int_t n, Float_t *p=0, Option_t *option="");
   virtual Int_t     Size() const;

   ClassDef(TPoints3D,1)  // Defines the abstract array of 3D points
};

inline void      TPoints3D::Delete(Option_t *opt){ TObject::Delete(opt);}
inline Int_t     TPoints3D::DistancetoPrimitive(Int_t px, Int_t py) {return fPoints?fPoints->DistancetoPrimitive(px,py):99999;}
inline Int_t     TPoints3D::GetLastPosition()  const   {return fPoints?fPoints->GetLastPosition():0;}
inline Int_t     TPoints3D::GetN()  const              {return fPoints?fPoints->GetN():0;}
inline Float_t  *TPoints3D::GetP()  const              {return fPoints?fPoints->GetP():0;}
inline Float_t   TPoints3D::GetX(Int_t idx)  const     {return fPoints?fPoints->GetX(idx):0;}
inline Float_t   TPoints3D::GetY(Int_t idx)  const     {return fPoints?fPoints->GetY(idx):0;}
inline Float_t   TPoints3D::GetZ(Int_t idx)  const     {return fPoints?fPoints->GetZ(idx):0;}
inline const Float_t  *TPoints3D::GetXYZ(Int_t idx)    {return fPoints?fPoints->GetXYZ(idx):0;}
inline Float_t  *TPoints3D::GetXYZ(Float_t *xyz,Int_t idx,Int_t num)  const
                          {return fPoints?fPoints->GetXYZ(xyz,idx,num):0;}
inline Option_t *TPoints3D::GetOption() const          {return fPoints?fPoints->GetOption():"";}
inline Int_t     TPoints3D::SetLastPosition(Int_t idx) {return fPoints?fPoints->SetLastPosition(idx):0;}
inline void      TPoints3D::SetOption(Option_t *option){if (fPoints) fPoints->SetOption(option);}
inline Int_t     TPoints3D::SetPoint(Int_t point, Float_t x, Float_t y, Float_t z){return fPoints?fPoints->SetPoint(point,x,y,z):0;}
inline Int_t     TPoints3D::SetPoints(Int_t n, Float_t *p, Option_t *option){return fPoints?fPoints->SetPoints(n,p,option):0;}

inline Int_t     TPoints3D::Size() const               {return fPoints?fPoints->Size():0;}

#endif


