// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   24/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPointsArray3D                                                       //
//                                                                      //
// A 3-D PolyLine.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#ifndef ROOT_TPointsArray3D
#define ROOT_TPointsArray3D


#include "TPoints3DABC.h"

#ifndef ROOT_TString
#include "TString.h"
#endif


class TPointsArray3D : public TPoints3DABC {

protected:
   Int_t        fN;            // Number of points
   Float_t     *fP;            // Array of 3-D coordinates  (x,y,z)
   TString      fOption;       // options
   UInt_t       fGLList;       // The list number for OpenGL view
   Int_t        fLastPoint;    // The index of the last filled point

public:
   TPointsArray3D();
   TPointsArray3D(Int_t n, Option_t *option="");
   TPointsArray3D(Int_t n, Float_t *p, Option_t *option="");
   TPointsArray3D(Int_t n, Float_t *x, Float_t *y, Float_t *z, Option_t *option="");
   TPointsArray3D(const TPointsArray3D &points);
   virtual ~TPointsArray3D();

   virtual void      Copy(TObject &points) const;
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py);
   virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual Int_t     GetLastPosition() const;
   virtual Int_t     GetN() const;
   virtual Float_t  *GetP() const;
   virtual Float_t   GetX(Int_t idx)  const;
   virtual Float_t   GetY(Int_t idx)  const;
   virtual Float_t   GetZ(Int_t idx)  const;
   virtual Float_t  *GetXYZ(Float_t *xyz,Int_t idx,Int_t num=1)  const;
   virtual const Float_t  *GetXYZ(Int_t idx);
   virtual Option_t *GetOption() const ;
   virtual Bool_t    Is3D() const;
   virtual void      ls(Option_t *option="") const;
   virtual void      PaintPoints(Int_t , Float_t *,Option_t *){;}
   virtual void      Print(Option_t *option="") const;
   virtual Int_t     SetLastPosition(Int_t idx);
   virtual void      SetOption(Option_t *option="");
   virtual Int_t     SetPoint(Int_t point, Float_t x, Float_t y, Float_t z); // *MENU*
   virtual Int_t     SetPoints(Int_t n, Float_t *p=0, Option_t *option="");
   virtual Int_t     Size() const;

   ClassDef(TPointsArray3D,1)  //A 3-D PolyLine
};


inline Int_t     TPointsArray3D::GetLastPosition()  const                   {return fLastPoint;}
inline Int_t     TPointsArray3D::GetN()  const                              {return fN;}
inline Float_t  *TPointsArray3D::GetP()  const                              {return fP;}
inline Float_t   TPointsArray3D::GetX(Int_t idx)  const                     {return fP[3*idx+0];}
inline Float_t   TPointsArray3D::GetY(Int_t idx)  const                     {return fP[3*idx+1];}
inline Float_t   TPointsArray3D::GetZ(Int_t idx)  const                     {return fP[3*idx+2];}
inline const Float_t  *TPointsArray3D::GetXYZ(Int_t idx)                    {return  &fP[3*idx+0];}
inline Float_t  *TPointsArray3D::GetXYZ(Float_t *xyz,Int_t idx,Int_t num)  const
                          {return (Float_t  *)memcpy(xyz,&fP[3*idx],3*num*sizeof(Float_t));}
inline Option_t *TPointsArray3D::GetOption() const                          {return fOption.Data();}
inline Bool_t    TPointsArray3D::Is3D() const                               {return kTRUE;}
inline void      TPointsArray3D::SetOption(Option_t *option)                {fOption = option;}

inline Int_t     TPointsArray3D::Size() const                               {return fLastPoint+1;}

#endif
