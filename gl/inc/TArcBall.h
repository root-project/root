// @(#)root/gl:$Name:  $:$Id: TArcBall.h,v 1.5 2004/09/14 15:37:34 rdm Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TArcBall
#define ROOT_TArcBall

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TPoint;

class TArcBall {
private:
   Double_t fThisRot[9];
   Double_t fLastRot[9];
   Double_t fTransform[16];
   Double_t fStVec[3];          //Saved click vector
   Double_t fEnVec[3];          //Saved drag vector
   Double_t fAdjustWidth;      //Mouse bounds width
   Double_t fAdjustHeight;     //Mouse bounds height
   //Non-copyable
   TArcBall(const TArcBall &);
   TArcBall & operator = (const TArcBall &);
   void ResetMatrices();
protected:
   void MapToSphere(const TPoint &NewPt, Double_t *NewVec)const;
public:
   TArcBall(UInt_t NewWidth, UInt_t NewHeight);

   void SetBounds(UInt_t NewWidth, UInt_t NewHeight)
   {
      fAdjustWidth  = 1.0f / ((NewWidth  - 1.) * 0.5);
      fAdjustHeight = 1.0f / ((NewHeight - 1.) * 0.5);
   }
   //Mouse down
   void Click(const TPoint &NewPt);
   //Mouse drag, calculate rotation
   void Drag(const TPoint &NewPt);
   const Double_t *GetRotMatrix()const
   {
      return fTransform;
   }
};

class TEqRow {
private:
   Double_t fData[4];
public:
   TEqRow();
   TEqRow(const Double_t *source);

   void SetRow(const Double_t *source);

   Double_t &operator [] (UInt_t ind)
   {
      return fData[ind];
   }
   Double_t operator [] (UInt_t ind)const
   {
      return fData[ind];
   }

   TEqRow &operator *= (Double_t x);
   TEqRow &operator /= (Double_t x);
   TEqRow &operator += (const TEqRow &row);
};

TEqRow operator * (const TEqRow &row, Double_t x);
TEqRow operator * (Double_t x, const TEqRow &row);
TEqRow operator / (const TEqRow &row, Double_t x);
TEqRow operator + (const TEqRow &row1, const TEqRow &row2);

class TToySolver {
private:
   TEqRow fMatrix[3];
   Int_t fBase[3];   
public:
   TToySolver(const Double_t *source);
   void GetSolution(Double_t *sink);
private:   
   void AddNewBV(UInt_t i, UInt_t j);
};

#endif

