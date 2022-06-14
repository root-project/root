// @(#)root/gl:$Id$
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

#include "Rtypes.h"

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
   TArcBall(UInt_t NewWidth = 100, UInt_t NewHeight = 100);
   virtual ~TArcBall() { }

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

   ClassDef(TArcBall,0) //ArcBall manipulator
};

#endif

