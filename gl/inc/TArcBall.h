// @(#)root/gl:$Name:  $:$Id:$
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** KempoApi: The Turloc Toolkit *****************************/
/** *    *                                                  **/
/** **  **  Filename: TArcBall.h                            **/
/**   **    Version:  Common                                **/
/**   **                                                    **/
/**                                                         **/
/**  TArcball class for mouse manipulation.                 **/
/**                                                         **/
/**                                                         **/
/**                                                         **/
/**                                                         **/
/**                              (C) 1999-2003 Tatewake.com **/
/**   History:                                              **/
/**   08/17/2003 - (TJG) - Creation                         **/
/**   09/23/2003 - (TJG) - Bug fix and optimization         **/
/**   09/25/2003 - (TJG) - Version for NeHe Basecode users  **/
/**                                                         **/
/*************************************************************/

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
   void MapToSphere(const class TPoint & NewPt, Double_t * NewVec)const;
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
   Double_t *GetRotMatrix();
};

#endif

