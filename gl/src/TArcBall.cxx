// @(#)root/gl:$Name:  $:$Id: TArcBall.cxx,v 1.4 2004/09/03 12:52:42 brun Exp $
// Author:  Timur Pocheptsov  03/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TArcBall.h"
#include "TPoint.h"
#include "TMath.h"

const Double_t Epsilon = 1.0e-5;

//Arcball sphere constants:
//Diameter is       2.0f
//Radius is         1.0f

//______________________________________________________________________________
inline void Vector3dCross(Double_t *NewObj, const Double_t * v1, const Double_t *v2)
{
   NewObj[0] = v1[1] * v2[2] - v1[2] * v2[1];
   NewObj[1] = v1[2] * v2[0] - v1[0] * v2[2];
   NewObj[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

//______________________________________________________________________________
inline Double_t Vector3dDot(const Double_t *NewObj, const Double_t *v1)
{
   return  NewObj[0] * v1[0] + NewObj[1] * v1[1] + NewObj[2] * v1[2];
}

//______________________________________________________________________________
inline Double_t Vector3dLengthSquared(const Double_t *NewObj)
{
   return  NewObj[0] * NewObj[0] + NewObj[1] * NewObj[1] + NewObj[2] * NewObj[2];
}

//______________________________________________________________________________
inline Double_t Vector3dLength(const Double_t *NewObj)
{
   return TMath::Sqrt(Vector3dLengthSquared(NewObj));
}

//______________________________________________________________________________
inline void Matrix3dSetZero(Double_t * NewObj)
{
   for (Int_t i = 0; i < 9; ++i)
      NewObj[i] = 0.;
}

//______________________________________________________________________________
inline void Matrix3dSetIdentity(Double_t *NewObj)
{
   Matrix3dSetZero(NewObj);
   //then set diagonal as 1
   NewObj[0] = NewObj[4] = NewObj[8] = 1.;
}

//______________________________________________________________________________
void Matrix3dSetRotationFromQuat4d(Double_t *NewObj, const Double_t *q1)
{
   Double_t n = (q1[0] * q1[0]) + (q1[1] * q1[1]) + (q1[2] * q1[2]) + (q1[3] * q1[3]);
   Double_t s = (n > 0.0f) ? (2.0f / n) : 0.0f;
   Double_t xs = q1[0] * s,  ys = q1[1] * s,  zs = q1[2] * s;
   Double_t wx = q1[3] * xs, wy = q1[3] * ys, wz = q1[3] * zs;
   Double_t xx = q1[0] * xs, xy = q1[0] * ys, xz = q1[0] * zs;
   Double_t yy = q1[1] * ys, yz = q1[1] * zs, zz = q1[2] * zs;

   NewObj[0] = 1.0f - (yy + zz); NewObj[3] = xy - wz;          NewObj[6] = xz + wy;
   NewObj[1] = xy + wz;          NewObj[4] = 1.0f - (xx + zz); NewObj[7] = yz - wx;
   NewObj[2] = xz - wy;          NewObj[5] = yz + wx;          NewObj[8] = 1.0f - (xx + yy);
}

//______________________________________________________________________________
void Matrix3dMulMatrix3d(Double_t *NewObj, const Double_t *m1)
{
   Double_t Result[9];

   Result[0] = (NewObj[0] * m1[0]) + (NewObj[3] * m1[1]) + (NewObj[6] * m1[2]);
   Result[3] = (NewObj[0] * m1[3]) + (NewObj[3] * m1[4]) + (NewObj[6] * m1[5]);
   Result[6] = (NewObj[0] * m1[6]) + (NewObj[3] * m1[7]) + (NewObj[6] * m1[8]);

   Result[1] = (NewObj[1] * m1[0]) + (NewObj[4] * m1[1]) + (NewObj[7] * m1[2]);
   Result[4] = (NewObj[1] * m1[3]) + (NewObj[4] * m1[4]) + (NewObj[7] * m1[5]);
   Result[7] = (NewObj[1] * m1[6]) + (NewObj[4] * m1[7]) + (NewObj[7] * m1[8]);

   Result[2] = (NewObj[2] * m1[0]) + (NewObj[5] * m1[1]) + (NewObj[8] * m1[2]);
   Result[5] = (NewObj[2] * m1[3]) + (NewObj[5] * m1[4]) + (NewObj[8] * m1[5]);
   Result[8] = (NewObj[2] * m1[6]) + (NewObj[5] * m1[7]) + (NewObj[8] * m1[8]);

   for (Int_t i = 0; i < 9; ++i)
      NewObj[i] = Result[i];
}

//______________________________________________________________________________
inline void Matrix4dSetRotationScaleFromMatrix4d(Double_t *NewObj, const Double_t *m1)
{
   NewObj[0] = m1[0]; NewObj[4] = m1[4]; NewObj[8] = m1[8];
   NewObj[1] = m1[1]; NewObj[5] = m1[5]; NewObj[9] = m1[9];
   NewObj[2] = m1[2]; NewObj[6] = m1[6]; NewObj[10] = m1[10];
}

//______________________________________________________________________________
inline Double_t Matrix4fSVD(const Double_t *NewObj, Double_t *rot3, Double_t *rot4)
{
   Double_t s = TMath::Sqrt(
                ( (NewObj[0] * NewObj[0]) + (NewObj[1] * NewObj[1]) + (NewObj[2] * NewObj[2]) +
                  (NewObj[4] * NewObj[4]) + (NewObj[5] * NewObj[5]) + (NewObj[6] * NewObj[6]) +
                  (NewObj[8] * NewObj[8]) + (NewObj[9] * NewObj[9]) + (NewObj[10] * NewObj[10]) ) / 3.0f );

   if (rot3) {
      rot3[0] = NewObj[0]; rot3[1] = NewObj[1]; rot3[2] = NewObj[2];
      rot3[3] = NewObj[4]; rot3[4] = NewObj[5]; rot3[5] = NewObj[6];
      rot3[6] = NewObj[8]; rot3[7] = NewObj[9]; rot3[8] = NewObj[10];

      Double_t n = 1. / TMath::Sqrt(NewObj[0] * NewObj[0] + NewObj[1] * NewObj[1] + NewObj[2] * NewObj[2] + 0.0001);

      rot3[0] *= n;
      rot3[1] *= n;
      rot3[2] *= n;

      n = 1. / TMath::Sqrt(NewObj[4] * NewObj[4] + NewObj[5] * NewObj[5] + NewObj[6] * NewObj[6] + 0.0001);
      rot3[3] *= n;
      rot3[4] *= n;
      rot3[5] *= n;

      n = 1.0f / TMath::Sqrt(NewObj[8] * NewObj[8] + NewObj[9] * NewObj[9] + NewObj[10] * NewObj[10] + 0.0001);
      rot3[6] *= n;
      rot3[7] *= n;
      rot3[8] *= n;
   }

   if (rot4) {
      if (rot4 != NewObj)
         Matrix4dSetRotationScaleFromMatrix4d(rot4, NewObj);

      Double_t n = 1. / TMath::Sqrt(NewObj[0] * NewObj[0] + NewObj[1] * NewObj[1] + NewObj[2] * NewObj[2] + 0.0001);

      rot4[0] *= n;
      rot4[1] *= n;
      rot4[2] *= n;

      n = 1. / TMath::Sqrt(NewObj[4] * NewObj[4] + NewObj[5] * NewObj[5] + NewObj[6] * NewObj[6] + 0.0001);
      rot4[4] *= n;
      rot4[5] *= n;
      rot4[6] *= n;

      n = 1. / TMath::Sqrt(NewObj[8] * NewObj[8] + NewObj[9] * NewObj[9] + NewObj[10] * NewObj[10] + 0.0001);
      rot4[8] *= n;
      rot4[9] *= n;
      rot4[10] *= n;
   }

   return s;
}

//______________________________________________________________________________
inline void Matrix4dSetRotationScaleFromMatrix3d(Double_t *NewObj, const Double_t *m1)
{
   NewObj[0] = m1[0]; NewObj[4] = m1[3]; NewObj[8] = m1[6];
   NewObj[1] = m1[1]; NewObj[5] = m1[4]; NewObj[9] = m1[7];
   NewObj[2] = m1[2]; NewObj[6] = m1[5]; NewObj[10] = m1[8];
}

//______________________________________________________________________________
inline void Matrix4dMulRotationScale(Double_t *NewObj, Double_t scale)
{
   NewObj[0] *= scale; NewObj[4] *= scale; NewObj[8] *= scale;
   NewObj[1] *= scale; NewObj[5] *= scale; NewObj[9] *= scale;
   NewObj[2] *= scale; NewObj[6] *= scale; NewObj[10] *= scale;
}

//______________________________________________________________________________
void Matrix4dSetRotationFromMatrix3d(Double_t *NewObj, const Double_t *m1)
{
   Double_t scale = Matrix4fSVD(NewObj, 0, 0);
   Matrix4dSetRotationScaleFromMatrix3d(NewObj, m1);
   Matrix4dMulRotationScale(NewObj, scale);
}

//______________________________________________________________________________
inline void TArcBall::MapToSphere(const TPoint &NewPt, Double_t *NewVec) const
{
   Double_t TempPt[] = {NewPt.fX, NewPt.fY};
   //Adjust point coords and scale down to range of [-1 ... 1]
   TempPt[0]  = TempPt[0] * fAdjustWidth  - 1.;
   TempPt[1]  = 1. - TempPt[1] * fAdjustHeight;
   //Compute the square of the length of the vector to the point from the center
   Double_t length = TempPt[0] * TempPt[0] + TempPt[1] * TempPt[1];
   //If the point is mapped outside of the sphere... (length > radius squared)
   if (length > 1.) {
      Double_t norm = 1.0f / TMath::Sqrt(length);
      //Return the "normalized" vector, a point on the sphere
      NewVec[0] = TempPt[0] * norm;
      NewVec[1] = TempPt[1] * norm;
      NewVec[2] = 0.;
   } else {   //Else it's on the inside
    //Return a vector to a point mapped inside the sphere sqrt(radius squared - length)
      NewVec[0] = TempPt[0];
      NewVec[1] = TempPt[1];
      NewVec[2] = TMath::Sqrt(1. - length);
   }
}

//______________________________________________________________________________
TArcBall::TArcBall(UInt_t Width, UInt_t Height)
            :fThisRot(), fLastRot(),
	          fTransform(), fStVec(),
	          fEnVec(), fAdjustWidth(0.),
	          fAdjustHeight(0.)
{
   SetBounds(Width, Height);
   ResetMatrices();
}

//______________________________________________________________________________
void TArcBall::Click(const TPoint &NewPt)
{
   //Mouse down
   MapToSphere(NewPt, fStVec);

   for (Int_t i = 0; i < 9; ++i)
      fLastRot[i] = fThisRot[i];
}

//______________________________________________________________________________
void TArcBall::Drag(const TPoint &NewPt)
{
   //Mouse drag, calculate rotation
   MapToSphere(NewPt, fEnVec);
   //Return the quaternion equivalent to the rotation
   Double_t NewRot[4] = {0.};
   Double_t Perp[3] = {0.};

   Vector3dCross(Perp, fStVec, fEnVec);
   //Compute the length of the perpendicular vector
   if (Vector3dLength(Perp) > Epsilon) {
   //We're ok, so return the perpendicular vector as the transform after all
      NewRot[0] = Perp[0];
      NewRot[1] = Perp[1];
      NewRot[2] = Perp[2];
      //In the quaternion values, w is cosine (theta / 2), where theta is rotation angle
      NewRot[3]= Vector3dDot(fStVec, fEnVec);
   } else  //if it's zero
      NewRot[0] = NewRot[1] = NewRot[2] = NewRot[3] = 0.;

   Matrix3dSetRotationFromQuat4d(fThisRot, NewRot);
   Matrix3dMulMatrix3d(fThisRot, fLastRot);
   Matrix4dSetRotationFromMatrix3d(fTransform, fThisRot);
}

//______________________________________________________________________________
void TArcBall::ResetMatrices()
{
   fTransform[0] = 1.f, fTransform[1] = fTransform[2] = fTransform[3] =
   fTransform[4] = 0.f, fTransform[5] = 1.f, fTransform[6] = fTransform[7] =
   fTransform[8] = fTransform[9] = 0.f, fTransform[10] = 1.f, fTransform[11] =
   fTransform[12] = fTransform[13] = fTransform[14] = 0.f, fTransform[15] = 1.f;
   Matrix3dSetIdentity(fLastRot);
   Matrix3dSetIdentity(fThisRot);
}

//______________________________________________________________________________
TEqRow::TEqRow()
           :fData()
{
}

//______________________________________________________________________________
TEqRow::TEqRow(const Double_t *source)
{
   fData[0] = source[0];
   fData[1] = source[1];
   fData[2] = source[2];
   fData[3] = source[3];
}

//______________________________________________________________________________
void TEqRow::SetRow(const Double_t *source)
{
   fData[0] = source[0];
   fData[1] = source[1];
   fData[2] = source[2];
   fData[3] = source[3];
}

//______________________________________________________________________________
TEqRow &TEqRow::operator *= (Double_t x)
{
   fData[0] *= x;
   fData[1] *= x;
   fData[2] *= x;
   fData[3] *= x;

   return *this;
}

//______________________________________________________________________________
TEqRow &TEqRow::operator /= (Double_t x)
{
   fData[0] /= x;
   fData[1] /= x;
   fData[2] /= x;
   fData[3] /= x;

   return *this;
}

//______________________________________________________________________________
TEqRow &TEqRow::operator += (const TEqRow &row)
{
   fData[0] += row.fData[0];
   fData[1] += row.fData[1];
   fData[2] += row.fData[2];
   fData[3] += row.fData[3];

   return *this;
}

//______________________________________________________________________________
TEqRow operator * (const TEqRow &row, Double_t x)
{
   return TEqRow(row) *= x;
}

//______________________________________________________________________________
TEqRow operator * (Double_t x, const TEqRow &row)
{
   return TEqRow(row) *= x;
}

//______________________________________________________________________________
TEqRow operator / (const TEqRow &row, Double_t x)
{
   return TEqRow(row) /= x;
}

//______________________________________________________________________________
TEqRow operator + (const TEqRow &r1, const TEqRow &r2)
{
   return TEqRow(r1)+=r2;
}

//______________________________________________________________________________
TToySolver::TToySolver(const Double_t *source)
{
   fMatrix[0].SetRow(source);
   fMatrix[1].SetRow(source + 4);
   fMatrix[2].SetRow(source + 8);
   fBase[0] = fBase[1] = fBase[2] = -1;
}

//______________________________________________________________________________
void TToySolver::GetSolution(Double_t *sink)
{
   for (UInt_t i = 0; i < 3; ++i) {
      for (UInt_t j = 0; j < 3; ++j) {
         if (fMatrix[i][j] != 0.) AddNewBV(i, j);
      }   
   }
   
   //j for vc 6.0 :))
   for (UInt_t j = 0; j < 3; ++j) {
      if (fBase[j] >= 0)sink[fBase[j]] = fMatrix[j][3];
   }
}

//______________________________________________________________________________
void TToySolver::AddNewBV(UInt_t row, UInt_t col)
{
   Double_t tmp1 = fMatrix[row][col];
   fMatrix[row] /= tmp1;
   
   for (UInt_t i = 0; i < 3; ++i) {
      if (i != row) {
         Double_t tmp2 = fMatrix[i][col];
         fMatrix[i] += -tmp2 * fMatrix[row];
      }
   }
   
   fBase[row] = col;
}
