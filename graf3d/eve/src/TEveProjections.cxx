// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveProjections.h"
#include "TEveUtil.h"

//==============================================================================
//==============================================================================
// TEveProjection
//==============================================================================

//______________________________________________________________________________
//
// Base-class for non-linear projections.
//
// Enables to define an external center of distortion and a scale to
// fixate a bounding box of a projected point.

ClassImp(TEveProjection);

Float_t TEveProjection::fgEps = 0.005f;

//______________________________________________________________________________
TEveProjection::TEveProjection(TEveVector& center) :
   fType          (kPT_Unknown),
   fGeoMode       (kGM_Unknown),
   fName          (0),
   fCenter        (center.fX, center.fY, center.fZ),
   fDistortion    (0.0f),
   fFixR          (300), fFixZ          (400),
   fPastFixRFac   (0),   fPastFixZFac   (0),
   fScaleR        (1),   fScaleZ        (1),
   fPastFixRScale (1),   fPastFixZScale (1),
   fLowLimit(-std::numeric_limits<Float_t>::infinity(),
             -std::numeric_limits<Float_t>::infinity(),
             -std::numeric_limits<Float_t>::infinity()),
   fUpLimit ( std::numeric_limits<Float_t>::infinity(),
              std::numeric_limits<Float_t>::infinity(),
              std::numeric_limits<Float_t>::infinity())

{
   // Constructor.
}

//______________________________________________________________________________
void TEveProjection::ProjectVector(TEveVector& v)
{
   // Project TEveVector.

   ProjectPoint(v.fX, v.fY, v.fZ);
}

//______________________________________________________________________________
void TEveProjection::UpdateLimit()
{
   // Update convergence in +inf and -inf.

   if (fDistortion == 0.0f)
      return;

   Float_t lim = 1.0f/fDistortion + fFixR;
   Float_t *c  = GetProjectedCenter();
   fUpLimit .Set( lim + c[0],  lim + c[1], c[2]);
   fLowLimit.Set(-lim + c[0], -lim + c[1], c[2]);
}

//______________________________________________________________________________
void TEveProjection::SetDistortion(Float_t d)
{
   // Set distortion.

   fDistortion    = d;
   fScaleR        = 1.0f + fFixR*fDistortion;
   fScaleZ        = 1.0f + fFixZ*fDistortion;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveProjection::SetFixR(Float_t r)
{
   // Set fixed radius.

   fFixR          = r;
   fScaleR        = 1 + fFixR*fDistortion;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveProjection::SetFixZ(Float_t z)
{
   // Set fixed radius.

   fFixZ          = z;
   fScaleZ        = 1 + fFixZ*fDistortion;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveProjection::SetPastFixRFac(Float_t x)
{
   // Set 2's-exponent for relative scaling beyond FixR.

   fPastFixRFac   = x;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
}

//______________________________________________________________________________
void TEveProjection::SetPastFixZFac(Float_t x)
{
   // Set 2's-exponent for relative scaling beyond FixZ.

   fPastFixZFac   = x;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
}

//______________________________________________________________________________
void TEveProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   // Get vector for axis in a projected space.

   for (Int_t i=0; i<3; i++)
   {
      vec[i] = (i==screenAxis) ? 1.0f : 0.0f;
   }
}

//______________________________________________________________________________
Float_t TEveProjection::GetValForScreenPos(Int_t i, Float_t sv)
{
   // Inverse projection.

   static const TEveException eH("TEveProjection::GetValForScreenPos ");

   Float_t xL, xM, xR;
   TEveVector vec;
   TEveVector dirVec;
   SetDirectionalVector(i, dirVec);
   if (fDistortion > 0.0f && ((sv > 0 && sv > fUpLimit[i]) || (sv < 0 && sv < fLowLimit[i])))
      throw(eH + Form("screen value '%f' out of limit '%f'.", sv, sv > 0 ? fUpLimit[i] : fLowLimit[i]));

   TEveVector zero; ProjectVector(zero);
   // search from -/+ infinity according to sign of screen value
   if (sv > zero[i])
   {
      xL = 0; xR = 1000;
      while (1)
      {
         vec.Mult(dirVec, xR); ProjectVector(vec);
         // printf("positive projected %f, value %f,xL, xR ( %f, %f)\n", vec[i], sv, xL, xR);
         if (vec[i] > sv || vec[i] == sv) break;
         xL = xR; xR *= 2;
      }
   }
   else if (sv < zero[i])
   {
      xR = 0; xL = -1000;
      while (1)
      {
         vec.Mult(dirVec, xL); ProjectVector(vec);
         // printf("negative projected %f, value %f,xL, xR ( %f, %f)\n", vec[i], sv, xL, xR);
         if (vec[i] < sv || vec[i] == sv) break;
         xR = xL; xL *= 2;
      }
   }
   else
   {
      return 0.0f;
   }

   do
   {
      xM = 0.5f * (xL + xR);
      vec.Mult(dirVec, xM);
      ProjectVector(vec);
      // printf("safr xL=%f, xR=%f; vec[i]=%f, sv=%f\n", xL, xR, vec[i], sv);
      if (vec[i] > sv)
         xR = xM;
      else
         xL = xM;
   } while (TMath::Abs(vec[i] - sv) >= fgEps);

   return xM;
}

//______________________________________________________________________________
Float_t TEveProjection::GetScreenVal(Int_t i, Float_t x)
{
   // Project point on given axis and return projected value.

   TEveVector dv;
   SetDirectionalVector(i, dv); dv = dv*x;
   ProjectVector(dv);
   return dv[i];
}


//==============================================================================
//==============================================================================
// TEveRhoZProjection
//==============================================================================

//______________________________________________________________________________
//
// Transformation from 3D to 2D. X axis represent Z coordinate. Y axis have value of
// radius with a sign of Y coordinate.

ClassImp(TEveRhoZProjection);

//______________________________________________________________________________
TEveRhoZProjection::TEveRhoZProjection(TEveVector& center) :
   TEveProjection(center)
{
   // Constructor.

   fType = kPT_RhoZ;
   fName = "RhoZ";
}

//______________________________________________________________________________
void TEveRhoZProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                      EPProc_e proc )
{
   // Project point.

   using namespace TMath;

   if (proc == kPP_Plane || proc == kPP_Full)
   {
      // project
      y = Sign((Float_t)Sqrt(x*x+y*y), y);
      x = z;
   }
   if (proc == kPP_Distort || proc == kPP_Full)
   {
      // move to center
      x -= fProjectedCenter.fX;
      y -= fProjectedCenter.fY;

      // distort
      if (x > fFixZ)
         x =  fFixZ + fPastFixZScale*(x - fFixZ);
      else if (x < -fFixZ)
         x = -fFixZ + fPastFixZScale*(x + fFixZ);
      else
         x =  x * fScaleZ / (1.0f + Abs(x)*fDistortion);

      if (y > fFixR)
         y =  fFixR + fPastFixRScale*(y - fFixR);
      else if (y < -fFixR)
         y = -fFixR + fPastFixRScale*(y + fFixR);
      else
         y =  y * fScaleR / (1.0f + Abs(y)*fDistortion);

      // move back from center
      x += fProjectedCenter.fX;
      y += fProjectedCenter.fY;
   }
   z = 0.0f;
}

//______________________________________________________________________________
void TEveRhoZProjection::SetCenter(TEveVector& v)
{
   // Set center of distortion (virtual method).

   fCenter = v;

   Float_t r = TMath::Sqrt(v.fX*v.fX + v.fY*v.fY);
   fProjectedCenter.fX = fCenter.fZ;
   fProjectedCenter.fY = TMath::Sign(r, fCenter.fY);
   fProjectedCenter.fZ = 0;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveRhoZProjection::UpdateLimit()
{
   // Update convergence in +inf and -inf.

   if (fDistortion == 0.0f)
      return;

   Float_t limR = 1.0f/fDistortion + fFixR;
   Float_t limZ = 1.0f/fDistortion + fFixZ;
   Float_t *c   = GetProjectedCenter();
   fUpLimit .Set( limZ + c[0],  limR + c[1], c[2]);
   fLowLimit.Set(-limZ + c[0], -limR + c[1], c[2]);
}

//______________________________________________________________________________
void TEveRhoZProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   // Get direction in the unprojected space for axis index in the
   // projected space.
   // This is virtual method from base-class TEveProjection.

   if (screenAxis == 0)
      vec.Set(0.0f, 0.0f, 1.0f);
   else if (screenAxis == 1)
      vec.Set(0.0f, 1.0f, 0.0f);

}
//______________________________________________________________________________
Bool_t TEveRhoZProjection::AcceptSegment(TEveVector& v1, TEveVector& v2,
                                         Float_t tolerance)
{
   // Check if segment of two projected points is valid.

   Float_t a = fProjectedCenter.fY;
   Bool_t val = kTRUE;
   if ((v1.fY <  a && v2.fY > a) || (v1.fY > a && v2.fY < a))
   {
      val = kFALSE;
      if (tolerance > 0)
      {
         Float_t a1 = TMath::Abs(v1.fY - a), a2 = TMath::Abs(v2.fY - a);
         if (a1 < a2)
         {
            if (a1 < tolerance) { v1.fY = a; val = kTRUE; }
         }
         else
         {
            if (a2 < tolerance) { v2.fY = a; val = kTRUE; }
         }
      }
   }
   return val;
}


//==============================================================================
//==============================================================================
// TEveRPhiProjection
//==============================================================================

//______________________________________________________________________________
//
// XY projection with distortion around given center.

ClassImp(TEveRPhiProjection);

//______________________________________________________________________________
TEveRPhiProjection::TEveRPhiProjection(TEveVector& center) :
   TEveProjection(center)
{
   // Constructor.

   fType    = kPT_RPhi;
   fGeoMode = kGM_Polygons;
   fName    = "RhoPhi";
}

//______________________________________________________________________________
void TEveRPhiProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                      EPProc_e proc)
{
   // Project point.

   using namespace TMath;

   if (proc != kPP_Plane)
   {
      x -= fCenter.fX;
      y -= fCenter.fY;
      Float_t phi = (x == 0.0f && y == 0.0f) ? 0.0f : ATan2(y, x);
      Float_t r   = Sqrt(x*x + y*y);

      if (r > fFixR)
         r =  fFixR + fPastFixRScale*(r - fFixR);
      else if (r < -fFixR)
         r = -fFixR + fPastFixRScale*(r + fFixR);
      else
         r =  r * fScaleR / (1.0f + r*fDistortion);

      x = r*Cos(phi) + fCenter.fX;
      y = r*Sin(phi) + fCenter.fY;
   }
   z = 0.0f;
}
