// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveProjections.h>
#include <TEveUtil.h>

//______________________________________________________________________________
// TEveProjection
//
// Base-class for non-linear projections.
//
// Enables to define an external center of distortion and a scale to
// fixate a bounding box of a projected point.

ClassImp(TEveProjection)

Float_t TEveProjection::fgEps = 0.005f;

//______________________________________________________________________________
TEveProjection::TEveProjection(TEveVector& center) :
   fType(PT_Unknown),
   fGeoMode(GM_Unknown),
   fName(0),
   fCenter(center.x, center.y, center.z),
   fDistortion(0.0f),
   fFixedRadius(300),
   fScale(1.0f)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveProjection::ProjectVector(TEveVector& v)
{
   // Project TEveVector.

   ProjectPoint(v.x, v.y, v.z);
}

//______________________________________________________________________________
void TEveProjection::UpdateLimit()
{
   // Update convergence in +inf and -inf.

   if ( fDistortion == 0.0f )
      return;

   Float_t lim =  1.0f/fDistortion + fFixedRadius;
   Float_t* c = GetProjectedCenter();
   fUpLimit.Set(lim + c[0], lim + c[1], c[2]);
   fLowLimit.Set(-lim + c[0], -lim + c[1], c[2]);
}

//______________________________________________________________________________
void TEveProjection::SetDistortion(Float_t d)
{
   // Set distortion.

   fDistortion=d;
   fScale = 1+fFixedRadius*fDistortion;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveProjection::SetFixedRadius(Float_t r)
{
   // Set fixed radius.

   fFixedRadius=r;
   fScale = 1 + fFixedRadius*fDistortion;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   // Get vector for axis in a projected space.

   for (Int_t i=0; i<3; i++)
   {
      vec[i] = (i==screenAxis) ? 1. : 0.;
   }
}

//______________________________________________________________________________
Float_t TEveProjection::GetValForScreenPos(Int_t i, Float_t sv)
{
   // Inverse projection.

   static const TEveException eH("TEveProjection::GetValForScreenPos ");

   Float_t xL, xM, xR;
   TEveVector V, DirVec;
   SetDirectionalVector(i, DirVec);
   if (fDistortion > 0.0f && ((sv > 0 && sv > fUpLimit[i]) || (sv < 0 && sv < fLowLimit[i])))
      throw(eH + Form("screen value '%f' out of limit '%f'.", sv, sv > 0 ? fUpLimit[i] : fLowLimit[i]));

   TEveVector zero; ProjectVector(zero);
   // search from -/+ infinity according to sign of screen value
   if (sv > zero[i])
   {
      xL = 0; xR = 1000;
      while (1)
      {
         V.Mult(DirVec, xR); ProjectVector(V);
         // printf("positive projected %f, value %f,xL, xR ( %f, %f)\n", V[i], sv, xL, xR);
         if (V[i] > sv || V[i] == sv) break;
         xL = xR; xR *= 2;
      }
   }
   else if (sv < zero[i])
   {
      xR = 0; xL = -1000;
      while (1)
      {
         V.Mult(DirVec, xL); ProjectVector(V);
         // printf("negative projected %f, value %f,xL, xR ( %f, %f)\n", V[i], sv, xL, xR);
         if (V[i] < sv || V[i] == sv) break;
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
      V.Mult(DirVec, xM);
      ProjectVector(V);
      if (V[i] > sv)
         xR = xM;
      else
         xL = xM;
   } while(TMath::Abs(V[i] - sv) >= fgEps);

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


//______________________________________________________________________________
// TEveRhoZProjection
//
// Transformation from 3D to 2D. X axis represent Z coordinate. Y axis have value of
// radius with a sign of Y coordinate.

ClassImp(TEveRhoZProjection)

//______________________________________________________________________________
void TEveRhoZProjection::SetCenter(TEveVector& v)
{
   // Set center of distortion (virtual method).

   fCenter = v;

   Float_t R = TMath::Sqrt(v.x*v.x+v.y*v.y);
   fProjectedCenter.x = fCenter.z;
   fProjectedCenter.y = TMath::Sign(R, fCenter.y);
   fProjectedCenter.z = 0;
   UpdateLimit();
}

//______________________________________________________________________________
void TEveRhoZProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,  PProc_e proc )
{
   // Project point.

   using namespace TMath;

   if(proc == PP_Plane || proc == PP_Full)
   {
      // project
      y = Sign((Float_t)Sqrt(x*x+y*y), y);
      x = z;
   }
   if(proc == PP_Distort || proc == PP_Full)
   {
      // move to center
      x -= fProjectedCenter.x;
      y -= fProjectedCenter.y;
      // distort
      y = (y*fScale) / (1.0f + Abs(y)*fDistortion);
      x = (x*fScale) / (1.0f + Abs(x)*fDistortion);
      // move back from center
      x += fProjectedCenter.x;
      y += fProjectedCenter.y;
   }
   z = 0.0f;
}

//______________________________________________________________________________
void TEveRhoZProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   // Get direction in the unprojected space for axis index in the projected space.
   // This is virtual method from base-class TEveProjection.

   if(screenAxis == 0)
      vec.Set(0., 0., 1);
   else if (screenAxis == 1)
      vec.Set(0., 1., 0);

}
//______________________________________________________________________________
Bool_t TEveRhoZProjection::AcceptSegment(TEveVector& v1, TEveVector& v2, Float_t tolerance)
{
   // Check if segment of two projected points is valid.

   Float_t a = fProjectedCenter.y;
   Bool_t val = kTRUE;
   if((v1.y <  a && v2.y > a) || (v1.y > a && v2.y < a))
   {
      val = kFALSE;
      if (tolerance > 0)
      {
         Float_t a1 = TMath::Abs(v1.y - a), a2 = TMath::Abs(v2.y - a);
         if (a1 < a2)
         {
            if (a1 < tolerance) { v1.y = a; val = kTRUE; }
         }
         else
         {
            if (a2 < tolerance) { v2.y = a; val = kTRUE; }
         }
      }
   }
   return val;
}


//______________________________________________________________________________
// TEveCircularFishEyeProjection
//
// XY projection with distortion around given center.

ClassImp(TEveCircularFishEyeProjection)

//______________________________________________________________________________
void TEveCircularFishEyeProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                                 PProc_e proc)
{
   // Project point.

   using namespace TMath;

   if (proc != PP_Plane)
   {
      x -= fCenter.x;
      y -= fCenter.y;
      Float_t phi = x == 0.0 && y == 0.0 ? 0.0 : ATan2(y,x);
      Float_t R = Sqrt(x*x+y*y);
      // distort
      Float_t NR = (R*fScale) / (1.0f + R*fDistortion);
      x = NR*Cos(phi) + fCenter.x;
      y = NR*Sin(phi) + fCenter.y;
   }
   z = 0.0f;
}

