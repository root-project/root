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
TEveProjection::TEveProjection() :
   fType          (kPT_Unknown),
   fGeoMode       (kGM_Unknown),
   fName          (0),
   fCenter        (),
   fUsePreScale   (kFALSE),
   fDistortion    (0.0f),
   fFixR          (300), fFixZ          (400),
   fPastFixRFac   (0),   fPastFixZFac   (0),
   fScaleR        (1),   fScaleZ        (1),
   fPastFixRScale (1),   fPastFixZScale (1),
   fMaxTrackStep  (5),
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
void TEveProjection::PreScalePoint(Float_t& v0, Float_t& v1)
{
   // Pre-scale point (v0, v1) in projected coordinates:
   //   RhoZ ~ (rho, z)
   //   RPhi ~ (r, phi), scaling phi doesn't make much sense.

   if (!fPreScales[0].empty())
   {
      Bool_t invp = kFALSE;
      if (v0 < 0) {
         v0    = -v0;
         invp = kTRUE;
      }
      vPreScale_i i = fPreScales[0].begin();
      while (v0 > i->fMax)
         ++i;
      v0 = i->fOffset + (v0 - i->fMin)*i->fScale;
      if (invp)
         v0 = -v0;
   }
   if (!fPreScales[1].empty())
   {
      Bool_t invp = kFALSE;
      if (v1 < 0) {
         v1    = -v1;
         invp = kTRUE;
      }
      vPreScale_i i = fPreScales[1].begin();
      while (v1 > i->fMax)
         ++i;
      v1 = i->fOffset + (v1 - i->fMin)*i->fScale;
      if (invp)
         v1 = -v1;
   }
}

//______________________________________________________________________________
void TEveProjection::AddPreScaleEntry(Int_t coord, Float_t value, Float_t scale)
{
   // Add new scaling range for given coordinate.
   // Arguments:
   //  coord    0 ~ x, 1 ~ y;
   //  value    value of input coordinate from which to apply this scale;
   //  scale    the scale to apply from value onwards.
   //
   // NOTE: If pre-scaling is combined with center-displaced then
   // the scale of the central region should be 1. This limitation
   // can be removed but will cost CPU.

   static const TEveException eh("TEveProjection::AddPreScaleEntry ");

   if (coord < 0 || coord > 1)
      throw (eh + "coordinate out of range.");

   const Float_t infty  = std::numeric_limits<Float_t>::infinity();

   vPreScale_t& vec = fPreScales[coord];

   if (vec.empty())
   {
      if (value == 0)
      {
         vec.push_back(PreScaleEntry_t(0, infty, 0, scale));
      }
      else
      {
         vec.push_back(PreScaleEntry_t(0, value, 0, 1));
         vec.push_back(PreScaleEntry_t(value, infty, value, scale));
      }
   }
   else
   {
      PreScaleEntry_t& prev = vec.back();
      if (value <= prev.fMin)
         throw (eh + "minimum value not larger than previous one.");

      prev.fMax = value;
      Float_t offset =  prev.fOffset + (prev.fMax - prev.fMin)*prev.fScale;
      vec.push_back(PreScaleEntry_t(value, infty, offset, scale));
   }
}

//______________________________________________________________________________
void TEveProjection::ChangePreScaleEntry(Int_t   coord, Int_t entry,
                                         Float_t new_scale)
{
   // Change scale for given entry and coordinate.
   //
   // NOTE: If the first entry you created used other value than 0,
   // one entry (covering range from 0 to this value) was created
   // automatically.

   static const TEveException eh("TEveProjection::ChangePreScaleEntry ");

   if (coord < 0 || coord > 1)
      throw (eh + "coordinate out of range.");

   vPreScale_t& vec = fPreScales[coord];
   Int_t        vs  = vec.size();
   if (entry < 0 || entry >= vs)
      throw (eh + "entry out of range.");

   vec[entry].fScale = new_scale;
   Int_t i0 = entry, i1 = entry + 1;
   while (i1 < vs)
   {
      PreScaleEntry_t e0 = vec[i0];
      vec[i1].fOffset = e0.fOffset + (e0.fMax - e0.fMin)*e0.fScale;
      i0 = i1++;
   }
}

//______________________________________________________________________________
void TEveProjection::ClearPreScales()
{
   // Clear all pre-scaling information.

   fPreScales[0].clear();
   fPreScales[1].clear();
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
TEveRhoZProjection::TEveRhoZProjection() :
   TEveProjection()
{
   // Constructor.

   fType = kPT_RhoZ;
   fName = "RhoZ";
}

//______________________________________________________________________________
void TEveRhoZProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                      EPProc_e proc)
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
      if (fUsePreScale)
         PreScalePoint(y, x);

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
TEveRPhiProjection::TEveRPhiProjection() :
   TEveProjection()
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
      Float_t r, phi;
      if (fUsePreScale)
      {
         r   = Sqrt(x*x + y*y);
         phi = (x == 0.0f && y == 0.0f) ? 0.0f : ATan2(y, x);
         PreScalePoint(r, phi);
         x = r*Cos(phi);
         y = r*Sin(phi);
      }

      x  -= fCenter.fX;
      y  -= fCenter.fY;
      r   = Sqrt(x*x + y*y);
      phi = (x == 0.0f && y == 0.0f) ? 0.0f : ATan2(y, x);

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
