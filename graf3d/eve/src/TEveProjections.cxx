// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TError.h"

#include "TEveProjections.h"
#include "TEveTrans.h"
#include "TEveUtil.h"

#include <limits>

/** \class TEveProjection
\ingroup TEve
Base-class for non-linear projections.

Enables to define an external center of distortion and a scale to
fixate a bounding box of a projected point.
*/

ClassImp(TEveProjection);

Float_t TEveProjection::fgEps    = 0.005f;
Float_t TEveProjection::fgEpsSqr = 0.000025f;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveProjection::TEveProjection() :
   fType          (kPT_Unknown),
   fGeoMode       (kGM_Unknown),
   fName          (0),
   fCenter        (),
   fDisplaceOrigin (kFALSE),
   fUsePreScale   (kFALSE),
   fDistortion    (0.0f),
   fFixR          (300), fFixZ          (400),
   fPastFixRFac   (0),   fPastFixZFac   (0),
   fScaleR        (1),   fScaleZ        (1),
   fPastFixRScale (1),   fPastFixZScale (1),
   fMaxTrackStep  (5)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Project float array.

void TEveProjection::ProjectPointfv(Float_t* v, Float_t d)
{
   ProjectPoint(v[0], v[1], v[2], d);
}

////////////////////////////////////////////////////////////////////////////////
/// Project double array.
/// This is a bit piggish as we convert the doubles to floats and back.

void TEveProjection::ProjectPointdv(Double_t* v, Float_t d)
{
   Float_t x = v[0], y = v[1], z = v[2];
   ProjectPoint(x, y, z, d);
   v[0] = x; v[1] = y; v[2] = z;
}

////////////////////////////////////////////////////////////////////////////////
/// Project TEveVector.

void TEveProjection::ProjectVector(TEveVector& v, Float_t d)
{
   ProjectPoint(v.fX, v.fY, v.fZ, d);
}

////////////////////////////////////////////////////////////////////////////////
/// Project float array, converting it to global coordinate system first if
/// transformation matrix is set.

void TEveProjection::ProjectPointfv(const TEveTrans* t, const Float_t* p, Float_t* v, Float_t d)
{
   v[0] = p[0]; v[1] = p[1]; v[2] = p[2];
   if (t)
   {
      t->MultiplyIP(v);
   }
   ProjectPoint(v[0], v[1], v[2], d);
}

////////////////////////////////////////////////////////////////////////////////
/// Project double array, converting it to global coordinate system first if
/// transformation matrix is set.
/// This is a bit piggish as we convert the doubles to floats and back.

void TEveProjection::ProjectPointdv(const TEveTrans* t, const Double_t* p, Double_t* v, Float_t d)
{
   Float_t x, y, z;
   if (t)
   {
      t->Multiply(p, v);
      x = v[0]; y = v[1]; z = v[2];
   }
   else
   {
      x = p[0]; y = p[1]; z = p[2];
   }
   ProjectPoint(x, y, z, d);
   v[0] = x; v[1] = y; v[2] = z;
}

////////////////////////////////////////////////////////////////////////////////
/// Project TEveVector, converting it to global coordinate system first if
/// transformation matrix is set.

void TEveProjection::ProjectVector(const TEveTrans* t, TEveVector& v, Float_t d)
{
   if (t)
   {
      t->MultiplyIP(v);
   }
   ProjectPoint(v.fX, v.fY, v.fZ, d);
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-scale single variable with pre-scale entry dim.

void TEveProjection::PreScaleVariable(Int_t dim, Float_t& v)
{
   if (!fPreScales[dim].empty())
   {
      Bool_t invp = kFALSE;
      if (v < 0) {
         v    = -v;
         invp = kTRUE;
      }
      vPreScale_i i = fPreScales[dim].begin();
      while (v > i->fMax)
         ++i;
      v = i->fOffset + (v - i->fMin)*i->fScale;
      if (invp)
         v = -v;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-scale point (x, y) in projected coordinates for 2D projections:
///  - RhoZ ~ (rho, z)
///  - RPhi ~ (r, phi), scaling phi doesn't make much sense.

void TEveProjection::PreScalePoint(Float_t& x, Float_t& y)
{
   PreScaleVariable(0, x);
   PreScaleVariable(1, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Pre-scale point (x, y, z) in projected coordinates for 3D projection.

void TEveProjection::PreScalePoint(Float_t& x, Float_t& y, Float_t& z)
{
   PreScaleVariable(0, x);
   PreScaleVariable(1, y);
   PreScaleVariable(2, z);
}

////////////////////////////////////////////////////////////////////////////////
/// Add new scaling range for given coordinate.
/// Arguments:
///  - coord    0 ~ x, 1 ~ y, 2 ~ z
///  - value    value of input coordinate from which to apply this scale;
///  - scale    the scale to apply from value onwards.
///
/// NOTE: If pre-scaling is combined with center-displaced then
/// the scale of the central region should be 1. This limitation
/// can be removed but will cost CPU.

void TEveProjection::AddPreScaleEntry(Int_t coord, Float_t value, Float_t scale)
{
   static const TEveException eh("TEveProjection::AddPreScaleEntry ");

   if (coord < 0 || coord > 2)
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

////////////////////////////////////////////////////////////////////////////////
/// Change scale for given entry and coordinate.
///
/// NOTE: If the first entry you created used other value than 0,
/// one entry (covering range from 0 to this value) was created
/// automatically.

void TEveProjection::ChangePreScaleEntry(Int_t   coord, Int_t entry,
                                         Float_t new_scale)
{
   static const TEveException eh("TEveProjection::ChangePreScaleEntry ");

   if (coord < 0 || coord > 2)
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

////////////////////////////////////////////////////////////////////////////////
/// Clear all pre-scaling information.

void TEveProjection::ClearPreScales()
{
   fPreScales[0].clear();
   fPreScales[1].clear();
   fPreScales[2].clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Set distortion.

void TEveProjection::SetDistortion(Float_t d)
{
   fDistortion    = d;
   fScaleR        = 1.0f + fFixR*fDistortion;
   fScaleZ        = 1.0f + fFixZ*fDistortion;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
}

////////////////////////////////////////////////////////////////////////////////
/// Set fixed radius.

void TEveProjection::SetFixR(Float_t r)
{
   fFixR          = r;
   fScaleR        = 1 + fFixR*fDistortion;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
}

////////////////////////////////////////////////////////////////////////////////
/// Set fixed radius.

void TEveProjection::SetFixZ(Float_t z)
{
   fFixZ          = z;
   fScaleZ        = 1 + fFixZ*fDistortion;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
}

////////////////////////////////////////////////////////////////////////////////
/// Set 2's-exponent for relative scaling beyond FixR.

void TEveProjection::SetPastFixRFac(Float_t x)
{
   fPastFixRFac   = x;
   fPastFixRScale = TMath::Power(10.0f, fPastFixRFac) / fScaleR;
}

////////////////////////////////////////////////////////////////////////////////
/// Get projected center.

Float_t* TEveProjection::GetProjectedCenter()
{
   static TEveVector zero;

   if (fDisplaceOrigin)
      return zero.Arr();
   else
      return fCenter.Arr();
}

////////////////////////////////////////////////////////////////////////////////
/// Set flag to displace for center.
/// This options is useful if want to have projected center
/// at (0, 0) position in projected coordinates and want to dismiss
/// gap around projected center in RhoZ projection.

void  TEveProjection::SetDisplaceOrigin(Bool_t x)
{
   fDisplaceOrigin = x;
   // update projected center
   SetCenter(fCenter);
}

////////////////////////////////////////////////////////////////////////////////
/// Set 2's-exponent for relative scaling beyond FixZ.

void TEveProjection::SetPastFixZFac(Float_t x)
{
   fPastFixZFac   = x;
   fPastFixZScale = TMath::Power(10.0f, fPastFixZFac) / fScaleZ;
}

////////////////////////////////////////////////////////////////////////////////
/// Find break-point on both sides of the discontinuity.
/// They still need to be projected after the call.
/// This is an obsolete version of the method that required manual
/// specification of precision -- this lead to (infrequent) infinite loops.

void TEveProjection::BisectBreakPoint(TEveVector& vL, TEveVector& vR, Float_t /*eps_sqr*/)
{
   static Bool_t warnedp = kFALSE;

   if (!warnedp)
   {
      Warning("BisectBreakPoint", "call with eps_sqr argument is obsolete - please use the new signature.");
      warnedp = kTRUE;
   }

   BisectBreakPoint(vL, vR, kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Find break-point on both sides of the discontinuity.
/// If project_result is true, the resulting break points will be projected
/// with given depth value.

void TEveProjection::BisectBreakPoint(TEveVector& vL, TEveVector& vR,
                                      Bool_t project_result, Float_t depth)
{
   TEveVector vM, vLP, vMP;
   Int_t n_loops = TMath::CeilNint(TMath::Log2(1e12 * (vL-vR).Mag2() / (0.5f*(vL+vR)).Mag2()) / 2);
   while (--n_loops >= 0)
   {
      vM.Mult(vL + vR, 0.5f);
      vLP.Set(vL); ProjectPoint(vLP.fX, vLP.fY, vLP.fZ, 0);
      vMP.Set(vM); ProjectPoint(vMP.fX, vMP.fY, vMP.fZ, 0);

      if (IsOnSubSpaceBoundrary(vMP))
      {
         vL.Set(vM);
         vR.Set(vM);
         break;
      }

      if (AcceptSegment(vLP, vMP, 0.0f))
      {
         vL.Set(vM);
      }
      else
      {
         vR.Set(vM);
      }
   }

   if (project_result)
   {
      ProjectVector(vL, depth);
      ProjectVector(vR, depth);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Method previously used by  TEveProjectionAxesGL. Now obsolete.

Float_t TEveProjection::GetLimit(Int_t, Bool_t)
{
  ::Warning("TEveProjection::GetLimits", "method is obsolete");

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Get vector for axis in a projected space.

void TEveProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   for (Int_t i=0; i<3; i++)
   {
      vec[i] = (i==screenAxis) ? 1.0f : 0.0f;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get center ortogonal to given axis index.

TEveVector TEveProjection::GetOrthogonalCenter(int i, TEveVector& centerOO)
{
   TEveVector dirVec;
   SetDirectionalVector(i, dirVec);

   TEveVector dirCenter;
   dirCenter.Mult(dirVec, fCenter.Dot(dirVec));
   centerOO = fCenter - dirCenter;


   return  centerOO;
}

////////////////////////////////////////////////////////////////////////////////
/// Inverse projection.

Float_t TEveProjection::GetValForScreenPos(Int_t axisIdx, Float_t sv)
{
   static const TEveException eH("TEveProjection::GetValForScreenPos ");

   static const int kMaxSteps = 5000;
   static const int kMaxVal = 10;

   Float_t xL, xM, xR;
   TEveVector vec;

   TEveVector dirVec;
   SetDirectionalVector(axisIdx, dirVec);

   TEveVector zero;
   if (fDisplaceOrigin) zero = fCenter;

   TEveVector zeroProjected = zero;
   ProjectVector(zeroProjected, 0.f);

   // search from -/+ infinity according to sign of screen value
   if (sv > zeroProjected[axisIdx])
   {
      xL = 0;
      xR = kMaxVal;

      int cnt = 0;
      while (cnt < kMaxSteps)
      {
         vec.Mult(dirVec, xR);
         if (fDisplaceOrigin) vec += fCenter;

         ProjectVector(vec, 0);
         if (vec[axisIdx] >= sv) break;
         xL = xR; xR *= 2;

         if (++cnt >= kMaxSteps)
            throw eH + Form("positive projected %f, value %f,xL, xR ( %f, %f)\n", vec[axisIdx], sv, xL, xR);
      }
   }
   else if (sv < zeroProjected[axisIdx])
   {
      xR = 0;
      xL = -kMaxVal;

      int cnt = 0;
      while (cnt < kMaxSteps)
      {
         vec.Mult(dirVec, xL);
         if (fDisplaceOrigin) vec += fCenter;

         ProjectVector(vec, 0);
         if (vec[axisIdx] <= sv) break;
         xR = xL; xL *= 2;
         if (++cnt >= kMaxSteps)
            throw eH + Form("negative projected %f, value %f,xL, xR ( %f, %f)\n", vec[axisIdx], sv, xL, xR);
      }
   }
   else
   {
      return 0.0f;
   }

   //  printf("search for value %f in rng[%f, %f] \n", sv, xL, xR);
   int cnt = 0;
   do
   {
      //printf("search value with bisection xL=%f, xR=%f; vec[axisIdx]=%f, sv=%f\n", xL, xR, vec[axisIdx], sv);
      xM = 0.5f * (xL + xR);
      vec.Mult(dirVec, xM);
      if (fDisplaceOrigin) vec += fCenter;
      ProjectVector(vec, 0);
      if (vec[axisIdx] > sv)
         xR = xM;
      else
         xL = xM;
      if (++cnt >= kMaxSteps)
         throw eH + Form("can't converge %f %f, l/r %f/%f, idx=%d\n", vec[axisIdx], sv, xL, xR, axisIdx);

   } while (TMath::Abs(vec[axisIdx] - sv) >= fgEps);


   return xM;
}

////////////////////////////////////////////////////////////////////////////////
/// Project point on given axis and return projected value.

Float_t TEveProjection::GetScreenVal(Int_t i, Float_t x, TEveVector& dirVec, TEveVector& /*oCenter*/)
{
   TEveVector pos = dirVec*x;

   if (fDisplaceOrigin)
      pos += fCenter;

   ProjectVector(pos , 0.f);

   return pos[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Project point on given axis and return projected value.

Float_t TEveProjection::GetScreenVal(Int_t i, Float_t x)
{
   TEveVector dirVec;
   SetDirectionalVector(i, dirVec);
   TEveVector oCenter;
   // GetOrthogonalCenter(i, oCenter);
   return GetScreenVal(i, x, dirVec, oCenter);
}

/** \class TEveRhoZProjection
\ingroup TEve
Transformation from 3D to 2D. X axis represent Z coordinate. Y axis have value of
radius with a sign of Y coordinate.
*/

ClassImp(TEveRhoZProjection);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveRhoZProjection::TEveRhoZProjection() :
   TEveProjection()
{
   fType = kPT_RhoZ;
   fName = "RhoZ";
}

////////////////////////////////////////////////////////////////////////////////
/// Project point.

void TEveRhoZProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                      Float_t  d, EPProc_e proc)
{
   using namespace TMath;

   if (fDisplaceOrigin) {
      x -= fCenter.fX;
      y -= fCenter.fY;
      z -= fCenter.fZ;
   }
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

      // distort

      if (!fDisplaceOrigin) {
         x -= fProjectedCenter.fX;
         y -= fProjectedCenter.fY;
      }

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

      if (!fDisplaceOrigin) {
         x += fProjectedCenter.fX;
         y += fProjectedCenter.fY;
      }
   }
   z = d;
}

////////////////////////////////////////////////////////////////////////////////
/// Set center of distortion (virtual method).

void TEveRhoZProjection::SetCenter(TEveVector& v)
{
   fCenter = v;

   if (fDisplaceOrigin)
   {
      fProjectedCenter.Set(0.f, 0.f, 0.f);
   }
   else
   {
      Float_t r = TMath::Sqrt(v.fX*v.fX + v.fY*v.fY);
      fProjectedCenter.fX = fCenter.fZ;
      fProjectedCenter.fY = TMath::Sign(r, fCenter.fY);
      fProjectedCenter.fZ = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get direction in the unprojected space for axis index in the
/// projected space.
/// This is virtual method from base-class TEveProjection.

void TEveRhoZProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   if (screenAxis == 0)
      vec.Set(0.0f, 0.0f, 1.0f);
   else if (screenAxis == 1)
      vec.Set(0.0f, 1.0f, 0.0f);

}

////////////////////////////////////////////////////////////////////////////////
/// Check if segment of two projected points is valid.
///
/// Move slightly one of the points if by shifting it by no more than
/// tolerance the segment can become acceptable.

Bool_t TEveRhoZProjection::AcceptSegment(TEveVector& v1, TEveVector& v2,
                                         Float_t tolerance) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Return sub-space id for the point.
/// 0 - upper half-space
/// 1 - lower half-space

Int_t TEveRhoZProjection::SubSpaceId(const TEveVector& v) const
{
   return v.fY > fProjectedCenter.fY ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Checks if point is on sub-space boundary.

Bool_t TEveRhoZProjection::IsOnSubSpaceBoundrary(const TEveVector& v) const
{
   return v.fY == fProjectedCenter.fY;
}

/** \class TEveRPhiProjection
\ingroup TEve
XY projection with distortion around given center.
*/

ClassImp(TEveRPhiProjection);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveRPhiProjection::TEveRPhiProjection() :
   TEveProjection()
{
   fType    = kPT_RPhi;
   fGeoMode = kGM_Polygons;
   fName    = "RhoPhi";
}

////////////////////////////////////////////////////////////////////////////////
/// Project point.

void TEveRPhiProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                      Float_t d, EPProc_e proc)
{
   using namespace TMath;

   if (fDisplaceOrigin)
   {
      x  -= fCenter.fX;
      y  -= fCenter.fY;
      z  -= fCenter.fZ;
   }

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

      if (!fDisplaceOrigin)
      {
         x  -= fCenter.fX;
         y  -= fCenter.fY;
      }

      r   = Sqrt(x*x + y*y);
      phi = (x == 0.0f && y == 0.0f) ? 0.0f : ATan2(y, x);

      if (r > fFixR)
         r =  fFixR + fPastFixRScale*(r - fFixR);
      else if (r < -fFixR)
         r = -fFixR + fPastFixRScale*(r + fFixR);
      else
         r =  r * fScaleR / (1.0f + r*fDistortion);

      x = r*Cos(phi);
      y = r*Sin(phi);

      if (!fDisplaceOrigin)
      {
         x += fCenter.fX;
         y += fCenter.fY;
      }
   }

   z = d;
}

/** \class TEveXZProjection
\ingroup TEve
XZ projection with distortion around given center.
*/

ClassImp(TEveXZProjection);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveXZProjection::TEveXZProjection() :
   TEveProjection()
{
   fType    = kPT_XZ;
   fGeoMode = kGM_Polygons;
   fName    = "XZ";
}

////////////////////////////////////////////////////////////////////////////////
/// Project point.

void TEveXZProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                    Float_t d, EPProc_e proc)
{
   using namespace TMath;
   
   if (fDisplaceOrigin)
   {
      x  -= fCenter.fX;
      y  -= fCenter.fY;
      z  -= fCenter.fZ;
   }

   // projection
   if (proc == kPP_Plane || proc == kPP_Full)
   {
      y = z;
      z = d;
   }
   if (proc != kPP_Distort || proc == kPP_Full)
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

      if (!fDisplaceOrigin)
      {
         x -= fProjectedCenter.fX;
         y -= fProjectedCenter.fY;
      }

      r   = Sqrt(x*x + y*y);
      phi = (x == 0.0f && y == 0.0f) ? 0.0f : ATan2(y, x);

      if (r > fFixR)
         r =  fFixR + fPastFixRScale*(r - fFixR);
      else if (r < -fFixR)
         r = -fFixR + fPastFixRScale*(r + fFixR);
      else
         r =  r * fScaleR / (1.0f + r*fDistortion);

      x = r*Cos(phi);
      y = r*Sin(phi);

      if (!fDisplaceOrigin)
      {
         x += fProjectedCenter.fX;
         y += fProjectedCenter.fY;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set center of distortion (virtual method).

void TEveXZProjection::SetCenter(TEveVector& v)
{
   fCenter = v;

   if (fDisplaceOrigin)
   {
      fProjectedCenter.Set(0.f, 0.f, 0.f);
   }
   else
   {
      fProjectedCenter.fX = fCenter.fX;
      fProjectedCenter.fY = fCenter.fZ;
      fProjectedCenter.fZ = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get direction in the unprojected space for axis index in the
/// projected space.
/// This is virtual method from base-class TEveProjection.

void TEveXZProjection::SetDirectionalVector(Int_t screenAxis, TEveVector& vec)
{
   if (screenAxis == 0)
      vec.Set(1.0f, 0.0f, 0.0f);
   else if (screenAxis == 1)
      vec.Set(0.0f, 0.0f, 1.0f);
}


/** \class TEve3DProjection
\ingroup TEve
3D scaling projection. One has to use pre-scaling to make any ise of this.
*/

ClassImp(TEve3DProjection);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEve3DProjection::TEve3DProjection() :
   TEveProjection()
{
   fType    = kPT_3D;
   fGeoMode = kGM_Unknown;
   fName    = "3D";
}

////////////////////////////////////////////////////////////////////////////////
/// Project point.

void TEve3DProjection::ProjectPoint(Float_t& x, Float_t& y, Float_t& z,
                                    Float_t /*d*/, EPProc_e proc)
{
   using namespace TMath;

   if (proc != kPP_Plane)
   {
      if (fUsePreScale)
      {
         PreScalePoint(x, y, z);
      }

      x -= fCenter.fX;
      y -= fCenter.fY;
      z -= fCenter.fZ;
   }
}
