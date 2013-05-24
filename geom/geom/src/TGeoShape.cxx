// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//____________________________________________________________________________
// TGeoShape - Base abstract class for all shapes. 
//____________________________________________________________________________
//
//
//   Shapes are geometrical objects that provide the basic modelling 
// functionality. They provide the definition of the LOCAL frame of coordinates,
// with respect to which they are defined. Any implementation of a shape deriving
// from the base TGeoShape class has to provide methods for :
//  - finding out if a point defined in their local frame is or not contained 
// inside;
//  - computing the distance from a local point to getting outside/entering the
// shape, given a known direction;
//  - computing the maximum distance in any direction from a local point that
// does NOT result in a boundary crossing of the shape (safe distance); 
//  - computing the cosines of the normal vector to the crossed shape surface,
// given a starting local point and an ongoing direction.
//   All the features above are globally managed by the modeller in order to
// provide navigation functionality. In addition to those, shapes have also to
// implement additional specific abstract methods :
//  - computation of the minimal box bounding the shape, given that this box have
// to be aligned with the local coordinates;
//  - algorithms for dividing the shape along a given axis and producing resulting
// divisions volumes.
//
//   The modeler currently provides a set of 16 basic shapes, which we will call
// primitives. It also provides a special class allowing the creation of shapes
// made as a result of boolean operations between primitives. These are called
// composite shapes and the composition operation can be recursive (composition
// of composites). This allows the creation of a quite large number of different
// shape topologies and combinations.
//
//   Shapes are named objects and register themselves to the manager class at 
// creation time. This is responsible for their final deletion. Shapes 
// can be created without name if their retreival by name is no needed. Generally
// shapes are objects that are usefull only at geometry creation stage. The pointer
// to a shape is in fact needed only when referring to a given volume and it is 
// always accessible at that level. A shape may be referenced by several volumes,
// therefore its deletion is not possible once volumes were defined based on it.
// 
// 
// 
// Creating shapes
//================
//   Shape objects embeed only the minimum set of parameters that are fully
// describing a valid physical shape. For instance, a tube is represented by
// its half length, the minimum radius and the maximum radius. Shapes are used
// togeather with media in order to create volumes, which in their turn 
// are the main components of the geometrical tree. A specific shape can be created
// stand-alone : 
//
//   TGeoBBox *box = new TGeoBBox("s_box", halfX, halfY, halfZ); // named
//   TGeoTube *tub = new TGeoTube(rmin, rmax, halfZ);            // no name
//   ...  (see each specific shape constructors)
//
//   Sometimes it is much easier to create a volume having a given shape in one 
// step, since shapes are not direcly linked in the geometrical tree but volumes 
// are :
//
//   TGeoVolume *vol_box = gGeoManager->MakeBox("BOX_VOL", "mat1", halfX, halfY, halfZ);
//   TGeoVolume *vol_tub = gGeoManager->MakeTube("TUB_VOL", "mat2", rmin, rmax, halfZ);
//   ...  (see MakeXXX() utilities in TGeoManager class)
//
//
// Shape queries
//===============
// Note that global queries related to a geometry are handled by the manager class.
// However, shape-related queries might be sometimes usefull.
//
// A) Bool_t TGeoShape::Contains(Double_t *point[3])
//   - this method returns true if POINT is actually inside the shape. The point
// has to be defined in the local shape reference. For instance, for a box having
// DX, DY and DZ half-lengths a point will be considered inside if :
//   | -DX <= point[0] <= DX
//   | -DY <= point[1] <= DY
//   | -DZ <= point[2] <= DZ
//
// B) Double_t TGeoShape::DistFromInside(Double_t *point[3], Double_t *dir[3],
//                                  Int_t iact, Double_t step, Double_t *safe)
//   - computes the distance to exiting a shape from a given point INSIDE, along
// a given direction. The direction is given by its director cosines with respect
// to the local shape coordinate system. This method provides additional
// information according the value of IACT input parameter :
//   IACT = 0     => compute only safe distance and fill it at the location 
//                   given by SAFE
//   IACT = 1     => a proposed STEP is supplied. The safe distance is computed
//                   first. If this is bigger than STEP than the proposed step
//                   is approved and returned by the method since it does not
//                   cross the shape boundaries. Otherwise, the distance to
//                   exiting the shape is computed and returned.
//   IACT = 2     => compute both safe distance and distance to exiting, ignoring
//                   the proposed step.
//   IACT > 2     => compute only the distance to exiting, ignoring anything else.
//
// C) Double_t TGeoShape::DistFromOutside(Double_t *point[3], Double_t *dir[3],
//                                  Int_t iact, Double_t step, Double_t *safe)
//   - computes the distance to entering a shape from a given point OUTSIDE. Acts
// in the same way as B).
//
// D) Double_t Safety(Double_t *point[3], Bool_t inside)
//
//   - compute maximum shift of a point in any direction that does not change its
// INSIDE/OUTSIDE state (does not cross shape boundaries). The state of the point
// have to be properly supplied.
//
// E) Double_t *Normal(Double_t *point[3], Double_t *dir[3], Bool_t inside)
//
//   - returns director cosines of normal to the crossed shape surface from a
// given point towards a direction. One has to specify if the point is inside 
// or outside shape. According to this, the normal will be outwards or inwards
// shape respectively. Normal components are statically stored by shape class,
// so it has to be copied after retreival in a different array. 
//
// Dividing shapes
//=================
//   Shapes can generally be divided along a given axis. Supported axis are
// X, Y, Z, Rxy, Phi, Rxyz. A given shape cannot be divided however on any axis.
// The general rule is that that divisions are possible on whatever axis that
// produces still known shapes as slices. The division of shapes should not be
// performed by TGeoShape::Divide() calls, but rather by TGeoVolume::Divide().
// The algorithm for dividing a specific shape is known by the shape object, but
// is always invoked in a generic way from the volume level. Details on how to
// do that can be found in TGeoVolume class. One can see how all division options
// are interpreted and which is their result inside specific shape classes.
//_____________________________________________________________________________
//
//Begin_Html
/*
<img src="gif/t_shape.jpg">
*/
//End_Html

#include "TObjArray.h"
#include "TEnv.h"
#include "TError.h"

#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "TVirtualGeoPainter.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

ClassImp(TGeoShape)

TGeoMatrix *TGeoShape::fgTransform = NULL;
Double_t    TGeoShape::fgEpsMch = 2.220446049250313e-16;
//_____________________________________________________________________________
TGeoShape::TGeoShape()
{
// Default constructor
   fShapeBits = 0;
   fShapeId   = 0;
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
      // gROOT->AddGeoManager(gGeoManager);
   }
//   fShapeId = gGeoManager->GetListOfShapes()->GetSize();
//   gGeoManager->AddShape(this);
}

//_____________________________________________________________________________
TGeoShape::TGeoShape(const char *name)
          :TNamed(name, "")
{
// Default constructor
   fShapeBits = 0;
   fShapeId   = 0;
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
      // gROOT->AddGeoManager(gGeoManager);
   }
   fShapeId = gGeoManager->GetListOfShapes()->GetSize();
   gGeoManager->AddShape(this);
}

//_____________________________________________________________________________
TGeoShape::~TGeoShape()
{
// Destructor
   if (gGeoManager) gGeoManager->GetListOfShapes()->Remove(this);
}

//_____________________________________________________________________________
void TGeoShape::CheckShape(Int_t testNo, Int_t nsamples, Option_t *option)
{
// Test for shape navigation methods. Summary for test numbers:
//  1: DistFromInside/Outside. Sample points inside the shape. Generate 
//    directions randomly in cos(theta). Compute DistFromInside and move the 
//    point with bigger distance. Compute DistFromOutside back from new point.
//    Plot d-(d1+d2)
//
   if (!gGeoManager) {
      Error("CheckShape","No geometry manager");
      return;
   }
   TGeoShape *shape = (TGeoShape*)this;
   gGeoManager->CheckShape(shape, testNo, nsamples, option);
}
   
//_____________________________________________________________________________
Double_t TGeoShape::ComputeEpsMch()
{
// Compute machine round-off double precision error as the smallest number that
// if added to 1.0 is different than 1.0.
   Double_t temp1 = 1.0; 
   Double_t temp2 = 1.0 + temp1;
   Double_t mchEps;
   while (temp2>1.0) {
      mchEps = temp1;
      temp1 /= 2;
      temp2 = 1.0 + temp1;
   }
   fgEpsMch = mchEps;
   return fgEpsMch;
}   
   
//_____________________________________________________________________________
Double_t TGeoShape::EpsMch()
{
   //static function returning the machine round-off error
   
   return fgEpsMch;
}
   
//_____________________________________________________________________________
const char *TGeoShape::GetName() const
{
// Get the shape name.
   if (!strlen(fName)) {
      return ((TObject *)this)->ClassName();
   }
   return TNamed::GetName();
}

//_____________________________________________________________________________
Int_t TGeoShape::ShapeDistancetoPrimitive(Int_t numpoints, Int_t px, Int_t py) const
{
// Returns distance to shape primitive mesh.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 9999;
   return painter->ShapeDistancetoPrimitive(this, numpoints, px, py);
}

//_____________________________________________________________________________
Bool_t TGeoShape::IsCloseToPhi(Double_t epsil, Double_t *point, Double_t c1, Double_t s1, Double_t c2, Double_t s2)
{
// True if point is closer than epsil to one of the phi planes defined by c1,s1 or c2,s2
   Double_t saf1 = TGeoShape::Big();
   Double_t saf2 = TGeoShape::Big();
   if (point[0]*c1+point[1]*s1 >= 0) saf1 = TMath::Abs(-point[0]*s1 + point[1]*c1);
   if (point[0]*c2+point[1]*s2 >= 0) saf2 = TMath::Abs(point[0]*s2 - point[1]*c2);
   Double_t saf = TMath::Min(saf1,saf2);
   if (saf<epsil) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
Bool_t TGeoShape::IsInPhiRange(Double_t *point, Double_t phi1, Double_t phi2)
{
// Static method to check if a point is in the phi range (phi1, phi2) [degrees]
   Double_t phi = TMath::ATan2(point[1], point[0]) * TMath::RadToDeg();
   while (phi<phi1) phi+=360.;
   Double_t ddp = phi-phi1;
   if (ddp>phi2-phi1) return kFALSE;
   return kTRUE;
}   

//_____________________________________________________________________________  
Bool_t TGeoShape::IsCrossingSemiplane(Double_t *point, Double_t *dir, Double_t cphi, Double_t sphi, Double_t &snext, Double_t &rxy)
{
// Compute distance from POINT to semiplane defined by PHI angle along DIR. Computes
// also radius at crossing point. This might be negative in case the crossing is
// on the other side of the semiplane.
   snext = rxy = TGeoShape::Big();
   Double_t nx = -sphi;
   Double_t ny = cphi;
   Double_t rxy0 = point[0]*cphi+point[1]*sphi;
   Double_t rdotn = point[0]*nx + point[1]*ny;
   if (TMath::Abs(rdotn)<TGeoShape::Tolerance()) {
      snext = 0.0;
      rxy = rxy0;
      return kTRUE;
   }
   if (rdotn<0) {
      rdotn = -rdotn;
   } else {
      nx = -nx;
      ny = -ny;
   }
   Double_t ddotn = dir[0]*nx + dir[1]*ny;
   if (ddotn<=0) return kFALSE;
   snext = rdotn/ddotn;
   rxy = rxy0+snext*(dir[0]*cphi+dir[1]*sphi);
   if (rxy<0) return kFALSE;
   return kTRUE;
}

//_____________________________________________________________________________  
Bool_t TGeoShape::IsSameWithinTolerance(Double_t a, Double_t b)
{
// Check if two numbers differ with less than a tolerance.
   if (TMath::Abs(a-b)<1.E-10) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________  
Bool_t TGeoShape::IsSegCrossing(Double_t x1, Double_t y1, Double_t x2, Double_t y2,Double_t x3, Double_t y3,Double_t x4, Double_t y4)
{
// Check if segments (A,B) and (C,D) are crossing,
// where: A(x1,y1), B(x2,y2), C(x3,y3), D(x4,y4)
   Double_t eps = TGeoShape::Tolerance();
   Bool_t stand1 = kFALSE;
   Double_t dx1 = x2-x1;
   Bool_t stand2 = kFALSE;
   Double_t dx2 = x4-x3;
   Double_t xm = 0.;
   Double_t ym = 0.;
   Double_t a1 = 0.;
   Double_t b1 = 0.;
   Double_t a2 = 0.;
   Double_t b2 = 0.;
   if (TMath::Abs(dx1) < eps) stand1 = kTRUE;
   if (TMath::Abs(dx2) < eps) stand2 = kTRUE;
   if (!stand1) {
      a1 = (x2*y1-x1*y2)/dx1;
      b1 = (y2-y1)/dx1;
   }   
   if (!stand2) {
      a2 = (x4*y3-x3*y4)/dx2;
      b2 = (y4-y3)/dx2;
   }   
   if (stand1 && stand2) {
      // Segments parallel and vertical
      if (TMath::Abs(x1-x3)<eps) {
         // Check if segments are overlapping
         if ((y3-y1)*(y3-y2)<-eps || (y4-y1)*(y4-y2)<-eps ||
             (y1-y3)*(y1-y4)<-eps || (y2-y3)*(y2-y4)<-eps) return kTRUE;
         return kFALSE;
      }
      // Different x values
      return kFALSE;
   }
   
   if (stand1) {
      // First segment vertical
      xm = x1;
      ym = a2+b2*xm;
   } else {
      if (stand2) {
         // Second segment vertical
         xm = x3;
         ym = a1+b1*xm;
      } else {
         // Normal crossing
         if (TMath::Abs(b1-b2)<eps) {
            // Parallel segments, are they aligned
            if (TMath::Abs(y3-(a1+b1*x3))>eps) return kFALSE;
            // Aligned segments, are they overlapping
            if ((x3-x1)*(x3-x2)<-eps || (x4-x1)*(x4-x2)<-eps ||
                (x1-x3)*(x1-x4)<-eps || (x2-x3)*(x2-x4)<-eps) return kTRUE;
            return kFALSE;
         }
         xm = (a1-a2)/(b2-b1);
         ym = (a1*b2-a2*b1)/(b2-b1);
      }
   }
   // Check if crossing point is both between A,B and C,D
   Double_t check = (xm-x1)*(xm-x2)+(ym-y1)*(ym-y2);
   if (check > -eps) return kFALSE;
   check = (xm-x3)*(xm-x4)+(ym-y3)*(ym-y4);
   if (check > -eps) return kFALSE;
   return kTRUE;
}         

//_____________________________________________________________________________
Double_t TGeoShape::DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1,
                                 Double_t s2, Double_t c2, Double_t sm, Double_t cm, Bool_t in)
{
// compute distance from point (inside phi) to both phi planes. Return minimum.
   Double_t sfi1=TGeoShape::Big();
   Double_t sfi2=TGeoShape::Big();
   Double_t s=0;
   Double_t un = dir[0]*s1-dir[1]*c1;
   if (!in) un=-un;
   if (un>0) {
      s=-point[0]*s1+point[1]*c1;
      if (!in) s=-s;
      if (s>=0) {
         s /= un;
         if (((point[0]+s*dir[0])*sm-(point[1]+s*dir[1])*cm)>=0) sfi1=s;
      }
   }
   un = -dir[0]*s2+dir[1]*c2;
   if (!in) un=-un;
   if (un>0) {
      s=point[0]*s2-point[1]*c2;
      if (!in) s=-s;
      if (s>=0) {
         s /= un;
         if ((-(point[0]+s*dir[0])*sm+(point[1]+s*dir[1])*cm)>=0) sfi2=s;
      }
   }
   return TMath::Min(sfi1, sfi2);
}

//_____________________________________________________________________________
void TGeoShape::NormalPhi(Double_t *point, Double_t *dir, Double_t *norm, Double_t c1, Double_t s1, Double_t c2, Double_t s2)
{
// Static method to compute normal to phi planes.
   Double_t saf1 = TGeoShape::Big();
   Double_t saf2 = TGeoShape::Big();
   if (point[0]*c1+point[1]*s1 >= 0) saf1 = TMath::Abs(-point[0]*s1 + point[1]*c1);
   if (point[0]*c2+point[1]*s2 >= 0) saf2 =  TMath::Abs(point[0]*s2 - point[1]*c2);
   Double_t c,s;
   if (saf1<saf2) {
      c=c1;
      s=s1;
   } else {
      c=c2;
      s=s2;
   }
   norm[2] = 0;
   norm[0] = -s;
   norm[1] = c;
   if (dir[0]*norm[0]+dir[1]*norm[1] < 0) { 
      norm[0] = s;
      norm[1] = -c;
   }
}           
 
//_____________________________________________________________________________
Double_t TGeoShape::SafetyPhi(Double_t *point, Bool_t in, Double_t phi1, Double_t phi2)
{
// Static method to compute safety w.r.t a phi corner defined by cosines/sines
// of the angles phi1, phi2.
   Bool_t inphi = TGeoShape::IsInPhiRange(point, phi1, phi2);
   if (inphi && !in) return -TGeoShape::Big(); 
   phi1 *= TMath::DegToRad();
   phi2 *= TMath::DegToRad();  
   Double_t c1 = TMath::Cos(phi1);
   Double_t s1 = TMath::Sin(phi1);
   Double_t c2 = TMath::Cos(phi2);
   Double_t s2 = TMath::Sin(phi2);
   Double_t rsq = point[0]*point[0]+point[1]*point[1];
   Double_t rproj = point[0]*c1+point[1]*s1;
   Double_t safsq = rsq-rproj*rproj;
   if (safsq<0) return 0.;
   Double_t saf1 = (rproj<0)?TGeoShape::Big():TMath::Sqrt(safsq);
   rproj = point[0]*c2+point[1]*s2;
   safsq = rsq-rproj*rproj;
   if (safsq<0) return 0.;   
   Double_t saf2 = (rproj<0)?TGeoShape::Big():TMath::Sqrt(safsq);
   Double_t safe = TMath::Min(saf1, saf2); // >0
   if (safe>1E10) {
      if (in) return TGeoShape::Big();
      return -TGeoShape::Big();
   }
   return safe;   
}        

//_____________________________________________________________________________
Double_t TGeoShape::SafetySeg(Double_t r, Double_t z, Double_t r1, Double_t z1, Double_t r2, Double_t z2, Bool_t outer)
{
// Compute distance from point of coordinates (r,z) to segment (r1,z1):(r2,z2)
   Double_t crossp = (z2-z1)*(r-r1)-(z-z1)*(r2-r1);
   crossp *= (outer) ? 1. : -1.;
   // Positive crossp means point on the requested side of the (1,2) segment
   if (crossp < 0) {
      if (((z-z1)*(z2-z)) > 0) return 0;
      return TGeoShape::Big();
   }   
   // Compute (1,P) dot (1,2)
   Double_t c1 = (z-z1)*(z2-z1)+(r-r1)*(r2-r1);
   // Negative c1 means point (1) is closest
   if (c1<1.E-10) return TMath::Sqrt((r-r1)*(r-r1)+(z-z1)*(z-z1));
   // Compute (2,P) dot (1,2)
   Double_t c2 = (z-z2)*(z2-z1)+(r-r2)*(r2-r1);
   // Positive c2 means point (2) is closest
   if (c2>-1.E-10) return TMath::Sqrt((r-r2)*(r-r2)+(z-z2)*(z-z2));
   // The closest point is between (1) and (2)
   c2 = (z2-z1)*(z2-z1)+(r2-r1)*(r2-r1);
   // projected length factor with respect to (1,2) length
   Double_t alpha = c1/c2;
   Double_t rp = r1 + alpha*(r2-r1);
   Double_t zp = z1 + alpha*(z2-z1);
   return TMath::Sqrt((r-rp)*(r-rp)+(z-zp)*(z-zp));
}

//_____________________________________________________________________________
void TGeoShape::SetShapeBit(UInt_t f, Bool_t set)
{
// Equivalent of TObject::SetBit.
   if (set) {
      SetShapeBit(f);
   } else {
      ResetShapeBit(f);
   }
}

//_____________________________________________________________________________
TGeoMatrix *TGeoShape::GetTransform()
{
// Returns current transformation matrix that applies to shape.
   return fgTransform;
}   

//_____________________________________________________________________________
void TGeoShape::SetTransform(TGeoMatrix *matrix)
{
// Set current transformation matrix that applies to shape.
   fgTransform = matrix;
}   

//_____________________________________________________________________________
void TGeoShape::TransformPoints(Double_t *points, UInt_t NbPnts) const
{
   // Tranform a set of points (LocalToMaster)
   UInt_t i,j;
   Double_t dmaster[3];
   if (fgTransform) {
      for (j = 0; j < NbPnts; j++) {
         i = 3*j;
         fgTransform->LocalToMaster(&points[i], dmaster);
         points[i]   = dmaster[0];
         points[i+1] = dmaster[1];
         points[i+2] = dmaster[2];
      }
      return;   
   }   
   if (!gGeoManager) return;
   Bool_t bomb = (gGeoManager->GetBombMode()==0)?kFALSE:kTRUE;

   for (j = 0; j < NbPnts; j++) {
      i = 3*j;
      if (gGeoManager->IsMatrixTransform()) {
         TGeoHMatrix *glmat = gGeoManager->GetGLMatrix();
         if (bomb) glmat->LocalToMasterBomb(&points[i], dmaster);
         else      glmat->LocalToMaster(&points[i], dmaster);
      } else {
         if (bomb) gGeoManager->LocalToMasterBomb(&points[i], dmaster);
         else      gGeoManager->LocalToMaster(&points[i],dmaster);
      }
      points[i]   = dmaster[0];
      points[i+1] = dmaster[1];
      points[i+2] = dmaster[2];
   }
}

//_____________________________________________________________________________
void TGeoShape::FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const
{
   // Fill the supplied buffer, with sections in desired frame
   // See TBuffer3D.h for explanation of sections, frame etc.
  
   // Catch this common potential error here
   // We have to set kRawSize (unless already done) to allocate buffer space 
   // before kRaw can be filled
   if (reqSections & TBuffer3D::kRaw) {
      if (!(reqSections & TBuffer3D::kRawSizes) && !buffer.SectionsValid(TBuffer3D::kRawSizes)) {
         R__ASSERT(kFALSE);
      }
   }

   if (reqSections & TBuffer3D::kCore) {
      // If writing core section all others will be invalid
      buffer.ClearSectionsValid();
 
      // Check/grab some objects we need
      if (!gGeoManager) { 
         R__ASSERT(kFALSE); 
         return; 
      }
      const TGeoVolume * paintVolume = gGeoManager->GetPaintVolume();
      if (!paintVolume) paintVolume = gGeoManager->GetTopVolume();
      if (!paintVolume) { 
         buffer.fID = const_cast<TGeoShape *>(this);
         buffer.fColor = 0;
         buffer.fTransparency = 0;
//         R__ASSERT(kFALSE); 
//         return; 
      } else {
         buffer.fID = const_cast<TGeoVolume *>(paintVolume);
         buffer.fColor = paintVolume->GetLineColor();

         buffer.fTransparency = paintVolume->GetTransparency();
         Double_t visdensity = gGeoManager->GetVisDensity();
         if (visdensity>0 && paintVolume->GetMedium()) {
            if (paintVolume->GetMaterial()->GetDensity() < visdensity) {
               buffer.fTransparency = 90;
            }
         }   
      }

      buffer.fLocalFrame = localFrame;
      Bool_t r1,r2=kFALSE;
      r1 = gGeoManager->IsMatrixReflection();
      if (paintVolume && paintVolume->GetShape()) {
         if (paintVolume->GetShape()->IsReflected()) {
         // Temporary trick to deal with reflected shapes.
         // Still lighting gets wrong...
            if (buffer.Type() < TBuffer3DTypes::kTube) r2 = kTRUE;
         }
      }      
      buffer.fReflection = ((r1&(!r2))|(r2&!(r1)));
      
      // Set up local -> master translation matrix
      if (localFrame) {
         TGeoMatrix * localMasterMat = 0;
         if (TGeoShape::GetTransform()) {
            localMasterMat = TGeoShape::GetTransform();
         } else {   
            localMasterMat = gGeoManager->GetCurrentMatrix();

         // For overlap drawing the correct matrix needs to obtained in 
         // from GetGLMatrix() - this should not be applied in the case
         // of composite shapes
            if (gGeoManager->IsMatrixTransform() && !IsComposite()) {
               localMasterMat = gGeoManager->GetGLMatrix();
            }
         }
         if (!localMasterMat) { 
            R__ASSERT(kFALSE); 
            return; 
         }
         localMasterMat->GetHomogenousMatrix(buffer.fLocalMaster);
      } else {
         buffer.SetLocalMasterIdentity();
      }

      buffer.SetSectionsValid(TBuffer3D::kCore);
   }
}

//_____________________________________________________________________________
Int_t TGeoShape::GetBasicColor() const
{
// Get the basic color (0-7).
   Int_t basicColor = 0; // TODO: Check on sensible fallback
   if (gGeoManager) {
      const TGeoVolume * volume = gGeoManager->GetPaintVolume();
      if (volume) {
            basicColor = ((volume->GetLineColor() %8) -1) * 4;
            if (basicColor < 0) basicColor = 0;
      }
   }
   return basicColor;
}

//_____________________________________________________________________________
const TBuffer3D &TGeoShape::GetBuffer3D(Int_t /*reqSections*/, Bool_t /*localFrame*/) const
{
   // Stub implementation to avoid forcing implementation at this stage
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);
   Warning("GetBuffer3D", "this must be implemented for shapes in a TGeoPainter hierarchy. This will be come a pure virtual fn eventually.");
   return buffer;
}

//_____________________________________________________________________________
const char *TGeoShape::GetPointerName() const
{
// Provide a pointer name containing uid.
   static TString name;
   Int_t uid = GetUniqueID();
   if (uid) name = TString::Format("p%s_%d", GetName(),uid);
   else     name = TString::Format("p%s", GetName());
   return name.Data();
}

//_____________________________________________________________________________
void TGeoShape::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute mouse actions on this shape.
   if (!gGeoManager) return;
   TVirtualGeoPainter *painter = gGeoManager->GetPainter();
   painter->ExecuteShapeEvent(this, event, px, py);
}

//_____________________________________________________________________________
void TGeoShape::Draw(Option_t *option)
{
// Draw this shape.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (option && strlen(option) > 0) {
      painter->DrawShape(this, option); 
   } else {
      painter->DrawShape(this, gEnv->GetValue("Viewer3D.DefaultDrawOption",""));
   }  
}

//_____________________________________________________________________________
void TGeoShape::Paint(Option_t *option)
{
// Paint this shape.
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (option && strlen(option) > 0) {
      painter->PaintShape(this, option); 
   } else {
      painter->PaintShape(this, gEnv->GetValue("Viewer3D.DefaultDrawOption",""));
   }  
}
