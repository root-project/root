// @(#)root/geom:$Name:  $:$Id: TGeoShape.cxx,v 1.9 2003/06/17 09:13:55 brun Exp $
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
// B) Double_t TGeoShape::DistToOut(Double_t *point[3], Double_t *dir[3],
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
// C) Double_t TGeoShape::DistToOut(Double_t *point[3], Double_t *dir[3],
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

#include "TGeoMatrix.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoShape.h"
#include "TVirtualGeoPainter.h"


const Double_t TGeoShape::kRadDeg = 180./TMath::Pi();
const Double_t TGeoShape::kDegRad = TMath::Pi()/180.;
const Double_t TGeoShape::kBig = 1E30;

ClassImp(TGeoShape)

//_____________________________________________________________________________
TGeoShape::TGeoShape()
{
// Default constructor
   fShapeId = 0;
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
   fShapeId = 0;
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
const char *TGeoShape::GetName() const
{
   if (!strlen(fName)) {
      return ((TObject *)this)->ClassName();
   }
   return TNamed::GetName();
}

//_____________________________________________________________________________
Int_t TGeoShape::ShapeDistancetoPrimitive(Int_t numpoints, Int_t px, Int_t py) const
{
   TVirtualGeoPainter *painter = gGeoManager->GetGeomPainter();
   if (!painter) return 9999;
   return painter->ShapeDistancetoPrimitive(this, numpoints, px, py);
}

//_____________________________________________________________________________
Double_t TGeoShape::ClosenessToCorner(Double_t *point, Bool_t in,
                                 Double_t *vertex, Double_t *normals, Double_t *cldir)
{
// Static method returning distance to closest point of a corner. The corner is 
// defined by vertex and normals to the 3 planes (in order X, Y, Z - norm[9]).
// also return unit vector pointing to this
  
   Double_t safe[3];  // closest distances to the 3 planes
   Double_t dvert[3]; // vector from vertex to point
   Int_t snorm = -1;
   Double_t close = 0;                                 
   memset(&safe[0], 0, 3*sizeof(Double_t));
   memset(cldir, 0, 3*sizeof(Double_t));
   Int_t i, j;
   for (i=0; i<3; i++)
      dvert[i]=point[i]-vertex[i];
   for (i=0; i<3; i++) {
      for (j=0; j<3; j++)
         safe[i]+=dvert[j]*normals[3*i+j];
   }   
   // point is inside
   if (in) {
      snorm = TMath::LocMax(3, &safe[0]);
      close = -safe[snorm];
      // check if point was outside corner
      if (close<0) return kBig;
      memcpy(cldir, &normals[3*snorm], 3*sizeof(Double_t));
      return close;   
   }
   // point is outside
   UInt_t nout=0;
   for (i=0; i<3; i++) {
      if (safe[i]>0) { 
         snorm = i;
         close = safe[i];
         nout++;
      }   
   }   
   // check if point is actually inside the corner (no visible plane)
   if (!nout) return kBig;
   if (nout==1) {
      // only one visible plane
      memcpy(cldir, &normals[3*snorm], 3*sizeof(Double_t));
      return close;
   }   
   if (nout==2) {
   // two faces visible
      Double_t calf = 0;
      Double_t s1=0;
      Double_t s2=0;
      for (j=0; j<3; j++) {
         if (safe[j]>0) {
            if (s1==0) s1=safe[j];
            else       s2=safe[j];
            continue;
         }   
         for (Int_t k=0; k<3; k++) 
            calf += normals[3*((j+1)%3)+k]*normals[3*((j+2)%3)+k]; 
      }
      close=TMath::Sqrt((s1*s1 + s2*s2 + 2.*s1*s2*calf)/(1. - calf*calf));
      return close;
   }   
   
   if (nout==3) {
   // an edge or even vertex more close than any of the planes
   // recompute closest distance
      close=0;
      for (i=0; i<3; i++) {
         if (safe[i]>0) close+=dvert[i]*dvert[i];
      } 
      close = TMath::Sqrt(close);
      for (i=0; i<3; i++)
         cldir[i] = dvert[i]/close;
      return close;           
   }
   return close; // never happens
}   

//_____________________________________________________________________________
Double_t TGeoShape::DistToCorner(Double_t *point, Double_t *dir, Bool_t in, 
                                 Double_t *vertex, Double_t *norm, Int_t &inorm)
{
// Static method to compute distance along a direction from inside/outside point to a corner.
// The corner is  defined by its normals to planes n1, n2, n3, and its vertex. 
// Also compute distance to closest plane belonging to corner, normal to this plane and
// normal to shape at intersection point.

// iact=0 :
 
//   printf("checking corner : %f %f %f\n", vertex[0], vertex[1], vertex[2]);
//   printf("normx : %f %f %f\n", norm[0], norm[1], norm[2]);
//   printf("normy : %f %f %f\n", norm[3], norm[4], norm[5]);
//   printf("normz : %f %f %f\n", norm[6], norm[7], norm[8]);
   Double_t safe[3];  // closest distances to the 3 planes
   Double_t dist[3];  // distances from point to each of the 3 planes along direction
   Double_t dvert[3]; // vector from vertex to point
   Double_t cosa[3];  // cosines of anles between direction and each normal
   Double_t snxt = kBig;
   inorm = -1;
   memset(&safe[0], 0, 3*sizeof(Double_t));
   memset(&cosa[0], 0, 3*sizeof(Double_t));
   Int_t i, j;
   
   for (i=0; i<3; i++) {
      dvert[i]=point[i]-vertex[i];
      dist[i] = kBig;
   }   
//   printf("dvert : %f %f %f\n", dvert[0], dvert[1], dvert[2]);
   for (i=0; i<3; i++) {
      for (j=0; j<3; j++) {
         safe[i]+=dvert[j]*norm[3*i+j];
         cosa[i]+=dir[j]*norm[3*i+j];
      }   
   }   
   // point is inside
   if (in) {
      if (safe[0]>0) return kBig;
      if (safe[1]>0) return kBig;
      if (safe[2]>0) return kBig;
      for (i=0; i<3; i++) 
         if (cosa[i]>0) dist[i]=-safe[i]/cosa[i];
      inorm = TMath::LocMin(3, &dist[0]);
      snxt = dist[inorm];
      return snxt;
   }
   // point is outside
   UInt_t npos=0;
   UInt_t nout=0;
   UInt_t npp=0;
   Double_t dvirt = kBig;
   snxt = 0;
   
   for (i=0; i<3; i++) {
      if (safe[i]>0) nout++;
      if (cosa[i]!=0) 
         dist[i]=-safe[i]/cosa[i];
      if (dist[i] < 0) continue;   
      npos++;
      if (safe[i]>0) {
      // crossing with visible plane
         npp++;
         if (snxt<dist[i]) {
         // most distant intersection point is the real one
            inorm = i;
            snxt = dist[i];
         }
      } else {
      // crossing with invisible plane      
            // compute distance to closest virtual intersection
            dvirt=TMath::Min(dvirt, dist[i]);
      }   
   }
//   printf("  safe : %f %f %f nout=%i\n", safe[0], safe[1], safe[2], nout);
//   printf("  dist : %f %f %f\n", dist[0], dist[1], dist[2]);
//   printf("  dist to next : %f\n", snxt);
//   printf("  closest virtual : %f\n", dvirt); 
//   printf("  inorm=%i snorm=%i\n", inorm, snorm);
//   printf("  nout=%i npos=%i npp=%i\n", nout, npos, npp);
   // select distance to closest plane
   if (!nout) {
   // point is actually inside the corner (no visible plane)
      inorm = -1;
      return kBig;
   }
   if (nout==1) {
   // only one face visible
      if (npp!=1 || snxt>dvirt)  {
         inorm = -1;
         return kBig;
      }
      return snxt;
   }      
   if (!npos) {
   // ray does not intersect any plane
      inorm = -1;
      return kBig;
   }   
   if (npp!=nout) {
   // ray ray does not intersect all visible faces
      inorm = -1;
      return kBig;
   }   
   if (snxt>dvirt) {
   // intersection with invisible plane closer than with real one -> no real intersection
//      close=kBig;
      inorm = -1;
      return kBig;
   }   
   return snxt;
}

//_____________________________________________________________________________
Int_t TGeoShape::GetVertexNumber(Bool_t vx, Bool_t vy, Bool_t vz)
{
// get visible vertex number for : box, trd1, trd2, trap, gtra, para shapes   
   Int_t imin, imax;
   if (!vz) {
      imin = 0;
      imax = 3;
   } else {
      imin = 4;
      imax = 7;
   }   
   if (!vx)
      imax=imin+1;
   else
      imin = imax-1;
   if(!vy) {
      if (!vx) return imin;
      return imax;
   }
   if (!vx) return imax;
   return imin;
}               

//_____________________________________________________________________________
Double_t TGeoShape::SafetyPhi(Double_t *point, Bool_t in, Double_t c1, Double_t s1, Double_t c2, Double_t s2)
{
// Static method to compute safety w.r.t a phi corner defined by cosines/sines
// of the angles phi1, phi2.
   Double_t saf1 = kBig;
   Double_t saf2 = kBig;
   if (point[0]*c1+point[1]*s1 >= 0) saf1 = -point[0]*s1 + point[1]*c1;
   if (point[0]*c2+point[1]*s2 >= 0) saf2 =  point[0]*s2 - point[1]*c2;
   if (in) {
      if (saf1<0) saf1=kBig;
      if (saf2<0) saf2=kBig;
      return TMath::Min(saf1,saf2);
   }
   if (saf1<0 && saf2<0) return TMath::Max(saf1,saf2);
   return TMath::Min(saf1,saf2);
}        

