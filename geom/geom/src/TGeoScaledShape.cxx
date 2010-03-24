// @(#)root/geom:$Id$
// Author: Andrei Gheata   26/09/05

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "Riostream.h"

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoScaledShape.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

//_____________________________________________________________________________
// TGeoScaledShape - A shape scaled by a TGeoScale transformation
//    
//Begin_Html
/*
<img src="gif/TGeoScaledShape.gif">
*/
//End_Html
//_____________________________________________________________________________


ClassImp(TGeoScaledShape)
   
//_____________________________________________________________________________
TGeoScaledShape::TGeoScaledShape()
{
// Default constructor
   fShape = 0;
   fScale = 0;
}   


//_____________________________________________________________________________
TGeoScaledShape::TGeoScaledShape(const char *name, TGeoShape *shape, TGeoScale *scale)
                :TGeoBBox(name,0,0,0)
{
// Constructor
   fShape = shape;
   fScale = scale;
   if (!fScale->IsRegistered()) fScale->RegisterYourself();
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoScaledShape::TGeoScaledShape(TGeoShape *shape, TGeoScale *scale)
{
// Constructor
   fShape = shape;
   fScale = scale;
   if (!fScale->IsRegistered()) fScale->RegisterYourself();
   ComputeBBox();
}

//_____________________________________________________________________________
TGeoScaledShape::~TGeoScaledShape()
{
// destructor
}

//_____________________________________________________________________________
Double_t TGeoScaledShape::Capacity() const
{
// Computes capacity of this shape [length^3]
   Double_t capacity = fShape->Capacity();
   const Double_t *scale = fScale->GetScale();
   capacity *= scale[0]*scale[1]*scale[2];
   return capacity;
}   
   
//_____________________________________________________________________________   
void TGeoScaledShape::ComputeBBox()
{
// Compute bounding box of the scaled shape
   if (!fShape) {
      Error("ComputeBBox", "Scaled shape %s without shape", GetName());
      return;
   } 
   TGeoBBox *box = (TGeoBBox*)fShape;
   const Double_t *orig = box->GetOrigin();
   Double_t point[3], master[3];
   point[0] = box->GetDX();
   point[1] = box->GetDY();
   point[2] = box->GetDZ();
   
   fScale->LocalToMaster(orig, fOrigin);
   fScale->LocalToMaster(point, master);
   fDX = TMath::Abs(master[0]);
   fDY = TMath::Abs(master[1]);
   fDZ = TMath::Abs(master[2]);
}   

//_____________________________________________________________________________   
void TGeoScaledShape::ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm)
{
// Compute normal to closest surface from POINT.
   Double_t local[3], ldir[3], lnorm[3];
   fScale->MasterToLocal(point,local);
   fScale->MasterToLocalVect(dir,ldir);
   TGeoMatrix::Normalize(ldir);
   fShape->ComputeNormal(local,ldir,lnorm);
//   fScale->LocalToMasterVect(lnorm, norm);
   fScale->MasterToLocalVect(lnorm, norm);
   TGeoMatrix::Normalize(norm);
}

//_____________________________________________________________________________
Bool_t TGeoScaledShape::Contains(Double_t *point) const
{
// Test if point is inside the scaled shape
   Double_t local[3];
   fScale->MasterToLocal(point,local);
   return fShape->Contains(local);
}

//_____________________________________________________________________________
Int_t TGeoScaledShape::DistancetoPrimitive(Int_t px, Int_t py)
{
// compute closest distance from point px,py to each vertex. Should not be called.
   Int_t n = fShape->GetNmeshVertices();
   return ShapeDistancetoPrimitive(n, px, py);
}

//_____________________________________________________________________________
Double_t TGeoScaledShape::DistFromInside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from inside point to surface of the scaled shape.
   Double_t local[3], ldir[3];
   Double_t lstep;
   fScale->MasterToLocal(point,local);
   lstep = fScale->MasterToLocal(step, dir);
   fScale->MasterToLocalVect(dir,ldir);
   TGeoMatrix::Normalize(ldir);
   Double_t dist = fShape->DistFromInside(local,ldir, iact, lstep, safe);
   if (iact<3 && safe) *safe = fScale->LocalToMaster(*safe);
   dist = fScale->LocalToMaster(dist, ldir);
   return dist;
}


//_____________________________________________________________________________
Double_t TGeoScaledShape::DistFromOutside(Double_t *point, Double_t *dir, Int_t iact, Double_t step, Double_t *safe) const
{
// Compute distance from outside point to surface of the scaled shape.
   Double_t local[3], ldir[3];
   Double_t lstep;
//   printf("DistFromOutside(%f,%f,%f,  %f,%f,%f)\n", point[0], point[1], point[2], dir[0], dir[1],dir[2]);
   fScale->MasterToLocal(point,local);
//   printf("local: %f,%f,%f\n", local[0],local[1], local[2]);
   lstep = fScale->MasterToLocal(step, dir);
   fScale->MasterToLocalVect(dir,ldir);
   TGeoMatrix::Normalize(ldir);
//   printf("localdir: %f,%f,%f\n",ldir[0],ldir[1],ldir[2]);
   Double_t dist = fShape->DistFromOutside(local,ldir, iact, lstep, safe);
//   printf("local distance: %f\n", dist);
   if (safe) *safe = fScale->LocalToMaster(*safe);
   dist = fScale->LocalToMaster(dist, ldir);
//   printf("converted distance: %f\n",dist);
   return dist;
}
   
//_____________________________________________________________________________
TGeoVolume *TGeoScaledShape::Divide(TGeoVolume * /*voldiv*/, const char *divname, Int_t /*iaxis*/, Int_t /*ndiv*/, 
                             Double_t /*start*/, Double_t /*step*/) 
{
// Cannot divide assemblies.
   Error("Divide", "Scaled shapes cannot be divided. Division volume %s not created", divname);
   return 0;
}   

//_____________________________________________________________________________
const TBuffer3D & TGeoScaledShape::GetBuffer3D(Int_t reqSections, Bool_t localFrame) const
{
// Fills a static 3D buffer and returns a reference.
   TBuffer3D &buffer = (TBuffer3D &)fShape->GetBuffer3D(reqSections, localFrame);

//   TGeoBBox::FillBuffer3D(buffer, reqSections, localFrame);
   Double_t halfLengths[3] = { fDX, fDY, fDZ };
   buffer.SetAABoundingBox(fOrigin, halfLengths);
   if (!buffer.fLocalFrame) {
      TransformPoints(buffer.fBBVertex[0], 8);
   }

   if ((reqSections & TBuffer3D::kRaw) && buffer.SectionsValid(TBuffer3D::kRawSizes)) {
      SetPoints(buffer.fPnts);
      if (!buffer.fLocalFrame) {
         TransformPoints(buffer.fPnts, buffer.NbPnts());
      }
   }
      
   return buffer;
}
//_____________________________________________________________________________
TGeoShape *TGeoScaledShape::GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const
{
// in case shape has some negative parameters, these has to be computed
// in order to fit the mother
   Error("GetMakeRuntimeShape", "Scaled shapes cannot be parametrized.");
   return NULL;
}

//_____________________________________________________________________________
void TGeoScaledShape::GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const
{
// Returns numbers of vertices, segments and polygons composing the shape mesh.
   fShape->GetMeshNumbers(nvert, nsegs, npols);
}
	
//_____________________________________________________________________________
void TGeoScaledShape::InspectShape() const
{
// print shape parameters
   printf("*** Shape %s: TGeoScaledShape ***\n", GetName());
   fScale->Print();
   fShape->InspectShape();
   TGeoBBox::InspectShape();
}

//_____________________________________________________________________________
Bool_t TGeoScaledShape::IsReflected() const
{
// Check if the scale transformation is a reflection.
   return fScale->IsReflection();
}

//_____________________________________________________________________________
TBuffer3D *TGeoScaledShape::MakeBuffer3D() const
{ 
   // Creates a TBuffer3D describing *this* shape.
   // Coordinates are in local reference frame.

   TBuffer3D *buff = fShape->MakeBuffer3D();
   if (buff) SetPoints(buff->fPnts);   
   return buff; 
}

//_____________________________________________________________________________
TGeoShape *TGeoScaledShape::MakeScaledShape(const char *name, TGeoShape *shape, TGeoScale *scale)
{
// Create a scaled shape starting from a non-scaled one.
   TGeoShape *new_shape;
   if (shape->IsA() == TGeoScaledShape::Class()) {
      TGeoScaledShape *sshape = (TGeoScaledShape*)shape;
      TGeoScale *old_scale = sshape->GetScale();
      TGeoShape *old_shape = sshape->GetShape();
      scale->SetScale(scale->GetScale()[0]*old_scale->GetScale()[0],
                      scale->GetScale()[1]*old_scale->GetScale()[1],
                      scale->GetScale()[2]*old_scale->GetScale()[2]);
      new_shape = new TGeoScaledShape(name, old_shape, scale);
      return new_shape;
   }   
   new_shape = new TGeoScaledShape(name, shape, scale);
   return new_shape;
}   

//_____________________________________________________________________________
void TGeoScaledShape::SetSegsAndPols(TBuffer3D &buff) const
{
// Fill TBuffer3D structure for segments and polygons.
   fShape->SetSegsAndPols(buff);
}

//_____________________________________________________________________________
Double_t TGeoScaledShape::Safety(Double_t *point, Bool_t in) const
{
// computes the closest distance from given point to this shape, according
// to option. The matching point on the shape is stored in spoint.
   Double_t local[3];
   fScale->MasterToLocal(point,local);
   Double_t safe = fShape->Safety(local,in);
   safe = fScale->LocalToMaster(safe);
   return safe;
}

//_____________________________________________________________________________
void TGeoScaledShape::SavePrimitive(ostream &out, Option_t *option)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TObject::TestBit(kGeoSavePrimitive)) return;
   out << "   // Shape: " << GetName() << " type: " << ClassName() << endl;
   if (!fShape || !fScale) {
      out << "##### Invalid shape or scale !. Aborting. #####" << endl;
      return;
   }
   fShape->SavePrimitive(out, option);
   TString sname = fShape->GetPointerName();
   const Double_t *sc = fScale->GetScale();
   out << "   // Scale factor:" << endl;
   out << "   TGeoScale *pScale = new TGeoScale(\"" << fScale->GetName() 
       << "\"," << sc[0] << "," << sc[1] << "," << sc[2] << ");" << endl;
   out << "   TGeoScaledShape *" << GetPointerName() << " = new TGeoScaledShape(\"" 
       << GetName() << "\"," << sname << ", pScale);" << endl;
}

//_____________________________________________________________________________
void TGeoScaledShape::SetPoints(Double_t *points) const
{
// Mesh points for scaled shapes.
   Int_t npts = fShape->GetNmeshVertices();
   fShape->SetPoints(points);
   Double_t master[3];
   for (Int_t i=0; i<npts; i++) {
      fScale->LocalToMaster(&points[3*i], master);
      memcpy(&points[3*i], master, 3*sizeof(Double_t));
   }   
}

//_____________________________________________________________________________
void TGeoScaledShape::SetPoints(Float_t *points) const
{
// Mesh points for scaled shapes.
   Int_t npts = fShape->GetNmeshVertices();
   fShape->SetPoints(points);
   Double_t master[3];
   Double_t local[3];
   Int_t index;
   for (Int_t i=0; i<npts; i++) {
      index = 3*i;
      local[0] = points[index];
      local[1] = points[index+1];
      local[2] = points[index+2];
      fScale->LocalToMaster(local, master);
      points[index] = master[0];
      points[index+1] = master[1];
      points[index+2] = master[2];
   }   
}
