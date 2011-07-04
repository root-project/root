// @(#)root/geom:$Id$
// Author: Andrei Gheata   30/10/01

//_____________________________________________________________________________
// TGeoPatternFinder - base finder class for patterns. 
//==================
//   A pattern is specifying a division type which applies only to a given
// shape type. The implemented patterns are for the moment equidistant slices
// on different axis. Implemented patterns are:
//
// TGeoPatternX - a X axis divison pattern
// TGeoPatternY - a Y axis divison pattern
// TGeoPatternZ - a Z axis divison pattern
// TGeoPatternParaX - a X axis divison pattern for PARA shape
// TGeoPatternParaY - a Y axis divison pattern for PARA shape
// TGeoPatternParaZ - a Z axis divison pattern for PARA shape
// TGeoPatternTrapZ - a Z axis divison pattern for TRAP or GTRA shapes
// TGeoPatternCylR - a cylindrical R divison pattern
// TGeoPatternCylPhi - a cylindrical phi divison pattern
// TGeoPatternSphR - a spherical R divison pattern
// TGeoPatternSphTheta - a spherical theta divison pattern
// TGeoPatternSphPhi - a spherical phi divison pattern
// TGeoPatternHoneycomb - a divison pattern specialized for honeycombs
//_____________________________________________________________________________

#include "Riostream.h"
#include "TObject.h"
#include "TGeoMatrix.h"
#include "TGeoPara.h"
#include "TGeoArb8.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TMath.h"

#include "TGeoPatternFinder.h"

ClassImp(TGeoPatternFinder)
ClassImp(TGeoPatternX)
ClassImp(TGeoPatternY)
ClassImp(TGeoPatternZ)
ClassImp(TGeoPatternParaX)
ClassImp(TGeoPatternParaY)
ClassImp(TGeoPatternParaZ)
ClassImp(TGeoPatternTrapZ)
ClassImp(TGeoPatternCylR)
ClassImp(TGeoPatternCylPhi)
ClassImp(TGeoPatternSphR)
ClassImp(TGeoPatternSphTheta)
ClassImp(TGeoPatternSphPhi)
ClassImp(TGeoPatternHoneycomb)
   

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder()
{
// Default constructor
   fMatrix     = 0;
   fCurrent    = -1;
   fNdivisions = 0;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
   fVolume     = 0;
   fNextIndex = -1;
}

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder(TGeoVolume *vol, Int_t ndiv)
{
// Default constructor
   fVolume     = vol;
   fMatrix     = 0;
   fCurrent    = -1;
   fNdivisions = ndiv;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
   fNextIndex = -1;
}

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder(const TGeoPatternFinder& pf) :
  TObject(pf),
  fStep(pf.fStep),
  fStart(pf.fStart),
  fEnd(pf.fEnd),
  fCurrent(pf.fCurrent),
  fNdivisions(pf.fNdivisions),
  fDivIndex(pf.fDivIndex),
  fMatrix(pf.fMatrix),
  fVolume(pf.fVolume),
  fNextIndex(pf.fNextIndex)
{ 
   //copy constructor
}

//_____________________________________________________________________________
TGeoPatternFinder& TGeoPatternFinder::operator=(const TGeoPatternFinder& pf)
{
   //assignment operator
   if(this!=&pf) {
      TObject::operator=(pf);
      fStep=pf.fStep;
      fStart=pf.fStart;
      fEnd=pf.fEnd;
      fCurrent=pf.fCurrent;
      fNdivisions=pf.fNdivisions;
      fDivIndex=pf.fDivIndex;
      fMatrix=pf.fMatrix;
      fVolume=pf.fVolume;
      fNextIndex = pf.fNextIndex;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoPatternFinder::~TGeoPatternFinder()
{
// Destructor
}

//______________________________________________________________________________
TGeoNode *TGeoPatternFinder::CdNext()
{
// Make next node (if any) current.
   if (fNextIndex < 0) return NULL;
   cd(fNextIndex);
   return GetNodeOffset(fCurrent);
}   

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternFinder::MakeCopy(Bool_t)
{
// Make a copy of this finder. Has to be overwritten by derived classes.
   return NULL;
}

//______________________________________________________________________________
void TGeoPatternFinder::SetRange(Double_t start, Double_t step, Int_t ndivisions)
{
// Set division range. Use this method only when dividing an assembly.
   fStart = start;
   fEnd = fStart + ndivisions*step;
   fStep = step;
   fNdivisions = ndivisions;
}
   
//______________________________________________________________________________
// TGeoPatternX - a X axis divison pattern
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternX::TGeoPatternX()
{
// Default constructor
}

//_____________________________________________________________________________
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoBBox*)vol->GetShape())->GetDX();
   fStart = -dx;
   fEnd = dx;
   fStep = 2*dx/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoBBox*)vol->GetShape())->GetDX();
   fStart = -dx;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
void TGeoPatternX::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent=idiv; 
   fMatrix->SetDx(fStart+idiv*fStep+0.5*fStep);
}

//_____________________________________________________________________________
void TGeoPatternX::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   matrix.SetDx(fStart+idiv*fStep+0.5*fStep);
}   

//_____________________________________________________________________________
TGeoPatternX::~TGeoPatternX()
{
// Destructor
}

//_____________________________________________________________________________
Bool_t TGeoPatternX::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t seg = (point[0]-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternX::FindNode(Double_t *point, const Double_t *dir)
{
// Find the cell corresponding to point and next cell along dir (if asked)
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[0]-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      if (dir[0]>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
Double_t TGeoPatternX::FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext)
{
// Compute distance to next division layer returning the index of next section.
// Point is in the frame of the divided volume.
   indnext = -1;
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[0])<TGeoShape::Tolerance()) return dist;
   if (fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[0]>0)?1:0;
   dist = (fStep*(fCurrent+inc)-point[0])/dir[0];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = fCurrent+inc;
   return dist;   
}   

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternX::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternX *finder = new TGeoPatternX(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternX::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
// TGeoPatternY - a Y axis divison pattern
//______________________________________________________________________________


//_____________________________________________________________________________
TGeoPatternY::TGeoPatternY()
{
// Default constructor
}

//_____________________________________________________________________________
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dy = ((TGeoBBox*)vol->GetShape())->GetDY();
   fStart = -dy;
   fEnd = dy;
   fStep = 2*dy/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dy = ((TGeoBBox*)vol->GetShape())->GetDY();
   fStart = -dy;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternY::~TGeoPatternY()
{
// Destructor
}

//_____________________________________________________________________________
void TGeoPatternY::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent=idiv; 
   fMatrix->SetDy(fStart+idiv*fStep+0.5*fStep);
}

//_____________________________________________________________________________
void TGeoPatternY::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   matrix.SetDy(fStart+idiv*fStep+0.5*fStep);
}   

//_____________________________________________________________________________
Bool_t TGeoPatternY::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t seg = (point[1]-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternY::FindNode(Double_t *point, const Double_t *dir)
{
// Find the cell corresponding to point and next cell along dir (if asked)
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[1]-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      if (dir[1]>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
Double_t TGeoPatternY::FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext)
{
// Compute distance to next division layer returning the index of next section.
// Point is in the frame of the divided volume.
   indnext = -1;
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[1])<TGeoShape::Tolerance()) return dist;
   if (fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[1]>0)?1:0;
   dist = (fStep*(fCurrent+inc)-point[1])/dir[1];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = fCurrent+inc;
   return dist;   
}   
   
//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternY::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternY *finder = new TGeoPatternY(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternY::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep;
}

//______________________________________________________________________________
// TGeoPatternZ - a Z axis divison pattern
//______________________________________________________________________________


//_____________________________________________________________________________
TGeoPatternZ::TGeoPatternZ()
{
// Default constructor
}
//_____________________________________________________________________________
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dz = ((TGeoBBox*)vol->GetShape())->GetDZ();
   fStart = -dz;
   fEnd = dz;
   fStep = 2*dz/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dz = ((TGeoBBox*)vol->GetShape())->GetDZ();
   fStart = -dz;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternZ::~TGeoPatternZ()
{
// Destructor
}
//_____________________________________________________________________________
void TGeoPatternZ::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent=idiv; 
   fMatrix->SetDz(((IsReflected())?-1.:1.)*(fStart+idiv*fStep+0.5*fStep));
}

//_____________________________________________________________________________
void TGeoPatternZ::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   matrix.SetDz(((IsReflected())?-1.:1.)*(fStart+idiv*fStep+0.5*fStep));
}   

//_____________________________________________________________________________
Bool_t TGeoPatternZ::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t seg = (point[2]-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternZ::FindNode(Double_t *point, const Double_t *dir)
{
// Find the cell corresponding to point and next cell along dir (if asked)
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[2]-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      if (dir[2]>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
Double_t TGeoPatternZ::FindNextBoundary(Double_t *point, Double_t *dir, Int_t &indnext)
{
// Compute distance to next division layer returning the index of next section.
// Point is in the frame of the divided volume.
   indnext = -1;
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[2])<TGeoShape::Tolerance()) return dist;
   if (fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[2]>0)?1:0;
   dist = (fStep*(fCurrent+inc)-point[2])/dir[2];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = fCurrent+inc;
   return dist;   
}   

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternZ::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternZ *finder = new TGeoPatternZ(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternZ::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
// TGeoPatternParaX - a X axis divison pattern for PARA shape
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternParaX::TGeoPatternParaX()
{
// Default constructor
}
//_____________________________________________________________________________
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoPara*)vol->GetShape())->GetX();
   fStart = -dx;
   fEnd = dx;
   fStep = 2*dx/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoPara*)vol->GetShape())->GetX();
   fStart = -dx;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaX::~TGeoPatternParaX()
{
// Destructor
}
//_____________________________________________________________________________
void TGeoPatternParaX::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent=idiv; 
   fMatrix->SetDx(fStart+idiv*fStep+0.5*fStep);
}

//_____________________________________________________________________________
Bool_t TGeoPatternParaX::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t txy = ((TGeoPara*)fVolume->GetShape())->GetTxy();
   Double_t txz = ((TGeoPara*)fVolume->GetShape())->GetTxz();
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t xt = point[0]-txz*point[2]-txy*(point[1]-tyz*point[2]);
   Double_t seg = (xt-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternParaX::FindNode(Double_t *point, const Double_t *dir)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t txy = ((TGeoPara*)fVolume->GetShape())->GetTxy();
   Double_t txz = ((TGeoPara*)fVolume->GetShape())->GetTxz();
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t xt = point[0]-txz*point[2]-txy*(point[1]-tyz*point[2]);
   Int_t ind = (Int_t)(1.+(xt-fStart)/fStep)-1;
   if (dir) {
      Double_t ttsq = txy*txy + (txz-txy*tyz)*(txz-txy*tyz);
      Double_t divdirx = 1./TMath::Sqrt(1.+ttsq);
      Double_t divdiry = -txy*divdirx;
      Double_t divdirz = -(txz-txy*tyz)*divdirx;
      Double_t dot = dir[0]*divdirx + dir[1]*divdiry + dir[2]*divdirz;
      fNextIndex = ind;
      if (dot>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternParaX::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternParaX *finder = new TGeoPatternParaX(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternParaX::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternParaX::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   matrix.SetDx(fStart+idiv*fStep+0.5*fStep);
}   

//______________________________________________________________________________
// TGeoPatternParaY - a Y axis divison pattern for PARA shape
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternParaY::TGeoPatternParaY()
{
// Default constructor
   fTxy = 0;
}
//_____________________________________________________________________________
TGeoPatternParaY::TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxy = ((TGeoPara*)vol->GetShape())->GetTxy();
   Double_t dy = ((TGeoPara*)vol->GetShape())->GetY();
   fStart = -dy;
   fEnd = dy;
   fStep = 2*dy/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaY::TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxy = ((TGeoPara*)vol->GetShape())->GetTxy();
   Double_t dy = ((TGeoPara*)vol->GetShape())->GetY();
   fStart = -dy;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaY::TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxy = ((TGeoPara*)vol->GetShape())->GetTxy();
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaY::~TGeoPatternParaY()
{
// Destructor
}
//_____________________________________________________________________________
void TGeoPatternParaY::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent = idiv;
   Double_t dy = fStart+idiv*fStep+0.5*fStep;
   fMatrix->SetDx(fTxy*dy);
   fMatrix->SetDy(dy);
}

//_____________________________________________________________________________
Bool_t TGeoPatternParaY::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t yt = point[1]-tyz*point[2];
   Double_t seg = (yt-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternParaY::FindNode(Double_t *point, const Double_t *dir)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t yt = point[1]-tyz*point[2];
   Int_t ind = (Int_t)(1.+(yt-fStart)/fStep) - 1;
   if (dir) {
      Double_t divdiry = 1./TMath::Sqrt(1.+tyz*tyz);
      Double_t divdirz = -tyz*divdiry;
      Double_t dot = dir[1]*divdiry + dir[2]*divdirz;
      fNextIndex = ind;
      if (dot>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternParaY::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternParaY *finder = new TGeoPatternParaY(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternParaY::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternParaY::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   Double_t dy = fStart+idiv*fStep+0.5*fStep;
   matrix.SetDx(fTxy*dy);
   matrix.SetDy(dy);
}   

//______________________________________________________________________________
// TGeoPatternParaZ - a Z axis divison pattern for PARA shape
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternParaZ::TGeoPatternParaZ()
{
// Default constructor
   fTxz = 0;
   fTyz = 0;
}
//_____________________________________________________________________________
TGeoPatternParaZ::TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxz = ((TGeoPara*)vol->GetShape())->GetTxz();
   fTyz = ((TGeoPara*)vol->GetShape())->GetTyz();
   Double_t dz = ((TGeoPara*)vol->GetShape())->GetZ();
   fStart = -dz;
   fEnd = dz;
   fStep = 2*dz/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternParaZ::TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxz = ((TGeoPara*)vol->GetShape())->GetTxz();
   fTyz = ((TGeoPara*)vol->GetShape())->GetTyz();
   Double_t dz = ((TGeoPara*)vol->GetShape())->GetZ();
   fStart = -dz;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternParaZ::TGeoPatternParaZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxz = ((TGeoPara*)vol->GetShape())->GetTxz();
   fTyz = ((TGeoPara*)vol->GetShape())->GetTyz();
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}

//_____________________________________________________________________________
TGeoPatternParaZ::~TGeoPatternParaZ()
{
// Destructor
}

//_____________________________________________________________________________
void TGeoPatternParaZ::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   fMatrix->SetDx(fTxz*dz);
   fMatrix->SetDy(fTyz*dz);
   fMatrix->SetDz((IsReflected())?-dz:dz);
}

//_____________________________________________________________________________
Bool_t TGeoPatternParaZ::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t seg = (point[2]-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternParaZ::FindNode(Double_t *point, const Double_t *dir)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)(1.+(zt-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      if (dir[2]>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternParaZ::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternParaZ *finder = new TGeoPatternParaZ(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternParaZ::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternParaZ::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   matrix.SetDx(fTxz*dz);
   matrix.SetDy(fTyz*dz);
   matrix.SetDz((IsReflected())?-dz:dz);
}   

//______________________________________________________________________________
// TGeoPatternTrapZ - a Z axis divison pattern for TRAP or GTRA shapes
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternTrapZ::TGeoPatternTrapZ()
{
// Default constructor
   fTxz = 0;
   fTyz = 0;
}
//_____________________________________________________________________________
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TMath::DegToRad())*TMath::Cos(phi*TMath::DegToRad());
   fTyz = TMath::Tan(theta*TMath::DegToRad())*TMath::Sin(phi*TMath::DegToRad());
   Double_t dz = ((TGeoArb8*)vol->GetShape())->GetDz();
   fStart = -dz;
   fEnd = dz;
   fStep = 2*dz/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TMath::DegToRad())*TMath::Cos(phi*TMath::DegToRad());
   fTyz = TMath::Tan(theta*TMath::DegToRad())*TMath::Sin(phi*TMath::DegToRad());
   Double_t dz = ((TGeoArb8*)vol->GetShape())->GetDz();
   fStart = -dz;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TMath::DegToRad())*TMath::Cos(phi*TMath::DegToRad());
   fTyz = TMath::Tan(theta*TMath::DegToRad())*TMath::Sin(phi*TMath::DegToRad());
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
   fMatrix->RegisterYourself();
}
//_____________________________________________________________________________
TGeoPatternTrapZ::~TGeoPatternTrapZ()
{
// Destructor
}
//_____________________________________________________________________________
void TGeoPatternTrapZ::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   fMatrix->SetDx(fTxz*dz);
   fMatrix->SetDy(fTyz*dz);
   fMatrix->SetDz((IsReflected())?-dz:dz);
}

//_____________________________________________________________________________
Bool_t TGeoPatternTrapZ::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t seg = (point[2]-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternTrapZ::FindNode(Double_t *point, const Double_t *dir)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)(1. + (zt-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      if (dir[2]>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternTrapZ::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternTrapZ *finder = new TGeoPatternTrapZ(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternTrapZ::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternTrapZ::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   matrix.SetDx(fTxz*dz);
   matrix.SetDy(fTyz*dz);
   matrix.SetDz((IsReflected())?-dz:dz);
}   

//______________________________________________________________________________
// TGeoPatternCylR - a cylindrical R divison pattern
//______________________________________________________________________________ 

//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR()
{
// Default constructor
   fMatrix = 0;
}
//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fMatrix     = gGeoIdentity;
// compute step, start, end
}
//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
   fMatrix     = gGeoIdentity;
// compute start, end
}
//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = gGeoIdentity;
}
//_____________________________________________________________________________
TGeoPatternCylR::~TGeoPatternCylR()
{
// Destructor
}

//_____________________________________________________________________________
Bool_t TGeoPatternCylR::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Double_t seg = (r-fStart)/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternCylR::FindNode(Double_t *point, const Double_t *dir)
{
// find the node containing the query point
   if (!fMatrix) fMatrix = gGeoIdentity;
   TGeoNode *node = 0;
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Int_t ind = (Int_t)(1. + (r-fStart)/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      Double_t dot = point[0]*dir[0] + point[1]*dir[1];
      if (dot>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternCylR::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternCylR *finder = new TGeoPatternCylR(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternCylR::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternCylR::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   

//______________________________________________________________________________
// TGeoPatternCylPhi - a cylindrical phi divison pattern
//______________________________________________________________________________ 

//_____________________________________________________________________________
TGeoPatternCylPhi::TGeoPatternCylPhi()
{
// Default constructor
   fSinCos = 0;
}
//_____________________________________________________________________________
TGeoPatternCylPhi::TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
   fStart = 0;
   fEnd = 0;
   fStep = 0;
   fMatrix = 0;
   fSinCos     = new Double_t[2*fNdivisions];
   for (Int_t i = 0; i<fNdivisions; i++) {
      fSinCos[2*i] = TMath::Sin(TMath::DegToRad()*(fStart+0.5*fStep+i*fStep));
      fSinCos[2*i+1] = TMath::Cos(TMath::DegToRad()*(fStart+0.5*fStep+i*fStep));
   }
}
//_____________________________________________________________________________
TGeoPatternCylPhi::TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
   fSinCos     = new Double_t[2*ndivisions];
   for (Int_t i = 0; i<fNdivisions; i++) {
      fSinCos[2*i] = TMath::Sin(TMath::DegToRad()*(fStart+0.5*fStep+i*fStep));
      fSinCos[2*i+1] = TMath::Cos(TMath::DegToRad()*(fStart+0.5*fStep+i*fStep));
   }
// compute start, end
}
//_____________________________________________________________________________
TGeoPatternCylPhi::TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   if (fStart<0) fStart+=360;
   fEnd        = end;
   if (fEnd<0) fEnd+=360;
   if ((end-start)<0)
      fStep       = (end-start+360)/ndivisions;
   else
      fStep       = (end-start)/ndivisions;
   fMatrix     = new TGeoRotation();
   fMatrix->RegisterYourself();
   fSinCos     = new Double_t[2*ndivisions];
   for (Int_t idiv = 0; idiv<ndivisions; idiv++) {
      fSinCos[2*idiv] = TMath::Sin(TMath::DegToRad()*(start+0.5*fStep+idiv*fStep));
      fSinCos[2*idiv+1] = TMath::Cos(TMath::DegToRad()*(start+0.5*fStep+idiv*fStep));
   }
}
//_____________________________________________________________________________
TGeoPatternCylPhi::~TGeoPatternCylPhi()
{
// Destructor
   if (fSinCos) delete [] fSinCos;
}
//_____________________________________________________________________________
void TGeoPatternCylPhi::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   fCurrent = idiv;
   ((TGeoRotation*)fMatrix)->FastRotZ(&fSinCos[2*idiv]);
}

//_____________________________________________________________________________
Bool_t TGeoPatternCylPhi::IsOnBoundary(const Double_t *point) const
{
// Checks if the current point is on division boundary
   Double_t phi = TMath::ATan2(point[1], point[0])*TMath::RadToDeg();
   if (phi<0) phi += 360;
   Double_t ddp = phi - fStart;
   if (ddp<0) ddp+=360;
   Double_t seg = ddp/fStep;
   Double_t diff = seg - Int_t(seg);
   if (diff>0.5) diff = 1.-diff;
   if (diff<1e-8) return kTRUE;
   return kFALSE;
}   

//_____________________________________________________________________________
TGeoNode *TGeoPatternCylPhi::FindNode(Double_t *point, const Double_t *dir)
{
// find the node containing the query point
   TGeoNode *node = 0;
   Double_t phi = TMath::ATan2(point[1], point[0])*TMath::RadToDeg();
   if (phi<0) phi += 360;
//   Double_t dphi = fStep*fNdivisions;
   Double_t ddp = phi - fStart;
   if (ddp<0) ddp+=360;
//   if (ddp>360) ddp-=360;
   Int_t ind = (Int_t)(1. + ddp/fStep) - 1;
   if (dir) {
      fNextIndex = ind;
      Double_t dot = point[0]*dir[1]-point[1]*dir[0];
      if (dot>0) fNextIndex++;
      else fNextIndex--;
      if ((fNextIndex<0) || (fNextIndex>=fNdivisions)) fNextIndex = -1;
   }   
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternCylPhi::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternCylPhi *finder = new TGeoPatternCylPhi(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoRotation *rot = new TGeoRotation(*fMatrix);
   rot->ReflectZ(kTRUE);
   rot->ReflectZ(kFALSE);
   rot->RegisterYourself();
   fMatrix = rot;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternCylPhi::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternCylPhi::Streamer(TBuffer &R__b)
{
   // Stream an object of class TGeoVolume.
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(TGeoPatternCylPhi::Class(), this);
      if (fNdivisions) {
         fSinCos     = new Double_t[2*fNdivisions];
         for (Int_t idiv = 0; idiv<fNdivisions; idiv++) {
            fSinCos[2*idiv] = TMath::Sin(TMath::DegToRad()*(fStart+0.5*fStep+idiv*fStep));
            fSinCos[2*idiv+1] = TMath::Cos(TMath::DegToRad()*(fStart+0.5*fStep+idiv*fStep));
         }
      }
   } else {
      R__b.WriteClassBuffer(TGeoPatternCylPhi::Class(), this);
   }
}

//_____________________________________________________________________________
void TGeoPatternCylPhi::UpdateMatrix(Int_t idiv, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
   matrix.FastRotZ(&fSinCos[2*idiv]);
}   

//______________________________________________________________________________
// TGeoPatternSphR - a spherical R divison pattern
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternSphR::TGeoPatternSphR()
{
// Default constructor
}
//_____________________________________________________________________________
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//_____________________________________________________________________________
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//_____________________________________________________________________________
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//_____________________________________________________________________________
TGeoPatternSphR::~TGeoPatternSphR()
{
// Destructor
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphR::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphR::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphR *finder = new TGeoPatternSphR(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphR::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternSphR::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   

//______________________________________________________________________________
// TGeoPatternSphTheta - a spherical theta divison pattern
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternSphTheta::TGeoPatternSphTheta()
{
// Default constructor
}
//_____________________________________________________________________________
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//_____________________________________________________________________________
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//_____________________________________________________________________________
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//_____________________________________________________________________________
TGeoPatternSphTheta::~TGeoPatternSphTheta()
{
// Destructor
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphTheta::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphTheta::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphTheta *finder = new TGeoPatternSphTheta(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphTheta::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternSphTheta::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   

//______________________________________________________________________________
// TGeoPatternSphPhi - a spherical phi divison pattern
//______________________________________________________________________________

//_____________________________________________________________________________
TGeoPatternSphPhi::TGeoPatternSphPhi()
{
// Default constructor
}
//_____________________________________________________________________________
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//_____________________________________________________________________________
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//_____________________________________________________________________________
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//_____________________________________________________________________________
TGeoPatternSphPhi::~TGeoPatternSphPhi()
{
// Destructor
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphPhi::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphPhi::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphPhi *finder = new TGeoPatternSphPhi(*this);
   if (!reflect) return finder;
   Reflect();
   TGeoCombiTrans *combi = new TGeoCombiTrans(*fMatrix);
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   combi->RegisterYourself();
   fMatrix = combi;
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphPhi::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//_____________________________________________________________________________
void TGeoPatternSphPhi::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   

//______________________________________________________________________________
// TGeoPatternHoneycomb - a divison pattern specialized for honeycombs
//______________________________________________________________________________   

//_____________________________________________________________________________
TGeoPatternHoneycomb::TGeoPatternHoneycomb()
{
// Default constructor
   fNrows       = 0;                 
   fAxisOnRows  = 0;            
   fNdivisions  = 0;             
   fStart       = 0;                 
}
//_____________________________________________________________________________
TGeoPatternHoneycomb::TGeoPatternHoneycomb(TGeoVolume *vol, Int_t nrows)
                     :TGeoPatternFinder(vol, nrows)
{
// Default constructor
   fNrows = nrows;
   fAxisOnRows  = 0;            
   fNdivisions  = 0;             
   fStart       = 0;                 
// compute everything else
}
//_____________________________________________________________________________
TGeoPatternHoneycomb::TGeoPatternHoneycomb(const TGeoPatternHoneycomb& pfh) :
  TGeoPatternFinder(pfh),
  fNrows(pfh.fNrows),
  fAxisOnRows(pfh.fAxisOnRows),
  fNdivisions(pfh.fNdivisions),
  fStart(pfh.fStart)
{ 
   //copy constructor
}
//_____________________________________________________________________________
TGeoPatternHoneycomb& TGeoPatternHoneycomb::operator=(const TGeoPatternHoneycomb& pfh) 
{
   //assignment operator
   if(this!=&pfh) {
      TGeoPatternFinder::operator=(pfh);
      fNrows=pfh.fNrows;
      fAxisOnRows=pfh.fAxisOnRows;
      fNdivisions=pfh.fNdivisions;
      fStart=pfh.fStart;
   } 
   return *this;
}
//_____________________________________________________________________________
TGeoPatternHoneycomb::~TGeoPatternHoneycomb()
{
// destructor
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternHoneycomb::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//_____________________________________________________________________________
void TGeoPatternHoneycomb::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   
