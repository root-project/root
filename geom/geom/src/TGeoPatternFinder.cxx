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

#include "TGeoPatternFinder.h"

#include "Riostream.h"
#include "TObject.h"
#include "TThread.h"
#include "TGeoMatrix.h"
#include "TGeoPara.h"
#include "TGeoArb8.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TMath.h"

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
   

//______________________________________________________________________________
TGeoPatternFinder::ThreadData_t::ThreadData_t() :
   fMatrix(0), fCurrent(-1), fNextIndex(-1)
{
   // Constructor.
}

//______________________________________________________________________________
TGeoPatternFinder::ThreadData_t::~ThreadData_t()
{
   // Destructor.

   if (fMatrix != gGeoIdentity) delete fMatrix;
}

//______________________________________________________________________________
TGeoPatternFinder::ThreadData_t& TGeoPatternFinder::GetThreadData() const
{
   Int_t tid = TGeoManager::ThreadId();
   TThread::Lock();
   if (tid >= fThreadSize)
   {
      fThreadData.resize(tid + 1);
      fThreadSize = tid + 1;
   }
   if (fThreadData[tid] == 0)
   {
      fThreadData[tid] = new ThreadData_t;
      fThreadData[tid]->fMatrix = CreateMatrix();
   }
   TThread::UnLock();
   return *fThreadData[tid];
}

//______________________________________________________________________________
void TGeoPatternFinder::ClearThreadData() const
{
   std::vector<ThreadData_t*>::iterator i = fThreadData.begin();
   while (i != fThreadData.end())
   {
      delete *i;
      ++i;
   }
   fThreadData.clear();
   fThreadSize = 0;
}

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder()
{
// Default constructor
   fNdivisions = 0;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
   fVolume     = 0;
   fThreadSize = 0;
}

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder(TGeoVolume *vol, Int_t ndiv)
{
// Default constructor
   fVolume     = vol;
   fNdivisions = ndiv;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
   fThreadSize = 0;
}

//_____________________________________________________________________________
TGeoPatternFinder::TGeoPatternFinder(const TGeoPatternFinder& pf) :
  TObject(pf),
  fStep(pf.fStep),
  fStart(pf.fStart),
  fEnd(pf.fEnd),
  fNdivisions(pf.fNdivisions),
  fDivIndex(pf.fDivIndex),
  fVolume(pf.fVolume)
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
      fNdivisions=pf.fNdivisions;
      fDivIndex=pf.fDivIndex;
      fVolume=pf.fVolume;
   } 
   return *this;
}

//_____________________________________________________________________________
TGeoPatternFinder::~TGeoPatternFinder()
{
// Destructor
   ClearThreadData();
}

//______________________________________________________________________________
Int_t TGeoPatternFinder::GetCurrent()
{
   // Return current index.
   return GetThreadData().fCurrent;
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternFinder::GetMatrix()
{
   // Return current matrix.
   return GetThreadData().fMatrix;
}

//______________________________________________________________________________
Int_t TGeoPatternFinder::GetNext() const
{
   // Get index of next division.
   return GetThreadData().fNextIndex;
}

//______________________________________________________________________________
void TGeoPatternFinder::SetNext(Int_t index)
{
   // Set index of next division.
   GetThreadData().fNextIndex = index;
}

//______________________________________________________________________________
TGeoNode *TGeoPatternFinder::CdNext()
{
// Make next node (if any) current.
   ThreadData_t& td = GetThreadData();
   if (td.fNextIndex < 0) return NULL;
   cd(td.fNextIndex);
   return GetNodeOffset(td.fCurrent);
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
}

//_____________________________________________________________________________
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}

//_____________________________________________________________________________
TGeoPatternX::~TGeoPatternX()
{
// Destructor
}

//_____________________________________________________________________________
void TGeoPatternX::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
   td.fMatrix->SetDx(fStart+idiv*fStep+0.5*fStep);
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternX::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[0]-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      if (dir[0]>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   ThreadData_t& td = GetThreadData();
   indnext = -1;
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[0])<TGeoShape::Tolerance()) return dist;
   if (td.fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[0]>0)?1:0;
   dist = (fStep*(td.fCurrent+inc)-point[0])/dir[0];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = td.fCurrent+inc;
   return dist;   
}   

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternX::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternX *finder = new TGeoPatternX(*this);
   if (!reflect) return finder;
   finder->Reflect();
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
}

//_____________________________________________________________________________
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
   td.fMatrix->SetDy(fStart+idiv*fStep+0.5*fStep);
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternY::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[1]-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      if (dir[1]>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   ThreadData_t& td = GetThreadData();
   indnext = -1;
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[1])<TGeoShape::Tolerance()) return dist;
   if (td.fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[1]>0)?1:0;
   dist = (fStep*(td.fCurrent+inc)-point[1])/dir[1];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = td.fCurrent+inc;
   return dist;   
}   
   
//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternY::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternY *finder = new TGeoPatternY(*this);
   if (!reflect) return finder;
   finder->Reflect();
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
}
//_____________________________________________________________________________
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
   td.fMatrix->SetDz(((IsReflected())?-1.:1.)*(fStart+idiv*fStep+0.5*fStep));
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternZ::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Int_t ind = (Int_t)(1.+(point[2]-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      if (dir[2]>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   ThreadData_t& td = GetThreadData();
   Double_t dist = TGeoShape::Big();
   if (TMath::Abs(dir[2])<TGeoShape::Tolerance()) return dist;
   if (td.fCurrent<0) {
      Error("FindNextBoundary", "Must call FindNode first");
      return dist;
   }   
   Int_t inc = (dir[2]>0)?1:0;
   dist = (fStep*(td.fCurrent+inc)-point[2])/dir[2];
   if (dist<0.) Error("FindNextBoundary", "Negative distance d=%g",dist);
   if (!inc) inc = -1;
   indnext = td.fCurrent+inc;
   return dist;   
}   

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternZ::MakeCopy(Bool_t reflect)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternZ *finder = new TGeoPatternZ(*this);
   if (!reflect) return finder;
   finder->Reflect();
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
}
//_____________________________________________________________________________
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
   td.fMatrix->SetDx(fStart+idiv*fStep+0.5*fStep);
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
   ThreadData_t& td = GetThreadData();
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
      td.fNextIndex = ind;
      if (dot>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
   return finder;
}
      
//______________________________________________________________________________
void TGeoPatternParaX::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternParaX::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent = idiv;
   Double_t dy = fStart+idiv*fStep+0.5*fStep;
   td.fMatrix->SetDx(fTxy*dy);
   td.fMatrix->SetDy(dy);
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t yt = point[1]-tyz*point[2];
   Int_t ind = (Int_t)(1.+(yt-fStart)/fStep) - 1;
   if (dir) {
      Double_t divdiry = 1./TMath::Sqrt(1.+tyz*tyz);
      Double_t divdirz = -tyz*divdiry;
      Double_t dot = dir[1]*divdiry + dir[2]*divdirz;
      td.fNextIndex = ind;
      if (dot>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
   return finder;
}
         
//______________________________________________________________________________
void TGeoPatternParaY::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternParaY::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   td.fMatrix->SetDx(fTxz*dz);
   td.fMatrix->SetDy(fTyz*dz);
   td.fMatrix->SetDz((IsReflected())?-dz:dz);
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)(1.+(zt-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      if (dir[2]>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
   return finder;
}
         
//______________________________________________________________________________
void TGeoPatternParaZ::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternParaZ::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+0.5*fStep;
   td.fMatrix->SetDx(fTxz*dz);
   td.fMatrix->SetDy(fTyz*dz);
   td.fMatrix->SetDz((IsReflected())?-dz:dz);
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)(1. + (zt-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      if (dir[2]>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternTrapZ::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternTrapZ::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoTranslation(0.,0.,0.);
   TGeoCombiTrans *combi = new TGeoCombiTrans();
   combi->ReflectZ(kTRUE);
   combi->ReflectZ(kFALSE);
   return combi;   
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
}
//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
}
//_____________________________________________________________________________
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
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
void TGeoPatternCylR::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
}

//_____________________________________________________________________________
TGeoNode *TGeoPatternCylR::FindNode(Double_t *point, const Double_t *dir)
{
// find the node containing the query point
   ThreadData_t& td = GetThreadData();
   if (!td.fMatrix) td.fMatrix = gGeoIdentity;
   TGeoNode *node = 0;
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Int_t ind = (Int_t)(1. + (r-fStart)/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      Double_t dot = point[0]*dir[0] + point[1]*dir[1];
      if (dot>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
   return finder;
}
      
//______________________________________________________________________________
void TGeoPatternCylR::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternCylR::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   return gGeoIdentity;
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
   ThreadData_t& td = GetThreadData();
   td.fCurrent = idiv;
   ((TGeoRotation*)td.fMatrix)->FastRotZ(&fSinCos[2*idiv]);
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
   ThreadData_t& td = GetThreadData();
   TGeoNode *node = 0;
   Double_t phi = TMath::ATan2(point[1], point[0])*TMath::RadToDeg();
   if (phi<0) phi += 360;
//   Double_t dphi = fStep*fNdivisions;
   Double_t ddp = phi - fStart;
   if (ddp<0) ddp+=360;
//   if (ddp>360) ddp-=360;
   Int_t ind = (Int_t)(1. + ddp/fStep) - 1;
   if (dir) {
      td.fNextIndex = ind;
      Double_t dot = point[0]*dir[1]-point[1]*dir[0];
      if (dot>0) td.fNextIndex++;
      else td.fNextIndex--;
      if ((td.fNextIndex<0) || (td.fNextIndex>=fNdivisions)) td.fNextIndex = -1;
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
   finder->Reflect();
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

//______________________________________________________________________________
TGeoMatrix* TGeoPatternCylPhi::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   if (!IsReflected()) return new TGeoRotation();
   TGeoRotation *rot = new TGeoRotation();
   rot->ReflectZ(kTRUE);
   rot->ReflectZ(kFALSE);
   return rot;   
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
void TGeoPatternSphR::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent = idiv;
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphR::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphR::MakeCopy(Bool_t)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphR *finder = new TGeoPatternSphR(*this);
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphR::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 1;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternSphR::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   return gGeoIdentity;
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
void TGeoPatternSphTheta::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphTheta::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphTheta::MakeCopy(Bool_t)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphTheta *finder = new TGeoPatternSphTheta(*this);
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphTheta::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 2;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternSphTheta::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   return gGeoIdentity;
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
void TGeoPatternSphPhi::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternSphPhi::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoPatternFinder *TGeoPatternSphPhi::MakeCopy(Bool_t)
{
// Make a copy of this finder. Reflect by Z if required.
   TGeoPatternSphPhi *finder = new TGeoPatternSphPhi(*this);
   return finder;
}
   
//______________________________________________________________________________
void TGeoPatternSphPhi::SavePrimitive(ostream &out, Option_t * /*option*/ /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".
   Int_t iaxis = 3;
   out << iaxis << ", " << fNdivisions << ", " << fStart << ", " << fStep; 
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternSphPhi::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   return gGeoIdentity;
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
void TGeoPatternHoneycomb::cd(Int_t idiv)
{
// Update current division index and global matrix to point to a given slice.
   ThreadData_t& td = GetThreadData();
   td.fCurrent=idiv; 
}
//_____________________________________________________________________________
TGeoNode *TGeoPatternHoneycomb::FindNode(Double_t * /*point*/, const Double_t * /*dir*/)
{
// find the node containing the query point
   return 0;
}

//______________________________________________________________________________
TGeoMatrix* TGeoPatternHoneycomb::CreateMatrix() const
{
   // Return new matrix of type used by  this finder.
   return gGeoIdentity;
}

//_____________________________________________________________________________
void TGeoPatternHoneycomb::UpdateMatrix(Int_t, TGeoHMatrix &matrix) const
{
// Fills external matrix with the local one corresponding to the given division
// index.
   matrix.Clear();
}   
