// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   30/10/01

/*************************************************************************
 * TGeoPatternFinder - base finder class for patterns. A pattern is specifying 
 *   a division type
 *************************************************************************/

#include "TObject.h"
#include "TGeoMatrix.h"
#include "TGeoPara.h"
#include "TGeoArb8.h"
#include "TGeoNode.h"
#include "TGeoManager.h"

#include "TGeoPatternFinder.h"

ClassImp(TGeoPatternFinder)
   

//-----------------------------------------------------------------------------
TGeoPatternFinder::TGeoPatternFinder()
{
// Default constructor
   fBasicCell  = 0;
   fMatrix     = 0;
   fCurrent    = 0;
   fNdivisions = 0;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternFinder::TGeoPatternFinder(TGeoVolume *vol, Int_t ndiv)
                  :TGeoFinder(vol)
{
// Default constructor
   fBasicCell  = 0;
   fMatrix     = 0;
   fCurrent    = -1;
   fNdivisions = ndiv;
   fDivIndex   = 0;
   fStep       = 0;
   fStart      = 0;
   fEnd        = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternFinder::~TGeoPatternFinder()
{
// Destructor
//   if (fMatrix && (fMatrix!=gGeoIdentity)) delete fMatrix;
}
//-----------------------------------------------------------------------------
TGeoVolume *TGeoPatternFinder::GetBasicVolume() const
{
// returns the pointer to the volume corresponding to a given division
   return fBasicCell;
}

/*************************************************************************
 * TGeoPatternX - a X axis divison pattern
 *   
 *************************************************************************/
ClassImp(TGeoPatternX)


//-----------------------------------------------------------------------------
TGeoPatternX::TGeoPatternX()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoBBox*)vol->GetShape())->GetDX();
   fStart = -dx;
   fEnd = dx;
   fStep = 2*dx/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoBBox*)vol->GetShape())->GetDX();
   fStart = -dx;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternX::TGeoPatternX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternX::~TGeoPatternX()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternX::FindNode(Double_t *point)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Int_t ind = (Int_t)((point[0]-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternY - a Y axis divison pattern
 *   
 *************************************************************************/
ClassImp(TGeoPatternY)


//-----------------------------------------------------------------------------
TGeoPatternY::TGeoPatternY()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dy = ((TGeoBBox*)vol->GetShape())->GetDY();
   fStart = -dy;
   fEnd = dy;
   fStep = 2*dy/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dy = ((TGeoBBox*)vol->GetShape())->GetDY();
   fStart = -dy;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternY::TGeoPatternY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternY::~TGeoPatternY()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternY::FindNode(Double_t *point)
{
// find the node containing the query point
   TGeoNode *node = 0;
   Int_t ind = (Int_t)((point[1]-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternZ - a Z axis divison pattern
 *   
 *************************************************************************/
ClassImp(TGeoPatternZ)


//-----------------------------------------------------------------------------
TGeoPatternZ::TGeoPatternZ()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dz = ((TGeoBBox*)vol->GetShape())->GetDZ();
   fStart = -dz;
   fEnd = dz;
   fStep = 2*dz/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dz = ((TGeoBBox*)vol->GetShape())->GetDZ();
   fStart = -dz;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternZ::TGeoPatternZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternZ::~TGeoPatternZ()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternZ::FindNode(Double_t *point)
{
// find the node containing the query point
   TGeoNode *node = 0;
   Int_t ind = (Int_t)((point[2]-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternParaX - a X axis divison pattern for PARA shape
 *   
 *************************************************************************/
ClassImp(TGeoPatternParaX)


//-----------------------------------------------------------------------------
TGeoPatternParaX::TGeoPatternParaX()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoPara*)vol->GetShape())->GetX();
   fStart = -dx;
   fEnd = dx;
   fStep = 2*dx/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t dx = ((TGeoPara*)vol->GetShape())->GetX();
   fStart = -dx;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternParaX::TGeoPatternParaX(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternParaX::~TGeoPatternParaX()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternParaX::FindNode(Double_t *point)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t txy = ((TGeoPara*)fVolume->GetShape())->GetTxy();
   Double_t txz = ((TGeoPara*)fVolume->GetShape())->GetTxz();
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t xt = point[0]-txz*point[2]-txy*(point[1]-tyz*point[2]);
   Int_t ind = (Int_t)((xt-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternParaY - a Y axis divison pattern for PARA shape
 *   
 *************************************************************************/
ClassImp(TGeoPatternParaY)


//-----------------------------------------------------------------------------
TGeoPatternParaY::TGeoPatternParaY()
{
// Default constructor
   fTxy = 0;
}
//-----------------------------------------------------------------------------
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
}
//-----------------------------------------------------------------------------
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
}
//-----------------------------------------------------------------------------
TGeoPatternParaY::TGeoPatternParaY(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fTxy = ((TGeoPara*)vol->GetShape())->GetTxy();
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternParaY::~TGeoPatternParaY()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoPatternParaY::cd(Int_t idiv)
{
   fCurrent = idiv;
   Double_t dy = fStart+idiv*fStep+fStep/2;
   ((TGeoTranslation*)fMatrix)->SetDx(fTxy*dy);
   ((TGeoTranslation*)fMatrix)->SetDy(dy);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternParaY::FindNode(Double_t *point)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t tyz = ((TGeoPara*)fVolume->GetShape())->GetTyz();
   Double_t yt = point[1]-tyz*point[2];
   Int_t ind = (Int_t)((yt-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternParaZ - a Z axis divison pattern for PARA shape
 *   
 *************************************************************************/
ClassImp(TGeoPatternParaZ)


//-----------------------------------------------------------------------------
TGeoPatternParaZ::TGeoPatternParaZ()
{
// Default constructor
   fTxz = 0;
   fTyz = 0;
}
//-----------------------------------------------------------------------------
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
}
//-----------------------------------------------------------------------------
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
}
//-----------------------------------------------------------------------------
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
}
//-----------------------------------------------------------------------------
TGeoPatternParaZ::~TGeoPatternParaZ()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoPatternParaZ::cd(Int_t idiv)
{
   fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+fStep/2;
   ((TGeoTranslation*)fMatrix)->SetDx(fTxz*dz);
   ((TGeoTranslation*)fMatrix)->SetDy(fTyz*dz);
   ((TGeoTranslation*)fMatrix)->SetDz(dz);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternParaZ::FindNode(Double_t *point)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)((zt-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternTrapZ - a Z axis divison pattern for TRAP or GTRA shapes
 *   
 *************************************************************************/

ClassImp(TGeoPatternTrapZ)

//-----------------------------------------------------------------------------
TGeoPatternTrapZ::TGeoPatternTrapZ()
{
// Default constructor
   fTxz = 0;
   fTyz = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Cos(phi*TGeoShape::kDegRad);
   fTyz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Sin(phi*TGeoShape::kDegRad);
   Double_t dz = ((TGeoArb8*)vol->GetShape())->GetDz();
   fStart = -dz;
   fEnd = dz;
   fStep = 2*dz/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t step)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Cos(phi*TGeoShape::kDegRad);
   fTyz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Sin(phi*TGeoShape::kDegRad);
   Double_t dz = ((TGeoArb8*)vol->GetShape())->GetDz();
   fStart = -dz;
   fEnd = fStart + ndivisions*step;
   fStep       = step;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternTrapZ::TGeoPatternTrapZ(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
             :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   Double_t theta = ((TGeoTrap*)vol->GetShape())->GetTheta();
   Double_t phi   = ((TGeoTrap*)vol->GetShape())->GetPhi();
   fTxz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Cos(phi*TGeoShape::kDegRad);
   fTyz = TMath::Tan(theta*TGeoShape::kDegRad)*TMath::Sin(phi*TGeoShape::kDegRad);
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = new TGeoTranslation(0,0,0);
}
//-----------------------------------------------------------------------------
TGeoPatternTrapZ::~TGeoPatternTrapZ()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoPatternTrapZ::cd(Int_t idiv)
{
   fCurrent = idiv;
   Double_t dz = fStart+idiv*fStep+fStep/2;
   ((TGeoTranslation*)fMatrix)->SetDx(fTxz*dz);
   ((TGeoTranslation*)fMatrix)->SetDy(fTyz*dz);
   ((TGeoTranslation*)fMatrix)->SetDz(dz);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternTrapZ::FindNode(Double_t *point)
{
// get the node division containing the query point
   TGeoNode *node = 0;
   Double_t zt = point[2];
   Int_t ind = (Int_t)((zt-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}


/*************************************************************************
 * TGeoPatternCylR - a cylindrical R divison pattern
 *   
 *************************************************************************/
 
ClassImp(TGeoPatternCylR)


//-----------------------------------------------------------------------------
TGeoPatternCylR::TGeoPatternCylR()
{
// Default constructor
   fMatrix = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fMatrix     = gGeoIdentity;
// compute step, start, end
}
//-----------------------------------------------------------------------------
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
   fMatrix     = gGeoIdentity;
// compute start, end
}
//-----------------------------------------------------------------------------
TGeoPatternCylR::TGeoPatternCylR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
   fMatrix     = gGeoIdentity;
}
//-----------------------------------------------------------------------------
TGeoPatternCylR::~TGeoPatternCylR()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternCylR::FindNode(Double_t *point)
{
// find the node containing the query point
   TGeoNode *node = 0;
   Double_t r = TMath::Sqrt(point[0]*point[0]+point[1]*point[1]);
   Int_t ind = (Int_t)((r-fStart+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}
/*************************************************************************
 * TGeoPatternCylPhi - a cylindrical phi divison pattern
 *   
 *************************************************************************/
 
ClassImp(TGeoPatternCylPhi)


//-----------------------------------------------------------------------------
TGeoPatternCylPhi::TGeoPatternCylPhi()
{
// Default constructor
   fSinCos = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternCylPhi::TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
   fStart = 0;
   fEnd = 0;
   fStep = 0;
   fMatrix = 0;
   fSinCos = 0;
}
//-----------------------------------------------------------------------------
TGeoPatternCylPhi::TGeoPatternCylPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
   fSinCos = 0;
// compute start, end
}
//-----------------------------------------------------------------------------
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
   fMatrix     = new TGeoRotation("rotdiv");
   fSinCos     = new Double_t[2*ndivisions];
   for (Int_t idiv = 0; idiv<ndivisions; idiv++) {
      fSinCos[2*idiv] = TMath::Sin(TGeoShape::kDegRad*(start+fStep/2+idiv*fStep));
      fSinCos[2*idiv+1] = TMath::Cos(TGeoShape::kDegRad*(start+fStep/2+idiv*fStep));
   }
}
//-----------------------------------------------------------------------------
TGeoPatternCylPhi::~TGeoPatternCylPhi()
{
// Destructor
   if (fMatrix) delete fMatrix;
   if (fSinCos) delete fSinCos;
}
//-----------------------------------------------------------------------------
void TGeoPatternCylPhi::cd(Int_t idiv)
{
   fCurrent = idiv;
   ((TGeoRotation*)fMatrix)->FastRotZ(&fSinCos[2*idiv]);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternCylPhi::FindNode(Double_t *point)
{
// find the node containing the query point
   TGeoNode *node = 0;
   Double_t phi = TMath::ATan2(point[1], point[0])*TGeoShape::kRadDeg;
   if (phi<0) phi += 360;
//   Double_t dphi = fStep*fNdivisions;
   Double_t ddp = phi - fStart;
   if (ddp<0) ddp+=360;
//   if (ddp>360) ddp-=360;
   Int_t ind = (Int_t)((ddp+fStep)/fStep)-1;
   if ((ind<0) || (ind>=fNdivisions)) return node; 
   node = GetNodeOffset(ind);
   cd(ind);
   return node;
}

/*************************************************************************
 * TGeoPatternSphR - a spherical R divison pattern
 *   
 *************************************************************************/


ClassImp(TGeoPatternSphR)


//-----------------------------------------------------------------------------
TGeoPatternSphR::TGeoPatternSphR()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphR::TGeoPatternSphR(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//-----------------------------------------------------------------------------
TGeoPatternSphR::~TGeoPatternSphR()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternSphR::FindNode(Double_t *point)
{
// find the node containing the query point
   return 0;
}

/*************************************************************************
 * TGeoPatternSphTheta - a spherical theta divison pattern
 *   
 *************************************************************************/


ClassImp(TGeoPatternSphTheta)


//-----------------------------------------------------------------------------
TGeoPatternSphTheta::TGeoPatternSphTheta()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphTheta::TGeoPatternSphTheta(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                    :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//-----------------------------------------------------------------------------
TGeoPatternSphTheta::~TGeoPatternSphTheta()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternSphTheta::FindNode(Double_t *point)
{
// find the node containing the query point
   return 0;
}

/*************************************************************************
 * TGeoPatternSphPhi - a spherical phi divison pattern
 *   
 *************************************************************************/


ClassImp(TGeoPatternSphPhi)


//-----------------------------------------------------------------------------
TGeoPatternSphPhi::TGeoPatternSphPhi()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
// compute step, start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t step)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStep       = step;
// compute start, end
}
//-----------------------------------------------------------------------------
TGeoPatternSphPhi::TGeoPatternSphPhi(TGeoVolume *vol, Int_t ndivisions, Double_t start, Double_t end)
                  :TGeoPatternFinder(vol, ndivisions)
{   
// constructor
   fStart      = start;
   fEnd        = end;
   fStep       = (end - start)/ndivisions;
}
//-----------------------------------------------------------------------------
TGeoPatternSphPhi::~TGeoPatternSphPhi()
{
// Destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternSphPhi::FindNode(Double_t *point)
{
// find the node containing the query point
   return 0;
}

/*************************************************************************
 * TGeoPatternHoneycomb - a divison pattern specialized for honeycombs
 *   
 *************************************************************************/

ClassImp(TGeoPatternHoneycomb)
   

//-----------------------------------------------------------------------------
TGeoPatternHoneycomb::TGeoPatternHoneycomb()
{
// Default constructor
   fNrows       = 0;                 
   fAxisOnRows  = 0;            
   fNdivisions  = 0;             
   fStart       = 0;                 
}
//-----------------------------------------------------------------------------
TGeoPatternHoneycomb::TGeoPatternHoneycomb(TGeoVolume *vol, Int_t nrows)
                     :TGeoPatternFinder(vol, nrows)
{
// Default constructor
   fNrows = nrows;
// compute everything else
}
//-----------------------------------------------------------------------------
TGeoPatternHoneycomb::~TGeoPatternHoneycomb()
{
// destructor
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoPatternHoneycomb::FindNode(Double_t *point)
{
// find the node containing the query point
   return 0;
}
