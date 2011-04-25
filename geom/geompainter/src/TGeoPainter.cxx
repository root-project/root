// @(#)root/geompainter:$Id$
// Author: Andrei Gheata   05/03/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
// TGeoPainter - class implementing all draw interfaces for a generic 3D viewer
// using TBuffer3D mechanism.
//______________________________________________________________________________

#include "TROOT.h"
#include "TClass.h"
#include "TColor.h"
#include "TPoint.h"
#include "TView.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TF1.h"
#include "TPluginManager.h"
#include "TVirtualPadEditor.h"
#include "TStopwatch.h"

#include "TPolyMarker3D.h"
#include "TVirtualGL.h"

#include "TGeoAtt.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoTrack.h"
#include "TGeoOverlap.h"
#include "TGeoChecker.h"
#include "TGeoPhysicalNode.h"
#include "TGeoCompositeShape.h"
#include "TGeoShapeAssembly.h"
#include "TGeoPainter.h"
#include "TMath.h"

#include "X3DBuffer.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualViewer3D.h"

ClassImp(TGeoPainter)

//______________________________________________________________________________
TGeoPainter::TGeoPainter(TGeoManager *manager) : TVirtualGeoPainter(manager)
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default constructor*-*-*-*-*-*-*-*-*
//*-*                  ====================================
   TVirtualGeoPainter::SetPainter(this);
   if (manager) fGeoManager = manager;
   else {
      Error("ctor", "No geometry loaded");
      return;
   }   
   fNsegments = fGeoManager->GetNsegments();
   fNVisNodes = 0;
   fBombX = 1.3;
   fBombY = 1.3;
   fBombZ = 1.3;
   fBombR = 1.3;
   fVisLevel = fGeoManager->GetVisLevel();
   fVisOption = fGeoManager->GetVisOption();
   fExplodedView = fGeoManager->GetBombMode();
   fVisBranch = "";
   fVolInfo = "";
   fVisLock = kFALSE;
   fIsRaytracing = kFALSE;
   fTopVisible = kFALSE;
   fPaintingOverlaps = kFALSE;
   fPlugin = 0;
   fVisVolumes = new TObjArray();
   fOverlap = 0;
   fGlobal = new TGeoHMatrix();
   fBuffer = new TBuffer3D(TBuffer3DTypes::kGeneric,20,3*20,0,0,0,0);
   fClippingShape = 0;
   fLastVolume = 0;
   fTopVolume = 0;
   fIsPaintingShape = kFALSE;
   memset(&fCheckedBox[0], 0, 6*sizeof(Double_t));
   
   fCheckedNode = fGeoManager->GetTopNode();
   fChecker = new TGeoChecker(fGeoManager);
   fIsEditable = kFALSE;
   DefineColors();
}
//______________________________________________________________________________
TGeoPainter::~TGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default destructor*-*-*-*-*-*-*-*-*
//*-*                  ===================================
   if (fChecker) delete fChecker;
   delete fVisVolumes;
   delete fGlobal;
   delete fBuffer;
   if (fPlugin) delete fPlugin;
}
//______________________________________________________________________________
void TGeoPainter::AddSize3D(Int_t numpoints, Int_t numsegs, Int_t numpolys)
{
//--- Add numpoints, numsegs, numpolys to the global 3D size.
   gSize3D.numPoints += numpoints;
   gSize3D.numSegs   += numsegs;
   gSize3D.numPolys  += numpolys;
}      
//______________________________________________________________________________
TVirtualGeoTrack *TGeoPainter::AddTrack(Int_t id, Int_t pdgcode, TObject *particle)
{
// Create a primary TGeoTrack.
   return (TVirtualGeoTrack*)(new TGeoTrack(id,pdgcode,0,particle));
}

//______________________________________________________________________________
void TGeoPainter::AddTrackPoint(Double_t *point, Double_t *box, Bool_t reset) 
{
// Average center of view of all painted tracklets and compute view box.
   static Int_t npoints = 0;
   static Double_t xmin[3] = {0,0,0};
   static Double_t xmax[3] = {0,0,0};
   Int_t i;
   if (reset) {
      memset(box, 0, 6*sizeof(Double_t));
      memset(xmin, 0, 3*sizeof(Double_t));
      memset(xmax, 0, 3*sizeof(Double_t));
      npoints = 0;
      return;
   }      
   if (npoints==0) {
      for (i=0; i<3; i++) xmin[i]=xmax[i]=0;
      npoints++;
   }
   npoints++;
   Double_t  ninv = 1./Double_t(npoints); 
   for (i=0; i<3; i++) {
      box[i] += ninv*(point[i]-box[i]);
      if (point[i]<xmin[i]) xmin[i]=point[i];
      if (point[i]>xmax[i]) xmax[i]=point[i];
      box[i+3] = 0.5*(xmax[i]-xmin[i]);
   } 
}
   
//______________________________________________________________________________
void TGeoPainter::BombTranslation(const Double_t *tr, Double_t *bombtr)
{
// get the new 'bombed' translation vector according current exploded view mode
   memcpy(bombtr, tr, 3*sizeof(Double_t));
   switch (fExplodedView) {
      case kGeoNoBomb:
         return;
      case kGeoBombXYZ:
         bombtr[0] *= fBombX;
         bombtr[1] *= fBombY;
         bombtr[2] *= fBombZ;
         return;
      case kGeoBombCyl:
         bombtr[0] *= fBombR;
         bombtr[1] *= fBombR;
         bombtr[2] *= fBombZ;
         return;
      case kGeoBombSph:
         bombtr[0] *= fBombR;
         bombtr[1] *= fBombR;
         bombtr[2] *= fBombR;
         return;
      default:
         return;
   }   
}

//_____________________________________________________________________________
void TGeoPainter::CheckBoundaryErrors(Int_t ntracks, Double_t radius)
{
// Check pushes and pulls needed to cross the next boundary with respect to the
// position given by FindNextBoundary. If radius is not mentioned the full bounding
// box will be sampled.
   fChecker->CheckBoundaryErrors(ntracks, radius);
}   

//_____________________________________________________________________________
void TGeoPainter::CheckBoundaryReference(Int_t icheck)
{
// Check the boundary errors reference file created by CheckBoundaryErrors method.
// The shape for which the crossing failed is drawn with the starting point in red
// and the extrapolated point to boundary (+/- failing push/pull) in yellow.
   fChecker->CheckBoundaryReference(icheck);
}   

//______________________________________________________________________________
void TGeoPainter::CheckGeometryFull(Bool_t checkoverlaps, Bool_t checkcrossings, Int_t ntracks, const Double_t *vertex)
{
// Geometry checking method (see: TGeoManager::CheckGeometry())
   fChecker->CheckGeometryFull(checkoverlaps,checkcrossings,ntracks,vertex);
}   

//______________________________________________________________________________
void TGeoPainter::CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const
{
// Geometry checking method (see TGeoChecker).
   fChecker->CheckGeometry(nrays, startx, starty, startz);
}   

//______________________________________________________________________________
void TGeoPainter::CheckOverlaps(const TGeoVolume *vol, Double_t ovlp, Option_t *option) const
{
// Check overlaps for the top volume of the geometry, within a limit OVLP. 
   fChecker->CheckOverlaps(vol, ovlp, option);
}

//______________________________________________________________________________
void TGeoPainter::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *option)
{
// check current point in the geometry
   fChecker->CheckPoint(x,y,z,option);
}   

//_____________________________________________________________________________
void TGeoPainter::CheckShape(TGeoShape *shape, Int_t testNo, Int_t nsamples, Option_t *option)
{
// Test for shape navigation methods. Summary for test numbers:
//  1: DistFromInside/Outside. Sample points inside the shape. Generate 
//    directions randomly in cos(theta). Compute DistFromInside and move the 
//    point with bigger distance. Compute DistFromOutside back from new point.
//    Plot d-(d1+d2)
//
   fChecker->CheckShape(shape, testNo, nsamples, option);
}

//______________________________________________________________________________
void TGeoPainter::ClearVisibleVolumes()
{
   //Clear the list of visible volumes
   //reset the kVisOnScreen bit for volumes previously in the list
   
   if (!fVisVolumes) return;
   TIter next(fVisVolumes);
   TGeoVolume *vol;
   while ((vol = (TGeoVolume*)next())) {
      vol->ResetAttBit(TGeoAtt::kVisOnScreen);
   }
   fVisVolumes->Clear();
}
      
      
//______________________________________________________________________________
void TGeoPainter::DefineColors() const
{
// Define 100 colors with increasing light intensities for each basic color (1-7)
// Register these colors at indexes starting with 1000.
   TColor::InitializeColors();
   TColor *color = gROOT->GetColor(1000);
   if (color) return;
   Int_t i,j;
   Float_t r,g,b,h,l,s;
   
   for (i=1; i<8; i++) {
      color = (TColor*)gROOT->GetListOfColors()->At(i);
      if (!color) {
         Warning("DefineColors", "No colors defined");
         return;
      }         
      color->GetHLS(h,l,s);
      for (j=0; j<100; j++) {
         l = 0.25+0.5*j/99.;
         TColor::HLS2RGB(h,l,s,r,g,b);
         new TColor(1000+(i-1)*100+j, r,g,b);
      }
   }           
}

//______________________________________________________________________________
Int_t TGeoPainter::GetColor(Int_t base, Float_t light) const
{
// Get index of a base color with given light intensity (0,1)
   const Int_t kBCols[8] = {1,2,3,5,4,6,7,1};
   TColor *tcolor = gROOT->GetColor(base);
   if (!tcolor) tcolor = new TColor(base, 0.5,0.5,0.5);
   Float_t r,g,b;
   tcolor->GetRGB(r,g,b);
   Int_t code = 0;
   if (r>0.5) code += 1;
   if (g>0.5) code += 2;
   if (b>0.5) code += 4;
   Int_t color, j;
   
   if (light<0.25) {
      j=0;
   } else {
      if (light>0.8) j=99;
      else j = Int_t(99*(light-0.25)/0.5);
   }   
   color = 1000 + (kBCols[code]-1)*100+j;
   return color;
}

//______________________________________________________________________________
TGeoVolume *TGeoPainter::GetDrawnVolume() const
{
// Get currently drawn volume.
   if (!gPad) return 0;
   return fTopVolume;
}         
 
//______________________________________________________________________________
Int_t TGeoPainter::DistanceToPrimitiveVol(TGeoVolume *volume, Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to a volume 
   const Int_t big = 9999;
   const Int_t inaxis = 7;
   const Int_t maxdist = 5;

   if (fTopVolume != volume) fTopVolume = volume;
   TView *view = gPad->GetView();
   if (!view) return big;   
   TGeoBBox *box;
   fGlobal->Clear();
   TGeoShape::SetTransform(fGlobal);

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   // return if point not in user area
   if (px < puxmin - inaxis) return big;
   if (py > puymin + inaxis) return big;
   if (px > puxmax + inaxis) return big;
   if (py < puymax - inaxis) return big;
   
   fCheckedNode = fGeoManager->GetTopNode();         
   gPad->SetSelected(view);
   Int_t dist = big;
//   Int_t id;
   
   if (fPaintingOverlaps) {
      TGeoVolume *crt;
      crt = fOverlap->GetFirstVolume();
      *fGlobal = fOverlap->GetFirstMatrix();
      dist = crt->GetShape()->DistancetoPrimitive(px,py);
      if (dist<maxdist) {
         gPad->SetSelected(crt);
         box = (TGeoBBox*)crt->GetShape();
         fGlobal->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }
      crt = fOverlap->GetSecondVolume();
      *fGlobal = fOverlap->GetSecondMatrix();
      dist = crt->GetShape()->DistancetoPrimitive(px,py);
      if (dist<maxdist) {
         gPad->SetSelected(crt);
         box = (TGeoBBox*)crt->GetShape();
         fGlobal->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }      
      return big;
   }
      // Compute distance to the right edge
   if ((puxmax+inaxis-px) < 40) {
      if ((py-puymax+inaxis) < 40) {
         // when the mouse points to the (40x40) right corner of the pad, the manager class is selected
         gPad->SetSelected(fGeoManager);
         fVolInfo = fGeoManager->GetName();
         box = (TGeoBBox*)volume->GetShape();
         memcpy(fCheckedBox, box->GetOrigin(), 3*sizeof(Double_t));
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }
      // when the mouse points to the (40 pix) right edge of the pad, the top volume is selected
      gPad->SetSelected(volume);
      fVolInfo = volume->GetName();
      box = (TGeoBBox*)volume->GetShape();
      memcpy(fCheckedBox, box->GetOrigin(), 3*sizeof(Double_t));
      fCheckedBox[3] = box->GetDX();
      fCheckedBox[4] = box->GetDY();
      fCheckedBox[5] = box->GetDZ();
      return 0;
   }   

   TGeoVolume *vol = volume;
   Bool_t vis = vol->IsVisible();
//   Bool_t drawDaughters = kTRUE;
   // Do we need to check a branch only?
   if (volume->IsVisBranch()) {
      if (!fGeoManager->IsClosed()) return big;
      fGeoManager->PushPath();
      fGeoManager->cd(fVisBranch.Data());
      while (fGeoManager->GetLevel()) {
         vol = fGeoManager->GetCurrentVolume();
         *fGlobal = gGeoManager->GetCurrentMatrix();
         dist = vol->GetShape()->DistancetoPrimitive(px,py);
         if (dist<maxdist) {
            fVolInfo = fVisBranch;
            box = (TGeoBBox*)vol->GetShape();
            fGeoManager->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
            fCheckedNode = gGeoManager->GetCurrentNode();
            if (fGeoManager->IsNodeSelectable()) gPad->SetSelected(fCheckedNode);
            else gPad->SetSelected(vol);
            fCheckedBox[3] = box->GetDX();
            fCheckedBox[4] = box->GetDY();
            fCheckedBox[5] = box->GetDZ();
            fGeoManager->PopPath();
            return 0;
         }
         fGeoManager->CdUp();
      }
      fGeoManager->PopPath();
      return dist;
   }      

   // Do I need to look for the top volume ?
   if ((fTopVisible && vis) || !vol->GetNdaughters() || !vol->IsVisDaughters() || vol->IsVisOnly()) {
      dist = vol->GetShape()->DistancetoPrimitive(px,py);
      if (dist<maxdist) {
         fVolInfo = vol->GetName();
         gPad->SetSelected(vol);
         box = (TGeoBBox*)vol->GetShape();
         memcpy(fCheckedBox, box->GetOrigin(), 3*sizeof(Double_t));
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }
      if (vol->IsVisOnly() || !vol->GetNdaughters() || !vol->IsVisDaughters())
         return dist;
   }      

   // Iterate the volume content
   TGeoIterator next(vol);
   next.SetTopName(TString::Format("%s_1",vol->GetName()));
   TGeoNode *daughter;

   Int_t level, nd;
   Bool_t last;

   while ((daughter=next())) {
      vol = daughter->GetVolume();
      level = next.GetLevel();
      nd = daughter->GetNdaughters();
      vis = daughter->IsVisible();
      if (volume->IsVisContainers()) {
         if (vis && level<=fVisLevel) {
            *fGlobal = next.GetCurrentMatrix();
            dist = vol->GetShape()->DistancetoPrimitive(px,py);
            if (dist<maxdist) {
               next.GetPath(fVolInfo);
               box = (TGeoBBox*)vol->GetShape();
               fGlobal->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
               fCheckedNode = daughter;
               if (fGeoManager->IsNodeSelectable()) gPad->SetSelected(fCheckedNode);
               else gPad->SetSelected(vol);
               fCheckedBox[3] = box->GetDX();
               fCheckedBox[4] = box->GetDY();
               fCheckedBox[5] = box->GetDZ();
               fGeoManager->PopPath();
               return 0;
            }
         }   
         // Check if we have to skip this branch
         if (level==fVisLevel || !daughter->IsVisDaughters()) {
            next.Skip();
            continue;
         }   
      } else if (volume->IsVisLeaves()) {
         last = ((nd==0) || (level==fVisLevel) || (!daughter->IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last) {
            *fGlobal = next.GetCurrentMatrix();
            dist = vol->GetShape()->DistancetoPrimitive(px,py);
            if (dist<maxdist) {
               next.GetPath(fVolInfo);
               box = (TGeoBBox*)vol->GetShape();
               fGlobal->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
               fCheckedNode = daughter;
               if (fGeoManager->IsNodeSelectable()) gPad->SetSelected(fCheckedNode);
               else gPad->SetSelected(vol);
               fCheckedBox[3] = box->GetDX();
               fCheckedBox[4] = box->GetDY();
               fCheckedBox[5] = box->GetDZ();
               fGeoManager->PopPath();
               return 0;
            }
         }
         // Check if we have to skip the branch
         if (last || !daughter->IsVisDaughters()) next.Skip();
      }
   }
   return dist;
}

//______________________________________________________________________________
void TGeoPainter::DefaultAngles()
{   
// Set default angles for the current view.
   if (gPad) {
      Int_t irep;
      TView *view = gPad->GetView();
      if (!view) return;
      view->SetView(-206,126,75,irep);
      ModifiedPad();
   }
}   

//______________________________________________________________________________
void TGeoPainter::DefaultColors()
{   
// Set default volume colors according to tracking media
   TIter next(fGeoManager->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next()))
      vol->SetLineColor(vol->GetMaterial()->GetDefaultColor());
   ModifiedPad();
}   

//______________________________________________________________________________
Int_t TGeoPainter::CountNodes(TGeoVolume *volume, Int_t rlevel) const
{
// Count number of visible nodes down to a given level.
   TGeoVolume *vol = volume;
   Int_t count = 0;
   Bool_t vis = vol->IsVisible();
   // Do I need to look for the top volume ?
   if ((fTopVisible && vis) || !vol->GetNdaughters() || !vol->IsVisDaughters() || vol->IsVisOnly()) 
      count++;
   // Is this the only volume?
   if (volume->IsVisOnly()) return count;

   // Do we need to check a branch only?
   if (volume->IsVisBranch()) {
      fGeoManager->PushPath();
      fGeoManager->cd(fVisBranch.Data());
      count = fGeoManager->GetLevel() + 1;
      fGeoManager->PopPath();
      return count;
   }
   // Iterate the volume content
   TGeoIterator next(vol);
   TGeoNode *daughter;
   Int_t level, nd;
   Bool_t last;
      
   while ((daughter=next())) {
//      vol = daughter->GetVolume();
      level = next.GetLevel();
      nd = daughter->GetNdaughters();
      vis = daughter->IsVisible();
      if (volume->IsVisContainers()) {
         if (vis && level<=rlevel) count++;
         // Check if we have to skip this branch
         if (level==rlevel || !daughter->IsVisDaughters()) {
            next.Skip();
            continue;
         }   
      } else if (volume->IsVisLeaves()) {
         last = ((nd==0) || (level==rlevel) || (!daughter->IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last) count++;
         // Check if we have to skip the branch
         if (last) next.Skip();
      }
   }      
   return count;    
}   

//______________________________________________________________________________
Int_t TGeoPainter::CountVisibleNodes()
{
// Count total number of visible nodes.
   Int_t maxnodes = fGeoManager->GetMaxVisNodes(); 
   Int_t vislevel = fGeoManager->GetVisLevel();
//   TGeoVolume *top = fGeoManager->GetTopVolume();
   TGeoVolume *top = fTopVolume;
   if (maxnodes <= 0  && top) {
      fNVisNodes = CountNodes(top, vislevel);
      SetVisLevel(vislevel);
      return fNVisNodes;
   }   
   //if (the total number of nodes of the top volume is less than maxnodes
   // we can visualize everything.
   //recompute the best visibility level
   if (!top) {
      SetVisLevel(vislevel);
      return 0;
   }   
   fNVisNodes = -1;
   Bool_t again = kFALSE;
   for (Int_t level = 1;level<20;level++) {
      vislevel = level;
      Int_t nnodes = CountNodes(top, level);
      if (top->IsVisOnly() || top->IsVisBranch()) {
         vislevel = fVisLevel;
         fNVisNodes = nnodes;
         break;
      }   
      if (nnodes > maxnodes) {
         vislevel--;
         break;
      }
      if (nnodes == fNVisNodes) {
         if (again) break;
         again = kTRUE;
      }   
      fNVisNodes = nnodes;
   }
   SetVisLevel(vislevel);
   return fNVisNodes;
}

//______________________________________________________________________________
void TGeoPainter::CheckEdit()
{
// Check if Ged library is loaded and load geometry editor classe.
   if (fIsEditable) return;
   if (!TClass::GetClass("TGedEditor")) return;
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TGeoManagerEditor"))) {
      if (h->LoadPlugin() == -1) return;
      h->ExecPlugin(0);
   }
   fIsEditable = kTRUE;
}      

//______________________________________________________________________________
void TGeoPainter::EditGeometry(Option_t *option)
{
// Start the geometry editor.
   if (!gPad) return;
   if (!fIsEditable) {
      if (!strlen(option)) gPad->GetCanvas()->GetCanvasImp()->ShowEditor();
      else TVirtualPadEditor::ShowEditor();
      CheckEdit();
   }   
   gPad->SetSelected(fGeoManager);
   gPad->GetCanvas()->Selected(gPad,fGeoManager,kButton1Down);   
}

//______________________________________________________________________________
void TGeoPainter::Draw(Option_t *option)
{
// Draw method.
   DrawVolume(fGeoManager->GetTopVolume(), option);
}

//______________________________________________________________________________
void TGeoPainter::DrawBatemanSol(TGeoBatemanSol *sol, Option_t *option)
{
// Draw the time evolution of a radionuclide.
   Int_t ncoeff = sol->GetNcoeff();
   if (!ncoeff) return;
   Double_t tlo=0., thi=0.;
   Double_t cn=0., lambda=0.;
   Int_t i;
   sol->GetRange(tlo, thi);
   Bool_t autorange = (thi==0.)?kTRUE:kFALSE;
   
   // Try to find the optimum range in time.
   if (autorange) tlo = 0.;
   sol->GetCoeff(0, cn, lambda);
   Double_t lambdamin = lambda;
   TString formula = "";
   for (i=0; i<ncoeff; i++) {
      sol->GetCoeff(i, cn, lambda);
      formula += TString::Format("%g*exp(-%g*x)",cn, lambda);
      if (i < ncoeff-1) formula += "+";
      if (lambda < lambdamin &&
          lambda > 0.) lambdamin = lambda;
   }
   if (autorange) thi = 10./lambdamin;
   formula += ";time[s]";
   formula += TString::Format(";Concentration_of_%s",sol->GetElement()->GetName());
   // Create a function
   TF1 *func = new TF1(TString::Format("conc%s",sol->GetElement()->GetName()), formula.Data(), tlo,thi);
   func->SetMinimum(1.e-3);
   func->SetMaximum(1.25*TMath::Max(sol->Concentration(tlo), sol->Concentration(thi)));
   func->SetLineColor(sol->GetLineColor());
   func->SetLineStyle(sol->GetLineStyle());
   func->SetLineWidth(sol->GetLineWidth());
   func->SetMarkerColor(sol->GetMarkerColor());
   func->SetMarkerStyle(sol->GetMarkerStyle());
   func->SetMarkerSize(sol->GetMarkerSize());
   func->Draw(option);
}   

//______________________________________________________________________________
void TGeoPainter::DrawVolume(TGeoVolume *vol, Option_t *option)
{
// Draw method.
   fTopVolume = vol;
   fLastVolume = 0;
   fIsPaintingShape = kFALSE;
//   if (fVisOption==kGeoVisOnly ||
//       fVisOption==kGeoVisBranch) fGeoManager->SetVisOption(kGeoVisLeaves);
   CountVisibleNodes();         
   TString opt = option;
   opt.ToLower();
   fPaintingOverlaps = kFALSE;
   fOverlap = 0;
   
   if (fVisLock) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   fTopVolume->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = TView::CreateView(11,0,0);
      // Set the view to perform a first autorange (frame) draw. 
      // TViewer3DPad will revert view to normal painting after this
      view->SetAutoRange(kTRUE);
      if (has_pad) gPad->Update();
   }
   Paint("range"); 
   view->SetAutoRange(kFALSE);    
   // If we are drawing into the pad, then the view needs to be
   // set to perspective
//   if (!view->IsPerspective()) view->SetPerspective();
   
   fLastVolume = fTopVolume;
 
         // Create a 3D viewer to paint us
   gPad->GetViewer3D(option);
}

//______________________________________________________________________________
void TGeoPainter::DrawShape(TGeoShape *shape, Option_t *option)
{
// Draw a shape.
   TString opt = option;
   opt.ToLower();
   fPaintingOverlaps = kFALSE;
   fOverlap = 0;
   fIsPaintingShape = kTRUE;
   
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this shape to pad
   shape->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = TView::CreateView(11,0,0);
      // Set the view to perform a first autorange (frame) draw. 
      // TViewer3DPad will revert view to normal painting after this
      view->SetAutoRange(kTRUE);
      if (has_pad) gPad->Update();
   }
   PaintShape(shape,"range");   
   view->SetAutoRange(kTRUE);   
   // Create a 3D viewer to paint us
   gPad->GetViewer3D(option);
}

//______________________________________________________________________________
void TGeoPainter::DrawOverlap(void *ovlp, Option_t *option)
{
// Draw an overlap.
   TString opt = option;
   fIsPaintingShape = kFALSE;
   TGeoOverlap *overlap = (TGeoOverlap*)ovlp;
   if (!overlap) return;
   
   fPaintingOverlaps = kTRUE;
   fOverlap = overlap;
   opt.ToLower();
   if (fVisLock) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   overlap->AppendPad(option);

   // Create a 3-D view
         // Create a 3D viewer to paint us
   gPad->GetViewer3D(option);
   TView *view = gPad->GetView();
   if (!view) {
      view = TView::CreateView(11,0,0);
      // Set the view to perform a first autorange (frame) draw. 
      // TViewer3DPad will revert view to normal painting after this
      view->SetAutoRange(kTRUE);
      PaintOverlap(ovlp, "range");
      overlap->GetPolyMarker()->Draw("SAME");
      if (has_pad) gPad->Update();
   }

   // If we are drawing into the pad, then the view needs to be
   // set to perspective
//   if (!view->IsPerspective()) view->SetPerspective();
   fVisLock = kTRUE;
}


//______________________________________________________________________________
void TGeoPainter::DrawOnly(Option_t *option)
{
// Draw only one volume.
   TString opt = option;
   opt.ToLower();
   if (fVisLock) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   fPaintingOverlaps = kFALSE;
   fIsPaintingShape = kFALSE;
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   fTopVolume = fGeoManager->GetCurrentVolume();
   fTopVolume->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = TView::CreateView(11,0,0);
      // Set the view to perform a first autorange (frame) draw. 
      // TViewer3DPad will revert view to normal painting after this
      view->SetAutoRange(kTRUE);
      fVisOption = kGeoVisOnly;
      if (has_pad) gPad->Update();
   }

   // If we are drawing into the pad, then the view needs to be
   // set to perspective
//   if (!view->IsPerspective()) view->SetPerspective();
   fVisLock = kTRUE;
}

//______________________________________________________________________________
void TGeoPainter::DrawCurrentPoint(Int_t color)
{
// Draw current point in the same view.
   if (!gPad) return;
   if (!gPad->GetView()) return;
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(color);
   const Double_t *point = fGeoManager->GetCurrentPoint();
   pm->SetNextPoint(point[0], point[1], point[2]);
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.5);
   pm->Draw("SAME");
}

//______________________________________________________________________________
void TGeoPainter::DrawPanel()
{
}

//______________________________________________________________________________
void TGeoPainter::DrawPath(const char *path)
{
// Draw all volumes for a given path.
   fVisOption=kGeoVisBranch;
   fVisBranch=path; 
   fIsPaintingShape = kFALSE;
   fTopVolume = fGeoManager->GetTopVolume();
   fTopVolume->SetVisRaytrace(kFALSE);
   DrawVolume(fTopVolume,"");   
}

//______________________________________________________________________________
void TGeoPainter::EstimateCameraMove(Double_t tmin, Double_t tmax, Double_t *start, Double_t *end)
{
// Estimate camera movement between tmin and tmax for best track display
   if (!gPad) return;
   TIter next(gPad->GetListOfPrimitives());
   TVirtualGeoTrack *track;
   TObject *obj;
   Int_t ntracks = 0;
   Double_t *point = 0;
   AddTrackPoint(point, start, kTRUE);
   while ((obj=next())) {
      if (strcmp(obj->ClassName(), "TGeoTrack")) continue;
      track = (TVirtualGeoTrack*)obj;
      ntracks++;
      track->PaintCollect(tmin, start);
   }
   
   if (!ntracks) return;
   next.Reset();
   AddTrackPoint(point, end, kTRUE);
   while ((obj=next())) {
      if (strcmp(obj->ClassName(), "TGeoTrack")) continue;
      track = (TVirtualGeoTrack*)obj;
      if (!track) continue;
      track->PaintCollect(tmax, end);
   }   
}

//______________________________________________________________________________
void TGeoPainter::ExecuteManagerEvent(TGeoManager * /*geom*/, Int_t event, Int_t /*px*/, Int_t /*py*/)
{
// Execute mouse actions on a given volume.
   if (!gPad) return;
   gPad->SetCursor(kPointer);
   switch (event) {
      case kButton1Down:
         if (!fIsEditable) CheckEdit();
   }          
}
   
//______________________________________________________________________________
void TGeoPainter::ExecuteShapeEvent(TGeoShape * /*shape*/, Int_t event, Int_t /*px*/, Int_t /*py*/)
{
// Execute mouse actions on a given shape.
   if (!gPad) return;
   gPad->SetCursor(kHand);
   switch (event) {
      case kButton1Down:
         if (!fIsEditable) CheckEdit();
   }      
}

//______________________________________________________________________________
void TGeoPainter::ExecuteVolumeEvent(TGeoVolume * /*volume*/, Int_t event, Int_t /*px*/, Int_t /*py*/)
{
// Execute mouse actions on a given volume.
   if (!gPad) return;
   if (!fIsEditable) CheckEdit();
//   if (fIsRaytracing) return;
//   Bool_t istop = (volume==fTopVolume)?kTRUE:kFALSE;
//   if (istop) gPad->SetCursor(kHand);
//   else gPad->SetCursor(kPointer);
   gPad->SetCursor(kHand);
//   static Int_t width, color;
   switch (event) {
   case kMouseEnter:
//      width = volume->GetLineWidth();
//      color = volume->GetLineColor();
      break;
   
   case kMouseLeave:
//      volume->SetLineWidth(width);
//      volume->SetLineColor(color);
      break;

   case kButton1Down:
//      volume->SetLineWidth(3);
//      volume->SetLineColor(2);
//      gPad->Modified();
//      gPad->Update();
      break;
   
   case kButton1Up:
//      volume->SetLineWidth(width);
//      volume->SetLineColor(color);
//      gPad->Modified();
//      gPad->Update();
      break;
      
   case kButton1Double:
      gPad->SetCursor(kWatch);
      GrabFocus();
      break;
   }
}

//______________________________________________________________________________
const char *TGeoPainter::GetVolumeInfo(const TGeoVolume *volume, Int_t /*px*/, Int_t /*py*/) const
{
// Get some info about the current selected volume.
   static TString info;
   info = "";
   if (!gPad) return info;
   if (fPaintingOverlaps) {
      if (!fOverlap) {
         info =  "wrong overlapping flag";
         return info;
      }   
      TString ovtype, name;
      if (fOverlap->IsExtrusion()) ovtype="EXTRUSION";
      else ovtype = "OVERLAP";
      if (volume==fOverlap->GetFirstVolume()) name=volume->GetName();
      else name=fOverlap->GetSecondVolume()->GetName();
      info = TString::Format("%s: %s of %g", name.Data(), ovtype.Data(), fOverlap->GetOverlap());
      return info;
   }   
   else info = TString::Format("%s, shape=%s", fVolInfo.Data(), volume->GetShape()->ClassName());
   return info;
}

//______________________________________________________________________________
TGeoChecker *TGeoPainter::GetChecker()
{
// Create/return geometry checker.
   if (!fChecker) fChecker = new TGeoChecker(fGeoManager);
   return fChecker;
}
 
//______________________________________________________________________________
void TGeoPainter::GetViewAngles(Double_t &longitude, Double_t &latitude, Double_t &psi) 
{
// Get the current view angles.
   if (!gPad) return;
   TView *view = gPad->GetView();
   if (!view) return;
   longitude = view->GetLongitude();
   latitude = view->GetLatitude();
   psi = view->GetPsi();
}   

//______________________________________________________________________________
void TGeoPainter::GrabFocus(Int_t nfr, Double_t dlong, Double_t dlat, Double_t dpsi)
{
// Move focus to current volume
   if (!gPad) return;
   TView *view = gPad->GetView();
   if (!view) return;
   if (!fCheckedNode && !fPaintingOverlaps) {
      printf("Woops!!!\n");
      TGeoBBox *box = (TGeoBBox*)fGeoManager->GetTopVolume()->GetShape();
      memcpy(&fCheckedBox[0], box->GetOrigin(), 3*sizeof(Double_t));
      fCheckedBox[3] = box->GetDX();
      fCheckedBox[4] = box->GetDY();
      fCheckedBox[5] = box->GetDZ();
   }      
   view->SetPerspective();
   Int_t nvols = fVisVolumes->GetEntriesFast();
   Int_t nframes = nfr;
   if (nfr==0) {
      nframes = 1;
      if (nvols<1500) nframes=10;
      if (nvols<1000) nframes=20;
      if (nvols<200) nframes = 50;
      if (nvols<100) nframes = 100;
   }   
   view->MoveFocus(&fCheckedBox[0], fCheckedBox[3], fCheckedBox[4], fCheckedBox[5], nframes, dlong, dlat, dpsi);
}

//______________________________________________________________________________
TH2F *TGeoPainter::LegoPlot(Int_t ntheta, Double_t themin, Double_t themax,
                            Int_t nphi,   Double_t phimin, Double_t phimax,
                            Double_t rmin, Double_t rmax, Option_t *option)
{
// Generate a lego plot fot the top volume, according to option.
   return fChecker->LegoPlot(ntheta, themin, themax, nphi, phimin, phimax, rmin, rmax, option);   
}
//______________________________________________________________________________
void TGeoPainter::LocalToMasterVect(const Double_t *local, Double_t *master) const
{
// Convert a local vector according view rotation matrix
   for (Int_t i=0; i<3; i++)
      master[i] = -local[0]*fMat[i]-local[1]*fMat[i+3]-local[2]*fMat[i+6];
}

//______________________________________________________________________________
void TGeoPainter::ModifiedPad(Bool_t update) const
{
// Check if a pad and view are present and send signal "Modified" to pad.
   if (!gPad) return;
   if (update) {
      gPad->Update();
      return;
   }   
   TView *view = gPad->GetView();
   if (!view) return;
   view->SetViewChanged();
   gPad->Modified();
   if (gROOT->FromPopUp()) gPad->Update();
}   

//______________________________________________________________________________
void TGeoPainter::Paint(Option_t *option)
{
// Paint current geometry according to option.
   if (!fGeoManager || !fTopVolume) return;
   Bool_t is_padviewer = kTRUE;
   if (gPad) is_padviewer = (!strcmp(gPad->GetViewer3D()->ClassName(),"TViewer3DPad"))?kTRUE:kFALSE;
   
   fIsRaytracing = fTopVolume->IsRaytracing();
   if (fTopVolume->IsVisContainers()) fVisOption = kGeoVisDefault;
   else if (fTopVolume->IsVisLeaves()) fVisOption = kGeoVisLeaves;
   else if (fTopVolume->IsVisOnly()) fVisOption = kGeoVisOnly;
   else if (fTopVolume->IsVisBranch()) fVisOption = kGeoVisBranch;
   
   
   if (!fIsRaytracing || !is_padviewer) {
      if (fGeoManager->IsDrawingExtra()) {
         // loop the list of physical volumes
         fGeoManager->CdTop();
         TObjArray *nodeList = fGeoManager->GetListOfPhysicalNodes();
         Int_t nnodes = nodeList->GetEntriesFast();
         Int_t inode;
         TGeoPhysicalNode *node;
         for (inode=0; inode<nnodes; inode++) {
            node = (TGeoPhysicalNode*)nodeList->UncheckedAt(inode);
            PaintPhysicalNode(node, option);
         }
      } else {
         PaintVolume(fTopVolume,option);
      }                      
      fVisLock = kTRUE;
   } 
   // Check if we have to raytrace (only in pad)  
   if (fIsRaytracing && is_padviewer) Raytrace();
}

//______________________________________________________________________________
void TGeoPainter::PaintOverlap(void *ovlp, Option_t *option)
{
// Paint an overlap.
   if (!fGeoManager) return;
   TGeoOverlap *overlap = (TGeoOverlap *)ovlp;
   if (!overlap) return;
   Int_t color, transparency;
   if (fOverlap != overlap) fOverlap = overlap;
   TGeoShape::SetTransform(fGlobal);
   TGeoHMatrix *hmat = fGlobal;
   TGeoVolume *vol;
   TGeoVolume *vol1 = overlap->GetFirstVolume();
   TGeoVolume *vol2 = overlap->GetSecondVolume();
   TGeoHMatrix *matrix1 = overlap->GetFirstMatrix();
   TGeoHMatrix *matrix2 = overlap->GetSecondMatrix();
   //
   vol = vol1;
   *hmat = matrix1;
   fGeoManager->SetMatrixReflection(matrix1->IsReflection());
   if (!fVisLock) fVisVolumes->Add(vol);
   fGeoManager->SetPaintVolume(vol);
   color = vol->GetLineColor();
   transparency = vol->GetTransparency();
   vol->SetLineColor(kGreen);
   vol->SetTransparency(40);
   if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();
   PaintShape(*(vol->GetShape()),option);
   vol->SetLineColor(color);
   vol->SetTransparency(transparency);
   vol = vol2;
   *hmat = matrix2;
   fGeoManager->SetMatrixReflection(matrix2->IsReflection());
   if (!fVisLock) fVisVolumes->Add(vol);
   fGeoManager->SetPaintVolume(vol);
   color = vol->GetLineColor();
   transparency = vol->GetTransparency();
   vol->SetLineColor(kBlue);
   vol->SetTransparency(40);
   if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();
   PaintShape(*(vol->GetShape()),option);
   vol->SetLineColor(color);
   vol->SetTransparency(transparency);
   fGeoManager->SetMatrixReflection(kFALSE);
   fVisLock = kTRUE;
}

//______________________________________________________________________________
void TGeoPainter::PaintNode(TGeoNode *node, Option_t *option, TGeoMatrix* global)
{
// Paint recursively a node and its content accordind to visualization options.
   PaintVolume(node->GetVolume(), option, global);
} 

//______________________________________________________________________________
void TGeoPainter::PaintVolume(TGeoVolume *top, Option_t *option, TGeoMatrix* global)
{
// Paint recursively a node and its content accordind to visualization options.
   if (fTopVolume != top) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   fTopVolume = top;
   if (!fVisLevel) return;
   TGeoVolume *vol = top;
   if(global)
      *fGlobal = *global;
   else
      fGlobal->Clear();
   TGeoShape::SetTransform(fGlobal);
   Bool_t drawDaughters = kTRUE;
   Bool_t vis = (top->IsVisible() && !top->IsAssembly());
   Int_t transparency = 0;

   // Update pad attributes in case we need to paint VOL
   if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();

   // Do we need to draw a branch ?
   if (top->IsVisBranch()) {
      fGeoManager->PushPath();
      fGeoManager->cd(fVisBranch.Data());
      while (fGeoManager->GetLevel()) {
         vol = fGeoManager->GetCurrentVolume();
         if (!fVisLock) {
            fVisVolumes->Add(vol);
            vol->SetAttBit(TGeoAtt::kVisOnScreen);
         }   
         fGeoManager->SetPaintVolume(vol);
         transparency = vol->GetTransparency();
         vol->SetTransparency(40);
         if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();
         if (global) {
            *fGlobal  = *global;
            *fGlobal *= *fGeoManager->GetCurrentMatrix();
         } else {
            *fGlobal = fGeoManager->GetCurrentMatrix();
         }
         fGeoManager->SetMatrixReflection(fGlobal->IsReflection());
         PaintShape(*(vol->GetShape()),option);
         vol->SetTransparency(transparency);
         fGeoManager->CdUp();
      }
      fVisLock = kTRUE;   
      fGeoManager->PopPath();
      fGeoManager->SetMatrixReflection(kFALSE);
      return;
   }   
      
   // Do I need to draw the top volume ?
   if ((fTopVisible && vis) || !top->GetNdaughters() || !top->IsVisDaughters() || top->IsVisOnly()) {
      fGeoManager->SetPaintVolume(vol);
      fGeoManager->SetMatrixReflection(fGlobal->IsReflection());
      drawDaughters = PaintShape(*(vol->GetShape()),option);
      if (!fVisLock && !vol->TestAttBit(TGeoAtt::kVisOnScreen)) {
         fVisVolumes->Add(vol);
         vol->SetAttBit(TGeoAtt::kVisOnScreen);
      } 
      if (top->IsVisOnly() || !top->GetNdaughters() || !top->IsVisDaughters()) {
         fVisLock = kTRUE;
         return;
      }    
   }   

   // Iterate the volume content
   TGeoIterator next(vol);
   if (fPlugin) next.SetUserPlugin(fPlugin);
   TGeoNode *daughter;
//   TGeoMatrix *glmat;
   Int_t level, nd;
   Bool_t last;
   Int_t line_color=0, line_width=0, line_style=0;
   while ((daughter=next())) {
      vol = daughter->GetVolume();
      fGeoManager->SetPaintVolume(vol);
      level = next.GetLevel();
      nd = daughter->GetNdaughters();
      vis = daughter->IsVisible();
      drawDaughters = kTRUE;
      if (top->IsVisContainers()) {
         if (vis && level<=fVisLevel) {
            if (fPlugin) {
               line_color = vol->GetLineColor();
               line_width = vol->GetLineWidth();
               line_style = vol->GetLineStyle();
               transparency = vol->GetTransparency();
               fPlugin->ProcessNode();
            }   
            if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();
            if (global) {
               *fGlobal  = *global;
               *fGlobal *= *next.GetCurrentMatrix();
            } else {
               *fGlobal = next.GetCurrentMatrix();
            }
            fGeoManager->SetMatrixReflection(fGlobal->IsReflection());
            drawDaughters = PaintShape(*(vol->GetShape()),option);
            if (fPlugin) {
               vol->SetLineColor(line_color);
               vol->SetLineWidth(line_width);
               vol->SetLineStyle(line_style);
               vol->SetTransparency(transparency);
            }   
            if (!fVisLock && !daughter->IsOnScreen()) {
               fVisVolumes->Add(vol);
               vol->SetAttBit(TGeoAtt::kVisOnScreen);
            }   
         }   
         // Check if we have to skip this branch
         if (!drawDaughters || level==fVisLevel || !daughter->IsVisDaughters()) {
            next.Skip();
            continue;
         }   
      } else if (top->IsVisLeaves()) {
         last = ((nd==0) || (level==fVisLevel) || (!daughter->IsVisDaughters()))?kTRUE:kFALSE;
         if (vis && last) {
            if (fPlugin) {
               line_color = vol->GetLineColor();
               line_width = vol->GetLineWidth();
               line_style = vol->GetLineStyle();
               transparency = vol->GetTransparency();
               fPlugin->ProcessNode();
            }   
            if (!strstr(option,"range")) ((TAttLine*)vol)->Modify();
            if (global) {
               *fGlobal  = *global;
               *fGlobal *= *next.GetCurrentMatrix();
            } else {
               *fGlobal = next.GetCurrentMatrix();
            }
            fGeoManager->SetMatrixReflection(fGlobal->IsReflection());
            drawDaughters = PaintShape(*(vol->GetShape()),option);
            if (fPlugin) {
               vol->SetLineColor(line_color);
               vol->SetLineWidth(line_width);
               vol->SetLineStyle(line_style);
               vol->SetTransparency(transparency);
            }   
            if (!fVisLock && !daughter->IsOnScreen()) {
               fVisVolumes->Add(vol);
               vol->SetAttBit(TGeoAtt::kVisOnScreen);
            }   
         }
         // Check if we have to skip the branch
         if (!drawDaughters || last || !daughter->IsVisDaughters()) next.Skip();
      }
   }
   if (fPlugin) fPlugin->SetIterator(0);
   fGeoManager->SetMatrixReflection(kFALSE);
   fVisLock = kTRUE;
}

//______________________________________________________________________________
Bool_t TGeoPainter::PaintShape(const TGeoShape & shape, Option_t *  option ) const
{
   // Paint the supplied shape into the current 3D viewer
   Bool_t addDaughters = kTRUE;

   TVirtualViewer3D * viewer = gPad->GetViewer3D();

   if (!viewer || shape.IsA()==TGeoShapeAssembly::Class()) {
      return addDaughters;
   }

   // For non-composite shapes we are the main paint method & perform the negotation 
   // with the viewer here
   if (!shape.IsComposite()) {
      // Does viewer prefer local frame positions?
      Bool_t localFrame = viewer->PreferLocalFrame();
      // Perform first fetch of buffer from the shape and try adding it
      // to the viewer
      const TBuffer3D & buffer = 
         shape.GetBuffer3D(TBuffer3D::kCore|TBuffer3D::kBoundingBox|TBuffer3D::kShapeSpecific, localFrame);
      Int_t reqSections = viewer->AddObject(buffer, &addDaughters);

      // If the viewer requires additional sections fetch from the shape (if possible)
      // and add again
      if (reqSections != TBuffer3D::kNone) {
         shape.GetBuffer3D(reqSections, localFrame);
         viewer->AddObject(buffer, &addDaughters);
      }
   }
   // Composite shapes have their own internal hierarchy of shapes, each
   // of which generate a filled TBuffer3D. Therefore we can't pass up a 
   // single buffer to here. So as a special case the TGeoCompositeShape
   // performs it's own painting & negotiation with the viewer.
   else {
      const TGeoCompositeShape * composite = static_cast<const TGeoCompositeShape *>(&shape);

      // We need the addDaughters flag returned from the viewer from paint
      // so can't use the normal TObject::Paint()
//      TGeoHMatrix *matrix = (TGeoHMatrix*)TGeoShape::GetTransform();
//      if (viewer->PreferLocalFrame()) matrix->Clear();
      addDaughters = composite->PaintComposite(option);
   }

   return addDaughters;
}

//______________________________________________________________________________
void TGeoPainter::PaintShape(TGeoShape *shape, Option_t *option)
{
// Paint an overlap.
   TGeoShape::SetTransform(fGlobal);
   fGlobal->Clear();
   fGeoManager->SetPaintVolume(0);
   PaintShape(*shape,option);
}

//______________________________________________________________________________
void TGeoPainter::PaintPhysicalNode(TGeoPhysicalNode *node, Option_t *option)
{
// Paints a physical node associated with a path.
   if (!node->IsVisible()) return;
   Int_t level = node->GetLevel();
   Int_t i, col, wid, sty;
   TGeoShape *shape;
   TGeoShape::SetTransform(fGlobal);
   TGeoHMatrix *matrix = fGlobal;
   TGeoVolume *vcrt;
   if (!node->IsVisibleFull()) {
      // Paint only last node in the branch
      vcrt  = node->GetVolume();
      if (!strstr(option,"range")) ((TAttLine*)vcrt)->Modify(); 
      shape = vcrt->GetShape();
      *matrix = node->GetMatrix();
      fGeoManager->SetMatrixReflection(matrix->IsReflection());
      fGeoManager->SetPaintVolume(vcrt);
      if (!node->IsVolAttributes() && !strstr(option,"range")) {
         col = vcrt->GetLineColor();
         wid = vcrt->GetLineWidth();
         sty = vcrt->GetLineStyle();
         vcrt->SetLineColor(node->GetLineColor());
         vcrt->SetLineWidth(node->GetLineWidth());
         vcrt->SetLineStyle(node->GetLineStyle());
         ((TAttLine*)vcrt)->Modify(); 
         PaintShape(*shape,option);
         vcrt->SetLineColor(col);
         vcrt->SetLineWidth(wid);
         vcrt->SetLineStyle(sty);
      } else {    
         PaintShape(*shape,option);
      }
   } else {
      // Paint full branch, except top node
      for (i=1;i<=level; i++) {
         vcrt  = node->GetVolume(i);
         if (!strstr(option,"range")) ((TAttLine*)vcrt)->Modify(); 
         shape = vcrt->GetShape();
         *matrix = node->GetMatrix(i);
         fGeoManager->SetMatrixReflection(matrix->IsReflection());
         fGeoManager->SetPaintVolume(vcrt);
         if (!node->IsVolAttributes() && !strstr(option,"range")) {
            col = vcrt->GetLineColor();
            wid = vcrt->GetLineWidth();
            sty = vcrt->GetLineStyle();
            vcrt->SetLineColor(node->GetLineColor());
            vcrt->SetLineWidth(node->GetLineWidth());
            vcrt->SetLineStyle(node->GetLineStyle());
            ((TAttLine*)vcrt)->Modify();
            PaintShape(*shape,option);
            vcrt->SetLineColor(col);
            vcrt->SetLineWidth(wid);
            vcrt->SetLineStyle(sty);
         } else {  
            PaintShape(*shape,option);
         }   
      }
   }
   fGeoManager->SetMatrixReflection(kFALSE);      
}   

//______________________________________________________________________________
void TGeoPainter::PrintOverlaps() const
{
// Print overlaps (see TGeoChecker::PrintOverlaps())
   fChecker->PrintOverlaps();
}   

//______________________________________________________________________________
void TGeoPainter::OpProgress(const char *opname, Long64_t current, Long64_t size, TStopwatch *watch, Bool_t last, Bool_t refresh)
{
// Text progress bar.
   fChecker->OpProgress(opname,current,size,watch,last,refresh);
}   
   
//______________________________________________________________________________
void TGeoPainter::RandomPoints(const TGeoVolume *vol, Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of a volume.
   fChecker->RandomPoints((TGeoVolume*)vol, npoints, option);
}   

//______________________________________________________________________________
void TGeoPainter::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Shoot nrays in the current drawn geometry
   fChecker->RandomRays(nrays, startx, starty, startz);
}   

//______________________________________________________________________________
void TGeoPainter::Raytrace(Option_t * /*option*/)
{
// Raytrace current drawn geometry
   if (!gPad || gPad->IsBatch()) return;   
   TView *view = gPad->GetView();
   if (!view) return;
   TGeoVolume *top = fGeoManager->GetTopVolume();
   if (top != fTopVolume) fGeoManager->SetTopVolume(fTopVolume);
   if (!view->IsPerspective()) view->SetPerspective();
   gVirtualX->SetMarkerSize(1);
   gVirtualX->SetMarkerStyle(1);
   Int_t i;
   Bool_t inclipst=kFALSE, inclip=kFALSE;
   Double_t krad = TMath::DegToRad();
   Double_t lat = view->GetLatitude();
   Double_t longit = view->GetLongitude();
   Double_t psi = view->GetPsi();
   Double_t c1 = TMath::Cos(psi*krad);
   Double_t s1 = TMath::Sin(psi*krad);
   Double_t c2 = TMath::Cos(lat*krad);
   Double_t s2 = TMath::Sin(lat*krad);
   Double_t s3 = TMath::Cos(longit*krad);
   Double_t c3 = -TMath::Sin(longit*krad);
   fMat[0] =  c1*c3 - s1*c2*s3;
   fMat[1] =  c1*s3 + s1*c2*c3;
   fMat[2] =  s1*s2;
      
   fMat[3] =  -s1*c3 - c1*c2*s3;
   fMat[4] = -s1*s3 + c1*c2*c3;
   fMat[5] =  c1*s2;
   
   fMat[6] =  s2*s3;
   fMat[7] =  -s2*c3;
   fMat[8] = c2; 
   Double_t u0, v0, du, dv;
   view->GetWindow(u0,v0,du,dv);
   Double_t dview = view->GetDview();
   Double_t dproj = view->GetDproj();
   Double_t local[3] = {0,0,1};
   Double_t dir[3], normal[3];   
   LocalToMasterVect(local,dir);
   Double_t min[3], max[3];
   view->GetRange(min, max);
   Double_t cov[3];
   for (i=0; i<3; i++) cov[i] = 0.5*(min[i]+max[i]);
   Double_t cop[3]; 
   for (i=0; i<3; i++) cop[i] = cov[i] - dir[i]*dview;
   fGeoManager->InitTrack(cop, dir);
   Bool_t outside = fGeoManager->IsOutside();
   fGeoManager->DoBackupState();
   if (fClippingShape) inclipst = inclip = fClippingShape->Contains(cop);
   Int_t px, py;
   Double_t xloc, yloc, modloc;
   Int_t pxmin,pxmax, pymin,pymax;
   pxmin = gPad->UtoAbsPixel(0);
   pxmax = gPad->UtoAbsPixel(1);
   pymin = gPad->VtoAbsPixel(1);
   pymax = gPad->VtoAbsPixel(0);
   TGeoNode *next, *nextnode;
   Double_t step,steptot;
   Double_t *norm;
   const Double_t *point = fGeoManager->GetCurrentPoint();
   Double_t *ppoint = (Double_t*)point;
   Double_t tosource[3];
   Double_t calf;
   Double_t phi = 0*krad;
   tosource[0] = -dir[0]*TMath::Cos(phi)+dir[1]*TMath::Sin(phi);
   tosource[1] = -dir[0]*TMath::Sin(phi)-dir[1]*TMath::Cos(phi);
   tosource[2] = -dir[2];
   
   Bool_t done;
//   Int_t istep;
   Int_t base_color, color;
   Double_t light;
   Double_t stemin=0, stemax=TGeoShape::Big();
   TPoint *pxy = new TPoint[1];
   TGeoVolume *nextvol;
   Int_t up;
   Int_t ntotal = pxmax*pymax;
   Int_t nrays = 0;
   TStopwatch *timer = new TStopwatch();
   timer->Start();
   for (px=pxmin; px<pxmax; px++) {
      for (py=pymin; py<pymax; py++) {
         if ((nrays%100)==0) OpProgress("Raytracing",nrays,ntotal,timer,kFALSE);
         nrays++;
         base_color = 1;
         steptot = 0;
         inclip = inclipst;
         xloc = gPad->AbsPixeltoX(pxmin+pxmax-px);
         xloc = xloc*du-u0;
         yloc = gPad->AbsPixeltoY(pymin+pymax-py);
         yloc = yloc*dv-v0;
         modloc = TMath::Sqrt(xloc*xloc+yloc*yloc+dproj*dproj);  
         local[0] = xloc/modloc;
         local[1] = yloc/modloc;
         local[2] = dproj/modloc;
         LocalToMasterVect(local,dir);
         fGeoManager->DoRestoreState();
         fGeoManager->SetOutside(outside);
         fGeoManager->SetCurrentPoint(cop);
         fGeoManager->SetCurrentDirection(dir);
//         fGeoManager->InitTrack(cop,dir);
         // current ray pointing to pixel (px,py)
         done = kFALSE;
         norm = 0;
         // propagate to the clipping shape if any
         if (fClippingShape) {
            if (inclip) {
               stemin = fClippingShape->DistFromInside(cop,dir,3);
               stemax = TGeoShape::Big();
            } else {
               stemax = fClippingShape->DistFromOutside(cop,dir,3);
               stemin = 0;
            }
         }         
               
         while (!done) {
            if (fClippingShape) {
               if (stemin>1E10) break;
               if (stemin>0) {
                  // we are inside clipping shape
                  fGeoManager->SetStep(stemin);
                  next = fGeoManager->Step();
                  steptot = 0;
                  stemin = 0;
                  up = 0;
                  while (next) {
                     // we found something after clipping region
                     nextvol = next->GetVolume();
                     if (nextvol->TestAttBit(TGeoAtt::kVisOnScreen)) {
                        done = kTRUE;
                        base_color = nextvol->GetLineColor();
                        fClippingShape->ComputeNormal(ppoint, dir, normal);
                        norm = normal;
                        break;
                     }
                     up++;
                     next = fGeoManager->GetMother(up);
                  }
                  if (done) break;
                  inclip = fClippingShape->Contains(ppoint);
                  fGeoManager->SetStep(1E-3);
                  while (inclip) {
                     fGeoManager->Step();
                     inclip = fClippingShape->Contains(ppoint);
                  }   
                  stemax = fClippingShape->DistFromOutside(ppoint,dir,3);
               }
            }              
            nextnode = fGeoManager->FindNextBoundaryAndStep();
            step = fGeoManager->GetStep();
            if (step>1E10) break;
            steptot += step;
            next = nextnode;
            // Check the step
            if (fClippingShape) {
               if (steptot>stemax) {
                  steptot = 0;
                  inclip = fClippingShape->Contains(ppoint);
                  if (inclip) {
                     stemin = fClippingShape->DistFromInside(ppoint,dir,3);
                     stemax = TGeoShape::Big();
                     continue;
                  } else {
                     stemin = 0;
                     stemax = fClippingShape->DistFromOutside(ppoint,dir,3);  
                  }
               }
            }      
            // Check if next node is visible
            if (!nextnode) continue;
            nextvol = nextnode->GetVolume();
            if (nextvol->TestAttBit(TGeoAtt::kVisOnScreen)) {
               done = kTRUE;
               base_color = nextvol->GetLineColor();
               next = nextnode;
               break;
            }
         }
         if (!done) continue;
         // current ray intersect a visible volume having color=base_color
//         if (!norm) norm = fGeoManager->FindNormal(kFALSE);
         if (!norm) norm = fGeoManager->FindNormalFast();
         if (!norm) continue;
         calf = norm[0]*tosource[0]+norm[1]*tosource[1]+norm[2]*tosource[2];
         light = 0.25+0.5*TMath::Abs(calf);
         color = GetColor(base_color, light);
         // Now we know the color of the pixel, just draw it
         gVirtualX->SetMarkerColor(color);         
         pxy[0].fX = px;
         pxy[0].fY = py;
         gVirtualX->DrawPolyMarker(1,pxy);
      }
   } 
   delete [] pxy;      
   timer->Stop();
   fChecker->OpProgress("Raytracing",nrays,ntotal,timer,kTRUE);
   delete timer;
}

//______________________________________________________________________________
TGeoNode *TGeoPainter::SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil,
                                    const char* g3path)
{
// shoot npoints randomly in a box of 1E-5 arround current point.
// return minimum distance to points outside
   return fChecker->SamplePoints(npoints, dist, epsil, g3path);
}

//______________________________________________________________________________
void TGeoPainter::SetBombFactors(Double_t bombx, Double_t bomby, Double_t bombz, Double_t bombr)
{
//--- Set cartesian and radial bomb factors for translations
   fBombX = bombx;
   fBombY = bomby;
   fBombZ = bombz;
   fBombR = bombr;
   if (IsExplodedView()) ModifiedPad();
}          

//______________________________________________________________________________
void TGeoPainter::SetExplodedView(Int_t ibomb)    
{
   // set type of exploding view
   if ((ibomb<0) || (ibomb>3)) {
      Warning("SetExplodedView", "exploded view can be 0-3");
      return;
   }
   if ((Int_t)ibomb==fExplodedView) return;   
   Bool_t change = (gPad==0)?kFALSE:kTRUE;

   if (ibomb==kGeoNoBomb) {
      change &= ((fExplodedView==kGeoNoBomb)?kFALSE:kTRUE);
   }
   if (ibomb==kGeoBombXYZ) {
      change &= ((fExplodedView==kGeoBombXYZ)?kFALSE:kTRUE);
   }
   if (ibomb==kGeoBombCyl) {
      change &= ((fExplodedView==kGeoBombCyl)?kFALSE:kTRUE);
   }
   if (ibomb==kGeoBombSph) {
      change &= ((fExplodedView==kGeoBombSph)?kFALSE:kTRUE);
   }
   fExplodedView = ibomb;
   if (change) ModifiedPad(); 
}

//______________________________________________________________________________
void TGeoPainter::SetNsegments(Int_t nseg)    
{
// Set number of segments to approximate circles
   if (nseg<3) {
      Warning("SetNsegments", "number of segments should be > 2");
      return;
   }
   if (fNsegments==nseg) return;
   fNsegments = nseg;
   ModifiedPad();
}

//______________________________________________________________________________
void TGeoPainter::SetNmeshPoints(Int_t npoints) {
// Set number of points to be generated on the shape outline when checking for overlaps.
   fChecker->SetNmeshPoints(npoints);
}   

//______________________________________________________________________________
void TGeoPainter::SetCheckedNode(TGeoNode *node) {
// Select a node to be checked for overlaps. All overlaps not involving it will
// be ignored.
   fChecker->SetSelectedNode(node);
}   

//______________________________________________________________________________
void TGeoPainter::SetVisLevel(Int_t level) {
// Set default level down to which visualization is performed
   if (level==fVisLevel && fLastVolume==fTopVolume) return;
   fVisLevel=level;
   if (!fTopVolume) return;
   if (fVisLock) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   if (!fLastVolume) {
//      printf("--- Drawing   %6d nodes with %d visible levels\n",fNVisNodes,fVisLevel);
      return;
   }   
   if (!gPad) return;
   if (gPad->GetView()) {
//      printf("--- Drawing   %6d nodes with %d visible levels\n",fNVisNodes,fVisLevel);
      ModifiedPad();
   }
}

//______________________________________________________________________________
void TGeoPainter::SetTopVisible(Bool_t vis)
{
// Set top geometry volume as visible.
   if (fTopVisible==vis) return;
   fTopVisible = vis;
   ModifiedPad();
}
   
//-----------------------------------------------------------------------------
void TGeoPainter::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
   if ((fVisOption<0) || (fVisOption>4)) {
      Warning("SetVisOption", "wrong visualization option");
      return;
   }
   
   if (option == kGeoVisChanged) {
      if (fVisLock) {
         ClearVisibleVolumes();
         fVisLock = kFALSE;
      }   
      ModifiedPad();
      return;
   }
   
   if (fTopVolume) {
      TGeoAtt *att = (TGeoAtt*)fTopVolume;
      att->SetAttBit(TGeoAtt::kVisBranch,kFALSE);
      att->SetAttBit(TGeoAtt::kVisContainers,kFALSE);
      att->SetAttBit(TGeoAtt::kVisOnly,kFALSE);
      switch (option) {
         case kGeoVisDefault:
            att->SetAttBit(TGeoAtt::kVisContainers,kTRUE);
            break;
         case kGeoVisLeaves:
            break;
         case kGeoVisOnly:   
            att->SetAttBit(TGeoAtt::kVisOnly,kTRUE);
            break;
      } 
   }            

   if (fVisOption==option) return;   
   fVisOption=option;
   if (fVisLock) {
      ClearVisibleVolumes();
      fVisLock = kFALSE;
   }   
   ModifiedPad();
}

//______________________________________________________________________________
Int_t TGeoPainter::ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const   
{   
//  Returns distance between point px,py on the pad an a shape.
   const Int_t inaxis = 7;
   const Int_t maxdist = 5;
   const Int_t big = 9999;
   Int_t dist = big;
   if (!gPad) return dist;
   TView *view = gPad->GetView();
   if (!(numpoints && view)) return dist;
   if (shape->IsA()==TGeoShapeAssembly::Class()) return dist;

   if (fIsPaintingShape) {
      Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
      Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
      Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
      Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
      // return if point not in user area
      if (px < puxmin - inaxis) return big;
      if (py > puymin + inaxis) return big;
      if (px > puxmax + inaxis) return big;
      if (py < puymax - inaxis) return big;
      if ((puxmax+inaxis-px) < 40) {
         // when the mouse points to the (40 pix) right edge of the pad, the manager class is selected
         gPad->SetSelected(fGeoManager);
         return 0;
      }
   }

   fBuffer->SetRawSizes(numpoints, 3*numpoints, 0, 0, 0, 0);
   Double_t *points = fBuffer->fPnts;
   shape->SetPoints(points);
   Double_t dpoint2, x1, y1, xndc[3];
   Double_t dmaster[3];
   Int_t j;
   for (Int_t i=0; i<numpoints; i++) {
      j = 3*i;
      TGeoShape::GetTransform()->LocalToMaster(&points[j], dmaster); 
      points[j]=dmaster[0]; points[j+1]=dmaster[1]; points[j+2]=dmaster[2];
      view->WCtoNDC(&points[j], xndc);
      x1 = gPad->XtoAbsPixel(xndc[0]);
      y1 = gPad->YtoAbsPixel(xndc[1]);
      dpoint2 = (px-x1)*(px-x1) + (py-y1)*(py-y1);
      if (dpoint2 < dist) dist=(Int_t)dpoint2;
   }
   if (dist > 100) return dist;
   dist = Int_t(TMath::Sqrt(Double_t(dist)));
   if (dist<maxdist && fIsPaintingShape) gPad->SetSelected((TObject*)shape);
   return dist;
}

//______________________________________________________________________________
void TGeoPainter::Test(Int_t npoints, Option_t *option)
{
// Check time of finding "Where am I" for n points.
   fChecker->Test(npoints, option);
}   

//______________________________________________________________________________
void TGeoPainter::TestOverlaps(const char* path)
{
//--- Geometry overlap checker based on sampling. 
   fChecker->TestOverlaps(path);
}   

//______________________________________________________________________________
Bool_t TGeoPainter::TestVoxels(TGeoVolume *vol)
{
// Check voxels efficiency per volume.
   return fChecker->TestVoxels(vol);
}   

//______________________________________________________________________________
void TGeoPainter::UnbombTranslation(const Double_t *tr, Double_t *bombtr)
{
// get the new 'unbombed' translation vector according current exploded view mode
   memcpy(bombtr, tr, 3*sizeof(Double_t));
   switch (fExplodedView) {
      case kGeoNoBomb:
         return;
      case kGeoBombXYZ:
         bombtr[0] /= fBombX;
         bombtr[1] /= fBombY;
         bombtr[2] /= fBombZ;
         return;
      case kGeoBombCyl:
         bombtr[0] /= fBombR;
         bombtr[1] /= fBombR;
         bombtr[2] /= fBombZ;
         return;
      case kGeoBombSph:
         bombtr[0] /= fBombR;
         bombtr[1] /= fBombR;
         bombtr[2] /= fBombR;
         return;
      default:
         return;
   }   
}
   
//______________________________________________________________________________
Double_t TGeoPainter::Weight(Double_t precision, Option_t *option)
{
// Compute weight [kg] of the current volume.
   return fChecker->Weight(precision, option);
}
   
   
