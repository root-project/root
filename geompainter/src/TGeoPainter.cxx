// @(#)root/geompainter:$Name:  $:$Id: TGeoPainter.cxx,v 1.27 2003/08/28 14:09:08 brun Exp $
// Author: Andrei Gheata   05/03/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TColor.h"
#include "TPoint.h"
#include "TView.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TPad.h"
#include "TH2F.h"

#include "TPolyMarker3D.h"
#include "TVirtualGL.h"

#include "TGeoSphere.h"
#include "TGeoPcon.h"
#include "TGeoTorus.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoManager.h"
#include "TGeoTrack.h"
#include "TGeoOverlap.h"
#include "TGeoChecker.h"
#include "TGeoPainter.h"

ClassImp(TGeoPainter)

//______________________________________________________________________________
TGeoPainter::TGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default constructor*-*-*-*-*-*-*-*-*
//*-*                  ====================================
   TVirtualGeoPainter::SetPainter(this);
   fNsegments = 20;
   fBombX = 1.3;
   fBombY = 1.3;
   fBombZ = 1.3;
   fBombR = 1.3;
   fVisLevel = 3;
   fVisOption = kGeoVisDefault;
   fExplodedView = 0;
   fVisBranch = "";
   fVisLock = kFALSE;
   fIsRaytracing = kFALSE;
   fTopVisible = kFALSE;
   fPaintingOverlaps = kFALSE;
   fVisVolumes = new TObjArray();
   fOverlap = 0;
   fMatrix = 0;
   fClippingShape = 0;
   memset(&fCheckedBox[0], 0, 6*sizeof(Double_t));
   
   if (gGeoManager) fGeom = gGeoManager;
   else Error("ctor", "No geometry loaded");
   fCheckedNode = fGeom->GetTopNode();
   fChecker = new TGeoChecker(fGeom);
   DefineColors();
}
//______________________________________________________________________________
TGeoPainter::~TGeoPainter()
{
//*-*-*-*-*-*-*-*-*-*-*Geometry painter default destructor*-*-*-*-*-*-*-*-*
//*-*                  ===================================
   if (fChecker) delete fChecker;
   delete fVisVolumes;
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
//______________________________________________________________________________
void TGeoPainter::CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const
{
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
//______________________________________________________________________________
void TGeoPainter::DefineColors() const
{
// Define 100 colors with increasing light intensities for each basic color (1-7)
// Register these colors at indexes starting with 300.
   TColor *color;
   Int_t i,j;
   Float_t r,g,b,h,l,s;
   
   for (i=1; i<8; i++) {
      color = (TColor*)gROOT->GetListOfColors()->At(i);
      color->GetHLS(h,l,s);
      for (j=0; j<100; j++) {
         l = 0.8*j/99.;
         TColor::HLS2RGB(h,l,s,r,g,b);
         new TColor(300+(i-1)*100+j, r,g,b);
      }
   }           
}

//______________________________________________________________________________
Int_t TGeoPainter::GetColor(Int_t base, Float_t light) const
{
// Get index of a base color with given light intensity (0,1)
   Int_t color, j;
   Int_t c = base%8;
   if (c==0) return c;
   if (light<0) {
      j=0;
   } else {
      if (light>0.8) j=99;
      else j = Int_t(99*light/0.8);
   }   
   color = 300 + (c-1)*100+j;
   return color;
}

//______________________________________________________________________________
Int_t TGeoPainter::DistanceToPrimitiveVol(TGeoVolume *vol, Int_t px, Int_t py)
{
// compute the closest distance of approach from point px,py to a volume 
   const Int_t big = 9999;
   const Int_t inaxis = 7;
   const Int_t maxdist = 5;
   
   TGeoBBox *box;

   Int_t puxmin = gPad->XtoAbsPixel(gPad->GetUxmin());
   Int_t puymin = gPad->YtoAbsPixel(gPad->GetUymin());
   Int_t puxmax = gPad->XtoAbsPixel(gPad->GetUxmax());
   Int_t puymax = gPad->YtoAbsPixel(gPad->GetUymax());
   // return if point not in user area
   if (px < puxmin - inaxis) return big;
   if (py > puymin + inaxis) return big;
   if (px > puxmax + inaxis) return big;
   if (py < puymax - inaxis) return big;
   
   TView *view = gPad->GetView();
   if (!view) return big;
   Int_t dist = big;
   Int_t id;
   
   if (fPaintingOverlaps) {
      TGeoVolume *crt;
      if (fOverlap->IsExtrusion()) {
         crt = fOverlap->GetVolume();
         fMatrix = gGeoIdentity;
         dist = crt->GetShape()->DistancetoPrimitive(px,py);
         if (dist<maxdist) {
            gPad->SetSelected(crt);
            box = (TGeoBBox*)crt->GetShape();
            fMatrix->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
            fCheckedBox[3] = box->GetDX();
            fCheckedBox[4] = box->GetDY();
            fCheckedBox[5] = box->GetDZ();
            return 0;
         }
      }   
      crt = fOverlap->GetNode(0)->GetVolume();
      fMatrix = fOverlap->GetNode(0)->GetMatrix();
      dist = crt->GetShape()->DistancetoPrimitive(px,py);
      if (dist<maxdist) {
         gPad->SetSelected(crt);
         box = (TGeoBBox*)crt->GetShape();
         fMatrix->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }
      if (fOverlap->IsExtrusion()) {
         gPad->SetSelected(view);
         return big;
      }
      crt = fOverlap->GetNode(1)->GetVolume();
      fMatrix = fOverlap->GetNode(1)->GetMatrix();
      dist = crt->GetShape()->DistancetoPrimitive(px,py);
      if (dist<maxdist) {
         gPad->SetSelected(crt);
         box = (TGeoBBox*)crt->GetShape();
         fMatrix->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
         fCheckedBox[3] = box->GetDX();
         fCheckedBox[4] = box->GetDY();
         fCheckedBox[5] = box->GetDZ();
         return 0;
      }      
      gPad->SetSelected(view);
      return dist;
   }
   
   if (fGeom->GetTopVolume() == vol) fGeom->CdTop();
   Int_t level = fGeom->GetLevel();
   TGeoNode *current = fGeom->GetCurrentNode();
   if (vol != current->GetVolume()) return 9999;
   Bool_t vis=(current->IsVisible() && (level || (!level && fTopVisible)) && fGeom->IsInPhiRange())?kTRUE:kFALSE;
   TGeoNode *node = 0;
   Int_t nd = vol->GetNdaughters();
   Bool_t last = kFALSE;
   fCheckedNode = fGeom->GetTopNode();
   switch (fVisOption) {
      case kGeoVisDefault:
         if (vis && (level<=fVisLevel)) { 
            dist = vol->GetShape()->DistancetoPrimitive(px,py);
            if (dist<maxdist) {
               gPad->SetSelected(vol);
	             fCheckedNode = current;
	             box = (TGeoBBox*)vol->GetShape();
	             fGeom->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
	             fCheckedBox[3] = box->GetDX();
	             fCheckedBox[4] = box->GetDY();
	             fCheckedBox[5] = box->GetDZ();
               return 0;
            }
         }
         // check daughters
         if (level<fVisLevel) {
            if ((!nd) || (!current->IsVisDaughters())) return dist;
            for (id=0; id<nd; id++) {
               node = vol->GetNode(id);
               fGeom->CdDown(id);
               dist = DistanceToPrimitiveVol(node->GetVolume(),px, py);
               if (dist==0) return 0;
               fGeom->CdUp();
            }
         }
         break;
      case kGeoVisLeaves:
         last = ((nd==0) || (level==fVisLevel))?kTRUE:kFALSE;
         if (vis && (last || (!current->IsVisDaughters()))) {
            dist = vol->GetShape()->DistancetoPrimitive(px, py);
            if (dist<maxdist) {
               gPad->SetSelected(vol);
	             fCheckedNode = current;
	             box = (TGeoBBox*)vol->GetShape();
	             fGeom->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
	             fCheckedBox[3] = box->GetDX();
	             fCheckedBox[4] = box->GetDY();
	             fCheckedBox[5] = box->GetDZ();
               return 0;
            }
         }
         if (last || (!current->IsVisDaughters())) return dist;
         for (id=0; id<nd; id++) {
            node = vol->GetNode(id);
            fGeom->CdDown(id);
            dist = DistanceToPrimitiveVol(node->GetVolume(),px,py);
            if (dist==0) return 0;
            fGeom->CdUp();
         }
         break;
      case kGeoVisOnly:
         dist = vol->GetShape()->DistancetoPrimitive(px, py);
         if (dist<maxdist) {
            gPad->SetSelected(vol);
            fCheckedNode = current;
	          box = (TGeoBBox*)vol->GetShape();
	          fGeom->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
	          fCheckedBox[3] = box->GetDX();
	          fCheckedBox[4] = box->GetDY();
	          fCheckedBox[5] = box->GetDZ();
            return 0;
         }
         break;
      case kGeoVisBranch:
         fGeom->cd(fVisBranch);
         while (fGeom->GetLevel()) {
            if (fGeom->GetCurrentVolume()->IsVisible()) {
               dist = fGeom->GetCurrentVolume()->GetShape()->DistancetoPrimitive(px, py);
               if (dist<maxdist) {
                  gPad->SetSelected(fGeom->GetCurrentVolume());
  	          fCheckedNode = current;
	          box = (TGeoBBox*)fGeom->GetCurrentVolume()->GetShape();
	          fGeom->LocalToMaster(box->GetOrigin(), &fCheckedBox[0]);
	          fCheckedBox[3] = box->GetDX();
	          fCheckedBox[4] = box->GetDY();
	          fCheckedBox[5] = box->GetDZ();
                  return 0;
               }
            }   
            fGeom->CdUp();
         }
         gPad->SetSelected(view);
	 fCheckedNode = gGeoManager->GetTopNode();      
         return big;   
      default:
	 fCheckedNode = gGeoManager->GetTopNode();      
         return big;
   }       
   if ((dist>maxdist) && !fGeom->GetLevel()) gPad->SetSelected(view);
   fCheckedNode = gGeoManager->GetTopNode();      
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
      gPad->Modified();
      gPad->Update();
   }
}   
//______________________________________________________________________________
void TGeoPainter::DefaultColors()
{   
// Set default volume colors according to tracking media
   TIter next(fGeom->GetListOfVolumes());
   TGeoVolume *vol;
   while ((vol=(TGeoVolume*)next()))
      vol->SetLineColor(vol->GetMaterial()->GetDefaultColor());
   if (gPad) {
      if (gPad->GetView()) {
         gPad->Modified();
         gPad->Update();
      }
   }
}   
//______________________________________________________________________________
void TGeoPainter::Draw(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   fPaintingOverlaps = kFALSE;
   fOverlap = 0;
   if (fVisOption==kGeoVisOnly) fGeom->SetVisOption(kGeoVisDefault);
   
   if (fVisLock) {
      fVisVolumes->Clear();
      fVisLock = kFALSE;
   }   
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   fGeom->GetTopVolume()->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = new TView(11);
      view->SetAutoRange(kTRUE);
      fGeom->GetTopVolume()->Paint("range");
      view->SetAutoRange(kFALSE);
      if (has_pad) gPad->Update();
   }
   if (!view->IsPerspective()) view->SetPerspective();
   fVisLock = kTRUE;
   printf("--- number of nodes on screen : %i\n", fVisVolumes->GetEntriesFast());
}
//______________________________________________________________________________
void TGeoPainter::DrawOverlap(void *ovlp, Option_t *option)
{
   TString opt = option;
   TGeoOverlap *overlap = (TGeoOverlap*)ovlp;
   if (!overlap) return;
   
   fPaintingOverlaps = kTRUE;
   fOverlap = overlap;
   opt.ToLower();
   if (fVisLock) {
      fVisVolumes->Clear();
      fVisLock = kFALSE;
   }   
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   overlap->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = new TView(11);
      view->SetAutoRange(kTRUE);
      PaintOverlap(overlap, "range");
      view->SetAutoRange(kFALSE);
      overlap->GetPolyMarker()->Draw("SAME");
      if (has_pad) gPad->Update();
   }
   if (!view->IsPerspective()) view->SetPerspective();
   fVisLock = kTRUE;
   printf("--- number of nodes on screen : %i\n", fVisVolumes->GetEntriesFast());
}
//______________________________________________________________________________
void TGeoPainter::DrawOnly(Option_t *option)
{
   TString opt = option;
   opt.ToLower();
   if (fVisLock) {
      fVisVolumes->Clear();
      fVisLock = kFALSE;
   }   
   fPaintingOverlaps = kFALSE;
   Bool_t has_pad = (gPad==0)?kFALSE:kTRUE;
   // Clear pad if option "same" not given
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!opt.Contains("same")) gPad->Clear();
   // append this volume to pad
   fGeom->GetCurrentVolume()->AppendPad(option);

   // Create a 3-D view
   TView *view = gPad->GetView();
   if (!view) {
      view = new TView(11);
      view->SetAutoRange(kTRUE);
      fVisOption = kGeoVisOnly;
      fGeom->GetCurrentVolume()->Paint("range");
      view->SetAutoRange(kFALSE);
      if (has_pad) gPad->Update();
   }
   if (!view->IsPerspective()) view->SetPerspective();
   fVisLock = kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoPainter::DrawCurrentPoint(Int_t color)
{
// Draw current point in the same view.
   if (!gPad) return;
   if (!gPad->GetView()) return;
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(color);
   Double_t *point = fGeom->GetCurrentPoint();
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
   fGeom->GetTopVolume()->Draw();   
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
      if (!track) continue;
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
void TGeoPainter::ExecuteVolumeEvent(TGeoVolume *volume, Int_t event, Int_t /*px*/, Int_t /*py*/)
{
// Execute mouse actions on a given volume.
   if (!gPad) return;
   gPad->SetCursor(kHand);
   if (fIsRaytracing) return;
   static Int_t width, color;
   switch (event) {
   case kMouseEnter:
      width = volume->GetLineWidth();
      color = volume->GetLineColor();
      volume->SetLineWidth(3);
      volume->SetLineColor(2);
      gPad->Modified();
      gPad->Update();
      break;
   
   case kMouseLeave:
      volume->SetLineWidth(width);
      volume->SetLineColor(color);
      gPad->Modified();
      gPad->Update();
      break;
   
   case kButton1Double:
      gPad->SetCursor(kWatch);
      GrabFocus();
//      volume->Draw();
      break;
   }
}
//______________________________________________________________________________
char *TGeoPainter::GetVolumeInfo(const TGeoVolume *volume, Int_t /*px*/, Int_t /*py*/) const
{
   const char *snull = "";
   if (!gPad) return (char*)snull;
   static char info[128];
   if (fPaintingOverlaps) {
      if (!fOverlap) {
         sprintf(info, "wrong overlapping flag");
         return info;
      }   
      TString ovtype, name;
      if (fOverlap->IsExtrusion()) {
         ovtype="EXTRUSION";
         if (volume==fOverlap->GetVolume()) name=volume->GetName();
         else name=fOverlap->GetNode(0)->GetName();
      } else {
         ovtype = "OVERLAP";
         if (volume==fOverlap->GetNode(0)->GetVolume()) name=fOverlap->GetNode(0)->GetName();
         else name=fOverlap->GetNode(1)->GetName();
      }   
      sprintf(info, "%s: %s of %g", name.Data(), ovtype.Data(), fOverlap->GetOverlap());
      return info;
   }   
   else sprintf(info,"%s, shape=%s", fGeom->GetPath(), volume->GetShape()->ClassName());
   return info;
}
//______________________________________________________________________________
TGeoChecker *TGeoPainter::GetChecker()
{
// Create/return geometry checker.
   if (!fChecker) fChecker = new TGeoChecker(fGeom);
   return fChecker;
}
 
//______________________________________________________________________________
void TGeoPainter::GetViewAngles(Double_t &longitude, Double_t &latitude, Double_t &psi) 
{
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
      TGeoBBox *box = (TGeoBBox*)fGeom->GetTopVolume()->GetShape();
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
Bool_t TGeoPainter::IsOnScreen(const TGeoNode *node) const
{
// check if this node is drawn. Assumes that this node is current
   
   //the following algorithm loops on all visible volumes: not very efficient
   //the solution is to build a sorted table of pointers and do a binary search
   if (fVisVolumes->IndexOf(node->GetVolume()) < 0) return kFALSE;
   return kTRUE;      
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
void TGeoPainter::ModifiedPad() const
{
// Check if a pad and view are present and send signal "Modified" to pad.
   if (!gPad) return;
   TView *view = gPad->GetView();
   if (!view) return;
   view->SetViewChanged();
   gPad->Modified();
   gPad->Update();
}   
//______________________________________________________________________________
void TGeoPainter::Paint(Option_t *option)
{
// Paint current geometry according to option.
//   printf("PaintNode(%s)\n", option);
   if (!fGeom) return;
   if (fVisOption==kGeoVisOnly) {
      fGeom->GetCurrentNode()->Paint(option);
      return;
   }
   fGeom->CdTop();
   TGeoNode *top = fGeom->GetTopNode();
   top->Paint(option);
   fVisLock = kTRUE;
   TString opt(option);
   opt.ToLower();
   if (strcmp(opt.Data(),"range") && fIsRaytracing) Raytrace();
}
//______________________________________________________________________________
void TGeoPainter::PaintOverlap(void *ovlp, Option_t *option)
{
// Paint an overlap.
   if (!fGeom) return;
   TGeoOverlap *overlap = (TGeoOverlap *)ovlp;
   if (!overlap) return;
   if (fOverlap != overlap) fOverlap = overlap;
   TGeoHMatrix *hmat = new TGeoHMatrix(); // id matrix
   TGeoVolume *vol = overlap->GetVolume();
   TGeoNode *node1, *node2;
   if (fOverlap->IsExtrusion()) {
      if (!fVisLock) fVisVolumes->Add(vol);
      fOverlap->SetLineColor(3);
      fOverlap->SetLineWidth(vol->GetLineWidth());
      vol->GetShape()->PaintNext(hmat, option);
      node1 = overlap->GetNode(0);
      *hmat = node1->GetMatrix();
      vol = node1->GetVolume();
      if (!fVisLock) fVisVolumes->Add(vol);
      fOverlap->SetLineColor(4);
      fOverlap->SetLineWidth(vol->GetLineWidth());
      vol->GetShape()->PaintNext(hmat, option);
   } else {
      node1 = overlap->GetNode(0);
      vol = node1->GetVolume();
      fOverlap->SetLineColor(3);
      fOverlap->SetLineWidth(vol->GetLineWidth());
      *hmat = node1->GetMatrix();
      if (!fVisLock) fVisVolumes->Add(vol);
      vol->GetShape()->PaintNext(hmat, option);
      node2 = overlap->GetNode(1);
      vol = node2->GetVolume();
      fOverlap->SetLineColor(4);
      fOverlap->SetLineWidth(vol->GetLineWidth());
      *hmat = node2->GetMatrix();
      if (!fVisLock) fVisVolumes->Add(vol);
      vol->GetShape()->PaintNext(hmat, option);
   }     
   delete hmat;
   fVisLock = kTRUE;
}
//______________________________________________________________________________
void TGeoPainter::PaintShape(X3DBuffer *buff, Bool_t rangeView, TGeoHMatrix *glmat)
{
//*-*-*-*-*Paint 3-D shape in current pad with its current attributes*-*-*-*-*
//*-*      ==========================================================
//
// rangeView = kTRUE - means no real painting
//                     just estimate the range
//                     of this shape only

    //*-* Paint in the pad
    //*-* Convert to the master system

    if (!buff) return;
    if (!fGeom) return;
    TGeoVolume *vol = fGeom->GetCurrentVolume();
    Float_t *point = &(buff->points[0]);
    Double_t dlocal[3];
    Double_t dmaster[3];
    if (fGeom) {
       for (Int_t j = 0; j < buff->numPoints; j++) {
           dlocal[0]=point[3*j]; dlocal[1]=point[3*j+1]; dlocal[2]=point[3*j+2];
           if (glmat) {
              glmat->LocalToMaster(&dlocal[0],&dmaster[0]);
           } else {   
              if (IsExplodedView()) 
                 fGeom->LocalToMasterBomb(&dlocal[0],&dmaster[0]);
              else   
                 fGeom->LocalToMaster(&dlocal[0],&dmaster[0]);
           }      
//           printf("point %i : %g %g %g\n", j,dmaster[0],dmaster[1],dmaster[2]);
           point[3*j]=dmaster[0]; point[3*j+1]=dmaster[1]; point[3*j+2]=dmaster[2];
       }
    }
    
    Float_t x0, y0, z0, x1, y1, z1;
    const Int_t kExpandView = 2;
    int i0;

    x0 = x1 = buff->points[0];
    y0 = y1 = buff->points[1];
    z0 = z1 = buff->points[2];

    if (!rangeView) {
      if (!fPaintingOverlaps) {
         ((TAttLine*)vol)->Modify();  //Change line attributes only if necessary
         ((TAttFill*)vol)->Modify();  //Change fill area attributes only if necessary
      } else {
         ((TAttLine*)fOverlap)->Modify();
      }   
    }

    for (Int_t i = 0; i < buff->numSegs; i++) {
        i0 = 3*buff->segs[3*i+1];
        Float_t *ptpoints_0 = &(buff->points[i0]);
        i0 = 3*buff->segs[3*i+2];
        Float_t *ptpoints_3 = &(buff->points[i0]);
        if (!rangeView) gPad->PaintLine3D(ptpoints_0, ptpoints_3);
        else {
            x0 = ptpoints_0[0] < x0 ? ptpoints_0[0] : x0;
            y0 = ptpoints_0[1] < y0 ? ptpoints_0[1] : y0;
            z0 = ptpoints_0[2] < z0 ? ptpoints_0[2] : z0;
            x1 = ptpoints_3[0] > x1 ? ptpoints_3[0] : x1;
            y1 = ptpoints_3[1] > y1 ? ptpoints_3[1] : y1;
            z1 = ptpoints_3[2] > z1 ? ptpoints_3[2] : z1;
        }
    }
    if (rangeView)
    {
      TView *view = gPad->GetView();
      if (view->GetAutoRange()) view->SetRange(x0,y0,z0,x1,y1,z1,kExpandView);
    }
}

//______________________________________________________________________________
void *TGeoPainter::MakeBox3DBuffer(const TGeoVolume *vol)
{
// Create a box 3D buffer for a given shape.
   X3DPoints *buff = new X3DPoints;
   const Int_t numpoints = 8;

   buff->numPoints = 8;

   Double_t *points = new Double_t[3*numpoints];
   TGeoShape *shape = vol->GetShape();

   shape->SetPoints(points);

   buff->points = points;
   return buff;
}   

//______________________________________________________________________________
void TGeoPainter::PaintBox(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint any type of box with 8 vertices
   const Int_t numpoints = 8;

//*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   shape->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView  && gPad->GetView3D()) gVirtualGL->PaintBrik(points);

 //==  for (Int_t i = 0; i < numpoints; i++)
 //            gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   

   Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
   if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }   

//*-* Allocate memory for segments *-*

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = 8;
        buff->numSegs   = 12;
        buff->numPolys  = (is3d)?6:0;
    }

//*-* Allocate memory for points *-*

    buff->points = points;
    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {
        buff->segs[ 0] = c;    buff->segs[ 1] = 0;    buff->segs[ 2] = 1;
        buff->segs[ 3] = c+1;  buff->segs[ 4] = 1;    buff->segs[ 5] = 2;
        buff->segs[ 6] = c+1;  buff->segs[ 7] = 2;    buff->segs[ 8] = 3;
        buff->segs[ 9] = c;    buff->segs[10] = 3;    buff->segs[11] = 0;
        buff->segs[12] = c+2;  buff->segs[13] = 4;    buff->segs[14] = 5;
        buff->segs[15] = c+2;  buff->segs[16] = 5;    buff->segs[17] = 6;
        buff->segs[18] = c+3;  buff->segs[19] = 6;    buff->segs[20] = 7;
        buff->segs[21] = c+3;  buff->segs[22] = 7;    buff->segs[23] = 4;
        buff->segs[24] = c;    buff->segs[25] = 0;    buff->segs[26] = 4;
        buff->segs[27] = c+2;  buff->segs[28] = 1;    buff->segs[29] = 5;
        buff->segs[30] = c+1;  buff->segs[31] = 2;    buff->segs[32] = 6;
        buff->segs[33] = c+3;  buff->segs[34] = 3;    buff->segs[35] = 7;
    }

//*-* Allocate memory for polygons *-*

    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];
       if (buff->polys) {
           buff->polys[ 0] = c;   buff->polys[ 1] = 4;  buff->polys[ 2] = 0;
           buff->polys[ 3] = 9;   buff->polys[ 4] = 4;  buff->polys[ 5] = 8;
           buff->polys[ 6] = c+1; buff->polys[ 7] = 4;  buff->polys[ 8] = 1;
           buff->polys[ 9] = 10;  buff->polys[10] = 5;  buff->polys[11] = 9;
           buff->polys[12] = c;   buff->polys[13] = 4;  buff->polys[14] = 2;
           buff->polys[15] = 11;  buff->polys[16] = 6;  buff->polys[17] = 10;
           buff->polys[18] = c+1; buff->polys[19] = 4;  buff->polys[20] = 3;
           buff->polys[21] = 8;   buff->polys[22] = 7;  buff->polys[23] = 11;
           buff->polys[24] = c+2; buff->polys[25] = 4;  buff->polys[26] = 0;
           buff->polys[27] = 3;   buff->polys[28] = 2;  buff->polys[29] = 1;
           buff->polys[30] = c+3; buff->polys[31] = 4;  buff->polys[32] = 4;
           buff->polys[33] = 5;   buff->polys[34] = 6;  buff->polys[35] = 7;
       }
    }
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}

//______________________________________________________________________________
void TGeoPainter::PaintCompositeShape(TGeoVolume *vol, Option_t *option)
{
// paint a composite shape
   PaintBox(vol->GetShape(), option);
}

//______________________________________________________________________________
void *TGeoPainter::MakeTorus3DBuffer(const TGeoVolume *vol)
{
// Create a torus 3D buffer for a given shape.
   Int_t n = fNsegments+1;
   TGeoShape *shape = vol->GetShape();
   TGeoTorus *tor = (TGeoTorus*)shape;
   if (!tor) return 0;
   X3DPoints *buff = new X3DPoints;
   Int_t numpoints = n*(n-1);
   Bool_t hasrmin = (tor->GetRmin()>0)?kTRUE:kFALSE;
   Bool_t hasphi  = (tor->GetDphi()<360)?kTRUE:kFALSE;
   if (hasrmin) numpoints *= 2;
   else if (hasphi) numpoints += 2;
   Double_t *points = new Double_t[3*numpoints];
   if (!points) return 0;

   shape->SetPoints(points);
   buff->points = points;
   return buff;
}   

//______________________________________________________________________________
void *TGeoPainter::MakeTube3DBuffer(const TGeoVolume *vol)
{
// Create a box 3D buffer for a given shape.
   X3DPoints *buff = new X3DPoints;
   Int_t n = fNsegments;
   const Int_t numpoints = 4*n;

   buff->numPoints = numpoints;

   Double_t *points = new Double_t[3*numpoints];
   TGeoShape *shape = vol->GetShape();

   shape->SetPoints(points);

   buff->points = points;
   return buff;
}   

//______________________________________________________________________________
void TGeoPainter::PaintTorus(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint a torus in pad or x3d
   Int_t i, j;
   const Int_t n = fNsegments+1;
   Int_t indx, indp, startcap=0;
   
   TGeoTorus *tor = (TGeoTorus*)shape;
   if (!tor) return;
   Int_t numpoints = n*(n-1);
   Bool_t hasrmin = (tor->GetRmin()>0)?kTRUE:kFALSE;
   Bool_t hasphi  = (tor->GetDphi()<360)?kTRUE:kFALSE;
   if (hasrmin) numpoints *= 2;
   else if (hasphi) numpoints += 2;

   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   shape->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
//   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points,-n,2);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);
   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   
   
    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints =   numpoints;
        buff->numSegs = (2*n-1)*(n-1);
        buff->numPolys = (n-1)*(n-1);
        if (hasrmin) {
           buff->numSegs   += (2*n-1)*(n-1);
           buff->numPolys  += (n-1)*(n-1);
        }   
        if (is3d && hasphi)  {
           buff->numSegs   += 2*(n-1);
           buff->numPolys  += 2*(n-1);
        }   
//        if (!is3d) buff->numPolys = 0;
    }

    buff->points = points;

    Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }
//*-* Allocate memory for segments *-*

    indp = n*(n-1); // start index for points on inner surface
    buff->segs = new Int_t[buff->numSegs*3];
    memset(buff->segs, 0, buff->numSegs*3*sizeof(Int_t));
    if (buff->segs) {
       // outer surface phi circles = n*(n-1) -> [0, n*(n-1) -1]
       // connect point j with point j+1 on same row
       indx = 0;
       for (i = 0; i < n; i++) { // rows [0,n-1]
          for (j = 0; j < n-1; j++) {  // points on a row [0, n-2]
             buff->segs[indx+(i*(n-1)+j)*3] = c;
             buff->segs[indx+(i*(n-1)+j)*3+1] = i*(n-1)+j;   // j on row i
             buff->segs[indx+(i*(n-1)+j)*3+2] = i*(n-1)+((j+1)%(n-1)); // j+1 on row i
          }
       }
       indx += 3*n*(n-1);
       // outer surface generators = (n-1)*(n-1) -> [n*(n-1), (2*n-1)*(n-1) -1]
       // connect point j on row i with point j on row i+1
       for (i = 0; i < n-1; i++) { // rows [0, n-2]
          for (j = 0; j < n-1; j++) {  // points on a row [0, n-2]
             buff->segs[indx+(i*(n-1)+j)*3] = c;
             buff->segs[indx+(i*(n-1)+j)*3+1] = i*(n-1)+j;     // j on row i
             buff->segs[indx+(i*(n-1)+j)*3+2] = (i+1)*(n-1)+j; // j on row i+1
          }
       }
       indx += 3*(n-1)*(n-1);
       startcap = (2*n-1)*(n-1);
                
       if (hasrmin) {
          // inner surface phi circles = n*(n-1) -> [(2*n-1)*(n-1), (3*n-1)*(n-1) -1]
          // connect point j with point j+1 on same row
          for (i = 0; i < n; i++) { // rows [0, n-1]
             for (j = 0; j < n-1; j++) {  // points on a row [0, n-2]
                buff->segs[indx+(i*(n-1)+j)*3] = c;              // lighter color
                buff->segs[indx+(i*(n-1)+j)*3+1] = indp + i*(n-1)+j;   // j on row i
                buff->segs[indx+(i*(n-1)+j)*3+2] = indp + i*(n-1)+((j+1)%(n-1)); // j+1 on row i
             }
          }
          indx += 3*n*(n-1);
          // inner surface generators = (n-1)*n -> [(3*n-1)*(n-1), (4*n-2)*(n-1) -1]
          // connect point j on row i with point j on row i+1
          for (i = 0; i < n-1; i++) { // rows [0, n-2]
             for (j = 0; j < n-1; j++) {  // points on a row [0, n-2]
                buff->segs[indx+(i*(n-1)+j)*3] = c;                // lighter color
                buff->segs[indx+(i*(n-1)+j)*3+1] = indp + i*(n-1)+j;     // j on row i
                buff->segs[indx+(i*(n-1)+j)*3+2] = indp + (i+1)*(n-1)+j; // j on row i+1
             }
          }
          indx += 3*(n-1)*(n-1);
          startcap = (4*n-2)*(n-1);
       }   

       if (is3d && hasphi) {
          if (hasrmin) {
           // endcaps = 2*(n-1) -> [(4*n-2)*(n-1), 4*n*(n-1)-1]
             i = 0;
             for (j = 0; j < n-1; j++) { 
                buff->segs[indx+j*3] = c+1;
                buff->segs[indx+j*3+1] = (n-1)*i+j;     // outer j on row 0
                buff->segs[indx+j*3+2] = indp+(n-1)*i+j; // inner j on row 0
             }
             indx += 3*(n-1);
             i = n-1;   
             for (j = 0; j < n-1; j++) { 
                buff->segs[indx+j*3] = c+1;
                buff->segs[indx+j*3+1] = (n-1)*i+j;     // outer j on row n-1
                buff->segs[indx+j*3+2] = indp+(n-1)*i+j; // inner j on row n-1
             }
             indx += 3*(n-1);
          } else {
             i = 0;
             for (j = 0; j < n-1; j++) { 
                buff->segs[indx+j*3] = c+1;
                buff->segs[indx+j*3+1] = (n-1)*i+j;     // outer j on row 0
                buff->segs[indx+j*3+2] = n*(n-1);       // center of first endcap
             }
             indx += 3*(n-1);
             i = n-1;   
             for (j = 0; j < n-1; j++) { 
                buff->segs[indx+j*3] = c+1;
                buff->segs[indx+j*3+1] = (n-1)*i+j;     // outer j on row n-1
                buff->segs[indx+j*3+2] = n*(n-1)+1;     // center of second endcap
             }
             indx += 3*(n-1);
          }
       }
    }               
                   
//*-* Allocate memory for polygons *-*

    indx = 0;
    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];
       memset(buff->polys, 0, buff->numPolys*6*sizeof(Int_t));
       if (buff->polys) {
          // outer surface = (n-1)*(n-1) -> [0, (n-1)*(n-1)-1]
          // normal pointing out
          for (i=0; i<n-1; i++) {
             for (j=0; j<n-1; j++) {
                buff->polys[indx++] = c;
                buff->polys[indx++] = 4;
                buff->polys[indx++] = (n-1)*i+j; // seg j on outer row i
                buff->polys[indx++] = n*(n-1)+(n-1)*i+j; // generator j on outer row i
                buff->polys[indx++] = (n-1)*(i+1)+j; // seg j on outer row i+1
                buff->polys[indx++] = n*(n-1)+(n-1)*i+((j+1)%(n-1)); // generator j+1 on outer row i
             }   
          }
          if (hasrmin) {
             indp = (2*n-1)*(n-1); // start index of inner segments
             // inner surface = (n-1)*(n-1) -> [(n-1)*(n-1), 2*(n-1)*(n-1)-1] 
             // normal pointing out 
             for (i=0; i<n-1; i++) {
                for (j=0; j<n-1; j++) {
                   buff->polys[indx++] = c;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = indp+(n-1)*i+j; // seg j on inner row i
                   buff->polys[indx++] = indp+n*(n-1)+(n-1)*i+((j+1)%(n-1)); // generator j+1 on inner row i
                   buff->polys[indx++] = indp+(n-1)*(i+1)+j; // seg j on inner row i+1
                   buff->polys[indx++] = indp+n*(n-1)+(n-1)*i+j; // generator j on inner row i
                }   
             }
          }   
          if (hasphi) {
             // endcaps = 2*(n-1) -> [2*(n-1)*(n-1), 2*n*(n-1)-1]
             i=0; // row 0
             Int_t np = (hasrmin)?4:3;
             for (j=0; j<n-1; j++) {
                buff->polys[indx++] = c+1;
                buff->polys[indx++] = np;
                buff->polys[indx++] = (n-1)*i+j;         // seg j on outer row 0
                buff->polys[indx++] = startcap+((j+1)%(n-1)); // endcap j+1 on row 0
                if (hasrmin)
                   buff->polys[indx++] = indp+(n-1)*i+j; // seg j on inner row 0
                buff->polys[indx++] = startcap+j;        // endcap j on row 0
             }   

             i=n-1; // row n-1
             for (j=0; j<n-1; j++) {
                buff->polys[indx++] = c+1;
                buff->polys[indx++] = np;
                buff->polys[indx++] = (n-1)*i+j;         // seg j on outer row n-1
                buff->polys[indx++] = startcap+(n-1)+j;      // endcap j on row n-1
                if (hasrmin)
                   buff->polys[indx++] = indp+(n-1)*i+j; // seg j on inner row n-1
                buff->polys[indx++] = startcap+(n-1)+((j+1)%(n-1));    // endcap j+1 on row n-1
             } 
          }
       }
    }           
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}

//______________________________________________________________________________
void TGeoPainter::PaintTube(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint tubes
   Int_t i, j;
   Int_t n = fNsegments;
   const Int_t numpoints = 4*n;

//*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   shape->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, n, 2);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        if (is3d) {
           buff->numSegs   = n*8;
           buff->numPolys  = n*4;
        } else {                        
           buff->numSegs   = n*6;
           buff->numPolys  = 0;
        }   
    }


//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }   

//*-* Allocate memory for segments *-*

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {
        for (i = 0; i < 4; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c;
                buff->segs[(i*n+j)*3+1] = i*n+j;
                buff->segs[(i*n+j)*3+2] = i*n+j+1;
            }
            buff->segs[(i*n+j-1)*3+2] = i*n;
        }
        for (i = 4; i < 6; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c+1;
                buff->segs[(i*n+j)*3+1] = (i-4)*n+j;
                buff->segs[(i*n+j)*3+2] = (i-2)*n+j;
            }
        }
        if (is3d) {
           for (i = 6; i < 8; i++) {
              for (j = 0; j < n; j++) {
                 buff->segs[(i*n+j)*3  ] = c;
                 buff->segs[(i*n+j)*3+1] = 2*(i-6)*n+j;
                 buff->segs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
              }
           }
        }
    }
//*-* Allocate memory for polygons *-*

    Int_t indx = 0;

    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];
       if (buff->polys) {
           for (i = 0; i < 2; i++) {
               for (j = 0; j < n; j++) {
                   indx = 6*(i*n+j);
                   buff->polys[indx  ] = c;
                   buff->polys[indx+1] = 4;
                   buff->polys[indx+2] = i*n+j;
                   buff->polys[indx+3] = (4+i)*n+j;
                   buff->polys[indx+4] = (2+i)*n+j;
                   buff->polys[indx+5] = (4+i)*n+j+1;
               }
               buff->polys[indx+5] = (4+i)*n;
           }
           for (i = 2; i < 4; i++) {
               for (j = 0; j < n; j++) {
                   indx = 6*(i*n+j);
                   buff->polys[indx  ] = c+i;
                   buff->polys[indx+1] = 4;
                   buff->polys[indx+2] = (i-2)*2*n+j;
                   buff->polys[indx+3] = (4+i)*n+j;
                   buff->polys[indx+4] = ((i-2)*2+1)*n+j;
                   buff->polys[indx+5] = (4+i)*n+j+1;
               }
               buff->polys[indx+5] = (4+i)*n;
           }
       }
    }
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    if (buff->points)   delete [] buff->points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}

//______________________________________________________________________________
void *TGeoPainter::MakeTubs3DBuffer(const TGeoVolume *vol)
{
// Create a box 3D buffer for a given shape.
   X3DPoints *buff = new X3DPoints;

   const Int_t n = fNsegments+1;
   const Int_t numpoints = 4*n;

   //*-* Allocate memory for points *-*

   Double_t *points = new Double_t[3*numpoints];

   buff->numPoints =   numpoints;

   TGeoShape *shape = vol->GetShape();
   shape->SetPoints(points);
   buff->points = points;
   return buff;
}   
//______________________________________________________________________________
void TGeoPainter::PaintTubs(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint tubes
   Int_t i, j;
   const Int_t n = fNsegments+1;
   const Int_t numpoints = 4*n;

   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;

   shape->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points,-n,2);

//==   for (i = 0; i < numpoints; i++)
//==            gNode->Local2Master(&points[3*i],&points[3*i]);
   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   
   
    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints =   numpoints;
        if (is3d)  {
           buff->numSegs   = 2*numpoints;
           buff->numPolys  = numpoints-2;
        } else { 
           buff->numSegs   = 6*n+4;
           buff->numPolys  = 0;
        }   
    }

    buff->points = points;

    Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }
//*-* Allocate memory for segments *-*

    buff->segs = new Int_t[buff->numSegs*3];
    memset(buff->segs, 0, buff->numSegs*3*sizeof(Int_t));
    if (buff->segs) {
        for (i = 0; i < 4; i++) {
            for (j = 1; j < n; j++) {
                buff->segs[(i*n+j-1)*3  ] = c;
                buff->segs[(i*n+j-1)*3+1] = i*n+j-1;
                buff->segs[(i*n+j-1)*3+2] = i*n+j;
            }
        }
        for (i = 4; i < 6; i++) {
            for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c+1;
                buff->segs[(i*n+j)*3+1] = (i-4)*n+j;
                buff->segs[(i*n+j)*3+2] = (i-2)*n+j;
            }
        }
        if (is3d) {
           for (i = 6; i < 8; i++) {
              for (j = 0; j < n; j++) {
                buff->segs[(i*n+j)*3  ] = c;
                buff->segs[(i*n+j)*3+1] = 2*(i-6)*n+j;
                buff->segs[(i*n+j)*3+2] = (2*(i-6)+1)*n+j;
              }
           }
        } else {   
           buff->segs[6*n*3] = c;
           buff->segs[6*n*3+1] = 0;
           buff->segs[6*n*3+2] = n;
           buff->segs[6*n*3+3] = c;
           buff->segs[6*n*3+4] = n-1;
           buff->segs[6*n*3+5] = 2*n-1;
           buff->segs[6*n*3+6] = c;
           buff->segs[6*n*3+7] = 2*n;
           buff->segs[6*n*3+8] = 3*n;
           buff->segs[6*n*3+9] = c;
           buff->segs[6*n*3+10] = 3*n-1;
           buff->segs[6*n*3+11] = 4*n-1;
        }   
    }

//*-* Allocate memory for polygons *-*

    Int_t indx = 0;
    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];
       memset(buff->polys, 0, buff->numPolys*6*sizeof(Int_t));
       if (buff->polys) {
           for (i = 0; i < 2; i++) {
               for (j = 0; j < n-1; j++) {
                   buff->polys[indx++] = c;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = i*n+j;
                   buff->polys[indx++] = (4+i)*n+j;
                   buff->polys[indx++] = (2+i)*n+j;
                   buff->polys[indx++] = (4+i)*n+j+1;
               }
           }
           for (i = 2; i < 4; i++) {
               for (j = 0; j < n-1; j++) {
                   buff->polys[indx++] = c+i;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = (i-2)*2*n+j;
                   buff->polys[indx++] = (4+i)*n+j;
                   buff->polys[indx++] = ((i-2)*2+1)*n+j;
                   buff->polys[indx++] = (4+i)*n+j+1;
               }
           }
           buff->polys[indx++] = c+2;
           buff->polys[indx++] = 4;
           buff->polys[indx++] = 6*n;
           buff->polys[indx++] = 4*n;
           buff->polys[indx++] = 7*n;
           buff->polys[indx++] = 5*n;

           buff->polys[indx++] = c+2;
           buff->polys[indx++] = 4;
           buff->polys[indx++] = 7*n-1;
           buff->polys[indx++] = 5*n-1;
           buff->polys[indx++] = 8*n-1;
           buff->polys[indx++] = 6*n-1;
       }
    }
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}
//______________________________________________________________________________
void *TGeoPainter::MakeSphere3DBuffer(const TGeoVolume *vol)
{
// Create a box 3D buffer for a given shape.
   X3DPoints *buff = new X3DPoints;

   TGeoShape *shape = vol->GetShape();
   ((TGeoSphere*)shape)->SetNumberOfDivisions(fNsegments);
   const Int_t n = ((TGeoSphere*)shape)->GetNumberOfDivisions()+1;
   Int_t nz = ((TGeoSphere*)shape)->GetNz()+1;
   if (nz < 2) return 0;
   Int_t numpoints = 2*n*nz;
   if (numpoints <= 0) return 0;

   //*-* Allocate memory for points *-*

   Double_t *points = new Double_t[3*numpoints];

   buff->numPoints = numpoints;

   shape->SetPoints(points);
   buff->points = points;
   return buff;
}

//______________________________________________________________________________
void TGeoPainter::PaintSphere(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint a sphere
   Int_t i, j;
   ((TGeoSphere*)shape)->SetNumberOfDivisions(fNsegments);
   const Int_t n = ((TGeoSphere*)shape)->GetNumberOfDivisions()+1;
   Double_t ph1 = ((TGeoSphere*)shape)->GetPhi1();
   Double_t ph2 = ((TGeoSphere*)shape)->GetPhi2();
   Int_t nz = ((TGeoSphere*)shape)->GetNz()+1;
   if (nz < 2) return;
   Int_t numpoints = 2*n*nz;
   if (numpoints <= 0) return;
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   shape->SetPoints(points);

   Bool_t rangeView = option && *option && strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, -n, nz);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);
   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   

   Bool_t specialCase = kFALSE;

   if (TMath::Abs(TMath::Sin(2*(ph2 - ph1))) <= 0.01)  //mark this as a very special case, when
         specialCase = kTRUE;                                  //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
        buff->numPolys  = (is3d)?(2*(nz*n-1+(specialCase == kTRUE))):0;
    }

//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }

//*-* Allocate memory for segments *-*

    Int_t indx, indx2, k;
    indx = indx2 = 0;

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {

        //inside & outside spheres, number of segments: 2*nz*(n-1)
        //             special case number of segments: 2*nz*n
        for (i = 0; i < nz*2; i++) {
            indx2 = i*n;
            for (j = 1; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2+j;
            }
            if (specialCase) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2;
            }
        }

        //bottom & top lines, number of segments: 2*n
        for (i = 0; i < 2; i++) {
            indx2 = i*(nz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside spheres, number of segments: 2*(nz-1)*n
        for (i = 0; i < (nz-1); i++) {

            //inside sphere
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside sphere
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(nz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (nz-1); i++) {
                for (j = 0; j < 2; j++) {
                    buff->segs[indx++] = c;
                    buff->segs[indx++] =  2*i    * n + j*(n-1);
                    buff->segs[indx++] = (2*i+1) * n + j*(n-1);
                }
            }
        }
    }


    Int_t m = n - 1 + (specialCase == kTRUE);

//*-* Allocate memory for polygons *-*

    indx = 0;
    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];

       if (buff->polys) {

           //bottom & top, number of polygons: 2*(n-1)
           // special case number of polygons: 2*n
           for (i = 0; i < 2; i++) {
               for (j = 0; j < n-1; j++) {
                   buff->polys[indx++] = c+3;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = 2*nz*m+i*n+j;
                   buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                   buff->polys[indx++] = 2*nz*m+i*n+j+1;
                   buff->polys[indx++] = i*(nz*2-2)*m+j;
               }
               if (specialCase) {
                   buff->polys[indx++] = c+3;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = 2*nz*m+i*n+j;
                   buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                   buff->polys[indx++] = 2*nz*m+i*n;
                   buff->polys[indx++] = i*(nz*2-2)*m+j;
               }
           }


           //inside & outside, number of polygons: (nz-1)*2*(n-1)
           for (k = 0; k < (nz-1); k++) {
               for (i = 0; i < 2; i++) {
                   for (j = 0; j < n-1; j++) {
                       buff->polys[indx++] = c+i;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = (2*k+i*1)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                       buff->polys[indx++] = (2*k+i*1+2)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j+1;
                   }
                   if (specialCase) {
                       buff->polys[indx++] = c+i;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = (2*k+i*1)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                       buff->polys[indx++] = (2*k+i*1+2)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n;
                   }
               }
           }


           //left & right sections, number of polygons: 2*(nz-1)
           //          special case number of polygons: 0
           if (!specialCase) {
               indx2 = nz*2*(n-1);
               for (k = 0; k < (nz-1); k++) {
                   for (i = 0; i < 2; i++) {
                       buff->polys[indx++] = c+2;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*nz*n+2*(k-1)+i;
                       buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                       buff->polys[indx++] = indx2+2*nz*n+2*k+i;
                       buff->polys[indx++] = indx2+(2*k+3)*n+i*(n-1);
                   }
               }
               buff->polys[indx-8] = indx2+n;
               buff->polys[indx-2] = indx2+2*n-1;
           }
       }
    }
    
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}

//______________________________________________________________________________
void *TGeoPainter::MakePcon3DBuffer(const TGeoVolume *vol)
{
// Create a box 3D buffer for a given shape.
   X3DPoints *buff = new X3DPoints;

   TGeoShape *shape = vol->GetShape();
   const Int_t n = ((TGeoPcon*)shape)->GetNsegments()+1;
   Int_t nz = ((TGeoPcon*)shape)->GetNz();
   if (nz < 2) return 0;
   Int_t numpoints =  nz*2*n;
   if (numpoints <= 0) return 0;
   Double_t *points = new Double_t[3*numpoints];
   shape->SetPoints(points);
   buff->numPoints = numpoints;
   buff->points = points;
   return buff;
}

//______________________________________________________________________________
void TGeoPainter::PaintPcon(TGeoShape *shape, Option_t *option, TGeoHMatrix *glmat)
{
// paint a pcon
   Int_t i, j;
   const Int_t n = ((TGeoPcon*)shape)->GetNsegments()+1;
   Int_t nz = ((TGeoPcon*)shape)->GetNz();
   if (nz < 2) return;
   Int_t numpoints =  nz*2*n;
   if (numpoints <= 0) return;
   Double_t dphi = ((TGeoPcon*)shape)->GetDphi();
   //*-* Allocate memory for points *-*

   Float_t *points = new Float_t[3*numpoints];
   if (!points) return;
   shape->SetPoints(points);

   Bool_t rangeView = strcmp(option,"range")==0 ? kTRUE : kFALSE;
   if (!rangeView && gPad->GetView3D()) gVirtualGL->PaintCone(points, -n, nz);

 //==  for (i = 0; i < numpoints; i++)
 //==          gNode->Local2Master(&points[3*i],&points[3*i]);

   Bool_t is3d = kFALSE;
   if (strstr(option, "x3d")) is3d=kTRUE;   

      Bool_t specialCase = kFALSE;

   if (dphi == 360)           //mark this as a very special case, when
        specialCase = kTRUE;     //we have to draw this PCON like a TUBE

    X3DBuffer *buff = new X3DBuffer;
    if (buff) {
        buff->numPoints = numpoints;
        buff->numSegs   = 4*(nz*n-1+(specialCase == kTRUE));
        buff->numPolys  = (is3d)?(2*(nz*n-1+(specialCase == kTRUE))):0;
    }

//*-* Allocate memory for points *-*

    buff->points = points;

    Int_t c = ((fGeom->GetCurrentVolume()->GetLineColor() % 8) - 1) * 4;     // Basic colors: 0, 1, ... 7
    if (c < 0) c = 0;
   if (fPaintingOverlaps) {
      if (fOverlap->IsExtrusion()) {
         if (fOverlap->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      } else {
         if (fOverlap->GetNode(0)->GetVolume()->GetShape()==shape) c=8;
         else c=12;
      }   
   }

//*-* Allocate memory for segments *-*

    Int_t indx, indx2, k;
    indx = indx2 = 0;

    buff->segs = new Int_t[buff->numSegs*3];
    if (buff->segs) {

        //inside & outside circles, number of segments: 2*nz*(n-1)
        //             special case number of segments: 2*nz*n
        for (i = 0; i < nz*2; i++) {
            indx2 = i*n;
            for (j = 1; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2+j;
            }
            if (specialCase) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j-1;
                buff->segs[indx++] = indx2;
            }
        }

        //bottom & top lines, number of segments: 2*n
        for (i = 0; i < 2; i++) {
            indx2 = i*(nz-1)*2*n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n+j;
            }
        }

        //inside & outside cilindres, number of segments: 2*(nz-1)*n
        for (i = 0; i < (nz-1); i++) {

            //inside cilinder
            indx2 = i*n*2;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+2;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
            //outside cilinder
            indx2 = i*n*2+n;
            for (j = 0; j < n; j++) {
                buff->segs[indx++] = c+3;
                buff->segs[indx++] = indx2+j;
                buff->segs[indx++] = indx2+n*2+j;
            }
        }

        //left & right sections, number of segments: 2*(nz-2)
        //          special case number of segments: 0
        if (!specialCase) {
            for (i = 1; i < (nz-1); i++) {
                for (j = 0; j < 2; j++) {
                    buff->segs[indx++] = c;
                    buff->segs[indx++] =  2*i    * n + j*(n-1);
                    buff->segs[indx++] = (2*i+1) * n + j*(n-1);
                }
            }
        }
    }


    Int_t m = n - 1 + (specialCase == kTRUE);

//*-* Allocate memory for polygons *-*

    indx = 0;

    buff->polys = 0;
    if (is3d) {
       buff->polys = new Int_t[buff->numPolys*6];

       if (buff->polys) {

           //bottom & top, number of polygons: 2*(n-1)
           // special case number of polygons: 2*n
           for (i = 0; i < 2; i++) {
               for (j = 0; j < n-1; j++) {
                   buff->polys[indx++] = c+3;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = 2*nz*m+i*n+j;
                   buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                   buff->polys[indx++] = 2*nz*m+i*n+j+1;
                   buff->polys[indx++] = i*(nz*2-2)*m+j;
               }
               if (specialCase) {
                   buff->polys[indx++] = c+3;
                   buff->polys[indx++] = 4;
                   buff->polys[indx++] = 2*nz*m+i*n+j;
                   buff->polys[indx++] = i*(nz*2-2)*m+m+j;
                   buff->polys[indx++] = 2*nz*m+i*n;
                   buff->polys[indx++] = i*(nz*2-2)*m+j;
               }
           }


           //inside & outside, number of polygons: (nz-1)*2*(n-1)
           for (k = 0; k < (nz-1); k++) {
               for (i = 0; i < 2; i++) {
                   for (j = 0; j < n-1; j++) {
                       buff->polys[indx++] = c+i;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = (2*k+i*1)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                       buff->polys[indx++] = (2*k+i*1+2)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j+1;
                   }
                   if (specialCase) {
                       buff->polys[indx++] = c+i;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = (2*k+i*1)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n+j;
                       buff->polys[indx++] = (2*k+i*1+2)*m+j;
                       buff->polys[indx++] = nz*2*m+(2*k+i*1+2)*n;
                   }
               }
           }


           //left & right sections, number of polygons: 2*(nz-1)
           //          special case number of polygons: 0
           if (!specialCase) {
               indx2 = nz*2*(n-1);
               for (k = 0; k < (nz-1); k++) {
                   for (i = 0; i < 2; i++) {
                       buff->polys[indx++] = c+2;
                       buff->polys[indx++] = 4;
                       buff->polys[indx++] = k==0 ? indx2+i*(n-1) : indx2+2*nz*n+2*(k-1)+i;
                       buff->polys[indx++] = indx2+2*(k+1)*n+i*(n-1);
                       buff->polys[indx++] = indx2+2*nz*n+2*k+i;
                       buff->polys[indx++] = indx2+(2*k+3)*n+i*(n-1);
                   }
               }
               buff->polys[indx-8] = indx2+n;
               buff->polys[indx-2] = indx2+2*n-1;
           }
       }
    }
    //*-* Paint in the pad
    PaintShape(buff,rangeView, glmat);

    if (is3d) {
        if(buff && buff->points && buff->segs)
            FillX3DBuffer(buff);
        else {
            gSize3D.numPoints -= buff->numPoints;
            gSize3D.numSegs   -= buff->numSegs;
            gSize3D.numPolys  -= buff->numPolys;
        }
    }

    delete [] points;
    if (buff->segs)     delete [] buff->segs;
    if (buff->polys)    delete [] buff->polys;
    if (buff)           delete    buff;
}
//______________________________________________________________________________
void TGeoPainter::PaintNode(TGeoNode *node, Option_t *option)
{
// paint recursively a node and its content accordind to visualization options
   TGeoNode *daughter = 0;
   TGeoVolume *vol = node->GetVolume();
   Int_t nd = node->GetNdaughters();
   Bool_t last = kFALSE;
   Int_t level = fGeom->GetLevel();
   Bool_t vis=(node->IsVisible() && (level || (!level && fTopVisible)) && fGeom->IsInPhiRange())?kTRUE:kFALSE;
   Int_t id;
   switch (fVisOption) {
      case kGeoVisDefault:
         if (vis && (level<=fVisLevel)) {
            if (!fIsRaytracing) vol->GetShape()->Paint(option);
            if (!fVisLock) fVisVolumes->Add(vol);
         }   
            // draw daughters
         if (level<fVisLevel) {
            if ((!nd) || (!node->IsVisDaughters())) return;
            for (id=0; id<nd; id++) {
               daughter = node->GetDaughter(id);
               fGeom->CdDown(id);
               PaintNode(daughter, option);
               fGeom->CdUp();
            }
         }
         break;
      case kGeoVisLeaves:
         if (level>fVisLevel) return;
         last = ((nd==0) || (level==fVisLevel))?kTRUE:kFALSE;
         if (vis && (last || (!node->IsVisDaughters()))) {
            if (!fIsRaytracing) vol->GetShape()->Paint(option);
            if (!fVisLock) fVisVolumes->Add(vol);
         }            
         if (last || (!node->IsVisDaughters())) return;
         for (id=0; id<nd; id++) {
            daughter = node->GetDaughter(id);
            fGeom->CdDown(id);
            PaintNode(daughter, option);
            fGeom->CdUp();
         }
         break;
      case kGeoVisOnly:
         if (!fIsRaytracing) vol->GetShape()->Paint(option);
         if (!fVisLock) fVisVolumes->Add(vol);
         break;
      case kGeoVisBranch:
         fGeom->cd(fVisBranch);
         while (fGeom->GetLevel()) {
            if (fGeom->GetCurrentVolume()->IsVisible()) {
               if (!fIsRaytracing) fGeom->GetCurrentVolume()->GetShape()->Paint(option);
               if (!fVisLock) fVisVolumes->Add(fGeom->GetCurrentVolume());
            }   
            fGeom->CdUp();
         }
         break;
      default:
         return;
   }
} 

//______________________________________________________________________________
void TGeoPainter::PrintOverlaps() const
{
   fChecker->PrintOverlaps();
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
   if (!view || ! view->IsPerspective()) return;
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
   fGeom->InitTrack(cop, dir);
   if (fClippingShape) inclipst = inclip = fClippingShape->Contains(cop);
   Int_t px, py;
   Double_t xloc, yloc, modloc;
   Int_t pxmin,pxmax, pymin,pymax;
   pxmin = gPad->UtoAbsPixel(0);
   pxmax = gPad->UtoAbsPixel(1);
   pymin = gPad->VtoAbsPixel(1);
   pymax = gPad->VtoAbsPixel(0);
   TGeoNode *next;
   Double_t step,steptot;
//   Double_t dotni;
   Double_t *norm;
   Double_t *point = fGeom->GetCurrentPoint();
//   Double_t ndc[3];
//   Int_t ppx,ppy;
//   Double_t refl[3];
   Double_t tosource[3];
   Double_t calf;
   Double_t phi = 0*krad;
   tosource[0] = -dir[0]*TMath::Cos(phi)+dir[1]*TMath::Sin(phi);
   tosource[1] = -dir[0]*TMath::Sin(phi)-dir[1]*TMath::Cos(phi);
   tosource[2] = -dir[2];
   
   Bool_t done;
   Int_t istep;
   Int_t base_color, color;
   Double_t light;
   Double_t stemin=0, stemax=1E30;
   TPoint *pxy = new TPoint[1];
   Int_t npoints = (pxmax-pxmin)*(pymax-pymin);
   Int_t n10 = npoints/10;
   Int_t ipoint = 0;
   for (px=pxmin; px<pxmax; px++) {
      for (py=pymin; py<pymax; py++) {         
         ipoint++;
         if (n10) {
            if ((ipoint%n10) == 0) printf("%i percent\n", 10*Int_t(Double_t(ipoint)/Double_t(n10)));
         }
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
         fGeom->InitTrack(cop,dir);
//         fGeom->CdTop();
//         fGeom->SetCurrentPoint(cop);
//         fGeom->SetCurrentDirection(dir);
         // current ray pointing to pixel (px,py)
         done = kFALSE;
         norm = 0;
         // propagate to the clipping shape if any
         if (fClippingShape) {
            if (inclip) {
               stemin = fClippingShape->DistToOut(cop,dir,3);
               stemax = 1E30;
            } else {
               stemax = fClippingShape->DistToIn(cop,dir,3);
               stemin = 0;
            }
         }         
               
         while (!done) {
            if (fClippingShape) {
               if (stemin>1E10) break;
               if (stemin>0) {
                  gGeoManager->SetStep(stemin);
                  next = gGeoManager->Step();
                  steptot = 0;
                  stemin = 0;
                  if (next) {
                     TGeoVolume *nextvol = next->GetVolume();
                     if (fVisVolumes->IndexOf(nextvol) >= 0) {
                        done = kTRUE;
                        base_color = nextvol->GetLineColor();
                        fClippingShape->ComputeNormal(point, dir, normal);
                        norm = normal;
                        break;
                     }
                  }
                  inclip = kTRUE;
                  stemax = fClippingShape->DistToOut(point,dir,3);
               }
            }              
            fGeom->FindNextBoundary();
            step = fGeom->GetStep();
            if (step>1E10) {
//               printf("pixels :%i,%i  (%f, %f, %f, %f, %f, %f)\n",px,py,
//                      cop[0],cop[1],cop[2],dir[0],dir[1],dir[2]);
               break;
            }   
            steptot += step;
            next = fGeom->Step();
            istep = 0;
            if (!fGeom->IsEntering()) fGeom->SetStep(1E-3);
            while (!fGeom->IsEntering()) {
               istep++;
               if (istep>1E2) break;
               steptot += 1E-3+1E-6;
               next = fGeom->Step();
            }
            if (istep>1E2) {
//               printf("Woops: Wrong dist from: (%f, %f, %f, %f, %f, %f)\n",
//                      cop[0],cop[1],cop[2],dir[0],dir[1],dir[2]);
//               return;
               break; 
            }     
            if (fClippingShape) {
               if (steptot>stemax) {
                  steptot = 0;
                  inclip = fClippingShape->Contains(point);
                  if (inclip) {
                     stemin = fClippingShape->DistToOut(point,dir,3);
                     stemax = 1E30;
                     continue;
                  } else {
                     stemin = 0;
                     stemax = fClippingShape->DistToIn(point,dir,3);  
                  }
               }
            }      
            if (next) {
               TGeoVolume *nextvol = next->GetVolume();
               if (fVisVolumes->IndexOf(nextvol) >= 0) {
                  done = kTRUE;
                  base_color = next->GetVolume()->GetLineColor();
                  break;
               }
            }
         }
         if (!done) continue;
         // current ray intersect a visible volume having color=base_color
//         view->WCtoNDC(point,ndc);
//         ppx = gPad->XtoPixel(ndc[0]);
//         ppy = gPad->YtoPixel(ndc[1]);
         if (!norm) norm = fGeom->FindNormal(kFALSE);
         if (!norm) {
            printf("Woops: Wrong norm from: (%f, %f, %f, %f, %f, %f)\n",
                   cop[0],cop[1],cop[2],dir[0],dir[1],dir[2]);
            break;       
         }
//         dotni = dir[0]*norm[0]+dir[1]*norm[1]+dir[2]*norm[2];
//         for (i=0; i<3; i++) refl[i] = dir[i] - 2.*dotni*norm[i];
//         calf = refl[0]*tosource[0]+refl[1]*tosource[1]+refl[2]*tosource[2];
         calf = norm[0]*tosource[0]+norm[1]*tosource[1]+norm[2]*tosource[2];
         light = 0.8*TMath::Abs(calf);
         color = GetColor(base_color, light);

         // Go back to cross again the boundary

/*         
         fGeom->SetCurrentDirection(-dir[0], -dir[1], -dir[2]);
         fGeom->FindNextBoundary();
         fGeom->Step();
         
         // Now shoot the ray according to light direction
         
         fGeom->SetCurrentDirection(tosource);
         done = kFALSE;
         while (!done) {
            fGeom->FindNextBoundary();
            step = fGeom->GetStep();
            if (step>1E10) break;
            next = fGeom->Step();
            istep = 0;
            if (!fGeom->IsEntering()) fGeom->SetStep(1E-3);
            while (!fGeom->IsEntering()) {
               istep++;
               if (istep>1E3) break;
               next = fGeom->Step();
            }
            if (istep>1E3) break;   
            if (next && next->IsOnScreen()) done = kTRUE;
         }
         if (done) color = GetColor(base_color,0);         
*/
         // Now we know the color of the pixel, just draw it
         gVirtualX->SetMarkerColor(color);         
         pxy[0].fX = px;
         pxy[0].fY = py;
//         printf("current pix: (%i, %i) real pix: (%i, %i)\n", px,py,ppx,ppy);
         gVirtualX->DrawPolyMarker(1,pxy);
      }
   } 
   delete [] pxy;      
}

//-----------------------------------------------------------------------------
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
   if (IsExplodedView()) {
      if (gPad) {
         gPad->Modified();
         gPad->Update();
      }
   }
}          
//______________________________________________________________________________
void TGeoPainter::Sizeof3D(const TGeoVolume *vol) const
{
//   Compute size of the 3d object "vol".
   if (fGeom->GetTopVolume() == vol) fGeom->CdTop();
   TGeoNode *node = 0;
   Int_t nd = vol->GetNdaughters();
   TGeoShape *shape = vol->GetShape();
   Bool_t last = kFALSE;
   Int_t level = fGeom->GetLevel();
   TGeoNode *current = fGeom->GetCurrentNode();
   Bool_t vis=(current->IsVisible() && (level || (!level && fTopVisible)) && fGeom->IsInPhiRange())?kTRUE:kFALSE;
   Int_t id;
   switch (fVisOption) {
      case kGeoVisDefault:
         if (vis && (level<=fVisLevel)) 
            shape->Sizeof3D();
            // draw daughters
         if (level<fVisLevel) {
            if ((!nd) || (!current->IsVisDaughters())) return;
            for (id=0; id<nd; id++) {
               node = vol->GetNode(id);
               fGeom->CdDown(id);
               Sizeof3D(node->GetVolume());
               fGeom->CdUp();
            }
         }
         break;
      case kGeoVisLeaves:
         last = ((nd==0) || (level==fVisLevel))?kTRUE:kFALSE;
         if (vis && (last || (!current->IsVisDaughters())))
            shape->Sizeof3D();
         if (last || (!current->IsVisDaughters())) return;
         for (id=0; id<nd; id++) {
            node = vol->GetNode(id);
            fGeom->CdDown(id);
            Sizeof3D(node->GetVolume());
            fGeom->CdUp();
         }
         break;
      case kGeoVisOnly:
         shape->Sizeof3D();
         break;
      case kGeoVisBranch:
         fGeom->cd(fVisBranch);
         while (fGeom->GetLevel()) {
            if (fGeom->GetCurrentVolume()->IsVisible()) 
               fGeom->GetCurrentVolume()->GetShape()->Sizeof3D();
            fGeom->CdUp();   
         }   
         break;
      default:
         return;
   }          
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
   if (change && gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }   
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
   if (!gPad) return;
   if (gPad->GetView()) {    
      gPad->Modified();
      gPad->Update();
   }
}
//______________________________________________________________________________
void TGeoPainter::SetVisLevel(Int_t level) {
// set default level down to which visualization is performed
   if (level<=0) {
      Warning("SetVisLevel", "visualization level should be >0");
      return;
   }   
   if (level==fVisLevel) return;
   fVisLevel=level;
   if (fVisLock) {
      fVisVolumes->Clear();
      fVisLock = kFALSE;
   }   
   if (!gPad) return;
   if (gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }
}
//-----------------------------------------------------------------------------
void TGeoPainter::SetTopVisible(Bool_t vis)
{
   if (fTopVisible==vis) return;
   fTopVisible = vis;
   if (!gPad) return;
   if (gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }
}
   
//-----------------------------------------------------------------------------
void TGeoPainter::SetVisOption(Int_t option) {
// set drawing mode :
// option=0 (default) all nodes drawn down to vislevel
// option=1           leaves and nodes at vislevel drawn
// option=2           path is drawn
   if ((fVisOption<0) || (fVisOption>3)) {
      Warning("SetVisOption", "wrong visualization option");
      return;
   }
   if (fVisOption==option) return;   
   fVisOption=option;
   if (fVisLock) {
      fVisVolumes->Clear();
      fVisLock = kFALSE;
   }   
   if (!gPad) return;
   if (gPad->GetView()) {
      gPad->Modified();
      gPad->Update();
   }
}
//-----------------------------------------------------------------------------
Int_t TGeoPainter::ShapeDistancetoPrimitive(const TGeoShape *shape, Int_t numpoints, Int_t px, Int_t py) const   
{   
//  Returns distance between point px,py on the pad an a shape.
  Int_t dist = 9999;
   TView *view = gPad->GetView();
   if (!(numpoints && view)) return dist;
   Float_t *points = new Float_t[3*numpoints];
   shape->SetPoints(points);
   Float_t dpoint2, x1, y1, xndc[3];
   Double_t dlocal[3], dmaster[3];
   for (Int_t i=0; i<numpoints; i++) {
      dlocal[0]=points[3*i]; dlocal[1]=points[3*i+1]; dlocal[2]=points[3*i+2];
      if (fPaintingOverlaps) {
         fMatrix->LocalToMaster(&dlocal[0], &dmaster[0]);
      } else if (IsExplodedView())
         fGeom->LocalToMasterBomb(&dlocal[0], &dmaster[0]);
      else   
         fGeom->LocalToMaster(&dlocal[0], &dmaster[0]);
      points[3*i]=dmaster[0]; points[3*i+1]=dmaster[1]; points[3*i+2]=dmaster[2];
      view->WCtoNDC(&points[3*i], xndc);
      x1 = gPad->XtoAbsPixel(xndc[0]);
      y1 = gPad->YtoAbsPixel(xndc[1]);
      dpoint2 = (px-x1)*(px-x1) + (py-y1)*(py-y1);
      if (dpoint2 < dist) dist=(Int_t)dpoint2;
   }
   delete [] points;
   return Int_t(TMath::Sqrt(Double_t(dist)));
}
//______________________________________________________________________________
void TGeoPainter::Test(Int_t npoints, Option_t *option)
{
// Check time of finding "Where am I" for n points.
   fChecker->Test(npoints, option);
}   
//-----------------------------------------------------------------------------
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
   return fChecker->Weight(precision, option);
}
   
   
