// @(#)root/geom:$Name:  $:$Id: TGeoChecker.cxx,v 1.1 2002/07/15 15:32:25 brun Exp $
// Author: Andrei Gheata   01/11/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// A simple geometry checker. Points can be randomly generated inside the 
// bounding  box of a node. For each point the distance to the nearest surface
// and the corresponting point on that surface are computed. These points are 
// stored in a tree and can be directly visualized within ROOT
// A second algoritm is shooting multiple rays from a given point to a geometry
// branch and storing the intersection points with surfaces in same tree. 
// Rays can be traced backwords in order to find overlaps by comparing direct 
// and inverse points.
//Begin_Html
/*
<img src="gif/t_checker.jpg">
*/
//End_Html

#include "TVirtualPad.h"
#include "TNtuple.h"
#include "TRandom3.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "TStopwatch.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoChecker.h"


// statics and globals

ClassImp(TGeoChecker)

//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker()
{
// Default constructor
   fGeom = 0;
   fTreePts      = 0; 
}
//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker(TGeoManager *geom)
{
// Constructor for a given geometry
   fGeom = geom;
   fTreePts = 0;
}
//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker(const char *treename, const char *filename)
{
// constructor
   fGeom = gGeoManager;
   fTreePts = 0;
}
//-----------------------------------------------------------------------------
TGeoChecker::~TGeoChecker()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoChecker::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *)
{
//--- Draw point (x,y,z) over the picture of the daughers of the volume containing this point.
//   Generates a report regarding the path to the node containing this point and the distance to
//   the closest boundary.

   Double_t point[3];
   Double_t local[3];
   Double_t dir[3];
   point[0] = x;
   point[1] = y;
   point[2] = z;
   memset(&dir[0], 0, 3*sizeof(Double_t));
   dir[2] = 1.;
   // init dummy track from current point
   fGeom->InitTrack(&point[0], &dir[0]);
   // get current node
   TGeoNode *node = fGeom->GetCurrentNode();
   printf("===  Check current point ===\n");
   printf("Current point : x=%f y=%f z=%f\n", point[0], point[1], point[2]);
   printf("  - path : %s\n", fGeom->GetPath());
   // get corresponding volume
   TGeoVolume *vol = node->GetVolume();
   // compute safety distance (distance to boundary ignored)
   fGeom->FindNextBoundary();
   Double_t close = fGeom->GetSafeDistance();
   if (close>1E10) {
      printf("Safety not implemented for shape %s\n",
             vol->GetShape()->ClassName());
   } else {
      printf("Safety radius : %f\n", close);
   }   
   fGeom->MasterToLocal(&point[0], &local[0]);
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(2);
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.5);
   pm->SetNextPoint(local[0], local[1], local[2]);
   if (vol->GetNdaughters()) {
      fGeom->SetVisLevel(1);
      vol->Draw();
   } else {
      vol->DrawOnly();
   }   
   pm->Draw("SAME");
   gPad->Modified();
   gPad->Update();
}  
//______________________________________________________________________________
void TGeoChecker::RandomPoints(TGeoVolume *vol, Int_t npoints, Option_t *option)
{
// Draw random points in the bounding box of a volume.
   if (!vol) return;
   gRandom = new TRandom3();
   vol->VisibleDaughters(kTRUE);
   vol->Draw();
   TString opt = option;
   opt.ToLower();
   TObjArray *pm = new TObjArray(128);
   TPolyMarker3D *marker = 0;
   const TGeoShape *shape = vol->GetShape();
   TGeoBBox *box = (TGeoBBox *)shape;
   Double_t dx = box->GetDX();
   Double_t dy = box->GetDY();
   Double_t dz = box->GetDZ();
   Double_t ox = (box->GetOrigin())[0];
   Double_t oy = (box->GetOrigin())[1];
   Double_t oz = (box->GetOrigin())[2];
   Double_t *xyz = new Double_t[3];
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   TGeoNode *node = 0;
   printf("Start... %i points\n", npoints);
   Int_t i=0;
   Int_t igen=0;
   Int_t ic = 0;
   Int_t n10 = npoints/10;
   Double_t ratio=0;
   while (igen<npoints) {
      xyz[0] = ox-dx+2*dx*gRandom->Rndm();
      xyz[1] = oy-dy+2*dy*gRandom->Rndm();
      xyz[2] = oz-dz+2*dz*gRandom->Rndm();
      fGeom->SetCurrentPoint(xyz);
      igen++;
      if (n10) {
         if ((igen%n10) == 0) printf("%i percent\n", Int_t(100*igen/npoints));
      }  
      node = fGeom->FindNode();
      if (!node) continue;
      if (!node->IsOnScreen()) continue;
      // draw only points in overlapping/non-overlapping volumes
      if (opt.Contains("many") && !node->IsOverlapping()) continue;
      if (opt.Contains("only") && node->IsOverlapping()) continue;
      ic = node->GetColour();
      if (ic >= 128) ic = 0;
      marker = (TPolyMarker3D*)pm->At(ic);
      if (!marker) {
         marker = new TPolyMarker3D();
         marker->SetMarkerColor(ic);
         marker->SetMarkerStyle(8);
         marker->SetMarkerSize(0.4);
         pm->AddAt(marker, ic);
      }
      marker->SetNextPoint(xyz[0], xyz[1], xyz[2]);
      i++;
   }
   printf("Number of visible points : %i\n", i);
   ratio = (Double_t)i/(Double_t)igen;
   printf("efficiency : %g\n", ratio);
   for (Int_t m=0; m<128; m++) {
      marker = (TPolyMarker3D*)pm->At(m);
      if (marker) marker->Draw("SAME");
   }
   fGeom->GetTopVolume()->VisibleDaughters(kFALSE);
   printf("---Daughters of %s made invisible.\n", fGeom->GetTopVolume()->GetName());
   printf("---Make them visible with : gGeoManager->GetTopVolume()->VisibleDaughters();\n");
   delete pm;
   delete xyz;
}   
//-----------------------------------------------------------------------------
void TGeoChecker::RandomRays(Int_t nrays, Double_t startx, Double_t starty, Double_t startz)
{
// Randomly shoot nrays from point (startx,starty,startz) and plot intersections 
// with surfaces for current top node.
   TObjArray *pm = new TObjArray(128);
   TPolyLine3D *line = 0;
   gRandom = new TRandom3();
   TGeoVolume *vol=fGeom->GetTopVolume();
   vol->VisibleDaughters(kTRUE);

   Double_t start[3];
   Double_t dir[3];
   Double_t *point = fGeom->GetCurrentPoint();
   vol->Draw();
   printf("Start... %i rays\n", nrays);
   TGeoNode *node, *startnode, *endnode;
   Bool_t vis1,vis2, is_sentering, is_entering, is_null;
   Int_t i=0;
   Int_t ipoint;
   Int_t itot=0;
   Int_t n10=nrays/10;
   Double_t theta,phi, step;
   while (itot<nrays) {
      itot++;
      ipoint = 0;
      if (n10) {
         if ((itot%n10) == 0) printf("%i percent\n", Int_t(100*itot/nrays));
      }
      start[0] = startx;
      start[1] = starty;
      start[2] = startz;
      phi = 2*TMath::Pi()*gRandom->Rndm();
      theta= TMath::ACos(1.-2.*gRandom->Rndm());
//      theta = 0.5*TMath::Pi();
      dir[0]=TMath::Sin(theta)*TMath::Cos(phi);
      dir[1]=TMath::Sin(theta)*TMath::Sin(phi);
      dir[2]=TMath::Cos(theta);
      fGeom->InitTrack(&start[0], &dir[0]);
      line = 0;
      startnode = fGeom->GetCurrentNode();
      if (fGeom->IsOutside()) startnode=0;
//      if (startnode) printf("start %s\n", startnode->GetName());
      vis1 = (startnode)?(startnode->IsOnScreen()):kFALSE;
      if (vis1) {
         line = new TPolyLine3D(2);
         line->SetLineColor(startnode->GetVolume()->GetLineColor());
         line->SetPoint(ipoint++, startx, starty, startz);
         i++;
         pm->Add(line);
      }
      // find the node that will be crossed first      
      node = fGeom->FindNextBoundary();
      is_sentering = fGeom->IsStepEntering();
      // find where we end-up
      endnode = fGeom->Step();
      if (fGeom->IsOutside()) endnode=0;
//      if (endnode) printf("endnode %s\n", endnode->GetName());
      step = fGeom->GetStep();
      vis2 = (endnode)?(endnode->IsOnScreen()):kFALSE;
      is_entering = fGeom->IsEntering();
//      if (is_entering) printf("entering\n");
      is_null = fGeom->IsNullStep();
      while (step<1E10) {
         if (ipoint>0) {
         // old visible node had an entry point -> finish segment
            line->SetPoint(ipoint, point[0], point[1], point[2]);
            ipoint = 0;
            line   = 0;
         }
         if (is_entering && vis2 && (startnode!=endnode)) {
            // create new segment
            line = new TPolyLine3D(2);   
            line->SetLineColor(endnode->GetVolume()->GetLineColor());
            line->SetPoint(ipoint++, point[0], point[1], point[2]);
            i++;
            pm->Add(line);
         }
         // now see if we can make an other step
         if (endnode==0) break;
         if (is_null) break;
         startnode = endnode;    
         node = fGeom->FindNextBoundary();
         is_sentering = fGeom->IsStepEntering();
         endnode = fGeom->Step();
         if (fGeom->IsOutside()) endnode=0;
//         if (endnode) printf("%s\n", endnode->GetName());
         step = fGeom->GetStep();
         vis2 = (endnode)?(endnode->IsOnScreen()):kFALSE;
         is_entering = fGeom->IsEntering();
         is_null = fGeom->IsNullStep();
      }      
   }   
   // draw all segments
   for (Int_t m=0; m<pm->GetEntriesFast(); m++) {
      line = (TPolyLine3D*)pm->At(m);
      if (line) line->Draw("SAME");
   }
   printf("number of segments : %i\n", i);
   fGeom->GetTopVolume()->VisibleDaughters(kFALSE);
   printf("---Daughters of %s made invisible.\n", fGeom->GetTopVolume()->GetName());
   printf("---Make them visible with : gGeoManager->GetTopVolume()->VisibleDaughters();\n");
   delete pm;
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoChecker::SamplePoints(Int_t npoints, Double_t &dist, Double_t epsil,
                                    const char* g3path)
{
// shoot npoints randomly in a box of 1E-5 arround current point.
// return minimum distance to points outside
   // make sure that path to current node is updated
   // get the response of tgeo
   TGeoNode *node = fGeom->FindNode();
   TGeoNode *nodegeo = 0;
   TGeoNode *nodeg3 = 0;
   TGeoNode *solg3 = 0;
   if (!node) {dist=-1; return 0;}
   gRandom = new TRandom3();
   Bool_t hasg3 = kFALSE;
   if (strlen(g3path)) hasg3 = kTRUE;
   char geopath[200];
   sprintf(geopath, "%s\n", fGeom->GetPath());
   dist = 1E10;
   TString common = "";
   // cd to common path
   Double_t point[3];
   Double_t closest[3];
   TGeoNode *node1 = 0;
   TGeoNode *node_close = 0;
   dist = 1E10;
   Double_t dist1 = 0;
   // initialize size of random box to epsil
   Double_t eps[3];
   eps[0] = epsil; eps[1]=epsil; eps[2]=epsil;
   Double_t *pointg = fGeom->GetCurrentPoint();
   if (hasg3) {
      TString spath = geopath;
      TString name = "";
      Int_t index=0;
      while (index>=0) {
         index = spath.Index("/", index+1);
         if (index>0) {
            name = spath(0, index);
            if (strstr(g3path, name.Data())) {
               common = name;
               continue;
            } else break;
         }
      }
      // if g3 response was given, cd to common path
      if (strlen(common.Data())) {
         while (strcmp(fGeom->GetPath(), common.Data()) && fGeom->GetLevel()) {
            nodegeo = fGeom->GetCurrentNode();
            fGeom->CdUp();
         }
         fGeom->cd(g3path);
         solg3 = fGeom->GetCurrentNode();
         while (strcmp(fGeom->GetPath(), common.Data()) && fGeom->GetLevel()) {
            nodeg3 = fGeom->GetCurrentNode();
            fGeom->CdUp();
         }
         if (!nodegeo) return 0;
         if (!nodeg3) return 0;
         fGeom->cd(common.Data());
         fGeom->MasterToLocal(fGeom->GetCurrentPoint(), &point[0]);
         Double_t xyz[3], local[3];
         for (Int_t i=0; i<npoints; i++) {
            xyz[0] = point[0] - eps[0] + 2*eps[0]*gRandom->Rndm();
            xyz[1] = point[1] - eps[1] + 2*eps[1]*gRandom->Rndm();
            xyz[2] = point[2] - eps[2] + 2*eps[2]*gRandom->Rndm();
            nodeg3->MasterToLocal(&xyz[0], &local[0]);
            if (!nodeg3->GetVolume()->Contains(&local[0])) continue;
            dist1 = TMath::Sqrt((xyz[0]-point[0])*(xyz[0]-point[0])+
                   (xyz[1]-point[1])*(xyz[1]-point[1])+(xyz[2]-point[2])*(xyz[2]-point[2]));
            if (dist1<dist) {
            // save node and closest point
               dist = dist1;
               node_close = solg3;
               // make the random box smaller
               eps[0] = TMath::Abs(point[0]-pointg[0]);
               eps[1] = TMath::Abs(point[1]-pointg[1]);
               eps[2] = TMath::Abs(point[2]-pointg[2]);
            }
         }
      }
      if (!node_close) dist = -1;
      return node_close;
   }

//   gRandom = new TRandom3();
   // save current point
   memcpy(&point[0], pointg, 3*sizeof(Double_t));
   for (Int_t i=0; i<npoints; i++) {
      // generate a random point in MARS
      pointg[0] = point[0] - eps[0] + 2*eps[0]*gRandom->Rndm();
      pointg[1] = point[1] - eps[1] + 2*eps[1]*gRandom->Rndm();
      pointg[2] = point[2] - eps[2] + 2*eps[2]*gRandom->Rndm();
      // check if new node is different from the old one
      if (node1!=node) {
         dist1 = TMath::Sqrt((point[0]-pointg[0])*(point[0]-pointg[0])+
                 (point[1]-pointg[1])*(point[1]-pointg[1])+(point[2]-pointg[2])*(point[2]-pointg[2]));
         if (dist1<dist) {
            dist = dist1;
            node_close = node1;
            memcpy(&closest[0], pointg, 3*sizeof(Double_t));
            // make the random box smaller
            eps[0] = TMath::Abs(point[0]-pointg[0]);
            eps[1] = TMath::Abs(point[1]-pointg[1]);
            eps[2] = TMath::Abs(point[2]-pointg[2]);
         }
      }
   }
   // restore the original point and path
   memcpy(pointg, &point[0], 3*sizeof(Double_t));
   fGeom->FindNode();  // really needed ?
   if (!node_close) dist=-1;
   return node_close;
}
//-----------------------------------------------------------------------------
void TGeoChecker::Test(Int_t npoints, Option_t *option)
{
   // Check time of finding "Where am I" for n points.
   gRandom= new TRandom3();
   Bool_t recheck = !strcmp(option, "RECHECK");
   if (recheck) printf("RECHECK\n");
   const TGeoShape *shape = fGeom->GetTopVolume()->GetShape();
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t *xyz = new Double_t[3*npoints];
   TStopwatch *timer = new TStopwatch();
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   timer->Start(kFALSE);
   Int_t i;
   for (i=0; i<npoints; i++) {
      xyz[3*i] = -dx+2*dx*gRandom->Rndm();
      xyz[3*i+1] = -dy+2*dy*gRandom->Rndm();
      xyz[3*i+2] = -dz+2*dz*gRandom->Rndm();
   }
   timer->Stop();
   printf("Generation time :\n");
   timer->Print();
   timer->Reset();
   TGeoNode *node, *node1;
   printf("Start... %i points\n", npoints);
   timer->Start(kFALSE);
   for (i=0; i<npoints; i++) {
      fGeom->SetCurrentPoint(xyz+3*i);
      if (recheck) fGeom->CdTop();
      node = fGeom->FindNode();
      if (recheck) {
         node1 = fGeom->FindNode();
         if (node1 != node) {
            printf("Difference for x=%g y=%g z=%g\n", xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
            printf(" from top : %s\n", node->GetName());
            printf(" redo     : %s\n", fGeom->GetPath());
         }
      }
   }
   timer->Stop();
   timer->Print();
   delete xyz;
   delete timer;
}
//-----------------------------------------------------------------------------
void TGeoChecker::TestOverlaps(const char* path)
{
//--- Geometry overlap checker based on sampling. 
   if (fGeom->GetTopVolume()!=fGeom->GetMasterVolume()) fGeom->RestoreMasterVolume();
   printf("Checking overlaps for path :\n");
   if (!fGeom->cd(path)) return;
   TGeoNode *checked = fGeom->GetCurrentNode();
   checked->InspectNode();
   // shoot 1E4 points in the shape of the current volume
   gRandom= new TRandom3();
   Int_t npoints = 1000000;
   Double_t big = 1E6;
   Double_t xmin = big;
   Double_t xmax = -big;
   Double_t ymin = big;
   Double_t ymax = -big;
   Double_t zmin = big;
   Double_t zmax = -big;
   TObjArray *pm = new TObjArray(128);
   TPolyMarker3D *marker = 0;
   TPolyMarker3D *markthis = new TPolyMarker3D();
   markthis->SetMarkerColor(5);
   TNtuple *ntpl = new TNtuple("ntpl","random points","x:y:z");
   TGeoShape *shape = fGeom->GetCurrentNode()->GetVolume()->GetShape();
   Double_t *point = new Double_t[3];
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t ox = (((TGeoBBox*)shape)->GetOrigin())[0];
   Double_t oy = (((TGeoBBox*)shape)->GetOrigin())[1];
   Double_t oz = (((TGeoBBox*)shape)->GetOrigin())[2];
   Double_t *xyz = new Double_t[3*npoints];
   Int_t i=0;
   printf("Generating %i points inside %s\n", npoints, fGeom->GetPath());
   while (i<npoints) {
      point[0] = ox-dx+2*dx*gRandom->Rndm();
      point[1] = oy-dy+2*dy*gRandom->Rndm();
      point[2] = oz-dz+2*dz*gRandom->Rndm();
      if (!shape->Contains(point)) continue;
      // convert each point to MARS
//      printf("local  %9.3f %9.3f %9.3f\n", point[0], point[1], point[2]);
      fGeom->LocalToMaster(point, &xyz[3*i]);
//      printf("master %9.3f %9.3f %9.3f\n", xyz[3*i], xyz[3*i+1], xyz[3*i+2]);
      xmin = TMath::Min(xmin, xyz[3*i]);
      xmax = TMath::Max(xmax, xyz[3*i]);
      ymin = TMath::Min(ymin, xyz[3*i+1]);
      ymax = TMath::Max(ymax, xyz[3*i+1]);
      zmin = TMath::Min(zmin, xyz[3*i+2]);
      zmax = TMath::Max(zmax, xyz[3*i+2]);
      i++;
   }
   delete point;
   ntpl->Fill(xmin,ymin,zmin);
   ntpl->Fill(xmax,ymin,zmin);
   ntpl->Fill(xmin,ymax,zmin);
   ntpl->Fill(xmax,ymax,zmin);
   ntpl->Fill(xmin,ymin,zmax);
   ntpl->Fill(xmax,ymin,zmax);
   ntpl->Fill(xmin,ymax,zmax);
   ntpl->Fill(xmax,ymax,zmax);
   ntpl->Draw("z:y:x");

   // shoot the poins in the geometry
   TGeoNode *node;
   TString cpath;
   Int_t ic=0;
   TObjArray *overlaps = new TObjArray();
   printf("using FindNode...\n");
   for (Int_t j=0; j<npoints; j++) {
      // always start from top level (testing only)
      fGeom->CdTop();
      fGeom->SetCurrentPoint(&xyz[3*j]);
      node = fGeom->FindNode();
      cpath = fGeom->GetPath();
      if (cpath.Contains(path)) {
         markthis->SetNextPoint(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
         continue;
      }
      // current point is found in an overlapping node
      if (!node) ic=128;
      else ic = node->GetVolume()->GetLineColor();
      if (ic >= 128) ic = 0;
      marker = (TPolyMarker3D*)pm->At(ic);
      if (!marker) {
         marker = new TPolyMarker3D();
         marker->SetMarkerColor(ic);
         pm->AddAt(marker, ic);
      }
      // draw the overlapping point
      marker->SetNextPoint(xyz[3*j], xyz[3*j+1], xyz[3*j+2]);
      if (node) {
         if (overlaps->IndexOf(node) < 0) overlaps->Add(node);
      }
   }
   // draw all overlapping points
   for (Int_t m=0; m<128; m++) {
      marker = (TPolyMarker3D*)pm->At(m);
//      if (marker) marker->Draw("SAME");
   }
   markthis->Draw("SAME");
   if (gPad) gPad->Update();
   // display overlaps
   if (overlaps->GetEntriesFast()) {
      printf("list of overlapping nodes :\n");
      for (i=0; i<overlaps->GetEntriesFast(); i++) {
         node = (TGeoNode*)overlaps->At(i);
         if (node->IsOverlapping()) printf("%s  MANY\n", node->GetName());
         else printf("%s  ONLY\n", node->GetName());
      }
   } else printf("No overlaps\n");
   delete ntpl;
   delete pm;
   delete xyz;
   delete overlaps;
}
//-----------------------------------------------------------------------------
void TGeoChecker::CreateTree(const char *treename, const char *filename)
{
// These points are stored in a tree and can be directly visualized within ROOT.
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::Generate(UInt_t npoint)
{
// Points are randomly generated inside the 
// bounding  box of a node. For each point the distance to the nearest surface
// and the corresponding point on that surface are computed.
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::Raytrace(Double_t *startpoint, UInt_t npoints)
{
// A second algoritm is shooting multiple rays from a given point to a geometry
// branch and storing the intersection points with surfaces in same tree. 
// Rays can be traced backwords in order to find overlaps by comparing direct 
// and inverse points.   
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::ShowPoints(Option_t *option)
{
// 
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
