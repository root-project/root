// @(#)root/geom:$Name:  $:$Id: TGeoChecker.cxx,v 1.25 2003/02/10 17:23:14 brun Exp $
// Author: Andrei Gheata   01/11/01
// CheckGeometry(), CheckOverlaps() by Mihaela Gheata

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
#include "TH2.h"
#include "TRandom3.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "TStopwatch.h"

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoOverlap.h"
#include "TGeoPainter.h"
#include "TGeoChecker.h"


// statics and globals

ClassImp(TGeoChecker)

//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker()
{
// Default constructor
   fGeom = 0;
   fTreePts      = 0; 
   fVsafe = 0;
}
//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker(TGeoManager *geom)
{
// Constructor for a given geometry
   fGeom = geom;
   fTreePts = 0;
   fVsafe = 0;
}
//-----------------------------------------------------------------------------
TGeoChecker::TGeoChecker(const char * /*treename*/, const char * /*filename*/)
{
// constructor
   fGeom = gGeoManager;
   fTreePts = 0;
   fVsafe = 0;
}
//-----------------------------------------------------------------------------
TGeoChecker::~TGeoChecker()
{
// Destructor
}
//-----------------------------------------------------------------------------
void TGeoChecker::CheckGeometry(Int_t nrays, Double_t startx, Double_t starty, Double_t startz) const
{
// Shoot nrays with random directions from starting point (startx, starty, startz)
// in the reference frame of this volume. Track each ray until exiting geometry, then
// shoot backwards from exiting point and compare boundary crossing points.
   Int_t i, j;
   Double_t start[3], end[3];
   Double_t dir[3];
   Double_t dummy[3];
   Double_t eps = 0.;
   Double_t *array1 = new Double_t[3*1000];
   Double_t *array2 = new Double_t[3*1000];
   TObjArray *pma = new TObjArray();
   TPolyMarker3D *pm;
   pm = new TPolyMarker3D();
   pm->SetMarkerColor(2); // error > eps
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.4);
   pma->AddAt(pm, 0);
   pm = new TPolyMarker3D();
   pm->SetMarkerColor(4); // point not found
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.4);
   pma->AddAt(pm, 1);
   pm = new TPolyMarker3D();
   pm->SetMarkerColor(6); // extra point back
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.4);
   pma->AddAt(pm, 2);
   Int_t nelem1, nelem2;
   Int_t dim1=1000, dim2=1000;
   if ((startx==0) && (starty==0) && (startz==0)) eps=1E-3;
   start[0] = startx+eps;
   start[1] = starty+eps;
   start[2] = startz+eps;
   Int_t n10=nrays/10;
   Double_t theta,phi;
   Double_t dw, dwmin, dx, dy, dz;
   Int_t ist1, ist2, ifound;
   for (i=0; i<nrays; i++) {
      if (n10) {
         if ((i%n10) == 0) printf("%i percent\n", Int_t(100*i/nrays));
      }
      phi = 2*TMath::Pi()*gRandom->Rndm();
      theta= TMath::ACos(1.-2.*gRandom->Rndm());
      dir[0]=TMath::Sin(theta)*TMath::Cos(phi);
      dir[1]=TMath::Sin(theta)*TMath::Sin(phi);
      dir[2]=TMath::Cos(theta);
      // shoot direct ray
      nelem1=nelem2=0;
//      printf("DIRECT %i\n", i);
      ShootRay(&start[0], dir[0], dir[1], dir[2], array1, nelem1, dim1);
      if (!nelem1) continue;
//      for (j=0; j<nelem1; j++) printf("%i : %f %f %f\n", j, array1[3*j], array1[3*j+1], array1[3*j+2]);
      memcpy(&end[0], &array1[3*(nelem1-1)], 3*sizeof(Double_t));
      // shoot ray backwards
//      printf("BACK %i\n", i);
      ShootRay(&end[0], -dir[0], -dir[1], -dir[2], array2, nelem2, dim2, &start[0]);
      if (!nelem2) {
         printf("#### NOTHING BACK ###########################\n");
         for (j=0; j<nelem1; j++) {
            pm = (TPolyMarker3D*)pma->At(0);
            pm->SetNextPoint(array1[3*j], array1[3*j+1], array1[3*j+2]);
	 }
	 continue;
      }	     
//      printf("BACKWARDS\n");
      Int_t k=nelem2>>1;
      for (j=0; j<k; j++) {
         memcpy(&dummy[0], &array2[3*j], 3*sizeof(Double_t));
         memcpy(&array2[3*j], &array2[3*(nelem2-1-j)], 3*sizeof(Double_t));
         memcpy(&array2[3*(nelem2-1-j)], &dummy[0], 3*sizeof(Double_t));
      }
//      for (j=0; j<nelem2; j++) printf("%i : %f ,%f ,%f   \n", j, array2[3*j], array2[3*j+1], array2[3*j+2]);	 
      if (nelem1!=nelem2) printf("### DIFFERENT SIZES : nelem1=%i nelem2=%i ##########\n", nelem1, nelem2);
      ist1 = ist2 = 0;
      // check first match
      
      dx = array1[3*ist1]-array2[3*ist2];
      dy = array1[3*ist1+1]-array2[3*ist2+1];
      dz = array1[3*ist1+2]-array2[3*ist2+2];
      dw = dx*dir[0]+dy*dir[1]+dz*dir[2];
      fGeom->SetCurrentPoint(&array1[3*ist1]);
      fGeom->FindNode();
//      printf("%i : %s (%g, %g, %g)\n", ist1, fGeom->GetPath(), 
//             array1[3*ist1], array1[3*ist1+1], array1[3*ist1+2]);
      if (TMath::Abs(dw)<1E-4) {
//         printf("   matching %i (%g, %g, %g)\n", ist2, array2[3*ist2], array2[3*ist2+1], array2[3*ist2+2]);
         ist2++;
      } else {
         printf("### NOT MATCHING %i f:(%f, %f, %f) b:(%f %f %f) DCLOSE=%f\n", ist2, array1[3*ist1], array1[3*ist1+1], array1[3*ist1+2], array2[3*ist2], array2[3*ist2+1], array2[3*ist2+2],dw);
         pm = (TPolyMarker3D*)pma->At(0);
         pm->SetNextPoint(array2[3*ist2], array2[3*ist2+1], array2[3*ist2+2]);
         if (dw<0) {
            // first boundary missed on way back
         } else {
            // first boundary different on way back
            ist2++;
         }
      }      
      
      while ((ist1<nelem1-1) && (ist2<nelem2)) {
         fGeom->SetCurrentPoint(&array1[3*ist1+3]);
         fGeom->FindNode();
//         printf("%i : %s (%g, %g, %g)\n", ist1+1, fGeom->GetPath(), 
//                array1[3*ist1+3], array1[3*ist1+4], array1[3*ist1+5]);
         
         dx = array1[3*ist1+3]-array1[3*ist1];
         dy = array1[3*ist1+4]-array1[3*ist1+1];
         dz = array1[3*ist1+5]-array1[3*ist1+2];
         // distance to next point
         dwmin = dx+dir[0]+dy*dir[1]+dz*dir[2];
         while (ist2<nelem2) {
            ifound = 0;
            dx = array2[3*ist2]-array1[3*ist1];
            dy = array2[3*ist2+1]-array1[3*ist1+1];
            dz = array2[3*ist2+2]-array1[3*ist1+2];
            dw = dx+dir[0]+dy*dir[1]+dz*dir[2];
	    if (TMath::Abs(dw-dwmin)<1E-4) {
	       ist1++;
	       ist2++;
	       break;
	    }   
            if (dw<dwmin) {
            // point found on way back. Check if close enough to ist1+1
               ifound++;
               dw = dwmin-dw;
               if (dw<1E-4) {
               // point is matching ist1+1
//                  printf("   matching %i (%g, %g, %g) DCLOSE=%g\n", ist2, array2[3*ist2], array2[3*ist2+1], array2[3*ist2+2], dw);
                  ist2++;
                  ist1++;
                  break;
               } else {
               // extra boundary found on way back   
                  fGeom->SetCurrentPoint(&array2[3*ist2]);
                  fGeom->FindNode();
     	          pm = (TPolyMarker3D*)pma->At(2);
	          pm->SetNextPoint(array2[3*ist2], array2[3*ist2+1], array2[3*ist2+2]);
                  printf("### EXTRA BOUNDARY %i :  %s found at DCLOSE=%f\n", ist2, fGeom->GetPath(), dw);
                  ist2++;
                  continue;
               }
            } else {
               if (!ifound) {
                  // point ist1+1 not found on way back
                  fGeom->SetCurrentPoint(&array1[3*ist1+3]);
                  fGeom->FindNode();
	          pm = (TPolyMarker3D*)pma->At(1);
	          pm->SetNextPoint(array2[3*ist1+3], array2[3*ist1+4], array2[3*ist1+5]);
                  printf("### BOUNDARY MISSED BACK #########################\n");
                  ist1++;
                  break;
               } else {
                  ist1++;
                  break;
               }
            }
         }               
      }                          
   }   
   pm = (TPolyMarker3D*)pma->At(0);
   pm->Draw("SAME");
   pm = (TPolyMarker3D*)pma->At(1);
   pm->Draw("SAME");
   pm = (TPolyMarker3D*)pma->At(2);
   pm->Draw("SAME");
   if (gPad) {
      gPad->Modified();
      gPad->Update();
   }      
   delete [] array1;
   delete [] array2;
}

//-----------------------------------------------------------------------------
void TGeoChecker::CheckOverlaps(const TGeoVolume *vol, Double_t ovlp, Option_t * /*option*/) const
{
// Check illegal overlaps for volume VOL within a limit OVLP.
   if (vol->GetFinder()) return;
   Int_t nd = vol->GetNdaughters();
   if (!nd) return;
   // first, test if daughters extrude their container
   TGeoShape *shapem = vol->GetShape();
   TGeoNode * node;
   TGeoMatrix *matrix;
   X3DPoints *buff;
   Bool_t extrude, isextrusion, isoverlapping;
   TGeoOverlap *nodeovlp = 0;
   Bool_t ismany;
   Double_t *points;
   Double_t local[3];
   Double_t point[3];
   Double_t safety = 1e30;  //TGeoShape::kBig
   Int_t id, ip;
   for (id=0; id<nd; id++) {
      node = vol->GetNode(id);
      buff = (X3DPoints*)node->GetVolume()->Make3DBuffer();
      if (!buff) {
         Error("CheckOverlaps", "could not fill X3D buffer for node %s", node->GetName());
         return;
      }	       
      matrix = node->GetMatrix();
      ismany = node->IsOverlapping();
      points = buff->points;
      isextrusion=kFALSE;
      // loop all points
      for (ip=0; ip<buff->numPoints; ip++) {
         memcpy(local, &points[3*ip], 3*sizeof(Double_t));
	       matrix->LocalToMaster(local, point);
	       extrude = !shapem->Contains(point);
	       if (extrude) {
	          safety = shapem->Safety(point, kFALSE);
	          if (safety<ovlp) extrude=kFALSE;
	       }    
	       if (extrude) {
            if (!isextrusion) {
               isextrusion = kTRUE;
               char *name = new char[20];
               sprintf(name,"%s_x_%i", vol->GetName(),id); 
               nodeovlp = new TGeoExtrusion(name, (TGeoVolume*)vol,id,safety);
               nodeovlp->SetNextPoint(point[0],point[1],point[2]);
               fGeom->AddOverlap(nodeovlp);
            } else {
               if (safety>nodeovlp->GetOverlap()) nodeovlp->SetOverlap(safety);
               nodeovlp->SetNextPoint(point[0],point[1],point[2]);
            }   
	       }
      }	     
      if (points) delete [] points;
      delete buff; 
   }
   // now check if the daughters overlap with each other
   if (nd<2) return;
   TGeoVoxelFinder *vox = vol->GetVoxels();
   TGeoNode *node1;
   TGeoMatrix *matrix1;
   TGeoShape *shape1;
   Bool_t ismany1, overlap;
   Int_t novlp;
   Int_t *ovlps;
   Int_t ko, io;
   for (id=0; id<nd; id++) {
      node = vol->GetNode(id);
      ismany = node->IsOverlapping();
      if (ismany) continue;
      shapem = node->GetVolume()->GetShape();
      matrix = node->GetMatrix();
      if (vox) {
         vox->FindOverlaps(id);
         ovlps = node->GetOverlaps(novlp);
         if (!ovlps) continue;
      } else continue;
      for (ko=0; ko<novlp; ko++) {
         io = ovlps[ko];
         if (io<id) continue;
         node1 = vol->GetNode(io);
         ismany1 = node1->IsOverlapping();
         if (ismany1) continue;
         matrix1 = node1->GetMatrix();
         buff = (X3DPoints*)node1->GetVolume()->Make3DBuffer();
         if (!buff) continue;   
         points = buff->points;
         // loop all points
         overlap = kFALSE;
         isoverlapping = kFALSE;
         for (ip=0; ip<buff->numPoints; ip++) {
            memcpy(local, &points[3*ip], 3*sizeof(Double_t));
	          matrix1->LocalToMaster(local, point);
            matrix->MasterToLocal(point, local); // now point in local reference of node
	          overlap = shapem->Contains(local);
	          if (overlap) {
	             safety = shapem->Safety(local, kTRUE);
	             if (safety<ovlp) overlap=kFALSE;
	          }    
	          if (overlap) {
               if (!isoverlapping) {
                  isoverlapping = kTRUE;
                  char *name = new char[20];
                  sprintf(name,"%s_o_%i_%i", vol->GetName(),id,io); 
                  nodeovlp = new TGeoNodeOverlap(name, (TGeoVolume*)vol,id,io,safety);
                  nodeovlp->SetNextPoint(point[0],point[1],point[2]);
                  fGeom->AddOverlap(nodeovlp);
               } else {
                  if (safety>nodeovlp->GetOverlap()) nodeovlp->SetOverlap(safety);
                  nodeovlp->SetNextPoint(point[0],point[1],point[2]);
               }     
	          }
         }	     
         if (points) delete [] points;
         delete buff; 
         shape1 = node1->GetVolume()->GetShape();
         buff = (X3DPoints*)node->GetVolume()->Make3DBuffer();
         if (!buff) continue;   
         points = buff->points;
         // loop all points
         for (ip=0; ip<buff->numPoints; ip++) {
            memcpy(local, &points[3*ip], 3*sizeof(Double_t));
	          matrix->LocalToMaster(local, point);
            matrix1->MasterToLocal(point, local); // now point in local reference of node
	          overlap = shape1->Contains(local);
	          if (overlap) {
	             safety = shape1->Safety(local, kTRUE);
	             if (safety<ovlp) overlap=kFALSE;
	          }    
	          if (overlap) {
               if (!isoverlapping) {
                  isoverlapping = kTRUE;
                  char *name = new char[20];
                  sprintf(name,"%s_o_%i_%i", vol->GetName(),id,io); 
                  nodeovlp = new TGeoNodeOverlap(name, (TGeoVolume*)vol,id,io,safety);
                  nodeovlp->SetNextPoint(point[0],point[1],point[2]);
                  fGeom->AddOverlap(nodeovlp);
               } else {
                  if (safety>nodeovlp->GetOverlap()) nodeovlp->SetOverlap(safety);
                  nodeovlp->SetNextPoint(point[0],point[1],point[2]);
               }     
	          }
         }
         if (points) delete [] points;
         delete buff;          
      }   	     
      node->SetOverlaps(0,0);
   }
}

//-----------------------------------------------------------------------------
void TGeoChecker::PrintOverlaps() const
{
// Print the current list of overlaps held by the manager class.
   TIter next(fGeom->GetListOfOverlaps());
   TGeoOverlap *ov;
   printf("=== Overlaps for %s ===\n", fGeom->GetName());
   while ((ov=(TGeoOverlap*)next())) ov->PrintInfo();
}

//-----------------------------------------------------------------------------
void TGeoChecker::CheckPoint(Double_t x, Double_t y, Double_t z, Option_t *)
{
//--- Draw point (x,y,z) over the picture of the daughers of the volume containing this point.
//   Generates a report regarding the path to the node containing this point and the distance to
//   the closest boundary.

   Double_t point[3];
   Double_t local[3];
   Double_t dir[3], localdir[3];
   point[0] = x;
   point[1] = y;
   point[2] = z;
   memset(dir, 0, 3*sizeof(Double_t));
   dir[2] = 1.;
   memset(localdir, 0, 3*sizeof(Double_t));
   localdir[2] = 1.;
   // init dummy track from current point
   TGeoVolume *vol = fGeom->GetTopVolume();
   if (fVsafe) {
      TGeoNode *old = fVsafe->GetNode("SAFETY_1");
      if (old) fVsafe->GetNodes()->RemoveAt(vol->GetNdaughters()-1);
   }   
//   if (vol != fGeom->GetMasterVolume()) fGeom->RestoreMasterVolume();
   TGeoNode *node = fGeom->FindNode(point[0], point[1], point[2]);
   fGeom->MasterToLocal(point, local);
   Double_t r = TMath::Sqrt(local[0]*local[0]+local[1]*local[1]+local[2]*local[2]);
   if (r>1E-5) { 
      localdir[0] = -local[0]/r;
      localdir[1] = -local[1]/r;
      localdir[2] = -local[2]/r;
   }
   fGeom->LocalToMasterVect(localdir, dir);   
   fGeom->SetCurrentDirection(dir);
   // get current node
   printf("===  Check current point ===\n");
   printf("Current point : x=%f y=%f z=%f\n", point[0], point[1], point[2]);
   printf("  - path : %s\n", fGeom->GetPath());
   // get corresponding volume
   if (node) vol = node->GetVolume();
   // compute safety distance (distance to boundary ignored)
   fGeom->FindNextBoundary();
   Double_t close = fGeom->GetSafeDistance();
   printf("Safety radius : %f\n", close);
   if (close>1E-4) {
      TGeoVolume *sph = fGeom->MakeSphere("SAFETY", vol->GetMedium(), 0, close, 0,180,0,360);
      sph->SetLineColor(2);
      sph->SetLineStyle(3);
      vol->AddNode(sph,1,new TGeoTranslation(local[0], local[1], local[2]));
      fVsafe = vol;
   }
   TPolyMarker3D *pm = new TPolyMarker3D();
   pm->SetMarkerColor(2);
   pm->SetMarkerStyle(8);
   pm->SetMarkerSize(0.5);
   pm->SetNextPoint(local[0], local[1], local[2]);
   if (vol->GetNdaughters()<2) fGeom->SetTopVisible();
   else fGeom->SetTopVisible(kFALSE);
   fGeom->SetVisLevel(1);
   if (!vol->IsVisible()) vol->SetVisibility(kTRUE);
   vol->Draw();
   pm->Draw("SAME");
   gPad->Modified();
   gPad->Update();
}  
//-----------------------------------------------------------------------------
TH2F *TGeoChecker::LegoPlot(Int_t ntheta, Double_t themin, Double_t themax,
                            Int_t nphi,   Double_t phimin, Double_t phimax,
                            Double_t /*rmin*/, Double_t /*rmax*/, Option_t *option)
{
// Generate a lego plot fot the top volume, according to option.
   TH2F *hist = new TH2F("lego", option, nphi, phimin, phimax, ntheta, themin, themax);
   
   Double_t degrad = TMath::Pi()/180.;
   Double_t theta, phi, step, matprop, x;
   Double_t start[3];
   Double_t dir[3];
   TGeoNode *startnode, *endnode;
   Int_t i;  // loop index for phi
   Int_t j;  // loop index for theta
   Int_t ntot = ntheta * nphi;
   Int_t n10 = ntot/10;
   Int_t igen = 0, iloop=0;
   printf("=== Lego plot sph. => nrays=%i\n", ntot);
   for (i=1; i<=nphi; i++) {
      for (j=1; j<=ntheta; j++) {
         igen++;
         if (n10) {
            if ((igen%n10) == 0) printf("%i percent\n", Int_t(100*igen/ntot));
         }  
         x = 0;
         theta = hist->GetYaxis()->GetBinCenter(j);
         phi   = hist->GetXaxis()->GetBinCenter(i)+1E-3;
         start[0] = start[1] = start[2] = 1E-3;
         dir[0]=TMath::Sin(theta*degrad)*TMath::Cos(phi*degrad);
         dir[1]=TMath::Sin(theta*degrad)*TMath::Sin(phi*degrad);
         dir[2]=TMath::Cos(theta*degrad);
         fGeom->InitTrack(&start[0], &dir[0]);
         startnode = fGeom->GetCurrentNode();
         if (fGeom->IsOutside()) startnode=0;
         if (startnode) {
            matprop = startnode->GetVolume()->GetMaterial()->GetRadLen();
         } else {
            matprop = 0.;
         }      
         fGeom->FindNextBoundary();
//         fGeom->IsStepEntering();
         // find where we end-up
         endnode = fGeom->Step();
         step = fGeom->GetStep();
         while (step<1E10) {
            // now see if we can make an other step
            iloop=0;
            while (!fGeom->IsEntering()) {
               iloop++;
               fGeom->SetStep(1E-3);
               step += 1E-3;
               endnode = fGeom->Step();
            }
            if (iloop>1000) printf("%i steps\n", iloop);   
            if (matprop>0) {
               x += step/matprop;
            }   
            if (endnode==0 && step>1E10) break;
            // generate an extra step to cross boundary
            startnode = endnode;    
            if (startnode) {
               matprop = startnode->GetVolume()->GetMaterial()->GetRadLen();
            } else {
               matprop = 0.;
            }      
            
            fGeom->FindNextBoundary();
            endnode = fGeom->Step();
            step = fGeom->GetStep();
         }
         hist->Fill(phi, theta, x); 
      }
   }
   return hist;           
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
      if ((ic<0) || (ic>=128)) ic = 0;
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
   Int_t istep= 0;
   Double_t *point = fGeom->GetCurrentPoint();
   vol->Draw();
   printf("Start... %i rays\n", nrays);
   TGeoNode *startnode, *endnode;
   Bool_t vis1,vis2;
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
      dir[0]=TMath::Sin(theta)*TMath::Cos(phi);
      dir[1]=TMath::Sin(theta)*TMath::Sin(phi);
      dir[2]=TMath::Cos(theta);
      startnode = fGeom->InitTrack(start[0],start[1],start[2], dir[0],dir[1],dir[2]);
      line = 0;
      if (fGeom->IsOutside()) startnode=0;
      vis1 = (startnode)?(startnode->IsOnScreen()):kFALSE;
      if (vis1) {
         line = new TPolyLine3D(2);
         line->SetLineColor(startnode->GetVolume()->GetLineColor());
         line->SetPoint(ipoint++, startx, starty, startz);
         i++;
         pm->Add(line);
      }
      // find where we end-up
      fGeom->FindNextBoundary();
      step = fGeom->GetStep();
      endnode = fGeom->Step();
      vis2 = (endnode)?(endnode->IsOnScreen()):kFALSE;
      while (step<1E10) {
         istep = 0;
         while (!fGeom->IsEntering()) {
            istep++;
            if (istep>1E4) break;
            fGeom->SetStep(1E-3);
            endnode = fGeom->Step();
            step += 1E-3;
         }      
         if (istep>1E4) break;
//         if (istep) printf("ADDED : %f (%i steps)\n", istep*1E-3, istep);
         vis2 = (endnode)?(endnode->IsOnScreen()):kFALSE;
         if (ipoint>0) {
         // old visible node had an entry point -> finish segment
            line->SetPoint(ipoint, point[0], point[1], point[2]);
            ipoint = 0;
            line   = 0;
         }
         if (vis2) {
            // create new segment
            line = new TPolyLine3D(2);   
            line->SetLineColor(endnode->GetVolume()->GetLineColor());
            line->SetPoint(ipoint++, point[0], point[1], point[2]);
            i++;
            pm->Add(line);
         } 
         // now see if we can make an other step
         if (endnode==0 && step>1E10) break;
         // generate an extra step to cross boundary
         startnode = endnode;    
         fGeom->FindNextBoundary();
         step = fGeom->GetStep();
         endnode = fGeom->Step();
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
void TGeoChecker::ShootRay(Double_t *start, Double_t dirx, Double_t diry, Double_t dirz, Double_t *array, Int_t &nelem, Int_t &dim, Double_t *endpoint) const
{
// Shoot one ray from start point with direction (dirx,diry,dirz). Fills input array
// with points just after boundary crossings.
//   Int_t array_dimension = 3*dim;
   nelem = 0;
   Int_t istep = 0;
   if (!dim) {
      printf("empty input array\n");
      return;
   }   
//   fGeom->CdTop();
   Double_t *point = fGeom->GetCurrentPoint();
   TGeoNode *endnode;
   Bool_t is_entering;
   Double_t step, forward;
   Double_t dir[3];
   dir[0] = dirx;
   dir[1] = diry;
   dir[2] = dirz;
   fGeom->InitTrack(start, &dir[0]);
   fGeom->GetCurrentNode();
//   printf("Start : (%f,%f,%f)\n", point[0], point[1], point[2]);
   fGeom->FindNextBoundary();
   step = fGeom->GetStep();
//   printf("---next : at step=%f\n", step);
   if (step>1E10) return;
   endnode = fGeom->Step();
   is_entering = fGeom->IsEntering();
   while (step<1E10) {
      if (endpoint) {
         forward = dirx*(endpoint[0]-point[0])+diry*(endpoint[1]-point[1])+dirz*(endpoint[2]-point[2]);
         if (forward<1E-3) {
//	    printf("exit : Passed start point. nelem=%i\n", nelem); 
	    return;
	 }
      }
      if (is_entering) {
         if (nelem>=dim) {
            Double_t *temparray = new Double_t[3*(dim+20)];
            memcpy(temparray, array, 3*dim*sizeof(Double_t));
            delete [] array;
            array = temparray;
	          dim += 20;
         }
         memcpy(&array[3*nelem], point, 3*sizeof(Double_t)); 
//         printf("%i (%f, %f, %f) step=%f\n", nelem, point[0], point[1], point[2], step);
         nelem++; 
      } else {
         if (endnode==0 && step>1E10) {
//	    printf("exit : NULL endnode. nelem=%i\n", nelem); 
	    return;
	 }    
         if (!fGeom->IsEntering()) {
//            if (startnode) printf("stepping %f from (%f, %f, %f) inside %s...\n", step,point[0], point[1], point[2], startnode->GetName());
//            else printf("stepping %f from (%f, %f, %f) OUTSIDE...\n", step,point[0], point[1], point[2]);
            istep = 0;
         }    
         while (!fGeom->IsEntering()) {
            istep++;
	    if (istep>1E3) {
//	       Error("ShootRay", "more than 1000 steps. Step was %f", step);
	       nelem = 0;
	       return;
	    }   
            fGeom->SetStep(1E-5);
            endnode = fGeom->Step();
         }
         if (istep>0) printf("%i steps\n", istep);   
         if (nelem>=dim) {
            Double_t *temparray = new Double_t[3*(dim+20)];
            memcpy(temparray, array, 3*dim*sizeof(Double_t));
            delete [] array;
            array = temparray;
	          dim += 20;
         }
         memcpy(&array[3*nelem], point, 3*sizeof(Double_t)); 
//         if (endnode) printf("%i (%f, %f, %f) step=%f\n", nelem, point[0], point[1], point[2], step);
         nelem++;   
         is_entering = kTRUE;
      }
      fGeom->FindNextBoundary();
      step = fGeom->GetStep();
//      printf("---next at step=%f\n", step);
      endnode = fGeom->Step();
      is_entering = fGeom->IsEntering();
   }
//   printf("exit : INFINITE step. nelem=%i\n", nelem);
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
   Double_t ox = (((TGeoBBox*)shape)->GetOrigin())[0];
   Double_t oy = (((TGeoBBox*)shape)->GetOrigin())[1];
   Double_t oz = (((TGeoBBox*)shape)->GetOrigin())[2];
   Double_t *xyz = new Double_t[3*npoints];
   TStopwatch *timer = new TStopwatch();
   printf("Random box : %f, %f, %f\n", dx, dy, dz);
   timer->Start(kFALSE);
   Int_t i;
   for (i=0; i<npoints; i++) {
      xyz[3*i] = ox-dx+2*dx*gRandom->Rndm();
      xyz[3*i+1] = oy-dy+2*dy*gRandom->Rndm();
      xyz[3*i+2] = oz-dz+2*dz*gRandom->Rndm();
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
Double_t TGeoChecker::CheckVoxels(TGeoVolume *vol, TGeoVoxelFinder *voxels, Double_t *xyz, Int_t npoints)
{
// count voxel timing
   TStopwatch timer;
   Double_t time;
   TGeoShape *shape = vol->GetShape();
   TGeoNode *node;
   TGeoMatrix *matrix;
   Double_t *point;
   Double_t local[3];
   Int_t *checklist;
   Int_t ncheck;

   timer.Start();
   for (Int_t i=0; i<npoints; i++) {
      point = xyz + 3*i;
      if (!shape->Contains(point)) continue;
      checklist = voxels->GetCheckList(point, ncheck);
      if (!checklist) continue;
      if (!ncheck) continue;
      for (Int_t id=0; id<ncheck; id++) {
         node = vol->GetNode(checklist[id]);
         matrix = node->GetMatrix();
         matrix->MasterToLocal(point, &local[0]);
         if (node->GetVolume()->GetShape()->Contains(&local[0])) break;
      }   
   }
   time = timer.CpuTime();
   return time;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoChecker::TestVoxels(TGeoVolume *vol, Int_t npoints)
{
// Returns optimal voxelization type for volume vol.
//   kFALSE - cartesian
//   kTRUE  - cylindrical
   TGeoVoxelFinder *voxels = vol->GetVoxels();
   if (!voxels) return kFALSE;
   gRandom= new TRandom3();
   const TGeoShape *shape = vol->GetShape();
   Double_t dx = ((TGeoBBox*)shape)->GetDX();
   Double_t dy = ((TGeoBBox*)shape)->GetDY();
   Double_t dz = ((TGeoBBox*)shape)->GetDZ();
   Double_t ox = (((TGeoBBox*)shape)->GetOrigin())[0];
   Double_t oy = (((TGeoBBox*)shape)->GetOrigin())[1];
   Double_t oz = (((TGeoBBox*)shape)->GetOrigin())[2];
   Double_t *xyz = new Double_t[3*npoints];
   Int_t i;
   // generate npoints
   for (i=0; i<npoints; i++) {
      xyz[3*i] = ox-dx+2*dx*gRandom->Rndm();
      xyz[3*i+1] = oy-dy+2*dy*gRandom->Rndm();
      xyz[3*i+2] = oz-dz+2*dz*gRandom->Rndm();
   }
   Bool_t voxtype = vol->IsCylVoxels();
   Double_t time1, time2;
   TGeoVoxelFinder *vox1, *vox2;
   // build both voxelization types
   if (voxtype) {
      printf("   default voxelization was cylindrical.\n");
      vox2 = voxels;
      vox1 = new TGeoVoxelFinder(vol);
      vox1->Voxelize("");
      if (!vol->GetVoxels()) {
         vol->SetVoxelFinder(voxels);
         delete xyz;
         return voxtype;
      }   
   } else {
      printf("   default voxelization was cartesian.\n");
      vox1 = voxels;
      vox2 = new TGeoCylVoxels(vol);
      vox2->Voxelize("");
      if (!vol->GetVoxels()) {
         vol->SetVoxelFinder(voxels);
         delete xyz;
         return voxtype;
      }   
   }   
   // count both voxelization timings
   time1 = CheckVoxels(vol, vox1, xyz, npoints);
   time2 = CheckVoxels(vol, vox2, xyz, npoints);

   printf("   --- time for XYZ : %g\n", time1);
   printf("   --- time for cyl : %g\n", time2);

   if (time1<time2) {
      printf("   best : XYZ\n");
      delete vox2;
      vol->SetVoxelFinder(vox1);
      vol->SetCylVoxels(kFALSE);
      delete xyz;
      return kFALSE;
   }
   printf("   best : cyl\n");
   delete vox1;
   vol->SetVoxelFinder(vox2);
   vol->SetCylVoxels(kTRUE);
   delete xyz;
   return kTRUE;      
}
//-----------------------------------------------------------------------------
void TGeoChecker::CreateTree(const char * /*treename*/, const char * /*filename*/)
{
// These points are stored in a tree and can be directly visualized within ROOT.
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
//-----------------------------------------------------------------------------
void TGeoChecker::Generate(UInt_t /*npoint*/)
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
void TGeoChecker::Raytrace(Double_t * /*startpoint*/, UInt_t /*npoints*/)
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
void TGeoChecker::ShowPoints(Option_t * /*option*/)
{
// 
//Begin_Html
/*
<img src=".gif">
*/
//End_Html
}
