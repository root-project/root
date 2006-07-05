// Program to check a TGeo geometry
// The first time you run this program, the geometry files will be taken
// from http://root.cern.ch/files (or if option "http is given as a command
// line argument (stressGeometry http*).
//   
//    How the program works
// If the file <geom_name>_ref.root does not exist, it is generated. The file
// contains a TTree with Npoints (default=100000) obtained with the following
// algorithm:
//   -a point is generated with a uniform distribution x,y,z in the master volume
//   -a direction theta, phi is generated uniformly in -2pi<phi<2pi and 0<theta<pi
//   -gGeoManager finds the geometry path for the point
//   -the number of boundaries (nbound), total length (length), safety distance
// from the starting point (safe) and number of radiation lengths (rad) from x,y,z 
// is calculated to the exit of the detector. The total number of crossings, detector
// weight and total number of radiation lengths for all tracks are stored as user info in the tree.
//
//  Using the file <geom_name>_ref.root (generated typically with a previous version
//  of the TGeo classes), the Npoints in the Tree are used to perform the
//  same operation with the new version.
//  In case of a disagreement, an error message is reported.
//
//  The ReadRef case is also used as a benchmark
//  The ROOTMARKS reported are relative to a Linux/P IV 2.8 GHz gcc3.2.3 machine
//  normalized at 800 ROOTMARKS when running with CINT.
//
// To run this script, do
//   stressGeometry
// or  stressGeometry *
// or  stressGeometry http*
// or  stressGeometry AlephAlice
// or from the ROOT command line
// root > .L stressGeometry.cxx  or .L stressGeometry.cxx+
// root > stressGeometry(exp_name); // where exp_name is the geometry file name without .root
// OR simply: stressGeometry(); to run tests for a set of geometries
//
// Authors: Rene Brun, Andrei Gheata, 22 march 2005

#include "TStopwatch.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoMedium.h"
#include "TGeoMaterial.h"
#include "TGeoBBox.h"
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TVectorD.h"
#include "TCanvas.h"
#include "TError.h"
#include "TApplication.h"

#ifndef __CINT__
void stressGeometry(const char*);

int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);
   if (argc > 1) stressGeometry(argv[1]);
   else          stressGeometry("*");
   return 0;
}

#endif
   
// data structure for one point
typedef struct {
   Double_t x,y,z,theta,phi;      // Initial track position and direction
   Int_t    nbound;               // Number of boundaries crossed until exit
   Float_t  length;               // Total length up to exit
   Float_t  safe;                 // Safety distance for the initial location
   Float_t  rad;                  // Number of radiation lengths up to exit
} p_t;
p_t p;
  
const char *exps[10] = {"aleph",      // 0
                        "alice",      // 1
                        "brahms",     // 2
                        "cdf",        // 3
                        "cms",        // 4
                        "hades",      // 5
                        "lhcbfull",   // 6
                        "star",       // 7
                        "babar",      // 8
                        "atlas"       // 10
};
// The timings below are on my machine PIV 3GHz
const Double_t cp_brun[10] = {0.8,
                              5.4,
                              1.3,
                              2.2,
                              7.5,
                              0.3,
                              2.0,
                              2.9,
                             17.5,
                             31.0};
// Bounding boxes for experiments
Double_t boxes[10][3] = {{600,600,500},     // aleph
                         {400,400,400},     // alice
                         {50,50,50},        // brahms
                         {500,500,500},     // cdf
                         {800,800,1000},    // cms
                         {250,250,200},     // hades
                         {6700,5000,19000}, // lhcb
                         {350,350,350},     // star
                         {300,300,400},     // babar
                         {1000,1000,1500}   // atlas
};                     
// Total and reference times    
Double_t tpstot = 0;
Double_t tpsref = 70.90; //time including the generation of the ref files
Bool_t testfailed = kFALSE;
                         
Int_t iexp[10];
void FindRad(Double_t x, Double_t y, Double_t z,Double_t theta, Double_t phi, Int_t &nbound, Float_t &length, Float_t &safe, Float_t &rad, Bool_t verbose=kFALSE);
void ReadRef(Int_t kexp);
void WriteRef(Int_t kexp);
void InspectRef(const char *exp="alice");

void stressGeometry(const char *exp="*") {
   gErrorIgnoreLevel = 10;
   
   printf("******************************************************************\n");
   printf("* STRESS GEOMETRY\n");
   TString opt = exp;
   opt.ToLower();
   Bool_t all = kFALSE;
   if (opt.Contains("*")) all = kTRUE;
   Int_t i;
   for (i=0; i<10; i++) {
      if (all) {
         iexp[i] = 1;
         continue;
      }
      if (opt.Contains(exps[i])) iexp[i] = 1;
      else                       iexp[i] = 0;
   }       
   Bool_t http = kFALSE;
   if (opt.Contains("http")) http = kTRUE;
   char fname[24];
   for (i=0; i<10; i++) {
      if (!iexp[i]) continue;
      sprintf(fname, "%s.root", exps[i]);
      if (gGeoManager) {
         delete gGeoManager;
         gGeoManager = 0;
      }   
      if (!http && !gSystem->AccessPathName(fname)) {
         TGeoManager::Import(fname);
      } else {
         //printf(" Accessing %s from http://root.cern.ch/files\n",fname);
         TGeoManager::Import(Form("http://root.cern.ch/files/%s",fname));
         if (!http && gGeoManager) {
            printf("Creating a local copy: %s\n",fname);
            gGeoManager->Export(fname);
         }
      }
      sprintf(fname, "%s_ref.root", exps[i]);
      if (gSystem->AccessPathName(fname)) {
         printf("File: %s does not exist, generating it\n", fname);
         WriteRef(i);
      }
   
      ReadRef(i);
//      InspectRef(exps[i]);
   }   
   if (all && tpstot>0) {
      Float_t rootmarks = 800*tpsref/tpstot;
      Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
      if (UNIX) {
         FILE *fp = gSystem->OpenPipe("uname -a", "r");
         char line[60];
         fgets(line,60,fp); line[59] = 0;
         printf("*  %s\n",line);
         gSystem->ClosePipe(fp);
      } else {
         const char *os = gSystem->Getenv("OS");
         if (!os) printf("*  Windows 95\n");
         else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
      }
      printf("******************************************************************\n");
      if (testfailed) printf("*  stressGeometry found bad points ............. FAILED\n");
      else          printf("*  stressGeometry .................................. OK\n");
      printf("******************************************************************\n");
      printf("*  CPU time in ReadRef = %6.2f seconds\n",tpstot);
      printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),gROOT->GetVersionDate(),gROOT->GetVersionTime());
   }
   printf("******************************************************************\n");
}

void ReadRef(Int_t kexp) {
   TStopwatch sw;
   char fname[24];
   sprintf(fname, "%s_ref.root", exps[kexp]);
   TFile f(fname);
   if (f.IsZombie()) return;
   TTree *TD = new TTree("TD","TGeo stress diff");
   TD->Branch("p",&p.x,"x/D:y/D:z/D:theta/D:phi/D:rad[4]/F");
   TTree *T = (TTree*)f.Get("T");
   T->SetBranchAddress("p",&p.x);
   Long64_t nentries = T->GetEntries();
   TVectorD *vref = (TVectorD *)T->GetUserInfo()->At(0);
   if (!vref) {
      printf(" ERROR: User info not found, regenerate reference file\n");
      return;
   }   
   TVectorD vect(4);
   TVectorD vect_ref = *vref;
   Int_t nbound;
   Float_t length, safe, rad;
   Float_t diff;
   Float_t diffmax = 1e-1;
   Int_t nbad = 0;
   vect(0) = 0;//gGeoManager->Weight(0.01, "va");
   for (Long64_t i=0;i<nentries;i++) {
      T->GetEntry(i);
      nbound = 0;
      length = 0.;
      safe = 0.;
      rad = 0.;
      FindRad(p.x,p.y,p.z, p.theta, p.phi, nbound, length, safe, rad);
      vect(1) += Double_t(nbound);
      vect(2) += length;
      vect(3) += rad;
      diff = 0;
      diff += TMath::Abs(nbound-p.nbound);
      diff += TMath::Abs(length-p.length);
      diff += TMath::Abs(safe-p.safe);
      diff += TMath::Abs(rad-p.rad);
      if (diff > diffmax) {
         nbad++;
         if (nbad < 10) {
            printf(" ==>Point %lld differs with diff = %g, x=%g, y=%g, z=%g\n",i,diff,p.x,p.y,p.z);
            printf("    p.nbound=%d, p.length=%g, p.safe=%g, p.rad=%g\n",
                        p.nbound,p.length,p.safe,p.rad);
            printf("      nbound=%d,   length=%g,   safe=%g,   rad=%g\n",
                        nbound,length,safe,rad);
         }
         TD->Fill();
         p.nbound = nbound;
         p.length = length;
         p.safe   = safe;
         p.rad    = rad;
         TD->Fill();
      }    
   }
   diff = 0.;
   for (Int_t j=1; j<4; j++) diff += TMath::Abs(vect_ref(j)-vect(j));
   if (diff > diffmax) {
//      printf("Total weight=%g   ref=%g\n", vect(0), vect_ref(0));
      printf("Total nbound=%g   ref=%g\n", vect(1), vect_ref(1));
      printf("Total length=%g   ref=%g\n", vect(2), vect_ref(2));
      printf("Total    rad=%g   ref=%g\n", vect(3), vect_ref(3));
      nbad++;  
   }   
      
   if (nbad) {
      testfailed = kTRUE;
      sprintf(fname, "%s_diff.root", exps[kexp]);
      TFile fdiff(fname,"RECREATE");
      TD->AutoSave();
      TD->Print();
   }   
   delete TD;
   
   Double_t cp = sw.CpuTime();
   tpstot += cp;
   if (nbad > 0) printf("*     stress %-15s  found %5d bad points ............. failed\n",exps[kexp],nbad);
   else          printf("*     stress %-15s: time/ref = %6.2f/%6.2f............ OK\n",exps[kexp],cp,cp_brun[kexp]);
}

void WriteRef(Int_t kexp) {
   TRandom3 r;
//   Double_t theta, phi;
   Double_t point[3];
   TVectorD vect(4);
   TGeoShape *top = gGeoManager->GetMasterVolume()->GetShape();
//   TGeoBBox *box = (TGeoBBox*)top;
   Double_t xmax = boxes[kexp][0]; //box->GetDX(); // 300;
   Double_t ymax = boxes[kexp][1]; //box->GetDY(); // 300;
   Double_t zmax = boxes[kexp][2]; //box->GetDZ(); // 500;
   char fname[24];
   sprintf(fname, "%s_ref.root", exps[kexp]);
   TFile f(fname,"recreate");
   TTree *T = new TTree("T","TGeo stress");
   T->Branch("p",&p.x,"x/D:y/D:z/D:theta/D:phi/D:nbound/I:length/F:safe/F:rad/F");
   T->GetUserInfo()->Add(&vect);
   Long64_t Npoints = 10000;
   Long64_t i = 0;
   vect(0) = 0; //gGeoManager->Weight(0.01, "va");
   while (i<Npoints) {
      p.x  = r.Uniform(-xmax,xmax);
      p.y  = r.Uniform(-ymax,ymax);
      p.z  = r.Uniform(-zmax,zmax);
      point[0] = p.x;
      point[1] = p.y;
      point[2] = p.z;
      if (top->Contains(point)) {
         p.phi   =  2*TMath::Pi()*r.Rndm();
         p.theta = TMath::ACos(1.-2.*r.Rndm());
         FindRad(p.x,p.y,p.z, p.theta, p.phi, p.nbound, p.length, p.safe, p.rad);
         vect(1) += Double_t(p.nbound);
         vect(2) += p.length;
         vect(3) += p.rad;
         T->Fill();
         i++;
      }
   }   
   T->AutoSave();
//   T->Print();
   delete T;
}

void FindRad(Double_t x, Double_t y, Double_t z,Double_t theta, Double_t phi, Int_t &nbound, Float_t &length, Float_t &safe, Float_t &rad, Bool_t verbose) {
   Double_t xp  = TMath::Sin(theta)*TMath::Cos(phi);
   Double_t yp  = TMath::Sin(theta)*TMath::Sin(phi);
   Double_t zp  = TMath::Cos(theta);
   Double_t snext;
   char path[256];
   Int_t ismall = 0;
   nbound = 0;
   length = 0.;
   safe   = 0.;
   rad    = 0.;
   TGeoMedium *med;
   gGeoManager->InitTrack(x,y,z,xp,yp,zp);
//   Double_t *point = gGeoManager->GetCurrentPoint();
   if (verbose) {
      printf("Track: (%15.10f,%15.10f,%15.10f,%15.10f,%15.10f,%15.10f)\n",
                       x,y,z,xp,yp,zp);
      sprintf(path, "%s", gGeoManager->GetPath());
   }                    
   TGeoNode *nextnode = gGeoManager->GetCurrentNode();
   safe = gGeoManager->Safety();
   while (nextnode) {
      med = 0;
      if (nextnode) med = nextnode->GetVolume()->GetMedium();
      else return;      
      nextnode = gGeoManager->FindNextBoundaryAndStep();
      nbound++;
      snext  = gGeoManager->GetStep();
      length += snext;
      if (med) {
         Double_t radlen = med->GetMaterial()->GetRadLen();
         if (radlen>1.e-5 && radlen<1.e10)
            rad += med->GetMaterial()->GetDensity()*snext/radlen;
         if (verbose) {
            printf(" STEP #%d: %s\n",nbound, path);
            printf("    step=%g  length=%g  rad=%g %s\n", snext,length,
                   med->GetMaterial()->GetDensity()*snext/med->GetMaterial()->GetRadLen(),med->GetName());
            sprintf(path, "%s", gGeoManager->GetPath());
         }   
      }
      if (snext<1.e-9) {
         ismall++;
         if (ismall > 3) {
            nbound -= ismall; 
//            nextnode = gGeoManager->FindNode();
//            printf("     (%15.10f,%15.10f,%15.10f,%15.10f,%15.10f,%15.10f)\n",
//                       point[0],point[1],point[2],xp,yp,zp);
//            printf("Small steps in: %s\n",gGeoManager->GetPath());
//            gGeoManager->InspectState();
            return;
         }   
      } else {
         ismall = 0;
      }      
   }   
}
  
void InspectDiff(Long64_t ientry=-1) {
   Int_t nbound = 0;   
   Float_t length = 0.;
   Float_t safe   = 0.;
   Float_t rad    = 0.;
   if (!gSystem->AccessPathName("alice.root")) {
      TGeoManager::Import("alice.root");
   } else {
      printf(" ERROR: To run this script you must copy the Alice geometry from\n");
      printf("        ftp://root.cern.ch/root/geom_name.root\n");
      return;
   }
   TFile f("alice_diff.root");
   if (f.IsZombie()) return;
   TTree *TD = (TTree*)f.Get("TD");
   TD->SetBranchAddress("p",&p.x);
   Long64_t nentries = TD->GetEntries();
   nentries = nentries>>1;
   if (ientry>=0 && ientry<nentries) {
      printf("DIFFERENCE #%lld\n", ientry);
      TD->GetEntry(2*ientry);
      printf("   NEW: nbound=%d  length=%g  safe=%g  rad=%g\n", p.nbound,p.length,p.safe,p.rad);
      TD->GetEntry(2*ientry+1);
      printf("   OLD: nbound=%d  length=%g  safe=%g  rad=%g\n", p.nbound,p.length,p.safe,p.rad);
      FindRad(p.x,p.y,p.z, p.theta, p.phi, nbound,length,safe,rad, kTRUE);
      return;
   }   
   for (Long64_t i=0;i<nentries;i++) {
      printf("DIFFERENCE #%lld\n", i);
      TD->GetEntry(2*i);
      printf("   NEW: nbound=%d  length=%g  safe=%g rad=%g\n", p.nbound,p.length,p.safe,p.rad);
      TD->GetEntry(2*i+1);
      printf("   OLD: nbound=%d  length=%g  safe=%g rad=%g\n", p.nbound,p.length,p.safe,p.rad);
      FindRad(p.x,p.y,p.z, p.theta, p.phi, nbound,length,safe,rad, kTRUE);
   }
}   

void InspectRef(const char *exp) {
// Inspect current reference.
   char fname[64];
   sprintf(fname, "%s_ref.root", exp);
   if (gSystem->AccessPathName(fname)) {
      printf("ERROR: file %s does not exist\n", fname);
      return;
   }
   TFile f(fname);
   if (f.IsZombie()) return;
   TTree *T = (TTree*)f.Get("T");
   Long64_t nentries = T->GetEntries();
   sprintf(fname, "Stress test for %s geometry", exp);
   TCanvas *c = new TCanvas("stress", fname,700,800);
   c->Divide(2,2,0.005,0.005);
   c->cd(1);
   gPad->SetLogy();
   T->Draw("p.nbound","","", nentries, 0);
   c->cd(2);
   gPad->SetLogy();
   T->Draw("p.length","","", nentries, 0);
   c->cd(3);
   gPad->SetLogy();
   T->Draw("p.safe","","", nentries, 0);
   c->cd(4);
   gPad->SetLogy();
   T->Draw("p.rad","","", nentries, 0);
   c->cd(0);
   c->SetFillColor(kYellow);
   TVectorD *vref = (TVectorD *)T->GetUserInfo()->At(0);
   TVectorD vect = *vref;
   printf("=====================================\n");
//   printf("Total weight:  %g [kg]\n", vect(0));
   printf("Total nbound:  %g boundaries crossed\n", vect(1));
   printf("Total length:  %g [m]\n", 0.01*vect(2));
   printf("Total nradlen: %f\n", vect(3));   
   printf("=====================================\n");
}
