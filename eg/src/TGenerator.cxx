// @(#)root/eg:$Name:  $:$Id: TGenerator.cxx,v 1.6 2002/01/23 17:52:47 rdm Exp $
// Author: Ola Nordmann   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenerator                                                            //
//                                                                      //
// Is an abstact base class, that defines the interface of ROOT and the //
// various event generators. Every event generator should inherit from  //
// TGenerator or its subclasses.                                         //
//                                                                      //
// Every class inherited from TGenerator knows already the interface to  //
// the /HEPEVT/ common block. So in the event creation of the various   //
// generators, the /HEPEVT/ common block should be filled               //
// The ImportParticles method then parses the result from the event        //
// generators into a TClonesArray of TParticle objects.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TGenerator.h"
#include "TDatabasePDG.h"
#include "TParticlePDG.h"
#include "TParticle.h"
#include "TObjArray.h"
#include "Hepevt.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TText.h"
#include "TPaveText.h"
#include "TClonesArray.h"
#include "Riostream.h"


ClassImp(TGenerator)

//______________________________________________________________________________
TGenerator::TGenerator(const char *name,const char *title): TNamed(name,title)
{
//
//  Event generator default constructor
//


  //  Initialize particles table
   TDatabasePDG::Instance();
   //TDatabasePDG *pdg = TDatabasePDG::Instance();
   //if (!pdg->ParticleList()) pdg->Init();

   fPtCut        = 0;
   fShowNeutrons = kTRUE;
   fParticles    =  new TObjArray(10000);
}

//______________________________________________________________________________
TGenerator::~TGenerator()
{
//
//  Event generator default destructor
//

   //do nothing
   if (fParticles) {
      fParticles->Delete();
      delete fParticles;
      fParticles = 0;
   }
}

//______________________________________________________________________________
TObjArray* TGenerator::ImportParticles(Option_t *option)
{
//
//  Default primary creation method. It reads the /HEPEVT/ common block which
//  has been filled by the GenerateEvent method. If the event generator does
//  not use the HEPEVT common block, This routine has to be overloaded by
//  the subclasses.
//  The default action is to store only the stable particles (ISTHEP = 1)
//  This can be demanded explicitly by setting the option = "Final"
//  If the option = "All", all the particles are stored.
//
  fParticles->Clear();
  Int_t numpart = HEPEVT.nhep;
  if (!strcmp(option,"") || !strcmp(option,"Final")) {
    for (Int_t i = 0; i<numpart; i++) {
      if (HEPEVT.isthep[i] == 1) {
//
//  Use the common block values for the TParticle constructor
//
        TParticle *p = new TParticle(
                                   HEPEVT.idhep[i],
                                   HEPEVT.isthep[i],
                                   HEPEVT.jmohep[i][0]-1,
                                   HEPEVT.jmohep[i][1]-1,
                                   HEPEVT.jdahep[i][0]-1,
                                   HEPEVT.jdahep[i][1]-1,

                                   HEPEVT.phep[i][0],
                                   HEPEVT.phep[i][1],
                                   HEPEVT.phep[i][2],
                                   HEPEVT.phep[i][3],
                                   HEPEVT.vhep[i][0],
                                   HEPEVT.vhep[i][1],
                                   HEPEVT.vhep[i][2],
                                   HEPEVT.vhep[i][3]);
        fParticles->Add(p);
        }
     }
  }
  else if (!strcmp(option,"All")) {
    for (Int_t i = 0; i<numpart; i++) {
      TParticle *p = new TParticle(
                                   HEPEVT.idhep[i],
                                   HEPEVT.isthep[i],
                                   HEPEVT.jmohep[i][0]-1,
                                   HEPEVT.jmohep[i][1]-1,
                                   HEPEVT.jdahep[i][0]-1,
                                   HEPEVT.jdahep[i][1]-1,

                                   HEPEVT.phep[i][0],
                                   HEPEVT.phep[i][1],
                                   HEPEVT.phep[i][2],
                                   HEPEVT.phep[i][3],
                                   HEPEVT.vhep[i][0],
                                   HEPEVT.vhep[i][1],
                                   HEPEVT.vhep[i][2],
                                   HEPEVT.vhep[i][3]);
      fParticles->Add(p);
    }
  }
  return fParticles;
}

//______________________________________________________________________________
Int_t TGenerator::ImportParticles(TClonesArray *particles, Option_t *option)
{
//
//  Default primary creation method. It reads the /HEPEVT/ common block which
//  has been filled by the GenerateEvent method. If the event generator does
//  not use the HEPEVT common block, This routine has to be overloaded by
//  the subclasses.
//  The function loops on the generated particles and store them in
//  the TClonesArray pointed by the argument particles.
//  The default action is to store only the stable particles (ISTHEP = 1)
//  This can be demanded explicitly by setting the option = "Final"
//  If the option = "All", all the particles are stored.
//
  if (particles == 0) return 0;
  TClonesArray &Particles = *particles;
  Particles.Clear();
  Int_t numpart = HEPEVT.nhep;
  if (!strcmp(option,"") || !strcmp(option,"Final")) {
    for (Int_t i = 0; i<numpart; i++) {
      if (HEPEVT.isthep[i] == 1) {
//
//  Use the common block values for the TParticle constructor
//
        new(Particles[i]) TParticle(
                                   HEPEVT.idhep[i],
                                   HEPEVT.isthep[i],
                                   HEPEVT.jmohep[i][0]-1,
                                   HEPEVT.jmohep[i][1]-1,
                                   HEPEVT.jdahep[i][0]-1,
                                   HEPEVT.jdahep[i][1]-1,

                                   HEPEVT.phep[i][0],
                                   HEPEVT.phep[i][1],
                                   HEPEVT.phep[i][2],
                                   HEPEVT.phep[i][3],
                                   HEPEVT.vhep[i][0],
                                   HEPEVT.vhep[i][1],
                                   HEPEVT.vhep[i][2],
                                   HEPEVT.vhep[i][3]);
        }
     }
  }
  else if (!strcmp(option,"All")) {
    for (Int_t i = 0; i<numpart; i++) {
      new(Particles[i]) TParticle(
                                   HEPEVT.idhep[i],
                                   HEPEVT.isthep[i],
                                   HEPEVT.jmohep[i][0]-1,
                                   HEPEVT.jmohep[i][1]-1,
                                   HEPEVT.jdahep[i][0]-1,
                                   HEPEVT.jdahep[i][1]-1,

                                   HEPEVT.phep[i][0],
                                   HEPEVT.phep[i][1],
                                   HEPEVT.phep[i][2],
                                   HEPEVT.phep[i][3],
                                   HEPEVT.vhep[i][0],
                                   HEPEVT.vhep[i][1],
                                   HEPEVT.vhep[i][2],
                                   HEPEVT.vhep[i][3]);
    }
  }
  return numpart;
}

//______________________________________________________________________________
void TGenerator::Browse(TBrowser *)
{
    Draw();
    gPad->Update();
}

//______________________________________________________________________________
Int_t TGenerator::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*Compute distance from point px,py to objects in event*-*-*-*
//*-*            =====================================================
//*-*

   const Int_t big = 9999;
   const Int_t inview = 0;
   Int_t dist = big;
   if (px > 50 && py > 50) dist = inview;
   return dist;
}

//______________________________________________________________________________
void TGenerator::Draw(Option_t *option)
{
//
//  Insert one event in the pad list
//

  // Create a default canvas if a canvas does not exist
   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
      gPad->GetVirtCanvas()->SetFillColor(13);
   }

   static Float_t rbox = 1000;
   Float_t rmin[3],rmax[3];
   TView *view = gPad->GetView();
   if (!strstr(option,"same")) {
      if (view) { view->GetRange(rmin,rmax); rbox = rmax[2];}
      gPad->Clear();
   }

   AppendPad(option);

   view = gPad->GetView();
   //    compute 3D view
   if (view) {
      view->GetRange(rmin,rmax);
      rbox = rmax[2];
   } else {
      view = new TView(1);
      view->SetRange(-rbox,-rbox,-rbox, rbox,rbox,rbox );
   }
   const Int_t kColorProton    = 4;
   const Int_t kColorNeutron   = 5;
   const Int_t kColorAntiProton= 3;
   const Int_t kColorPionPlus  = 6;
   const Int_t kColorPionMinus = 2;
   const Int_t kColorKaons     = 7;
   const Int_t kColorElectrons = 0;
   const Int_t kColorGamma     = 18;

   Int_t NProtons    = 0;
   Int_t NNeutrons   = 0;
   Int_t NAntiProtons= 0;
   Int_t NPionPlus   = 0;
   Int_t NPionMinus  = 0;
   Int_t NKaons      = 0;
   Int_t NElectrons  = 0;
   Int_t NGammas     = 0;

   Int_t ntracks = fParticles->GetEntriesFast();
   Int_t i,lwidth,color,lstyle;
   TParticlePDG *ap;
   TParticle *p;
   const char *name;
   Double_t etot,vx,vy,vz;
   Int_t ninvol = 0;
   for (i=0;i<ntracks;i++) {
      p = (TParticle*)fParticles->UncheckedAt(i);
      if(!p) continue;
      ap = (TParticlePDG*)p->GetPDG();
      vx = p->Vx();
      vy = p->Vy();
      vz = p->Vz();
      if  (vx*vx+vy*vy+vz*vz > rbox*rbox) continue;
      Float_t pt = p->Pt();
      if (pt < fPtCut) continue;
      etot = p->Energy();
      if (etot > 0.1) lwidth = Int_t(6*TMath::Log10(etot));
      else lwidth = 1;
      if (lwidth < 1) lwidth = 1;
      lstyle = 1;
      color = 0;
      name = ap->GetName();
      if (!strcmp(name,"n"))     { if (!fShowNeutrons) continue;
                                   color = kColorNeutron;    NNeutrons++;}
      if (!strcmp(name,"p"))     { color = kColorProton;     NProtons++;}
      if (!strcmp(name,"p bar")) { color = kColorAntiProton; NAntiProtons++;}
      if (!strcmp(name,"pi+"))   { color = kColorPionPlus;   NPionPlus++;}
      if (!strcmp(name,"pi-"))   { color = kColorPionMinus;  NPionMinus++;}
      if (!strcmp(name,"e+"))    { color = kColorElectrons;  NElectrons++;}
      if (!strcmp(name,"e-"))    { color = kColorElectrons;  NElectrons++;}
      if (!strcmp(name,"gamma")) { color = kColorGamma;      NGammas++; lstyle = 3; }
      if ( strstr(name,"K"))     { color = kColorKaons;      NKaons++;}
      p->SetLineColor(color);
      p->SetLineStyle(lstyle);
      p->SetLineWidth(lwidth);
      p->AppendPad();
      ninvol++;
   }

   // event title
   TPaveText *pt = new TPaveText(-0.94,0.85,-0.25,0.98,"br");
   pt->AddText((char*)GetName());
   pt->AddText((char*)GetTitle());
   pt->SetFillColor(42);
   pt->Draw();

   // Annotate color codes
   Int_t tcolor = 5;
   if (gPad->GetFillColor() == 10) tcolor = 4;
   TText *text = new TText(-0.95,-0.47,"Particles");
   text->SetTextAlign(12);
   text->SetTextSize(0.025);
   text->SetTextColor(tcolor);
   text->Draw();
   text->SetTextColor(kColorGamma);      text->DrawText(-0.95,-0.52,"(on screen)");
   text->SetTextColor(kColorGamma);      text->DrawText(-0.95,-0.57,"Gamma");
   text->SetTextColor(kColorProton);     text->DrawText(-0.95,-0.62,"Proton");
   text->SetTextColor(kColorNeutron);    text->DrawText(-0.95,-0.67,"Neutron");
   text->SetTextColor(kColorAntiProton); text->DrawText(-0.95,-0.72,"AntiProton");
   text->SetTextColor(kColorPionPlus);   text->DrawText(-0.95,-0.77,"Pion +");
   text->SetTextColor(kColorPionMinus);  text->DrawText(-0.95,-0.82,"Pion -");
   text->SetTextColor(kColorKaons);      text->DrawText(-0.95,-0.87,"Kaons");
   text->SetTextColor(kColorElectrons);  text->DrawText(-0.95,-0.92,"Electrons,etc.");

   text->SetTextColor(tcolor);
   text->SetTextAlign(32);
   char tcount[32];
   sprintf(tcount,"%d",ntracks);      text->DrawText(-0.55,-0.47,tcount);
   sprintf(tcount,"%d",ninvol);       text->DrawText(-0.55,-0.52,tcount);
   sprintf(tcount,"%d",NGammas);      text->DrawText(-0.55,-0.57,tcount);
   sprintf(tcount,"%d",NProtons);     text->DrawText(-0.55,-0.62,tcount);
   sprintf(tcount,"%d",NNeutrons);    text->DrawText(-0.55,-0.67,tcount);
   sprintf(tcount,"%d",NAntiProtons); text->DrawText(-0.55,-0.72,tcount);
   sprintf(tcount,"%d",NPionPlus);    text->DrawText(-0.55,-0.77,tcount);
   sprintf(tcount,"%d",NPionMinus);   text->DrawText(-0.55,-0.82,tcount);
   sprintf(tcount,"%d",NKaons);       text->DrawText(-0.55,-0.87,tcount);
   sprintf(tcount,"%d",NElectrons);   text->DrawText(-0.55,-0.92,tcount);

   sprintf(tcount,"Protons/Pions= %4f",Float_t(NProtons)/Float_t(NPionPlus+NPionMinus));
   text->SetTextAlign(12);
   text->DrawText(-0.45,-0.92,tcount);
   sprintf(tcount,"Kaons/Pions= %4f",Float_t(NKaons)/Float_t(NPionPlus+NPionMinus));
   text->DrawText(0.30,-0.92,tcount);
}


//______________________________________________________________________________
void TGenerator::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================

   if (gPad->GetView()) {
      gPad->GetView()->ExecuteRotateView(event, px, py);
      return;
   }
}

//______________________________________________________________________________
TParticle *TGenerator::GetParticle(Int_t i) const
{
//
//  Returns pointer to primary number i;
//

   if (!fParticles) return 0;
   Int_t n = fParticles->GetLast();
   if (i < 0 || i > n) return 0;
   return (TParticle*)fParticles->UncheckedAt(i);
}

//______________________________________________________________________________
void TGenerator::Paint(Option_t *)
{
//
//  Paint one event
//

}

//______________________________________________________________________________
void TGenerator::SetPtCut(Float_t ptcut)
{
//
//  Set Pt threshold below which primaries are not drawn
//
   fPtCut = ptcut;
   Draw();
   gPad->Update();
}

//______________________________________________________________________________
void TGenerator::SetViewRadius(Float_t rbox)
{
//
//  Set lower and upper values of the view range
//
   SetViewRange(-rbox,-rbox,-rbox,rbox,rbox,rbox);
}

//______________________________________________________________________________
void TGenerator::SetViewRange(Float_t xmin, Float_t ymin, Float_t zmin, Float_t xmax, Float_t ymax, Float_t zmax)
{
//
//  Set lower and upper values of the view range
//
   TView *view = gPad->GetView();
   if (!view) return;
   view->SetRange(xmin,ymin,zmin,xmax,ymax,zmax);

   Draw();
   gPad->Update();
}

//______________________________________________________________________________
void TGenerator::ShowNeutrons(Bool_t show)
{
//
//  Set flag to display or not neutrons
//
   fShowNeutrons = show;
   Draw();
   gPad->Update();
}
