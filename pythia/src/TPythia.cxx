// @(#)root/pythia:$Name:  $:$Id: TPythia.cxx,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Piotr Golonka   10/09/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TPythia                                                                    //
//                                                                            //
// TPythia is an interface class to F77 version of Pythia 5.7 and Jetset 7.4  //
// CERNLIB event generators, written by T.Sjostrand.                          //
// For details about these generators look at Pythia/Jetset manual:           //
//                                                                            //
//******************************************************************************
//**                                                                          **
//**                                                                          **
//**  PPP  Y   Y TTTTT H   H III   A        JJJJ EEEE TTTTT  SSS  EEEE TTTTT  **
//**  P  P  Y Y    T   H   H  I   A A          J E      T   S     E      T    **
//**  PPP    Y     T   HHHHH  I  AAAAA         J EEE    T    SSS  EEE    T    **
//**  P      Y     T   H   H  I  A   A      J  J E      T       S E      T    **
//**  P      Y     T   H   H III A   A       JJ  EEEE   T    SSS  EEEE   T    **
//**                                                                          **
//**                                                                          **
//**              *......*                  Welcome to the Lund Monte Carlo!  **
//**         *:::!!:::::::::::*                                               **
//**      *::::::!!::::::::::::::*            This is PYTHIA version 5.720    **
//**    *::::::::!!::::::::::::::::*        Last date of change: 29 Nov 1995  **
//**   *:::::::::!!:::::::::::::::::*                                         **
//**   *:::::::::!!:::::::::::::::::*         This is JETSET version 7.408    **
//**    *::::::::!!::::::::::::::::*!       Last date of change: 23 Aug 1995  **
//**      *::::::!!::::::::::::::* !!                                         **
//**      !! *:::!!:::::::::::*    !!                 Main author:            **
//**      !!     !* -><- *         !!              Torbjorn Sjostrand         **
//**      !!     !!                !!        Dept. of theoretical physics 2   **
//**      !!     !!                !!              University of Lund         **
//**      !!                       !!                Solvegatan 14A           **
//**      !!        ep             !!             S-223 62 Lund, Sweden       **
//**      !!                       !!          phone: +46 - 46 - 222 48 16    **
//**      !!                 pp    !!          E-mail: torbjorn@thep.lu.se    **
//**      !!   e+e-                !!                                         **
//**      !!                       !!                                         **
//**      !!                                                                  **
//**                                                                          **
//**                                                                          **
//** The latest program versions and documentation is found on WWW address    **
//** http://thep.lu.se/tf2/staff/torbjorn/Welcome.html                        **
//**                                                                          **
//** When you cite these programs, priority should always be given to the     **
//** latest published description. Currently this is                          **
//** T. Sjostrand, Computer Physics Commun. 82 (1994) 74.                     **
//** The most recent long description (unpublished) is                        **
//** T. Sjostrand, LU TP 95-20 and CERN-TH.7112/93 (revised August 1995).     **
//** Also remember that the programs, to a large extent, represent original   **
//** physics research. Other publications of special relevance to your        **
//** studies may therefore deserve separate mention.                          **
//**                                                                          **
//**                                                                          **
//******************************************************************************

#include "TPythia.h"
#include "Pcommon.h"
#include "TMCParticle.h"
#include "TParticle.h"

#include "TCanvas.h"
#include "TView.h"
#include "TROOT.h"
#include "TPaveText.h"

#ifndef WIN32
# define pytest pytest_
# define pyinit pyinit_
# define pyevnt pyevnt_
# define pystat pystat_
# define lulist lulist_
# define luexec luexec_
# define lucomp lucomp_
# define type_of_call
#else
# define pytest PYTEST
# define pyinit PYINIT
# define pyevnt PYEVNT
# define pystat PYSTAT
# define lulist LULIST
# define luexec LUEXEC
# define lucomp LUCOMP
# define type_of_call _stdcall
#endif

extern "C" void type_of_call pytest(Long_t &key);
#ifndef WIN32
extern "C" void type_of_call pyinit(char *frame, char *beam, char *target,
                                     float &win, Long_t l_frame, Long_t l_beam,
                                     Long_t l_target);
#else
extern "C" void type_of_call pyinit(char *frame,  Long_t l_frame,
                                    char *beam,   Long_t l_beam,
                                    char *target, Long_t l_target,
                                    float &win
                                    );
#endif
extern "C" void type_of_call pyevnt();
extern "C" void type_of_call pystat(Long_t &key);
extern "C" void type_of_call lulist(Long_t &key);
extern "C" void type_of_call luexec();
extern "C" int  type_of_call lucomp(Long_t &kf);

ClassImp(TPythia)


//______________________________________________________________________________
TPythia::TPythia() : TGenerator("Pythia","Pythia")
{
// TPythia constructor: creates a TClonesArray in which it will store all
// particles. Note that there may be only one functional TPythia object
// at a time, so it's not use to create more than one instance of it.

    delete fParticles; // was allocated as TObjArray in TGenerator

    fParticles = new TClonesArray("TMCParticle",50);
}

//______________________________________________________________________________
TPythia::~TPythia()
{
// Destroys the object, deletes and disposes all TMCParticles currently on list.

    if (fParticles) {
      fParticles->Delete();
      delete fParticles;
      fParticles = 0;
   }
}

//______________________________________________________________________________
void TPythia::Draw(Option_t *option)
{
// Event display - not supported for TPythia yet.

   if (!gPad) {
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
      gPad->GetCanvas()->SetFillColor(13);
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

   TPaveText *pt = new TPaveText(-0.94,0.85,-0.25,0.98,"br");
   pt->AddText((char*)GetName());
   pt->AddText((char*)GetTitle());
   pt->SetFillColor(42);
   pt->Draw();
}

//______________________________________________________________________________
TObjArray *TPythia::ImportParticles(Option_t *)
{
// Fills TClonesArray fParticles list with particles from common LUJETS.
// Old contents of a list are cleared. This function should be called after
// any change in common LUJETS, however GetParticles() method  calls it
// automatically - user don't need to care about it. In case you make a call
// to LuExec() you must call this method yourself to transfer new data from
// common LUJETS to the fParticles list.

   fParticles->Clear();

   Int_t numpart   = LUJETS.n;
   TClonesArray &a = *((TClonesArray*)fParticles);

   for (Int_t i = 0; i < numpart; i++) {
      new(a[i]) TMCParticle(LUJETS.k[0][i] ,
                            LUJETS.k[1][i] ,
                            LUJETS.k[2][i] ,
                            LUJETS.k[3][i] ,
                            LUJETS.k[4][i] ,

                            LUJETS.p[0][i] ,
                            LUJETS.p[1][i] ,
                            LUJETS.p[2][i] ,
                            LUJETS.p[3][i] ,
                            LUJETS.p[4][i] ,

                            LUJETS.v[0][i] ,
                            LUJETS.v[1][i] ,
                            LUJETS.v[2][i] ,
                            LUJETS.v[3][i] ,
                            LUJETS.v[4][i]);

   }
   return fParticles;
}

//______________________________________________________________________________
Int_t TPythia::ImportParticles(TClonesArray *particles, Option_t *option)
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
  Int_t numpart = LUJETS.n;
  if (!strcmp(option,"") || !strcmp(option,"Final")) {
    for (Int_t i = 0; i<numpart; i++) {
      if (LUJETS.k[1][i] == 1) {
//
//  Use the common block values for the TParticle constructor
//
        new(Particles[i]) TParticle(
                            LUJETS.k[1][i] ,
                            LUJETS.k[0][i] ,
                            LUJETS.k[2][i] ,
                            -1,
                            LUJETS.k[3][i] ,
                            LUJETS.k[4][i] ,

                            LUJETS.p[0][i] ,
                            LUJETS.p[1][i] ,
                            LUJETS.p[2][i] ,
                            LUJETS.p[3][i] ,

                            LUJETS.v[0][i] ,
                            LUJETS.v[1][i] ,
                            LUJETS.v[2][i] ,
                            LUJETS.v[3][i]);
      }
    }
  }
  else if (!strcmp(option,"All")) {
    for (Int_t i = 0; i<numpart; i++) {
        new(Particles[i]) TParticle(
                            LUJETS.k[1][i] ,
                            LUJETS.k[0][i] ,
                            LUJETS.k[2][i] ,
                            -1,
                            LUJETS.k[3][i] ,
                            LUJETS.k[4][i] ,

                            LUJETS.p[0][i] ,
                            LUJETS.p[1][i] ,
                            LUJETS.p[2][i] ,
                            LUJETS.p[3][i] ,

                            LUJETS.v[0][i] ,
                            LUJETS.v[1][i] ,
                            LUJETS.v[2][i] ,
                            LUJETS.v[3][i]);
    }
  }
  return numpart;
}

//====================== access to common PYSUBS ===============================

//______________________________________________________________________________
void TPythia::SetMSEL(Int_t sel)
{
// Sets a value of MSEL in common PYSUBS.
// (D=1) a switch to select between full user control and some preprogrammed
//  alternatives:
//       0 = full user control , desired subprocesses have to be switched on
//           using SetMSUB()
// (see documentation for further details).
// This setting should be done before a call to Initialize() or Pyinit().

   PYSUBS.msel=sel;
}

//______________________________________________________________________________
Int_t TPythia::GetMSEL() const
{
// returns a current value of MSEL in common PYSUBS.

   return PYSUBS.msel;
}

//______________________________________________________________________________
void TPythia::SetMSUB(Int_t isub, Bool_t msub)
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:SetMSUB(isub,msub):\n ");
      printf ("      isub=%i is out of range [1..200]!\n",isub);
      return;
   }

   Int_t value         = (msub != 0);
   PYSUBS.msub[isub-1] = value;

}

//______________________________________________________________________________
Bool_t TPythia::GetMSUB(Int_t isub) const
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:GetMSUB(isub): \n ");
      printf ("      isub=%i is out of range [1..200]!\n",isub);
      return 0;
   }

   return PYSUBS.msub[isub-1];

}


//______________________________________________________________________________
void TPythia::SetKFIN(Int_t i, Int_t j, Bool_t kfin)
{
   if ( i!=1 &&  i!=2 ) {
      printf("ERROR in TPythia::SetKFIN(i,j,kfin):\n");
      printf("      side: i=%i is neither 1(=beam) nor 2(=target)\n",i);
      return;
   }

   if ( j<-40 || j>40) {
      printf("ERROR in TPythia::SetKFIN(side,flavour,kfin):\n");
      printf("      flavour: j=%i is not in range [-40..40]",j);
      return;
   }

   Int_t value            = (kfin!=0);
   PYSUBS.kfin[j+40][i-1] = value;

}

//______________________________________________________________________________
Bool_t TPythia::GetKFIN(Int_t i, Int_t j) const
{
   if ( i!=1 && i!=2 ) {
      printf("ERROR in TPythia::GetKFIN(i,j):\n");
      printf("      side: i=%i is neither 1(=beam) nor 2(=target)\n",i);
      return 0;
   }

   if ( j<-40|| j>40) {
      printf("ERROR in TPythia::GetKFIN(i,j):\n");
      printf("      flavour: j=%i is not in range [-40..40]",j);
      return 0;
   }

   return PYSUBS.kfin[j+40][i-1];

}

//______________________________________________________________________________
void TPythia::SetCKIN(Int_t key, Float_t value)
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:SetCKIN(key,value): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return;
   }

   PYSUBS.ckin[key-1]=value;

}

//______________________________________________________________________________
Float_t TPythia::GetCKIN(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:GetCKIN(key): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return 0.0;
   }

   return PYSUBS.ckin[key-1];

}

//====================== access to common PYPARS ===============================

//______________________________________________________________________________
void TPythia::SetMSTP(Int_t key,Int_t value)
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:SetMSTP(key,value): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return;
   }

   PYPARS.mstp[key-1]=value;

}

//______________________________________________________________________________
Int_t TPythia::GetMSTP(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:GetMSTP(key): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return 0;
   }

   return PYPARS.mstp[key-1];

}

//______________________________________________________________________________
void TPythia::SetPARP(Int_t key,Float_t value)
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:SetPARP(key,value): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return;
   }

   PYPARS.parp[key-1]=value;

}

//______________________________________________________________________________
Float_t TPythia::GetPARP(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:GetPARP(key): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return 0;
   }

   return PYPARS.parp[key-1];

}


//______________________________________________________________________________
void TPythia::SetMSTI(Int_t key,Int_t value)
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:SetMSTI(key,value): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return;
   }

   PYPARS.msti[key-1]=value;

}


//______________________________________________________________________________
Int_t TPythia::GetMSTI(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:GetMSTI(key): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return 0;
   }

   return PYPARS.msti[key-1];

}


//______________________________________________________________________________
void TPythia::SetPARI(Int_t key,Float_t value)
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:SetPARI(key,value): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return;
   }

   PYPARS.pari[key-1]=value;

}

//______________________________________________________________________________
Float_t TPythia::GetPARI(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf ("ERROR in TPythia:GetPARI(key): \n ");
      printf ("      key=%i is out of range [1..200]!\n",key);
      return 0;
   }

   return PYPARS.pari[key-1];

}


//====================== access to common PYINT1 ===============================

//______________________________________________________________________________
void  TPythia::SetMINT(Int_t key, Int_t value)
{
   if (key<1 || key>400) {
      printf ("ERROR in TPythia:SetMINT(key,value): \n ");
      printf ("      key=%i is out of range [1..400]!\n",key);
      return;
   }

   PYINT1.mint[key-1]=value;

}

//______________________________________________________________________________
Int_t TPythia::GetMINT(Int_t key) const
{
   if (key<1 || key>400) {
      printf ("ERROR in TPythia:GetMINT(key): \n ");
      printf ("      key=%i is out of range [1..400]!\n",key);
      return 0;
   }

   return PYINT1.mint[key-1];

}


//______________________________________________________________________________
void  TPythia::SetVINT(Int_t key, Float_t value)
{
   if ( key<1 || key>400 ) {
      printf ("ERROR in TPythia:SetVINT(key,value): \n ");
      printf ("  array index: key=%i is out of range [1..400]!\n",key);
      return;
   }

   PYINT1.vint[key-1]=value;

}


//______________________________________________________________________________
Float_t TPythia::GetVINT(Int_t key) const
{
   if ( key<1 || key>400 ) {
      printf ("ERROR in TPythia:GetVINT(key): \n ");
      printf ("  array index: key=%i is out of range [0..400]!\n",key);
      return 0.0;
   }

   return PYINT1.vint[key-1];

}



//====================== access to common PYINT2 ===============================

//______________________________________________________________________________
void  TPythia::SetISET(Int_t isub,Int_t iset )
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:SetVSET(isub,iset): \n ");
      printf ("   isub=%i - out of range [1..200]!\n",isub);
      return;
   }

   if ( iset<-2 || iset>11 ) {
      printf ("ERROR in TPythia:SetVSET(isub,iset): \n ");
      printf ("      unsuported value of iset=%i - out of range [-2..11]!\n",iset);
      return;
   }

   PYINT2.iset[isub-1]=iset;

}

//______________________________________________________________________________
Int_t TPythia::GetISET(Int_t isub) const
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:GetVSET(isub): \n ");
      printf ("      isub=%i - out of range [1..200]!\n",isub);
      return 0;
   }

   return PYINT2.iset[isub-1];

}


//______________________________________________________________________________
void TPythia::SetKFPR(Int_t isub, Int_t j, Int_t kfpr)
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:SetKFPR(isub,j,kf): \n ");
      printf ("      isub=%i is out of range [1..200]!\n",isub);
      return;
   }

   if ( j!=1 && j!=2 ){
      printf ("ERROR in TPythia:SetKFPR(isub,j,kf): \n ");
      printf ("      j=%i is neither 1 nor 2 \n",j);
      return;
   }

   PYINT2.kfpr[j-1][isub-1]=kfpr;

}


//______________________________________________________________________________
Int_t TPythia::GetKFPR(Int_t isub, Int_t j) const
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:GetKFPR(isub,j): \n ");
      printf ("      isub=%i is out of range [1..200]!\n",isub);
      return 0;
   }

   if ( j!=1 && j!=2 ) {
      printf ("ERROR in TPythia:GetKFPR(isub,j): \n ");
      printf ("      j=%i is neither 1 nor 2 \n",j);
      return 0;
   }

   return PYINT2.kfpr[j-1][isub-1];

}

//______________________________________________________________________________
void  TPythia::SetCOEF(Int_t isub, Int_t j, Float_t coef)
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:SetCOEF(isub,j,coef): \n ");
      printf ("      isub=%i is out of range [1..200]!\n",isub);
      return;
   }

   if ( j<1 || j>20 ) {
      printf ("ERROR in TPythia:SetCOEF(isub,j,coef): \n ");
      printf ("      j=%i is out of range [1..20]",j);
      return;
   }

   PYINT2.coef[j-1][isub-1]=coef;

}

//______________________________________________________________________________
Float_t TPythia::GetCOEF(Int_t isub, Int_t j) const
{
   if ( isub<1 || isub>200 ) {
      printf ("ERROR in TPythia:GetCOEF(isub,j): \n ");
      printf ("      isub=%i is out of range [0..200]!\n",isub);
      return 0.0;
   }

   if ( j<1 && j>20 ) {
      printf ("ERROR in TPythia:GetCOEF(isub,j): \n ");
      printf ("  array index: j=%i is out of range [1..20] \n",j);
      return 0.0;
   }

   return PYINT2.coef[j-1][isub-1];

}


//______________________________________________________________________________
void TPythia::SetICOL(Int_t kf,Int_t i, Int_t j,Int_t value)
{
   if ( kf<1 || kf>40 ) {
      printf ("ERROR in TPythia:SetICOL(kf,i,j,value): \n ");
      printf ("      kf=%i is out of range [1..40]!\n",kf);
      return;
   }

   if ( i<1 || i>4 ) {
      printf ("ERROR in TPythia:SetICOL(kf,i,j,value): \n ");
      printf ("  array index: i=%i is out of range [1..4]!\n",i);
      return;
   }

   if ( j!=1 && j!=2 ) {
      printf ("ERROR in TPythia:SetICOL(kf,i,j,value): \n ");
      printf ("      j=%i is neither 1 nor 2 !\n",j);
      return;
   }

   PYINT2.icol[j-1][i-1][kf-1]=value;

}

//______________________________________________________________________________
Int_t  TPythia::GetICOL(Int_t kf,Int_t i, Int_t j) const
{
   if ( kf<1 || kf>40 ) {
      printf ("ERROR in TPythia:GetICOL(kf,i,j): \n ");
      printf ("      kf=%i is out of range [1..40]!\n",kf);
      return 0;
   }

   if ( i<1 || i>4 ) {
      printf ("ERROR in TPythia:GetICOL(kf,i,j): \n ");
      printf ("      i=%i is out of range [1..4]!\n",i);
      return 0;
   }

   if ( j!=1 && j!=2 ) {
      printf ("ERROR in TPythia:GetICOL(kf,i,j): \n ");
      printf ("      j=%i is neither 1 nor 2 !\n",kf);
      return 0;
   }

   return PYINT2.icol[j-1][i-1][kf-1];

}



//====================== access to common PYINT3 ===============================

//______________________________________________________________________________
Float_t TPythia::GetXSFX(Int_t side, Int_t kf) const
{
   if (side!=1 && side !=2) {
      printf("ERROR in TPythia::GetXSFX(side,kf):\n");
      printf("      side=%i is neither 1(=beam) nor 2(=target)\n",side);
      return 0.0;
   }

   if ( kf<-40 || kf >40 ) {
      printf("ERROR in TPythia::GetXSFX(side,kf):\n");
      printf("      kf=%i is out of range [-40..40]\n",kf);
      return 0.0;
   }

   return PYINT3.xsfx[kf+40][side-1];

}


//______________________________________________________________________________
Int_t TPythia::GetISIG(Int_t ichn, Int_t isig) const
{
   if ( ichn<1 || ichn>1000 ) {
      printf("ERROR in TPythia::GetISIG(ichn,isig):\n");
      printf("      ichn=%i is out of range [1..1000]\n",ichn);
      return 0;
   }

   if ( isig<1 || isig >3 ) {
      printf("ERROR in TPythia::GetISIG(ichn,isig):\n");
      printf("      isig=%i is out of range [1..3]\n",isig);
      return 0;
   }

   return PYINT3.isig[ichn-1][isig-1];

}


//______________________________________________________________________________
Float_t TPythia::GetSIGH(Int_t ichn) const
{
   if (ichn<1 || ichn >1000) {
      printf("ERROR in TPythia::GetSIGH(ichn):\n");
      printf("      ichn=%i is out of range [1..1000]\n",ichn);
      return 0.0;
   }

   return PYINT3.sigh[ichn-1];

}



//====================== access to common PYINT4 ===============================

//______________________________________________________________________________
Float_t TPythia::GetWIDP(Int_t kf, Int_t j) const
{
   if ( kf<21 || kf >40 ) {
      printf("ERROR in TPythia::GetWIDP(kf,j):\n");
      printf("      kf=%i is out of range [21..40]\n",kf);
      return 0.0;
   }

   if ( j<0 || j >40 ) {
      printf("ERROR in TPythia::GetWIDP(kf,j):\n");
      printf("      j=%i is out of range [0..40]\n",j);
      return 0.0;
   }

   return PYINT4.widp[j][kf-21];

}


//______________________________________________________________________________
Float_t TPythia::GetWIDE(Int_t kf, Int_t j) const
{
   if ( kf<21 || kf >40 ) {
      printf("ERROR in TPythia::GetWIDE(kf,j):\n");
      printf("      kf=%i is out of range [21..40]\n",kf);
      return 0.0;
   }

   if ( j<0 || j >40 ) {
      printf("ERROR in TPythia::GetWIDE(kf,j):\n");
      printf("      j=%i is out of range [0..40]\n",j);
      return 0.0;
   }

   return PYINT4.wide[j][kf-21];

}


//______________________________________________________________________________
Float_t TPythia::GetWIDS(Int_t kf, Int_t j) const
{
   if ( kf<21 || kf >40 ) {
      printf("ERROR in TPythia::GetWIDP(kf,j):\n");
      printf("      kf=%i is out of range [21..40]\n",kf);
      return 0.0;
   }

   if ( j<1 || j >3 ) {
      printf("ERROR in TPythia::GetWIDP(kf,j):\n");
      printf("      j=%i is out of range [1..3]\n",j);
      return 0.0;
   }

   return PYINT4.wids[j-1][kf-21];

}




//====================== access to common PYINT5 ===============================

//______________________________________________________________________________
Int_t TPythia::GetNGEN(Int_t isub, Int_t key) const
{
   if ( isub<0 || isub>200 ) {
      printf("ERROR in TPythia::GetNGEN(isub.key):\n");
      printf("      isub=%i is out of range [0..200]\n",isub);
      return 0;
   }

  if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::GetNGEN(isub.key):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return 0;
   }

   return PYINT5.ngen[key-1][isub];

}


//______________________________________________________________________________
Float_t TPythia::GetXSEC(Int_t isub, Int_t key) const
{
   if ( isub<0 || isub >200 ) {
      printf("ERROR in TPythia::GetXSEC(isub.key):\n");
      printf("      isub=%i is out of range [0..200]\n",isub);
      return 0.0;
   }

   if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::GetXSEC(isub.key):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return 0.0;
   }

   return PYINT5.xsec[key-1][isub];

}

//====================== access to common LUDATR ===============================

//______________________________________________________________________________
Int_t TPythia::GetMRLU(Int_t key) const
{
   if ( key<1 || key>6 ) {
      printf("ERROR in TPythia::GetMRLU(key):\n");
      printf("      key=%i is out of range [1..6]\n",key);
      return 0;
   }

   return LUDATR.mrlu[key-1];

}

//______________________________________________________________________________
void TPythia::SetMRLU(Int_t key, Int_t seed)
{
   if ( key<1 || key>6 ) {
      printf("ERROR in TPythia::SetMRLU(key,seed):\n");
      printf("      key=%i is out of range [1..6]\n",key);
      return;
   }

   LUDATR.mrlu[key-1] = seed;

}


//______________________________________________________________________________
Float_t TPythia::GetRRLU(Int_t key) const
{
   if ( key<1 || key>100 ) {
      printf("ERROR in TPythia::GetRRLU(key):\n");
      printf("      key=%i is out of range [1..100]\n",key);
      return 0.0;
   }

   return LUDATR.rrlu[key-1];

}

//______________________________________________________________________________
void TPythia::SetRRLU(Int_t key, Float_t value)
{
   if ( key<1 || key>100 ) {
      printf("ERROR in TPythia::SetRRLU(key,seed):\n");
      printf("      key=%i is out of range [1..100]\n",key);
      return;
   }

   LUDATR.rrlu[key-1] = value;

}

//====================== access to common LUDAT1 ===============================

//______________________________________________________________________________
Int_t TPythia::GetMSTU(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::GetMSTU(key):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return 0;
   }

   return LUDAT1.mstu[key-1];

}


//______________________________________________________________________________
void TPythia::SetMSTU(Int_t key, Int_t value)
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::SetMSTU(key,value):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return;
   }

   LUDAT1.mstu[key-1] = value;

}


//______________________________________________________________________________
Float_t TPythia::GetPARU(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::GetPARU(key):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return 0.0;
   }

   return LUDAT1.paru[key-1];

}


//______________________________________________________________________________
void TPythia::SetPARU(Int_t key, Float_t value)
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::SetPARU(key,value):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return;
   }

   LUDAT1.paru[key-1] = value;

}

//______________________________________________________________________________
Int_t TPythia::GetMSTJ(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::GetMSTJ(key):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return 0;
   }

   return LUDAT1.mstj[key-1];

}


//______________________________________________________________________________
void TPythia::SetMSTJ(Int_t key, Int_t value)
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::SetMSTJ(key,value):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return;
   }

   LUDAT1.mstj[key-1] = value;

}


//______________________________________________________________________________
Float_t TPythia::GetPARJ(Int_t key) const
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::GetPARJ(key):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return 0.0;
   }

   return LUDAT1.parj[key-1];

}


//______________________________________________________________________________
void TPythia::SetPARJ(Int_t key, Float_t value)
{
   if ( key<1 || key>200 ) {
      printf("ERROR in TPythia::SetPARJ(key,value):\n");
      printf("      key=%i is out of range [1..200]\n",key);
      return;
   }

   LUDAT1.parj[key-1] = value;

}





//====================== access to common LUDAT2 ===============================

//______________________________________________________________________________
Int_t TPythia::GetKCHG(Int_t kc,Int_t key) const
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::GetKCHG(kc,key):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return 0;
   }

   if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::GetKCHG(kc,key):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return 0;
   }

   return LUDAT2.kchg[key-1][kc-1];

}

//______________________________________________________________________________
void TPythia::SetKCHG(Int_t kc,Int_t key, Int_t value)
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::SetKCHG(kc,key,value):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return;
   }

   if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::SetKCHG(kc,key):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return;
   }

   LUDAT2.kchg[key-1][kc-1] = value;

}

//______________________________________________________________________________
Float_t TPythia::GetPMAS(Int_t kc,Int_t key) const
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::GetPMAS(kc,key):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return 0.0;
   }

   if ( key<1 || key>4 ) {
      printf("ERROR in TPythia::GetPMAS(kc,key):\n");
      printf("      key=%i is out of range [1..4]\n",key);
      return 0.0;
   }

   return LUDAT2.pmas[key-1][kc-1];

}

//______________________________________________________________________________
void TPythia::SetPMAS(Int_t kc,Int_t key, Float_t value)
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::SetPMAS(kc,key,value):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return;
   }

   if ( key<1 || key>4 ) {
      printf("ERROR in TPythia::SetPMAS(kc,key):\n");
      printf("      key=%i is out of range [1..4]\n",key);
      return;
   }

   LUDAT2.pmas[key-1][kc-1] = value;

}


//______________________________________________________________________________
Float_t TPythia::GetPARF(Int_t key) const
{

   if ( key<1 || key>2000 ) {
      printf("ERROR in TPythia::GetPARF(key):\n");
      printf("      key=%i is out of range [1..2000]\n",key);
      return 0.0;
   }

   return LUDAT2.parf[key-1];

}

//______________________________________________________________________________
void TPythia::SetPARF(Int_t key, Float_t value)
{
   if ( key<1 || key>2000 ) {
      printf("ERROR in TPythia::SetPARF(key,value):\n");
      printf("      key=%i is out of range [1..2000]\n",key);
      return;
   }

   LUDAT2.parf[key-1] = value;

}

//______________________________________________________________________________
Float_t TPythia::GetVCKM(Int_t i,Int_t j) const
{
   if ( i<1 || i>4 ) {
      printf("ERROR in TPythia::GetVCKM(i,j):\n");
      printf("      up generation index i=%i is out of range [1..4]\n",i);
      return 0;
   }
   if ( j<1 || j>4 ) {
      printf("ERROR in TPythia::GetVCKM(i,j):\n");
      printf("      down generation index j=%i is out of range [1..4]\n",j);
      return 0;
   }


   return LUDAT2.vckm[j-1][i-1];

}


//______________________________________________________________________________
void TPythia::SetVCKM(Int_t i,Int_t j,Float_t value)
{
   if ( i<1 || i>4 ) {
      printf("ERROR in TPythia::SetVCKM(i,j):\n");
      printf("      up generation index i=%i is out of range [1..4]\n",i);
      return;
   }
   if ( j<1 || j>4 ) {
      printf("ERROR in TPythia::SetVCKM(i,j):\n");
      printf("      down generation index j=%i is out of range [1..4]\n",j);
      return;
   }


   LUDAT2.vckm[j-1][i-1]=value;

}



//====================== access to common LUDAT3 ===============================

//______________________________________________________________________________
Int_t TPythia::GetMDCY(Int_t kc,Int_t key) const
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::GetMDCY(kc,key):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return 0;
   }

   if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::GetMDCY(kc,key):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return 0;
   }

   return LUDAT3.mdcy[key-1][kc-1];

}

//______________________________________________________________________________
void TPythia::SetMDCY(Int_t kc,Int_t key, Int_t value)
{
   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::GetMDCY(kc,key,value):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return;
   }

   if ( key<1 || key>3 ) {
      printf("ERROR in TPythia::GetMDCY(kc,key,value):\n");
      printf("      key=%i is out of range [1..3]\n",key);
      return;
   }

   LUDAT3.mdcy[key-1][kc-1]=value;

}

//______________________________________________________________________________
Int_t TPythia::GetMDME(Int_t idc,Int_t key) const
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::GetMDME(idc,key):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return 0;
   }

   if ( key<1 || key>2 ) {
      printf("ERROR in TPythia::GetMDME(idc,key):\n");
      printf("      key=%i is neither 1 nor 2 !\n",key);
      return 0;
   }

   return LUDAT3.mdme[key-1][idc-1];

}


//______________________________________________________________________________
void TPythia::SetMDME(Int_t idc,Int_t key,Int_t value)
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::GetMDME(idc,key,value):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return;
   }

   if ( key<1 || key>2 ) {
      printf("ERROR in TPythia::GetMDME(idc,key,value):\n");
      printf("      key=%i is neither 1 nor 2 !\n",key);
      return;
   }

   LUDAT3.mdme[key-1][idc-1]=value;

}


//______________________________________________________________________________
Float_t TPythia::GetBRAT(Int_t idc) const
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::GetBRAT(idc):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return 0.0;
   }

   return LUDAT3.brat[idc-1];

}


//______________________________________________________________________________
void TPythia::SetBRAT(Int_t idc,Float_t value)
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::SetBRAT(idc,value):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return;
   }

   LUDAT3.brat[idc-1] = value;

}


//______________________________________________________________________________
Int_t TPythia::GetKFDP(Int_t idc,Int_t j) const
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::GetKFDP(idc,j):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return 0;
   }

   if ( j<1 || j>5 ) {
      printf("ERROR in TPythia::GetKFDP(idc,j):\n");
      printf("      j=%i is out of range [1..5] !\n",j);
      return 0;
   }

   return LUDAT3.kfdp[j-1][idc-1];

}


//______________________________________________________________________________
void TPythia::SetKFDP(Int_t idc,Int_t j,Int_t value)
{
   if ( idc<1 || idc>2000 ) {
      printf("ERROR in TPythia::SetKFDP(idc,j,value):\n");
      printf("      idc=%i is out of range [1..2000]\n",idc);
      return;
   }

   if ( j<1 || j>5 ) {
      printf("ERROR in TPythia::SetKFDP(idc,j,value):\n");
      printf("      j=%i is out of range [1..5] !\n",j);
      return;
   }
   LUDAT3.kfdp[j-1][idc-1]=value;

}

//====================== access to common LUDAT4 ===============================

//______________________________________________________________________________
char *TPythia::GetCHAF(Int_t kc) const
{
   static char buf[9]="";

   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::GetCHAF(kc):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return 0;
   }

   strncpy(buf,LUDAT4.chaf[kc-1],8);
   buf[8]=0;

   return buf;

}


//______________________________________________________________________________
void TPythia::SetCHAF(Int_t kc,char *name)
{

   if ( kc<1 || kc>500 ) {
      printf("ERROR in TPythia::SetCHAF(kc,name):\n");
      printf("      kc=%i is out of range [1..500]\n",kc);
      return;
   }

   strncpy(LUDAT4.chaf[kc-1],name,8);


}



//====================== access to Pythia subroutines ===+======================


//______________________________________________________________________________
void TPythia::Initialize(const char *frame, const char *beam, const char *target, float win)
{
// Calls Pyinit with the same parameters after performing some checking,
// sets correct title. This method should preferably be called instead of Pyinit.
// PURPOSE: to initialize the generation procedure.
// ARGUMENTS: See documentation for details.
//    frame:  - specifies the frame of the experiment:
//                "CMS","FIXT","USER","FOUR","FIVE","NONE"
//    beam,
//    target: - beam and target particles (with additionaly cahrges, tildes or "bar":
//              e,nu_e,mu,nu_mu,tau,nu_tau,gamma,pi,n,p,Lambda,Sigma,Xi,Omega,
//              pomeron,reggeon
//    win:    - related to energy system:
//              for frame=="CMS" - total energy of system
//              for frame=="FIXT" - momentum of beam particle
//              for frame=="USER" - dummy - see documentation.
////////////////////////////////////////////////////////////////////////////////////

   char  cframe[4];
   strncpy(cframe,frame,4);
   char  cbeam[8];
   strncpy(cbeam,beam,8);
   char  ctarget[8];
   strncpy(ctarget,target,8);

   if ( (!strncmp(frame, "CMS"  ,3)) &&
        (!strncmp(frame, "FIXT" ,4)) &&
        (!strncmp(frame, "USER" ,4)) &&
        (!strncmp(frame, "FOUR" ,4)) &&
        (!strncmp(frame, "FIVE" ,4)) &&
        (!strncmp(frame, "NONE" ,4)) ) {
      printf("WARNING! In TPythia:Initialize():\n");
      printf(" specified frame=%s is neither of CMS,FIXT,USER,FOUR,FIVE,NONE\n",frame);
      printf(" resetting to \"CMS\" .");
      sprintf(cframe,"CMS");
   }

   if ( (!strncmp(beam, "e"       ,1)) &&
        (!strncmp(beam, "nu_e"    ,4)) &&
        (!strncmp(beam, "mu"      ,2)) &&
        (!strncmp(beam, "nu_mu"   ,5)) &&
        (!strncmp(beam, "tau"     ,3)) &&
        (!strncmp(beam, "nu_tau"  ,6)) &&
        (!strncmp(beam, "gamma"   ,5)) &&
        (!strncmp(beam, "pi"      ,2)) &&
        (!strncmp(beam, "n"       ,1)) &&
        (!strncmp(beam, "p"       ,1)) &&
        (!strncmp(beam, "Lambda"  ,6)) &&
        (!strncmp(beam, "Sigma"   ,5)) &&
        (!strncmp(beam, "Xi"      ,2)) &&
        (!strncmp(beam, "Omega"   ,5)) &&
        (!strncmp(beam, "pomeron" ,7)) &&
        (!strncmp(beam, "reggeon" ,7)) ) {
      printf("WARNING! In TPythia:Initialize():\n");
      printf(" specified beam=%s is unrecognized .\n",beam);
      printf(" resetting to \"p+\" .");
      sprintf(cbeam,"p+");
   }

   if ( (!strncmp(target, "e"       ,1)) &&
        (!strncmp(target, "nu_e"    ,4)) &&
        (!strncmp(target, "mu"      ,2)) &&
        (!strncmp(target, "nu_mu"   ,5)) &&
        (!strncmp(target, "tau"     ,3)) &&
        (!strncmp(target, "nu_tau"  ,6)) &&
        (!strncmp(target, "gamma"   ,5)) &&
        (!strncmp(target, "pi"      ,2)) &&
        (!strncmp(target, "n"       ,1)) &&
        (!strncmp(target, "p"       ,1)) &&
        (!strncmp(target, "Lambda"  ,6)) &&
        (!strncmp(target, "Sigma"   ,5)) &&
        (!strncmp(target, "Xi"      ,2)) &&
        (!strncmp(target, "Omega"   ,5)) &&
        (!strncmp(target, "pomeron" ,7)) &&
        (!strncmp(target, "reggeon" ,7)) ){
      printf("WARNING! In TPythia:Initialize():\n");
      printf(" specified target=%s is unrecognized.\n",target);
      printf(" resetting to \"p+\" .");
      sprintf(ctarget,"p+");
   }



   Pyinit(cframe, cbeam ,ctarget, win);

   char atitle[32];
   sprintf(atitle," %s-%s at %g GeV",cbeam,ctarget,win);
   SetTitle(atitle);

}


//______________________________________________________________________________
void TPythia::GenerateEvent()
{
// Generates one event nd automatically fills the fParticles list with new particles.
// This function should rathe be used instead of pyevnt();

   pyevnt();
   ImportParticles();
}

//______________________________________________________________________________
void TPythia::Pyinit(char *frame, char *beam, char *target, float win)
{
// Calls Pythia's PYINIT subroutine passing these parameters in a way accepted
// by FORTRAN routines. Yo should rather use Initialize() method instead of this
// one.

   Float_t lwin = win;
   Long_t  s1   = strlen(frame);
   Long_t  s2   = strlen(beam);
   Long_t  s3   = strlen(target);

#ifndef WIN32
   pyinit(frame, beam ,target, lwin, s1, s2, s3);
#else
   pyinit(frame, s1, beam , s2, target, s3, lwin);
#endif

}



//______________________________________________________________________________
void TPythia::Pyevnt()
{
// Calls Pythia's PYEVNT. You'd better use GenerateEvent() method instead.

   pyevnt();
}

//______________________________________________________________________________
void TPythia::Pystat(Int_t mstat)
{
// Calls Pythia's PYSTAT: prints out some statistics depending on value of key:
// see documentation for details...

   Long_t lkey = mstat;
   pystat(lkey);
}

//______________________________________________________________________________
void TPythia::Pytest(Int_t key)
{
// Calls Pythia's PYTEST routine - runs a set of tests to detect possible errors.

   Long_t lkey = key;
   pytest(lkey);
}

//______________________________________________________________________________
void TPythia::Lulist(Int_t mlist)
{
// Calls JetSet's LULUST routine - lists an event.

   Long_t lkey = mlist;
   lulist(lkey);
}

//______________________________________________________________________________
void TPythia::Luexec()
{
// Calls JetSet's LuExec routine - administrates the fragmentation and decay chain.

   luexec();
}

//______________________________________________________________________________
Int_t TPythia::Lucomp(Int_t kf)
{
   Long_t lkey = kf;
   return lucomp(lkey);
}




//______________________________________________________________________________
void TPythia::SetupTest()
{
// Exemplary setup of Pythia parameters:
// Switches on processes 102,123,124 (Higgs generation) and switches off
// interactions, fragmentation, ISR, FSR...

   SetMSEL(0);            // full user controll;

   SetMSUB(102,1);        // g + g -> H0
   SetMSUB(123,1);        // f + f' -> f + f' + H0
   SetMSUB(124,1);        // f + f' -> f" + f"' + H0


   SetPMAS(6,1,175.0);   // mass of TOP
   SetPMAS(25,1,300);    // mass of Higgs


   SetCKIN(1,290.0);     // range of allowed mass
   SetCKIN(2,310.0);

   SetMSTP(61,  0);      // switch off ISR
   SetMSTP(71,  0);      // switch off FSR
   SetMSTP(81,  0);      // switch off multiple interactions
   SetMSTP(111, 0);      // switch off fragmentation/decay


}
