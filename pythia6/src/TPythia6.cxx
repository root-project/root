// @(#)root/pythia6:$Name:  $:$Id: TPythia6.cxx,v 1.1.1.1 2000/05/16 17:00:48 rdm Exp $
// Author: Rene Brun   19/10/99
//
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TPythia6                                                                   //
//                                                                            //
// TPythia is an interface class to F77 version of Pythia 6.1                 //
// CERNLIB event generators, written by T.Sjostrand.                          //
// For the details about these generators look at Pythia/Jetset manual:       //
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
//**      !!                       !!         Copyright Torbjorn Sjostrand    **
//**      !!                                     and CERN, Geneva 1993        **
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

#include "TPythia6.h"

#include "TClonesArray.h"
#include "TMCParticle.h"
#include "TParticle.h"

TPythia6*  TPythia6::fgInstance = 0;

ClassImp(TPythia6)

extern "C" {
  void*  pythia6_common_block_address_(char*, int len);
  void   tpythia6_open_fortran_file_ (int* lun, char* name, int);
  void   tpythia6_close_fortran_file_(int* lun);
}

//------------------------------------------------------------------------------
TPythia6::Cleaner::Cleaner() {
}

//------------------------------------------------------------------------------
TPythia6::Cleaner::~Cleaner() {

  if (TPythia6::fgInstance) {
    delete TPythia6::fgInstance;
    TPythia6::fgInstance = 0;
  }
}

//------------------------------------------------------------------------------
//  constructor is not supposed to be called from the outside - only
//  Initialize() method
//------------------------------------------------------------------------------
TPythia6::TPythia6() : TGenerator("TPythia6","TPythia6") {
// TPythia6 constructor: creates a TClonesArray in which it will store all
// particles. Note that there may be only one functional TPythia6 object
// at a time, so it's not use to create more than one instance of it.

  delete fParticles; // was allocated as TObjArray in TGenerator

  fParticles = new TClonesArray("TMCParticle",50);

  // initialize common-blocks

  fPyjets = (Pyjets_t*) pythia6_common_block_address_((char*)"PYJETS",6);
  fPydat1 = (Pydat1_t*) pythia6_common_block_address_((char*)"PYDAT1",6);
  fPydat2 = (Pydat2_t*) pythia6_common_block_address_((char*)"PYDAT2",6);
  fPydat3 = (Pydat3_t*) pythia6_common_block_address_((char*)"PYDAT3",6);
  fPydat4 = (Pydat4_t*) pythia6_common_block_address_((char*)"PYDAT4",6);
  fPydatr = (Pydatr_t*) pythia6_common_block_address_((char*)"PYDATR",6);
  fPysubs = (Pysubs_t*) pythia6_common_block_address_((char*)"PYSUBS",6);
  fPypars = (Pypars_t*) pythia6_common_block_address_((char*)"PYPARS",6);
  fPyint1 = (Pyint1_t*) pythia6_common_block_address_((char*)"PYINT1",6);
  fPyint2 = (Pyint2_t*) pythia6_common_block_address_((char*)"PYINT2",6);
  fPyint3 = (Pyint3_t*) pythia6_common_block_address_((char*)"PYINT3",6);
  fPyint4 = (Pyint4_t*) pythia6_common_block_address_((char*)"PYINT4",6);
  fPyint5 = (Pyint5_t*) pythia6_common_block_address_((char*)"PYINT5",6);
  fPyint6 = (Pyint6_t*) pythia6_common_block_address_((char*)"PYINT6",6);
  fPyint7 = (Pyint7_t*) pythia6_common_block_address_((char*)"PYINT7",6);
  fPyint8 = (Pyint8_t*) pythia6_common_block_address_((char*)"PYINT8",6);
  fPyint9 = (Pyint9_t*) pythia6_common_block_address_((char*)"PYINT9",6);
  fPyuppr = (Pyuppr_t*) pythia6_common_block_address_((char*)"PYUPPR",6);
  fPymssm = (Pymssm_t*) pythia6_common_block_address_((char*)"PYMSSM",6);
  fPyssmt = (Pyssmt_t*) pythia6_common_block_address_((char*)"PYSSMT",6);
  fPyints = (Pyints_t*) pythia6_common_block_address_((char*)"PYINTS",6);
  fPybins = (Pybins_t*) pythia6_common_block_address_((char*)"PYBINS",6);
}

//------------------------------------------------------------------------------
 TPythia6::~TPythia6()
 {
// Destroys the object, deletes and disposes all TMCParticles currently on list.

    if (fParticles) {
      fParticles->Delete();
      delete fParticles;
      fParticles = 0;
   }
 }

//------------------------------------------------------------------------------
TPythia6* TPythia6::Instance() {
  // model of automatic memory cleanup suggested by Jim Kowalkovski:
  // destructor for local static variable `cleaner' is  always called in the end
  // of the job thus deleting the only TPythia6 instance

  static TPythia6::Cleaner cleaner;
  return fgInstance ? fgInstance : (fgInstance=new TPythia6()) ;
}





//______________________________________________________________________________
void TPythia6::GenerateEvent() {

  //  generate event and copy the information from /HEPEVT/ to fPrimaries

  pyevnt_();
  ImportParticles();
}

//______________________________________________________________________________
void TPythia6::OpenFortranFile(int lun, char* name) {
  tpythia6_open_fortran_file_(&lun, name, strlen(name));
}

//______________________________________________________________________________
void TPythia6::CloseFortranFile(int lun) {
  tpythia6_close_fortran_file_(&lun);
}


//______________________________________________________________________________
TObjArray *TPythia6::ImportParticles(Option_t *)
{
// Fills TObjArray fParticles list with particles from common LUJETS
// Old contents of a list are cleared. This function should be called after
// any change in common LUJETS, however GetParticles() method  calls it
// automatically - user don't need to care about it. In case you make a call
// to LuExec() you must call this method yourself to transfer new data from
// common LUJETS to the fParticles list.

   fParticles->Clear();
   Int_t numpart   = fPyjets->N;
   TClonesArray &a = *((TClonesArray*)fParticles);
   for (Int_t i = 0; i<numpart; i++) {
      new(a[i]) TMCParticle(fPyjets->K[0][i] ,
                            fPyjets->K[1][i] ,
                            fPyjets->K[2][i] ,
                            fPyjets->K[3][i] ,
                            fPyjets->K[4][i] ,

                            fPyjets->P[0][i] ,
                            fPyjets->P[1][i] ,
                            fPyjets->P[2][i] ,
                            fPyjets->P[3][i] ,
                            fPyjets->P[4][i] ,

                            fPyjets->V[0][i] ,
                            fPyjets->V[1][i] ,
                            fPyjets->V[2][i] ,
                            fPyjets->V[3][i] ,
                            fPyjets->V[4][i]);
   }
   return fParticles;
}

//______________________________________________________________________________
Int_t TPythia6::ImportParticles(TClonesArray *particles, Option_t *option)
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
  Int_t numpart = fPyjets->N;
  if (!strcmp(option,"") || !strcmp(option,"Final")) {
    for (Int_t i = 0; i<numpart; i++) {
      if (fPyjets->K[1][i] == 1) {
//
//  Use the common block values for the TParticle constructor
//
        new(Particles[i]) TParticle(
                            fPyjets->K[1][i] ,
                            fPyjets->K[0][i] ,
                            fPyjets->K[2][i] ,
                            -1,
                            fPyjets->K[3][i] ,
                            fPyjets->K[4][i] ,

                            fPyjets->P[0][i] ,
                            fPyjets->P[1][i] ,
                            fPyjets->P[2][i] ,
                            fPyjets->P[3][i] ,

                            fPyjets->V[0][i] ,
                            fPyjets->V[1][i] ,
                            fPyjets->V[2][i] ,
                            fPyjets->V[3][i]);
      }
    }
  }
  else if (!strcmp(option,"All")) {
    for (Int_t i = 0; i<=numpart; i++) {
        new(Particles[i]) TParticle(
                            fPyjets->K[1][i] ,
                            fPyjets->K[0][i] ,
                            fPyjets->K[2][i] ,
                            -1,
                            fPyjets->K[3][i] ,
                            fPyjets->K[4][i] ,

                            fPyjets->P[0][i] ,
                            fPyjets->P[1][i] ,
                            fPyjets->P[2][i] ,
                            fPyjets->P[3][i] ,

                            fPyjets->V[0][i] ,
                            fPyjets->V[1][i] ,
                            fPyjets->V[2][i] ,
                            fPyjets->V[3][i]);
    }
  }
  return numpart;
}

//______________________________________________________________________________
void TPythia6::Initialize(const char *frame, const char *beam, const char *target, float win)
{
// Calls PyInit with the same parameters after performing some checking,
// sets correct title. This method should preferably be called instead of PyInit.
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
      printf("WARNING! In TPythia6:Initialize():\n");
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
      printf("WARNING! In TPythia6:Initialize():\n");
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
      printf("WARNING! In TPythia6:Initialize():\n");
      printf(" specified target=%s is unrecognized.\n",target);
      printf(" resetting to \"p+\" .");
      sprintf(ctarget,"p+");
   }



   Pyinit(cframe, cbeam ,ctarget, win);

   char atitle[32];
   sprintf(atitle," %s-%s at %g GeV",cbeam,ctarget,win);
   SetTitle(atitle);

}


void TPythia6::Pyinit(char* frame, char* beam, char* target, double win) {
//------------------------------------------------------------------------------
// Calls Pyinit with the same parameters after performing some checking,
// sets correct title. This method should preferably be called instead of PyInit.
// PURPOSE: to initialize the generation procedure.
// ARGUMENTS: See documentation for details.
//    frame:  - specifies the frame of the experiment:
//                "CMS","FIXT","USER","FOUR","FIVE","NONE"
//    beam,
//    target: - beam and target particles (with additionaly charges,
//              tildes or "bar":
//              e,nu_e,mu,nu_mu,tau,nu_tau,gamma,pi,n,p,Lambda,Sigma,Xi,Omega,
//              pomeron,reggeon
//    win:    - related to energy system:
//              for frame=="CMS" - total energy of system
//              for frame=="FIXT" - momentum of beam particle
//              for frame=="USER" - dummy - see documentation.
//------------------------------------------------------------------------------

  pyinit_(frame,beam,target,&win,strlen(frame),strlen(beam),strlen(target));
}


int TPythia6::Pycomp(int kf) {
  return pycomp_(&kf);
}

void TPythia6::Pyedit(int medit) {
  pyedit_(&medit);
}

void TPythia6::Pyevnt() {
  pyevnt_();
}

void TPythia6::Pyexec() {
  pyexec_();
}

void TPythia6::Pyhepc(int mconv) {
  pyhepc_(&mconv);
}

void TPythia6::Pylist(int flag) {
  pylist_(&flag);
}

void TPythia6::Pyname(int kf, char* name) {
  pyname_(&kf,name,15);
				// cut trailing blanks to get C string

  for (int i=15; (i>=0) && (name[i] != ' '); i--) {
    name[i] = 0;
  }
}

double TPythia6::Pyr(int idummy) {
  return pyr_(&idummy);
}

void TPythia6::Pyrget(int lun, int move) {
  pyrget_(&lun,&move);
}

void TPythia6::Pyrset(int lun, int move) {
  pyrset_(&lun,&move);
}

void TPythia6::Pystat(int flag) {
  pystat_(&flag);
}

void TPythia6::Pytest(int flag) {
  pytest_(&flag);
}

void TPythia6::Pyupda(int mupda, int lun) {
  pyupda_(&mupda,&lun);
}

//______________________________________________________________________________
void TPythia6::SetupTest()
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
