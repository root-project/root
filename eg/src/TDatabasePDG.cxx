// @(#)root/eg:$Name:  $:$Id: TDatabasePDG.cxx,v 1.8 2001/03/06 10:06:37 brun Exp $
// Author: Pasha Murat   12/02/99

#include "TROOT.h"
#include "TEnv.h"
#include "TSystem.h"
#include "TDatabasePDG.h"
#include "TDecayChannel.h"
#include "TParticlePDG.h"

#ifdef WIN32
#include <strstrea.h>
#else
#include <strstream.h>
#endif

////////////////////////////////////////////////////////////////////////
//
//  Particle database manager class
//
//  This manager creates a list of particles which by default is 
//  initialised from with the constants used by PYTHIA6 (plus some 
//  other particles added). See definition and the format of the default 
//  particle list in $ROOTSYS/eg/src/pdg_table.txt
//
//  there are 2 ways of redefining the name of the file containing the 
//  particle properties
//
//  1. one can define the name in .rootrc file:
//
//  Root.DatabasePDG: $(HOME)/my_pdg_table.txt
//
//  2. one can use TDatabasePDG::ReadPDGTable method explicitly:
//
//     - TDatabasePDG *pdg = new TDatabasePDG();
//     - pdg->ReadPDGtable(filename)
//
//  See TParticlePDG for the description of a static particle properties.
//  See TParticle    for the description of a dynamic particle particle.
//
////////////////////////////////////////////////////////////////////////

ClassImp(TDatabasePDG)

TDatabasePDG*  TDatabasePDG::fgInstance = 0;
//______________________________________________________________________________
TDatabasePDG::TDatabasePDG(): TNamed("PDGDB","The PDG particle data base")
{
  // Create PDG database. Initialization of the DB has to be done via explicit
  // call to ReadDataBasePDG (also done by GetParticle methods)

   fParticleList  = 0;
   fListOfClasses = 0;
   if (fgInstance) {
     Warning("TDatabasePDG", "object already instantiated");
   }
   else {
     fgInstance = this;
     gROOT->GetListOfSpecials()->Add(this);
   }
}

//______________________________________________________________________________
TDatabasePDG::~TDatabasePDG()
{
   // Cleanup the PDG database.

   if (fParticleList) {
      fParticleList->Delete();
      delete fParticleList;
   }
				// classes do not own particles...
   if (fListOfClasses) delete fListOfClasses;
   gROOT->GetListOfSpecials()->Remove(this);
   fgInstance = 0;
}


//______________________________________________________________________________
TParticlePDG* TDatabasePDG::AddParticle(const char *name, const char *title,
					Double_t mass, Bool_t stable,
					Double_t width, Double_t charge,
					const char* ParticleClass, 
					Int_t PDGcode, 
					Int_t Anti,
					Int_t TrackingCode)
{
  //
  //  Particle definition normal constructor. If the particle is set to be
  //  stable, the decay width parameter does have no meaning and can be set to
  //  any value. The parameters granularity, LowerCutOff and HighCutOff are
  //  used for the construction of the mean free path look up tables. The
  //  granularity will be the number of logwise energy points for which the
  //  mean free path will be calculated.
  //

  TParticlePDG* old = GetParticle(PDGcode);

  if (old) {
    printf(" *** TDatabasePDG::AddParticle: particle with PDGcode=%d already defined\n",PDGcode);
    return 0;
  }

  TParticlePDG* p = new TParticlePDG(name, title, mass, stable, width,
				     charge, ParticleClass, PDGcode, Anti, 
				     TrackingCode);
  fParticleList->Add(p);

  TParticleClassPDG* pclass = GetParticleClass(ParticleClass);

  if (!pclass) {
    pclass = new TParticleClassPDG(ParticleClass);
    fListOfClasses->Add(pclass);
  }

  pclass->AddParticle(p);

  return p;
}

//______________________________________________________________________________
TParticlePDG* TDatabasePDG::AddAntiParticle(const char* Name, Int_t PdgCode) 
{
  // assuming particle has already been defined

  TParticlePDG* old = GetParticle(PdgCode);

  if (old) {
    printf(" *** TDatabasePDG::AddAntiParticle: can't redefine parameters\n");
    return NULL;
  }

  Int_t pdg_code  = abs(PdgCode);
  TParticlePDG* p = GetParticle(pdg_code);

  TParticlePDG* ap = AddParticle(Name,
				 Name, 
				 p->Mass(), 
				 1,
				 p->Width(),
				 -p->Charge(),
				 p->ParticleClass(),
				 PdgCode,
				 1,
				 p->TrackingCode());
  return ap;
}


//______________________________________________________________________________
TParticlePDG *TDatabasePDG::GetParticle(const char *name) const
{
   //
   //  Get a pointer to the particle object according to the name given
   //

  if (fParticleList == 0)  ((TDatabasePDG*)this)->ReadPDGTable();

  TParticlePDG *def = (TParticlePDG *)fParticleList->FindObject(name);
//     if (!def) {
//        Error("GetParticle","No match for %s exists!",name);
//     }
   return def;
}

//______________________________________________________________________________
TParticlePDG *TDatabasePDG::GetParticle(Int_t PDGcode) const
{
   //
   //  Get a pointer to the particle object according to the MC code number
   //

  if (fParticleList == 0)  ((TDatabasePDG*)this)->ReadPDGTable();

   TIter next(fParticleList);
   TParticlePDG *p;
   while ((p = (TParticlePDG *)next())) {
      if (p->PdgCode() == PDGcode) return p;
   }
   //   Error("GetParticle","No match for PDG code %d exists!",PDGcode);
   return 0;
}

//______________________________________________________________________________
void TDatabasePDG::Print(Option_t *option) const
{
   // Print contents of PDG database.

  if (fParticleList == 0)  ((TDatabasePDG*)this)->ReadPDGTable();

   TIter next(fParticleList);
   TParticlePDG *p;
   while ((p = (TParticlePDG *)next())) {
      p->Print(option);
   }
}

//______________________________________________________________________________
Int_t TDatabasePDG::ConvertIsajetToPdg(Int_t isaNumber)
{
//
//  Converts the ISAJET Particle number into the PDG MC number
//
  switch (isaNumber) {
    case     1 : return     2; //     UP        .30000E+00       .67
    case    -1 : return    -2; //     UB        .30000E+00      -.67
    case     2 : return     1; //     DN        .30000E+00      -.33
    case    -2 : return    -1; //     DB        .30000E+00       .33
    case     3 : return     3; //     ST        .50000E+00      -.33
    case    -3 : return    -3; //     SB        .50000E+00       .33
    case     4 : return     4; //     CH        .16000E+01       .67
    case    -4 : return    -4; //     CB        .16000E+01      -.67
    case     5 : return     5; //     BT        .49000E+01      -.33
    case    -5 : return    -5; //     BB        .49000E+01       .33
    case     6 : return     6; //     TP        .17500E+03       .67
    case    -6 : return    -6; //     TB        .17500E+03      -.67
    case     9 : return    21; //     GL       0.               0.00
    case    80 : return    24; //     W+        SIN2W=.23       1.00
    case   -80 : return   -24; //     W-        SIN2W=.23      -1.00
    case    90 : return    23; //     Z0        SIN2W=.23       0.00
    case   230 : return   311; //     K0        .49767E+00      0.00
    case  -230 : return  -311; //     AK0       .49767E+00      0.00
    case   330 : return   331; //     ETAP      .95760E+00      0.00
    case   340 : return     0; //     F-        .20300E+01     -1.00
    case  -340 : return     0; //     F+        .20300E+01      1.00
    case   440 : return   441; //     ETAC      .29760E+01      0.00
    case   111 : return   113; //     RHO0      .77000E+00      0.00
    case   121 : return   213; //     RHO+      .77000E+00      1.00
    case  -121 : return  -213; //     RHO-      .77000E+00     -1.00
    case   221 : return   223; //     OMEG      .78260E+00      0.00
    case   131 : return   323; //     K*+       .88810E+00      1.00
    case  -131 : return  -323; //     K*-       .88810E+00     -1.00
    case   231 : return   313; //     K*0       .89220E+00      0.00
    case  -231 : return  -313; //     AK*0      .89220E+00      0.00
    case   331 : return   333; //     PHI       .10196E+01      0.00
    case  -140 : return   421; //     D0
    case   140 : return  -421; //     D0 bar
    case   141 : return  -423; //     AD*0      .20060E+01      0.00
    case  -141 : return   423; //     D*0       .20060E+01      0.00
    case  -240 : return  -411; //     D+
    case   240 : return   411; //     D-
    case   241 : return  -413; //     D*-       .20086E+01     -1.00
    case  -241 : return   413; //     D*+       .20086E+01      1.00
    case   341 : return     0; //     F*-       .21400E+01     -1.00
    case  -341 : return     0; //     F*+       .21400E+01      1.00
    case   441 : return   443; //     JPSI      .30970E+01      0.00

                                        // B-mesons, Bc still missing
    case   250 : return   511; // B0
    case  -250 : return  -511; // B0 bar
    case   150 : return   521; // B+
    case  -150 : return  -521; // B-
    case   350 : return   531; // Bs  0
    case  -350 : return  -531; // Bs  bar
    case   351 : return   533; // Bs* 0
    case  -351 : return  -533; // Bs* bar
    case   450 : return   541; // Bc  +
    case  -450 : return  -541; // Bc  bar

    case  1140 : return  4222; //     SC++      .24300E+01      2.00
    case -1140 : return -4222; //     ASC--     .24300E+01     -2.00
    case  1240 : return  4212; //     SC+       .24300E+01      1.00
    case -1240 : return -4212; //     ASC-      .24300E+01     -1.00
    case  2140 : return  4122; //     LC+       .22600E+01      1.00
    case -2140 : return -4122; //     ALC-      .22600E+01     -1.00
    case  2240 : return  4112; //     SC0       .24300E+01      0.00
    case -2240 : return -4112; //     ASC0      .24300E+01      0.00
    case  1340 : return     0; //     USC.      .25000E+01      1.00
    case -1340 : return     0; //     AUSC.     .25000E+01     -1.00
    case  3140 : return     0; //     SUC.      .24000E+01      1.00
    case -3140 : return     0; //     ASUC.     .24000E+01     -1.00
    case  2340 : return     0; //     DSC.      .25000E+01      0.00
    case -2340 : return     0; //     ADSC.     .25000E+01      0.00
    case  3240 : return     0; //     SDC.      .24000E+01      0.00
    case -3240 : return     0; //     ASDC.     .24000E+01      0.00
    case  3340 : return     0; //     SSC.      .26000E+01      0.00
    case -3340 : return     0; //     ASSC.     .26000E+01      0.00
    case  1440 : return     0; //     UCC.      .35500E+01      2.00
    case -1440 : return     0; //     AUCC.     .35500E+01     -2.00
    case  2440 : return     0; //     DCC.      .35500E+01      1.00
    case -2440 : return     0; //     ADCC.     .35500E+01     -1.00
    case  3440 : return     0; //     SCC.      .37000E+01      1.00
    case -3440 : return     0; //     ASCC.     .37000E+01     -1.00
    case  1111 : return  2224; //     DL++      .12320E+01      2.00
    case -1111 : return -2224; //     ADL--     .12320E+01     -2.00
    case  1121 : return  2214; //     DL+       .12320E+01      1.00
    case -1121 : return -2214; //     ADL-      .12320E+01     -1.00
    case  1221 : return  2114; //     DL0       .12320E+01      0.00
    case -1221 : return -2114; //     ADL0      .12320E+01      0.00
    case  2221 : return   1114; //     DL-       .12320E+01     -1.00
    case -2221 : return -1114; //     ADL+      .12320E+01      1.00
    case  1131 : return  3224; //     S*+       .13823E+01      1.00
    case -1131 : return -3224; //     AS*-      .13823E+01     -1.00
    case  1231 : return  3214; //     S*0       .13820E+01      0.00
    case -1231 : return -3214; //     AS*0      .13820E+01      0.00
    case  2231 : return  3114; //     S*-       .13875E+01     -1.00
    case -2231 : return -3114; //     AS*+      .13875E+01      1.00
    case  1331 : return  3324; //     XI*0      .15318E+01      0.00
    case -1331 : return -3324; //     AXI*0     .15318E+01      0.00
    case  2331 : return  3314; //     XI*-      .15350E+01     -1.00
    case -2331 : return -3314; //     AXI*+     .15350E+01      1.00
    case  3331 : return  3334; //     OM-       .16722E+01     -1.00
    case -3331 : return -3334; //     AOM+      .16722E+01      1.00
    case  1141 : return     0; //     UUC*      .26300E+01      2.00
    case -1141 : return     0; //     AUUC*     .26300E+01     -2.00
    case  1241 : return     0; //     UDC*      .26300E+01      1.00
    case -1241 : return     0; //     AUDC*     .26300E+01     -1.00
    case  2241 : return     0; //     DDC*      .26300E+01      0.00
    case -2241 : return     0; //     ADDC*     .26300E+01      0.00
    case  1341 : return     0; //     USC*      .27000E+01      1.00
    case -1341 : return     0; //     AUSC*     .27000E+01     -1.00
    case  2341 : return     0; //     DSC*      .27000E+01      0.00
    case -2341 : return     0; //     ADSC*     .27000E+01      0.00
    case  3341 : return     0; //     SSC*      .28000E+01      0.00
    case -3341 : return     0; //     ASSC*     .28000E+01      0.00
    case  1441 : return     0; //     UCC*      .37500E+01      2.00
    case -1441 : return     0; //     AUCC*     .37500E+01     -2.00
    case  2441 : return     0; //     DCC*      .37500E+01      1.00
    case -2441 : return     0; //     ADCC*     .37500E+01     -1.00
    case  3441 : return     0; //     SCC*      .39000E+01      1.00
    case -3441 : return     0; //     ASCC*     .39000E+01     -1.00
    case  4441 : return     0; //     CCC*      .48000E+01      2.00
    case -4441 : return     0; //     ACCC*     .48000E+01     -2.00
    case    10 : return    22; // Photon
    case    12 : return    11; // Electron
    case   -12 : return   -11; // Positron
    case    14 : return    13; // Muon-
    case   -14 : return   -13; // Muon+
    case    16 : return    15; // Tau-
    case   -16 : return   -15; // Tau+
    case    11 : return    12; // Neutrino e
    case   -11 : return   -12; // Anti Neutrino e
    case    13 : return    14; // Neutrino Muon
    case   -13 : return   -14; // Anti Neutrino Muon
    case    15 : return    16; // Neutrino Tau
    case   -15 : return   -16; // Anti Neutrino Tau
    case   110 : return   111; // Pion0
    case   120 : return   211; // Pion+
    case  -120 : return  -211; // Pion-
    case   220 : return   221; // Eta
    case   130 : return   321; // Kaon+
    case  -130 : return  -321; // Kaon-
    case   -20 : return   130; // Kaon Long
    case    20 : return   310; // Kaon Short

                                        // baryons
    case  1120 : return  2212; // Proton
    case -1120 : return -2212; // Anti Proton
    case  1220 : return  2112; // Neutron
    case -1220 : return -2112; // Anti Neutron
    case  2130 : return  3122; // Lambda
    case -2130 : return -3122; // Lambda bar
    case  1130 : return  3222; // Sigma+
    case -1130 : return -3222; // Sigma bar -
    case  1230 : return  3212; // Sigma0
    case -1230 : return -3212; // Sigma bar 0
    case  2230 : return  3112; // Sigma-
    case -2230 : return -3112; // Sigma bar +
    case  1330 : return  3322; // Xi0
    case -1330 : return -3322; // Xi bar 0
    case  2330 : return  3312; // Xi-
    case -2330 : return -3312; // Xi bar +
    default :    return 0;      // isajet or pdg number does not exist
  }
}

//______________________________________________________________________________
void TDatabasePDG::ReadPDGTable(const char *FileName)
{
   // read list of particles from a file
   // if the particle list does not exist, it is created, otherwise
   // particles are added to the existing list
   // See $ROOTSYS/tutorials/pdg.dat to see the file format

  if (fParticleList == 0) {
    fParticleList  = new THashList;
    fListOfClasses = new TObjArray;
  }

  char         default_name[200];
  const char*  fn;

  if (FileName == "") {
    sprintf(default_name,"%s/eg/src/pdg_table.txt",gSystem->Getenv("ROOTSYS"));
    fn = gEnv->GetValue("Root.DatabasePDG",default_name);
  }
  else {
    fn = FileName;
  }

  FILE* file = fopen(fn,"r");
  if (file == 0) {
    Error("ReadPDGTable","Could not open PDG particle file %s",fn);
    return;
  }

  char      c[512];
  Int_t     class_number, anti, isospin, i3, spin, tracking_code;
  Int_t     ich, kf, nch, charge;
  char      name[30], class_name[30];
  Double_t  mass, width, branching_ratio;
  Int_t     dau[20];

  Int_t     idecay, decay_type, flavor, ndau;

  while ( (c[0]=getc(file)) != EOF) {

    if (c[0] != '#') {
      ungetc(c[0],file);
				// read channel number
      fscanf(file,"%i",&ich);
      fscanf(file,"%s",name  );
      fscanf(file,"%i",&kf   );
      fscanf(file,"%i",&anti );

      if (kf < 0) {
	AddAntiParticle(name,kf);
				// nothing more on this line
	fgets(c,200,file);
      }
      else {
	fscanf(file,"%i",&class_number);
	fscanf(file,"%s",class_name);
	fscanf(file,"%i",&charge);
	fscanf(file,"%le",&mass);
	fscanf(file,"%le",&width);
	fscanf(file,"%i",&isospin);
	fscanf(file,"%i",&i3);
	fscanf(file,"%i",&spin);
	fscanf(file,"%i",&flavor);
	fscanf(file,"%i",&tracking_code);
	fscanf(file,"%i",&nch);
				// nothing more on this line
	fgets(c,200,file);
	

				// create particle

	TParticlePDG* part = AddParticle(name,
					   name,
					   mass,
					   1,
					   width,
					   charge,
					   class_name,
					   kf,
					   anti,
					   tracking_code);

	if (nch) {
				// read in decay channels
	  int ich = 0;
	  while ( ((c[0]=getc(file)) != EOF) && (ich <nch)) {
	    if (c[0] != '#') {
	      ungetc(c[0],file);
	      
	      fscanf(file,"%i",&idecay);
	      fscanf(file,"%i",&decay_type);
	      fscanf(file,"%le",&branching_ratio);
	      fscanf(file,"%i",&ndau);
	      for (int idau=0; idau<ndau; idau++) {
		fscanf(file,"%i",&dau[idau]);
	      }
				// add decay channel

	      part->AddDecayChannel(decay_type,branching_ratio,ndau,dau);
	      ich++;
	    }
				// skip end of line
	    fgets(c,200,file);
	  }
	}
      }
    }
    else {
				// skip end of line
      fgets(c,200,file);
    }
  }
				// in the end loop over the antiparticles and
				// define their decay lists
  TIter it(fParticleList);

  Int_t code[20];
  TParticlePDG  *ap, *p, *daughter;
  TDecayChannel dc;

  while ((p = (TParticlePDG*) it.Next())) {

  				// define decay channels for antiparticles
    if (p->PdgCode() < 0) {
      ap = GetParticle(-p->PdgCode());
      nch = ap->NDecayChannels();
      for (int ich=0; ich<nch; ich++) {
	TDecayChannel* dc = ap->DecayChannel(ich);
	ndau = dc->NDaughters(); 
	for (int i=0; i<ndau; i++) {
					// conserve CPT

	  code[i] = dc->DaughterPdgCode(i);
	  daughter = GetParticle(code[i]);
	  if (daughter->AntiParticle()) {
					// this particle does have an 
					// antiparticle
	    code[i] = -code[i];
	  }
	}
	p->AddDecayChannel(dc->MatrixElementCode(),
			   dc->BranchingRatio(),
			   dc->NDaughters(),
			   code);
      }
      p->SetAntiParticle(ap);
      ap->SetAntiParticle(p);
    }
  }

  fclose(file);
  return;
}


//______________________________________________________________________________
void TDatabasePDG::Browse(TBrowser* b)
{
  if (fListOfClasses ) fListOfClasses->Browse(b);
}


//______________________________________________________________________________
Int_t TDatabasePDG::WritePDGTable(const char *filename)
{
   // write contents of the particle DB into a file

  Error("WritePDGTable"," not implemented yet");
  return 0;
/*
  if (1) return 0;

  FILE *file = fopen(filename,"w");
  if (file == 0) {
    Error("WritePDGTable","Could not open PDG particle file %s",filename);
    return -1;
  }

  fclose(file);
*/
}


