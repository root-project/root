// @(#)root/pythia8:$Name$:$Id$
// Author: Andreas Morsch   27/10/2007

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TPythia8
    \ingroup pythia8

TPythia8 is an interface class to C++ version of Pythia 8.1           
event generators, written by T.Sjostrand.

The user is assumed to be familiar with the Pythia package.
This class includes only a basic interface to Pythia8. Because Pythia8 is
also written in C++, its functions/classes can be called directly from a 
compiled C++ script.

To call Pythia functions not available in this interface a dictionary must
be generated.

See $ROOTSYS/tutorials/pythia/pythia8.C for an example of use from CINT.

Author: Andreas Morsch   27/10/2007

\verbatim
*------------------------------------------------------------------------------------*
 |                                                                                    |
 |  *------------------------------------------------------------------------------*  |
 |  |                                                                              |  |
 |  |                                                                              |  |
 |  |   PPP   Y   Y  TTTTT  H   H  III    A      Welcome to the Lund Monte Carlo!  |  |
 |  |   P  P   Y Y     T    H   H   I    A A     This is PYTHIA version 8.100      |  |
 |  |   PPP     Y      T    HHHHH   I   AAAAA    Last date of change: 20 Oct 2007  |  |
 |  |   P       Y      T    H   H   I   A   A                                      |  |
 |  |   P       Y      T    H   H  III  A   A    Now is 27 Oct 2007 at 18:26:53    |  |
 |  |                                                                              |  |
 |  |   Main author: Torbjorn Sjostrand; CERN/PH, CH-1211 Geneva, Switzerland,     |  |
 |  |     and Department of Theoretical Physics, Lund University, Lund, Sweden;    |  |
 |  |     phone: + 41 - 22 - 767 82 27; e-mail: torbjorn@thep.lu.se                |  |
 |  |   Author: Stephen Mrenna; Computing Division, Simulations Group,             |  |
 |  |     Fermi National Accelerator Laboratory, MS 234, Batavia, IL 60510, USA;   |  |
 |  |     phone: + 1 - 630 - 840 - 2556; e-mail: mrenna@fnal.gov                   |  |
 |  |   Author: Peter Skands; CERN/PH, CH-1211 Geneva, Switzerland,                |  |
 |  |     and Theoretical Physics Department,                                      |  |
 |  |     Fermi National Accelerator Laboratory, MS 106, Batavia, IL 60510, USA;   |  |
 |  |     phone: + 41 - 22 - 767 24 59; e-mail: skands@fnal.gov                    |  |
 |  |                                                                              |  |
 |  |   The main program reference is the 'Brief Introduction to PYTHIA 8.1',      |  |
 |  |   T. Sjostrand, S. Mrenna and P. Skands, arXiv:0710.3820                     |  |
 |  |                                                                              |  |
 |  |   The main physics reference is the 'PYTHIA 6.4 Physics and Manual',         |  |
 |  |   T. Sjostrand, S. Mrenna and P. Skands, JHEP05 (2006) 026 [hep-ph/0603175]. |  |
 |  |                                                                              |  |
 |  |   An archive of program versions and documentation is found on the web:      |  |
 |  |   http://www.thep.lu.se/~torbjorn/Pythia.html                                |  |
 |  |                                                                              |  |
 |  |   This program is released under the GNU General Public Licence version 2.   |  |
 |  |   Please respect the MCnet Guidelines for Event Generator Authors and Users. |  |
 |  |                                                                              |  |
 |  |   Disclaimer: this program comes without any guarantees.                     |  |
 |  |   Beware of errors and use common sense when interpreting results.           |  |
 |  |                                                                              |  |
 |  |   Copyright (C) 2007 Torbjorn Sjostrand                                      |  |
 |  |                                                                              |  |
 |  |                                                                              |  |
 |  *------------------------------------------------------------------------------*  |
 |                                                                                    |
 *------------------------------------------------------------------------------------*
 \endverbatim
*/

#include "TPythia8.h"

#include "TClonesArray.h"
#include "TParticle.h"
#include "TDatabasePDG.h"
#include "TLorentzVector.h"

ClassImp(TPythia8);

TPythia8*  TPythia8::fgInstance = 0;

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TPythia8::TPythia8(bool printBanner /*= true*/):
    TGenerator("TPythia8", "TPythia8"),
    fPythia(0),
    fNumberOfParticles(0)
{
   if (fgInstance)
      Fatal("TPythia8", "There's already an instance of TPythia8");

   delete fParticles; // was allocated as TObjArray in TGenerator

   fParticles = new TClonesArray("TParticle",50);
   // If only we could skip stating the default of the first parameter...:
   fPythia    = new Pythia8::Pythia("../share/Pythia8/xmldoc", printBanner);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with an xmlDir (eg "../xmldoc"

TPythia8::TPythia8(const char *xmlDir, bool printBanner /*= true*/):
    TGenerator("TPythia8", "TPythia8"),
    fPythia(0),
    fNumberOfParticles(0)
{
   if (fgInstance)
      Fatal("TPythia8", "There's already an instance of TPythia8");

   delete fParticles; // was allocated as TObjArray in TGenerator

   fParticles = new TClonesArray("TParticle",50);
   fPythia    = new Pythia8::Pythia(xmlDir, printBanner);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TPythia8::~TPythia8()
{
   if (fParticles) {
      fParticles->Delete();
      delete fParticles;
      fParticles = 0;
   }
   delete fPythia;
}

////////////////////////////////////////////////////////////////////////////////
/// Return an instance of TPythia8

TPythia8* TPythia8::Instance()
{
   return fgInstance ? fgInstance : (fgInstance = new TPythia8()) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization

Bool_t TPythia8::Initialize(Int_t idAin, Int_t idBin, Double_t ecms)
{
   AddParticlesToPdgDataBase();

   // Set arguments in Settings database.
   fPythia->settings.mode("Beams:idA",  idAin);
   fPythia->settings.mode("Beams:idB",  idBin);
   fPythia->settings.mode("Beams:frameType",  1);
   fPythia->settings.parm("Beams:eCM", ecms);

   return fPythia->init();

   //return fPythia->init(idAin, idBin, ecms);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization

Bool_t TPythia8::Initialize(Int_t idAin, Int_t idBin, Double_t eAin, Double_t eBin)
{
   AddParticlesToPdgDataBase();

   // Set arguments in Settings database.
   fPythia->settings.mode("Beams:idA",  idAin);
   fPythia->settings.mode("Beams:idB",  idBin);
   fPythia->settings.mode("Beams:frameType",  2);
   fPythia->settings.parm("Beams:eA",      eAin);
   fPythia->settings.parm("Beams:eB",      eBin);

   // Send on to common initialization.
   return fPythia->init();

   //return fPythia->init(idAin, idBin, eAin, eBin);
}

////////////////////////////////////////////////////////////////////////////////
/// Generate the next event

void TPythia8::GenerateEvent()
{
   fPythia->next();
   fNumberOfParticles  = fPythia->event.size() - 1;
   ImportParticles();
}
////////////////////////////////////////////////////////////////////////////////
/// Import particles from Pythia stack

Int_t TPythia8::ImportParticles(TClonesArray *particles, Option_t *option)
{
   if (particles == 0) return 0;
   TClonesArray &clonesParticles = *particles;
   clonesParticles.Clear();
   Int_t nparts=0;
   Int_t i;
   Int_t ioff = 0;
   fNumberOfParticles  = fPythia->event.size();
   if (fPythia->event[0].id() == 90) {
     ioff = -1;
   }

   if (!strcmp(option,"") || !strcmp(option,"Final")) {
      for (i = 0; i < fNumberOfParticles; i++) {
         if (fPythia->event[i].id() == 90) continue;
         if (fPythia->event[i].isFinal()) {
            new(clonesParticles[nparts]) TParticle(
                                                   fPythia->event[i].id(),
                                                   fPythia->event[i].isFinal(),
                                                   fPythia->event[i].mother1() + ioff,
                                                   fPythia->event[i].mother2() + ioff,
                                                   fPythia->event[i].daughter1() + ioff,
                                                   fPythia->event[i].daughter2() + ioff,
                                                   fPythia->event[i].px(),     // [GeV/c]
                                                   fPythia->event[i].py(),     // [GeV/c]
                                                   fPythia->event[i].pz(),     // [GeV/c]
                                                   fPythia->event[i].e(),      // [GeV]
                                                   fPythia->event[i].xProd(),  // [mm]
                                                   fPythia->event[i].yProd(),  // [mm]
                                                   fPythia->event[i].zProd(),  // [mm]
                                                   fPythia->event[i].tProd()); // [mm/c]
            nparts++;
         } // final state partice
      } // particle loop
   } else if (!strcmp(option,"All")) {
      for (i = 0; i < fNumberOfParticles; i++) {
         if (fPythia->event[i].id() == 90) continue;
         new(clonesParticles[nparts]) TParticle(
                                                fPythia->event[i].id(),
                                                fPythia->event[i].isFinal(),
                                                fPythia->event[i].mother1() + ioff,
                                                fPythia->event[i].mother2() + ioff,
                                                fPythia->event[i].daughter1() + ioff,
                                                fPythia->event[i].daughter2() + ioff,
                                                fPythia->event[i].px(),       // [GeV/c]
                                                fPythia->event[i].py(),       // [GeV/c]
                                                fPythia->event[i].pz(),       // [GeV/c]
                                                fPythia->event[i].e(),        // [GeV]
                                                fPythia->event[i].xProd(),    // [mm]
                                                fPythia->event[i].yProd(),    // [mm]
                                                fPythia->event[i].zProd(),    // [mm]
                                                fPythia->event[i].tProd());   // [mm/c]
         nparts++;
      } // particle loop
   }
   if(ioff==-1)     fNumberOfParticles--;
   return nparts;
}

////////////////////////////////////////////////////////////////////////////////
/// Import particles from Pythia stack

TObjArray* TPythia8::ImportParticles(Option_t* /* option */)
{
   fParticles->Clear();
   Int_t ioff = 0;
   Int_t numpart   = fPythia->event.size();
   if (fPythia->event[0].id() == 90) {
     numpart--;
     ioff = -1;
   }


   TClonesArray &a = *((TClonesArray*)fParticles);
   for (Int_t i = 1; i <= numpart; i++) {
      new(a[i]) TParticle(
         fPythia->event[i].id(),
         fPythia->event[i].isFinal(),
         fPythia->event[i].mother1()   + ioff,
         fPythia->event[i].mother2()   + ioff,
         fPythia->event[i].daughter1() + ioff,
         fPythia->event[i].daughter2() + ioff,
         fPythia->event[i].px(),       // [GeV/c]
         fPythia->event[i].py(),       // [GeV/c]
         fPythia->event[i].pz(),       // [GeV/c]
         fPythia->event[i].e(),        // [GeV]
         fPythia->event[i].xProd(),    // [mm]
         fPythia->event[i].yProd(),    // [mm]
         fPythia->event[i].zProd(),    // [mm]
         fPythia->event[i].tProd());   // [mm/c]
   }
   return fParticles;
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization

Int_t TPythia8::GetN() const
{
   return (fPythia->event.size() - 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Configuration

void TPythia8::ReadString(const char* string) const
{
   fPythia->readString(string);
}

////////////////////////////////////////////////////////////////////////////////
/// Configuration

void  TPythia8::ReadConfigFile(const char* string) const
{
  fPythia->readFile(string);
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::ListAll() const
{
   fPythia->settings.listAll();
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::ListChanged() const
{
   fPythia->settings.listChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::Plist(Int_t id) const
{
   fPythia->particleData.list(id);
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::PlistAll() const
{
   fPythia->particleData.listAll();
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::PlistChanged() const
{
   fPythia->particleData.listChanged();
}

////////////////////////////////////////////////////////////////////////////////
/// Print end of run statistics

void TPythia8::PrintStatistics() const
{
   fPythia->stat();
}

////////////////////////////////////////////////////////////////////////////////
/// Event listing

void TPythia8::EventListing() const
{
   fPythia->event.list();
}

////////////////////////////////////////////////////////////////////////////////
/// Add some pythia specific particle code to the data base

void TPythia8::AddParticlesToPdgDataBase()
{
   TDatabasePDG *pdgDB = TDatabasePDG::Instance();
   pdgDB->AddParticle("string","string", 0, kTRUE,
                      0, 0, "QCD string", 90);
   pdgDB->AddParticle("rho_diff0", "rho_diff0", 0, kTRUE,
                      0, 0, "QCD diffr. state", 9900110);
   pdgDB->AddParticle("pi_diffr+", "pi_diffr+", 0, kTRUE,
                      0, 1, "QCD diffr. state", 9900210);
   pdgDB->AddParticle("omega_di", "omega_di", 0, kTRUE,
                      0, 0, "QCD diffr. state", 9900220);
   pdgDB->AddParticle("phi_diff","phi_diff", 0, kTRUE,
                      0, 0, "QCD diffr. state", 9900330);
   pdgDB->AddParticle("J/psi_di", "J/psi_di", 0, kTRUE,
                      0, 0, "QCD diffr. state", 9900440);
   pdgDB->AddParticle("n_diffr0","n_diffr0",0,kTRUE,
                      0, 0, "QCD diffr. state", 9902110);
   pdgDB->AddParticle("p_diffr+","p_diffr+", 0, kTRUE,
                      0, 1, "QCD diffr. state", 9902210);
}

