// @(#)root/eg:$Name:  $:$Id: TDatabasePDG.h,v 1.1.1.1 2000/05/16 17:00:47 rdm Exp $
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//------------------------------------------------------------------------------
//  Jan 19 1999 P.Murat: this is the very  first draft, alot of functionality
//                       is still missing
//------------------------------------------------------------------------------
#ifndef ROOT_TDatabasePDG
#define ROOT_TDatabasePDG

#ifndef ROOT_TParticlePDG
#include "TParticlePDG.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif



class TDatabasePDG: public TNamed {

protected:
   THashList   *fParticleList;        // list of PDG particles

   static TDatabasePDG *fgInstance;   // protect against multiple instances

public:
   TDatabasePDG();
   virtual ~TDatabasePDG();
   virtual void   AddParticle(const char *name, const char *title,
                              Double_t Mass, Bool_t Stable,
                              Double_t DecayWidth, Double_t Charge, const char *Type,
                              Int_t pdgCode);
				
   virtual Int_t  ConvertIsajetToPdg(Int_t isaNumber);

   TParticlePDG  *GetParticle(Int_t pdgCode) const;
   TParticlePDG  *GetParticle(const char *name) const;
   virtual void   Init(); // function which does the real job of initializing the database
   const THashList *ParticleList() const { return fParticleList; }

   virtual void   Print(Option_t *opt = "") const;
   virtual void   ReadPDGTable(const char *filename);

   static TDatabasePDG *Instance();

  ClassDef(TDatabasePDG,1)  // PDG particle database
};

#endif
