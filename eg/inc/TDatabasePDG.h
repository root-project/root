// @(#)root/eg:$Name:  $:$Id: TDatabasePDG.h,v 1.3 2001/03/05 09:09:42 brun Exp $
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TDatabasePDG
#define ROOT_TDatabasePDG

#ifndef ROOT_TParticlePDG
#include "TParticlePDG.h"
#endif
#ifndef ROOT_THashList
#include "THashList.h"
#endif
#ifndef ROOT_TParticleClassPDG
#include "TParticleClassPDG.h"
#endif

class TDatabasePDG: public TNamed {

protected:
  static TDatabasePDG *fgInstance;	// protect against multiple instances
  THashList* fParticleList;		// list of PDG particles
  TObjArray* fListOfClasses;		// list of classes (leptons etc.)

  
public:

  TDatabasePDG();
  virtual ~TDatabasePDG();

  static TDatabasePDG*  Instance();

  virtual TParticlePDG*   AddParticle(const char*  Name, 
					const char*  Title,
					Double_t     Mass, 
					Bool_t       Stable,
					Double_t     DecayWidth, 
					Double_t     Charge, 
					const char*  ParticleClass,
					Int_t        PdgCode,
					Int_t        Anti=-1,
					Int_t        TrackingCode=0);

  virtual Int_t  ConvertIsajetToPdg(Int_t isaNumber);

  virtual TParticlePDG* AddAntiParticle(const char* Name, Int_t PdgCode);
				
  TParticlePDG  *GetParticle(Int_t pdgCode) const;
  TParticlePDG  *GetParticle(const char *name) const;

  TParticleClassPDG* GetParticleClass(const char* name) {
    return (TParticleClassPDG*) fListOfClasses->FindObject(name);
  }

  const THashList *ParticleList() const { return fParticleList; }

  virtual void   Print(Option_t *opt = "") const;

  Bool_t IsFolder() const { return kTRUE; }
  virtual void   Browse(TBrowser* b);

  virtual void   ReadPDGTable (const char *filename = "");
  virtual Int_t  WritePDGTable(const char *filename);

  ClassDef(TDatabasePDG,2)  // PDG particle database

};

#endif
