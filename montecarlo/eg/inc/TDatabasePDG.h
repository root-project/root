// @(#)root/eg:$Id$
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

#include "TParticlePDG.h"
#include "TParticleClassPDG.h"

class THashList;
class TExMap;

class TDatabasePDG: public TNamed {

protected:
   THashList           *fParticleList;     // list of PDG particles
   TObjArray           *fListOfClasses;    // list of classes (leptons etc.)
   mutable TExMap      *fPdgMap;           //!hash-map from pdg-code to particle

   // make copy-constructor and assigment protected since class cannot be copied
   TDatabasePDG(const TDatabasePDG& db)
     : TNamed(db), fParticleList(db.fParticleList),
     fListOfClasses(db.fListOfClasses), fPdgMap(0) { }

   TDatabasePDG& operator=(const TDatabasePDG& db)
   {if(this!=&db) {TNamed::operator=(db); fParticleList=db.fParticleList;
         fListOfClasses=db.fListOfClasses; fPdgMap=db.fPdgMap;}
      return *this;}

   void BuildPdgMap() const;

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

   virtual Int_t  ConvertGeant3ToPdg(Int_t Geant3Number) const;
   virtual Int_t  ConvertPdgToGeant3(Int_t pdgNumber) const;
   virtual Int_t  ConvertIsajetToPdg(Int_t isaNumber) const;

   virtual TParticlePDG* AddAntiParticle(const char* Name, Int_t PdgCode);

   TParticlePDG  *GetParticle(Int_t pdgCode) const;
   TParticlePDG  *GetParticle(const char *name) const;

   TParticleClassPDG* GetParticleClass(const char* name) {
      if (fParticleList == 0)  ((TDatabasePDG*)this)->ReadPDGTable();
      return (TParticleClassPDG*) fListOfClasses->FindObject(name);
   }

   const THashList *ParticleList() const { return fParticleList; }

   virtual void   Print(Option_t *opt = "") const;

   Bool_t IsFolder() const { return kTRUE; }
   virtual void   Browse(TBrowser* b);

   virtual void   ReadPDGTable (const char *filename = "");
   virtual Int_t  WritePDGTable(const char *filename);

   ClassDef(TDatabasePDG, 3);  // PDG particle database
};

#endif
