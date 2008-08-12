// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MyEvent                                                              //
//                                                                      //
// Description of event with particles, track and detector parameters   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef MYEVENT_H
#define MYEVENT_H

#include "constants.h"

#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TRef.h"
#include "TPolyLine3D.h"
#include "TVector3.h"
#include "MyParticle.h"
#include "MyDetector.h"
#include "TGListTree.h"
#include "TDatime.h"

class TGeoMaterial;

class EventHeader {

private:
   Int_t       fEvtNum;    // Event Identification
   Int_t       fRun;       // Run Identification
   TDatime     fDate;      // Date of the simulation
   Int_t       fPrimary;   // Type of the primary particle (PDG code)
   Double_t    fEnergy;    // Primary particle's energy

public:
   EventHeader() : fEvtNum(0), fRun(0), fPrimary(0), fEnergy(0.0) { }
   virtual ~EventHeader() { }
   void        Set(Int_t i, Int_t r, TDatime d, Int_t p, Double_t e)
               { fEvtNum = i; fRun = r; fDate = d; fPrimary = p; fEnergy = e; }
   Int_t       GetEvtNum() const { return fEvtNum; }
   Int_t       GetRun() const { return fRun; }
   TDatime     GetDate() const { return fDate; }
   Int_t       GetPrimary() const { return fPrimary; }
   Double_t    GetEnergy() const { return fEnergy; }

   ClassDef(EventHeader,1)  //Event Header
};

class MyEvent : public TObject {

private:
   Int_t           fId;                // Event identification (obsolete)
   Int_t           fTotalParticles;    // Total number of particles
   Int_t           fLast;              // Index of last particle
   Int_t           fAliveParticles;    // Number of still alive particles
   Int_t           fNparticles;        // Number of particles
   Int_t           fNseg;              // Number of track segments
   Int_t           fMatter;            // Material index
   EventHeader     fEvtHdr;            // Event header
   Double_t        fB;                 // Magnetic field
   Double_t        fEThreshold[16];    // Energy threshold for coloring tracks
   TClonesArray   *fParticles;         // ->array with all particles
   TRef            fLastParticle;      // Reference pointer to last particle
   MyDetector      fDetector;          // Detector

   static TClonesArray *fgParticles;   // Pointer on particles array

public :
   MyEvent();
   virtual ~MyEvent();
   void            Clear(Option_t *option ="");
   void            Reset(Option_t *option ="");
   void            Init(Int_t id, Int_t first_particle, Double_t E_0, Double_t B_0);
   void            SetB(Double_t newB) { fB = newB; }
   void            SetNseg(Int_t n) { fNseg = n; }
   void            SetHeader(Int_t, Int_t, TDatime, Int_t, Double_t);
   MyParticle     *AddParticle(Int_t, Int_t, const TVector3 &, const TVector3 &);

   Int_t           Id() { return fId; }
   Int_t           GetNparticles() const { return fNparticles; }
   Int_t           GetNseg() const { return fNseg; }
   Int_t           GetNAlives() { return fAliveParticles; }
   Int_t           GetTotal() { return fTotalParticles; }
   Int_t           GetLast() { return fLast; }
   Double_t        GetB() { return fB; }
   MyDetector     *GetDetector() { return &fDetector; }
   EventHeader    *GetHeader() { return &fEvtHdr; }
   TClonesArray   *GetParticles() { return fParticles; }
   MyParticle     *GetLastParticle() const {return (MyParticle*)fLastParticle.GetObject();}
   MyParticle     *GetParticle(Int_t at) const {return (MyParticle*)fParticles->At(at);}

   Int_t           Action(Int_t);
   Double_t        BremsProb(Int_t);
   Int_t           Bremsstrahlung(Int_t);
   Int_t           CheckDecayTime(Int_t id);
   void            CheckMatter(Int_t id);
   Int_t           Decay(Int_t id);
   void            DefineDecay(Int_t);
   void            DeleteParticle(Int_t);
   Int_t           DEDX(Int_t);
   Int_t           FindFreeId(Int_t *);
   void            MagneticField(Int_t);
   Int_t           Move(Int_t, TVector3 &);
   Double_t        PairProb(Int_t);
   Int_t           PairCreation(Int_t id);
   Int_t           ParticleColor(Int_t);
   void            ScatterAngle(Int_t);

   ClassDef(MyEvent,1)  //Event structure
};

#endif // MYEVENT_H
