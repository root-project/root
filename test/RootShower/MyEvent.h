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

class EventHeader {

private:

    Int_t       fEvtNum;
    Int_t       fRun;
    Long_t      fDate;
    Int_t       fPrimary;   // Type of primary particle
    Double_t    fEnergy;    // Energy of primary particle

public:
    EventHeader() : fEvtNum(0), fRun(0), fDate(0), fPrimary(0), fEnergy(0.0) { }
    virtual ~EventHeader() { }
    void        Set(Int_t i, Int_t r, Long_t d, Int_t p, Double_t e)
                { fEvtNum = i; fRun = r; fDate = d; fPrimary = p; fEnergy = e; }
    Int_t       GetEvtNum() const { return fEvtNum; }
    Int_t       GetRun() const { return fRun; }
    Long_t      GetDate() const { return fDate; }
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
    Int_t           fNtrack;            // Number of tracks
    Int_t           fNparticles;        // Number of particles
    Int_t           fNseg;              // Number of track segments
    EventHeader     fEvtHdr;            // Event header
    Double_t        fB;                 // Magnetic field
    Double_t        E_thresh[10];       // Energy threshold for coloring tracks
    TObjArray      *fTracks;            // ->array with all tracks
    TClonesArray   *fParticles;         // ->array with all particles
    TRef            fLastTrack;         // reference pointer to last track
    TRef            fLastParticle;      // reference pointer to last particle
    MyDetector      fDetector;          // Detector

    static TObjArray    *fgTracks;
    static TClonesArray *fgParticles;

public :

    MyEvent();
    virtual ~MyEvent();
    void            Clear(Option_t *option ="");
    static void     Reset(Option_t *option ="");
    void            Init(Int_t id, Int_t first_particle, Double_t E_0, Double_t B_0,
                         Int_t mat, Double_t dimx, Double_t dimy, Double_t dimz);
    void            CreateDetector(Int_t, Double_t, Double_t, Double_t);
    void            SetB(Double_t newB) { fB = newB; }
    void            SetNseg(Int_t n) { fNseg = n; }
    void            SetNtrack(Int_t n) { fNtrack = n; }
    void            SetHeader(Int_t, Int_t, Long_t, Int_t, Double_t);
    TPolyLine3D    *AddTrack(const TVector3 &, Int_t);
    TPolyLine3D    *AddTrack(Double_t, Double_t, Double_t, Int_t);
    MyParticle     *AddParticle(Int_t, Int_t, const TVector3 &, const TVector3 &);

    Int_t           Id() { return fId; }
    Int_t           GetNtrack() const { return fNtrack; }
    Int_t           GetNparticles() const { return fNparticles; }
    Int_t           GetNseg() const { return fNseg; }
    Int_t           GetNAlives() { return fAliveParticles; }
    Int_t           GetTotal() { return fTotalParticles; }
    Int_t           GetLast() { return fLast; }
    Double_t        GetB() { return fB; }
    MyDetector     *GetDetector() { return &fDetector; }
    EventHeader    *GetHeader() { return &fEvtHdr; }
    TObjArray      *GetTracks() const {return fTracks;}
    TClonesArray   *GetParticles() { return fParticles; }
    MyParticle     *GetLastParticle() const {return (MyParticle*)fLastParticle.GetObject();}
    TPolyLine3D    *GetTrack(Int_t at) const {return (TPolyLine3D*)fTracks->At(at);}
    MyParticle     *GetParticle(Int_t at) const {return (MyParticle*)fParticles->At(at);}

    void            DeleteParticle(Int_t);
    void            Magnetic_field(Int_t);
    Int_t           dE_dX(Int_t);
    Int_t           Action(Int_t);
    Int_t           Bremsstrahlung(Int_t);
    Int_t           Pair_production(Int_t);
    Int_t           Move(Int_t, const TVector3 &);
    Double_t        Pair_prob(Int_t);
    Double_t        Brems_prob(Int_t);
    void            Define_decay(Int_t);
    Int_t           FindFreeId(Int_t *);
    TVector3        FindOrtho(const TVector3 &);
    void            ScatterAngle(Int_t);
    Int_t           Particle_color(Int_t);
    Int_t           CheckDecayTime(Int_t id);
    Int_t           Decay(Int_t id);

    ClassDef(MyEvent,1)  //Event structure
};

#endif // MYEVENT_H
