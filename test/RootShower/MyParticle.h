// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MyParticle                                                           //
//                                                                      //
// Defines single particle class, with its status and decay parameters  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef MYPARTICLE_H
#define MYPARTICLE_H

#include "TGListTree.h"
#include "TVector3.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TRef.h"
#include "TPolyLine3D.h"
#include "TParticle.h"
#include "TParticlePDG.h"

class MyParticle : public TParticle {

private:
   Int_t       fId;            // Index of particle in array
   Int_t       fStatus;        // Particle's status (CREATED,ALIVE,DEAD)
   Int_t       fDecayType;     // Particle's decay type (bremstrahlung,pair production,decay)
   Int_t       fNChildren;     // Number of children
   TVector3   *fLocation;     // Particle's current location
   Double_t    fPassed;        // Distance actually covered
   Double_t    fEloss;         // Total Energy loss into the detector
   Double_t    fDecayLength;   // Calculated interaction length
   Double_t    fTimeOfDecay;   // Generated decay time
   Int_t       fChild[6];      // Array of children indexes

   Int_t       fNtrack;    
   TObjArray  *fTracks;       // ->array with all tracks

public :
   MyParticle();
   ~MyParticle();
   MyParticle(Int_t, Int_t, Int_t, Int_t, const TVector3 &, const TVector3 &, Double_t, Double_t, Double_t);
   MyParticle(Int_t, Int_t, Int_t, Int_t, const TVector3 &, const TVector3 &);
   TPolyLine3D  *AddTrack(const TVector3 &, Int_t);
   TPolyLine3D  *AddTrack(Double_t, Double_t, Double_t, Int_t);
   Int_t         GetId() { return fId; }
   Int_t         GetStatus() { return fStatus; }
   Int_t         GetDecayType() { return fDecayType; }
   TVector3     *GetpLocation() { return fLocation; }
   TVector3      GetvLocation() { return *fLocation; }
   TVector3      GetvMoment() { return TVector3(Px(),Py(),Pz()); }
   Double_t      GetPassed() { return fPassed; }
   Double_t      GetELoss() { return fEloss; }
   Double_t      GetDecayLength() { return fDecayLength; }
   Int_t         GetChildId(Int_t id) { return fChild[id]; }
   Double_t      GetTimeOfDecay() { return fTimeOfDecay; }
   Int_t         GetNChildren() { return fNChildren; }
   Int_t         GetNTracks() { return fNtrack+1; }
   Char_t       *GetObjectInfo(Int_t px, Int_t py) const;
   TPolyLine3D  *GetTrack(Int_t at) const {return (TPolyLine3D*)fTracks->At(at);}
   const Char_t *GetName() const;

   void          SetId(Int_t id) { fId = id; }
   void          SetNChildren(Int_t nb) { fNChildren = nb; }
   void          SetStatus(Int_t stat) { fStatus = stat; }
   void          SetDecayType(Int_t decay) { fDecayType = decay; }
   void          SetTimeOfDecay(Double_t time) { fTimeOfDecay = time; }
   void          SetLocation(const TVector3 &loc) { fLocation->SetX(loc.x());
                             fLocation->SetY(loc.y()); fLocation->SetZ(loc.z()); }
   void          SetLocation(Double_t lx, Double_t ly, Double_t lz) {
                             fLocation->SetX(lx); fLocation->SetY(ly); fLocation->SetZ(lz); }
   void          SetMoment(const TVector3 &mom);
   void          SetMoment(const TVector3 &mom, Double_t energy) {
                           SetMomentum(mom.x(), mom.y(), mom.z(), energy); }
   void          SetNextPoint(Int_t color);
   void          SetPassed(Double_t pass) { fPassed = pass; }
   void          AddELoss(Double_t eloss) { fEloss += eloss; }
   void          SetDecayLength(Double_t len) { fDecayLength = len; }
   void          SetChild(Int_t id, Int_t child_id) { fChild[id] = child_id; }
   void          GenerateTimeOfDecay();

   void          HighLight(); // *MENU*

   virtual void  Delete(Option_t *) { }
   virtual void  SetLineAttributes() { }
   virtual void  SetDrawOption(Option_t *) { }

   ClassDef(MyParticle,1)  //Event structure
};

#endif // MYPARTICLE_H
