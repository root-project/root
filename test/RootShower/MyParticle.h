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
#include "TParticle.h"

class MyParticle : public TParticle {

private:

    Int_t       fId;            // Index of particle in array
    Int_t       fstatus;        // Particle's status (CREATED,ALIVE,DEAD)
    Int_t       fdecay_type;    // Particle's decay type (bremstrahlung,pair production,decay)
    Int_t       fnChildren;     // Number of children
    TVector3    *flocation;     // Particle's current location
    Double_t    fpassed;        // Distance actually covered
    Double_t    fEloss;         // Total Energy loss into the detector
    Double_t    fdecay_length;  // Calculated interaction length
    Double_t    ftimeOfDecay;   // Generated decay time
    Int_t       fChild[6];      // Array of children indexes

public :

    MyParticle();
    ~MyParticle();
    MyParticle(Int_t, Int_t, Int_t, Int_t, const TVector3 &, const TVector3 &, Double_t, Double_t, Double_t);
    MyParticle(Int_t, Int_t, Int_t, Int_t, const TVector3 &, const TVector3 &);
    Int_t       GetId() { return fId; }
    Int_t       GetStatus() { return fstatus; }
    Int_t       GetDecayType() { return fdecay_type; }
    TVector3    *GetpLocation() { return flocation; }
    TVector3    GetvLocation() { return *flocation; }
    TVector3    GetvMoment() { return TVector3(Px(),Py(),Pz()); }
    Double_t    GetPassed() { return fpassed; }
    Double_t    GetELoss() { return fEloss; }
    Double_t    GetDecayLength() { return fdecay_length; }
    Int_t       GetChildId(Int_t id) { return fChild[id]; }
    Double_t    GetTimeOfDecay() { return ftimeOfDecay; }
    Int_t       GetNChildren() { return fnChildren; }
    Char_t     *GetObjectInfo(Int_t px, Int_t py) const;
    const Char_t *GetName() const;

    void        SetId(Int_t id) { fId = id; }
    void        SetNChildren(Int_t nb) { fnChildren = nb; }
    void        SetStatus(Int_t stat) { fstatus = stat; }
    void        SetDecayType(Int_t decay) { fdecay_type = decay; }
    void        SetTimeOfDecay(Double_t time) { ftimeOfDecay = time; }
    void        SetLocation(const TVector3 &loc) { flocation->SetX(loc.x());
                    flocation->SetY(loc.y()); flocation->SetZ(loc.z()); }
    void        SetLocation(Double_t lx, Double_t ly, Double_t lz) {
                    flocation->SetX(lx); flocation->SetY(ly); flocation->SetZ(lz); }
    void        SetMoment(const TVector3 &mom);
    void        SetMoment(const TVector3 &mom, Double_t energy) {
                    SetMomentum(mom.x(), mom.y(), mom.z(), energy); }
    void        SetPassed(Double_t pass) { fpassed = pass; }
    void        AddELoss(Double_t eloss) { fEloss += eloss; }
    void        SetDecayLength(Double_t len) { fdecay_length = len; }
    void        SetChild(Int_t id, Int_t child_id) { fChild[id] = child_id; }
    void        GenerateTimeOfDecay();

    ClassDef(MyParticle,1)  //Event structure
};

#endif // MYPARTICLE_H
