// @(#)root/pythia6:$Id$
// Author: Piotr Golonka   17/09/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMCParticle
#define ROOT_TMCParticle

#include "TObject.h"
#include "TAttLine.h"
#include "TPrimary.h"


class TMCParticle : public TObject, public TAttLine {

private:

   Int_t    fKS;            // status of particle       ( LUJETS K[1] )
   Int_t    fKF;            // KF flavour code          ( LUJETS K[2] )
   Int_t    fParent;        // parrent's id             ( LUJETS K[3] )
   Int_t    fFirstChild;    // id of first child        ( LUJETS K[4] )
   Int_t    fLastChild;     // id of last  child        ( LUJETS K[5] )

   Float_t  fPx;            // X momenta [GeV/c]        ( LUJETS P[1] )
   Float_t  fPy;            // Y momenta [GeV/c]        ( LUJETS P[2] )
   Float_t  fPz;            // Z momenta [GeV/c]        ( LUJETS P[3] )
   Float_t  fEnergy;        // Energy    [GeV]          ( LUJETS P[4] )
   Float_t  fMass;          // Mass      [Gev/c^2]      ( LUJETS P[5] )

   Float_t  fVx;            // X vertex  [mm]           ( LUJETS V[1] )
   Float_t  fVy;            // Y vertex  [mm]           ( LUJETS V[2] )
   Float_t  fVz;            // Z vertex  [mm]           ( LUJETS V[3] )
   Float_t  fTime;          // time of procuction [mm/c]( LUJETS V[4] )
   Float_t  fLifetime;      // proper lifetime [mm/c]   ( LUJETS V[5] )


public:
   TMCParticle() : fKS(0), fKF(0), fParent(0), fFirstChild(0),
     fLastChild(0), fPx(0), fPy(0), fPz(0), fEnergy(0), fMass(0),
     fVx(0), fVy(0), fVz(0), fTime(0), fLifetime(0) {}

            TMCParticle(Int_t kS, Int_t kF, Int_t parent,
                        Int_t firstchild, Int_t lastchild,
                        Float_t px, Float_t py, Float_t pz,
                        Float_t energy, Float_t mass,
                        Float_t vx, Float_t vy, Float_t vz,
                        Float_t time, Float_t lifetime) :

               fKS(kS),
               fKF(kF),
               fParent(parent),
               fFirstChild(firstchild),
               fLastChild(lastchild),
               fPx(px),
               fPy(py),
               fPz(pz),
               fEnergy(energy),
               fMass(mass),
               fVx(vx),
               fVy(vy),
               fVz(vz),
               fTime(time),
               fLifetime(lifetime) { }


   virtual             ~TMCParticle() { }

   Int_t       GetKS() const {return fKS;}
   Int_t       GetKF() const {return fKF;}
   Int_t       GetParent() const {return fParent;}
   Int_t       GetFirstChild() const {return fFirstChild;}
   Int_t       GetLastChild() const {return fLastChild;}

   Float_t     GetPx() const {return fPx;}
   Float_t     GetPy() const {return fPy;}
   Float_t     GetPz() const {return fPz;}
   Float_t     GetEnergy() const {return fEnergy;}
   Float_t     GetMass() const {return fMass;}

   Float_t     GetVx() const {return fVx;}
   Float_t     GetVy() const {return fVy;}
   Float_t     GetVz() const {return fVz;}
   Float_t     GetTime() const {return fTime;}
   Float_t     GetLifetime() const {return fLifetime;}
   virtual const char     *GetName() const;

   virtual void        SetKS(Int_t kS) {fKS=kS;}
   virtual void        SetKF(Int_t kF) {fKF=kF;}
   virtual void        SetParent(Int_t parent) {fParent=parent;}
   virtual void        SetFirstChild(Int_t first) {fFirstChild=first;}
   virtual void        SetLastChild(Int_t last) {fLastChild=last;}

   virtual void        SetPx(Float_t px) {fPx=px;}
   virtual void        SetPy(Float_t py) {fPy=py;}
   virtual void        SetPz(Float_t pz) {fPz=pz;}
   virtual void        SetEnergy(Float_t energy) {fEnergy=energy;}
   virtual void        SetMass(Float_t mass) {fMass=mass;}

   virtual void        SetVx(Float_t vx) {fVx=vx;}
   virtual void        SetVy(Float_t vy) {fVy=vy;}
   virtual void        SetVz(Float_t vz) {fVz=vz;}
   virtual void        SetTime(Float_t time) {fTime=time;}
   virtual void        SetLifetime(Float_t lifetime) {fLifetime=lifetime;}


   virtual void        ls(Option_t* option) const;

   ClassDef(TMCParticle,1)  // LUJETS particles data record.
};

#endif
