// @(#)root/eg:$Id$
// Author: Ola Nordmann   21/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPrimary                                                             //
// Is a small class in order to define the particles at the production  //
// vertex.                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TPrimary
#define ROOT_TPrimary

#include "TNamed.h"
#include "TAttLine.h"
#include "TAtt3D.h"
#include "X3DBuffer.h"

class TAttParticle;

class TPrimary : public TObject, public TAttLine, public TAtt3D {

protected:
        Int_t         fPart;         //Particle id produced
        Int_t         fFirstMother;  //Index of the first mother particle
        Int_t         fSecondMother; //Index of the second mother particle(if any)
        Int_t         fGeneration;   //Generation flag: last gen. (0) or not (1) or ghost (2)
        Double_t      fPx;           //Momentum in x direction in GeV/c
        Double_t      fPy;           //Momentum in y direction in GeV/c
        Double_t      fPz;           //Momentum in z direction in GeV/c
        Double_t      fEtot;         //Total energy in GeV
        Double_t      fVx;           //Production vertex x position in user units
        Double_t      fVy;           //Production vertex y position in user units
        Double_t      fVz;           //Production vertex z position in user units
        Double_t      fTime;         //Time of particle production in user units
        Double_t      fTimeEnd;      //Time of particle destruction (always in the pp-cms!)
        TString       fType;         //Indicator of primary type

public:
   TPrimary();
   TPrimary(Int_t part, Int_t first, Int_t second, Int_t gener,
            Double_t px, Double_t py, Double_t pz,
            Double_t etot, Double_t vx, Double_t vy, Double_t vz,
            Double_t time, Double_t timend, const char *type = "");
   virtual ~TPrimary();
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual void          ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual const TAttParticle  *GetParticle() const;
   virtual const char   *GetName() const;
   virtual const char   *GetTitle() const;
   virtual Int_t         GetFirstMother() const { return fFirstMother; }
   virtual Int_t         GetSecondMother() const { return fSecondMother; }
   virtual Int_t         GetGeneration() const { return fGeneration; }
   virtual Double_t      GetXMomentum() const { return fPx; }
   virtual Double_t      GetYMomentum() const { return fPy; }
   virtual Double_t      GetZMomentum() const { return fPz; }
   virtual Double_t      GetTotalEnergy() const { return fEtot; }
   virtual Double_t      GetXPosition() const { return fVx; }
   virtual Double_t      GetYPosition() const { return fVy; }
   virtual Double_t      GetZPosition() const { return fVz; }
   virtual Double_t      GetTime() const { return fTime; }
   virtual Double_t      GetTimeEnd() const { return fTimeEnd; }
   virtual const char   *GetType() const { return fType.Data(); }
   virtual void          Paint(Option_t *option = "");
   virtual void          Print(Option_t *option = "") const;
   virtual void          Sizeof3D() const;

   ClassDef(TPrimary,1)  //TPrimary vertex particle information
};

#endif
