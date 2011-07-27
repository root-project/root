// @(#)root/eg:$Id$
// Author: Rene Brun , Federico Carminati  26/04/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParticle: defines  equivalent of HEPEVT particle                    //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TParticle
#define ROOT_TParticle

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAtt3D
#include "TAtt3D.h"
#endif
#ifndef ROOT_TLorentzVector
#include "TLorentzVector.h"
#endif

class TParticlePDG;

class TParticle : public TObject, public TAttLine, public TAtt3D {


protected:

  Int_t          fPdgCode;              // PDG code of the particle
  Int_t          fStatusCode;           // generation status code
  Int_t          fMother[2];            // Indices of the mother particles
  Int_t          fDaughter[2];          // Indices of the daughter particles
  Float_t        fWeight;               // particle weight

  Double_t       fCalcMass;             // Calculated mass

  Double_t       fPx;                   // x component of momentum
  Double_t       fPy;                   // y component of momentum
  Double_t       fPz;                   // z component of momentum
  Double_t       fE;                    // Energy

  Double_t       fVx;                   // x of production vertex
  Double_t       fVy;                   // y of production vertex
  Double_t       fVz;                   // z of production vertex
  Double_t       fVt;                   // t of production vertex

  Double_t       fPolarTheta;           // Polar angle of polarisation
  Double_t       fPolarPhi;             // azymutal angle of polarisation

  mutable TParticlePDG* fParticlePDG;   //! reference to the particle record in PDG database
  //----------------------------------------------------------------------------
  //  functions
  //----------------------------------------------------------------------------
public:
                                // ****** constructors and destructor
   TParticle();

   TParticle(Int_t pdg, Int_t status,
             Int_t mother1, Int_t mother2,
             Int_t daughter1, Int_t daughter2,
             Double_t px, Double_t py, Double_t pz, Double_t etot,
             Double_t vx, Double_t vy, Double_t vz, Double_t time);

   TParticle(Int_t pdg, Int_t status,
             Int_t mother1, Int_t mother2,
             Int_t daughter1, Int_t daughter2,
             const TLorentzVector &p,
             const TLorentzVector &v);

   TParticle(const TParticle &part);

   virtual ~TParticle();

   TParticle& operator=(const TParticle&);

//   virtual TString* Name   () const { return fName.Data(); }
//   virtual char*  GetName()   const { return fName.Data(); }

   Int_t          GetStatusCode   ()            const { return fStatusCode;                                     }
   Int_t          GetPdgCode      ()            const { return fPdgCode;                                        }
   Int_t          GetFirstMother  ()            const { return fMother[0];                                      }
   Int_t          GetMother       (Int_t i)     const { return fMother[i];                                      }
   Int_t          GetSecondMother ()            const { return fMother[1];                                      }
   Bool_t         IsPrimary       ()            const { return fMother[0]>-1 ? kFALSE : kTRUE;                  } //Is this particle primary one?
   Int_t          GetFirstDaughter()            const { return fDaughter[0];                                    }
   Int_t          GetDaughter     (Int_t i)     const { return fDaughter[i];                                    }
   Int_t          GetLastDaughter ()            const { return fDaughter[1];                                    }
   Double_t       GetCalcMass     ()            const { return fCalcMass;                                       }
   Double_t       GetMass         ()            const;
   Int_t          GetNDaughters   ()            const { return fDaughter[1]>0 ? fDaughter[1]-fDaughter[0]+1 : 0;}
   Float_t        GetWeight       ()            const { return fWeight;                                         }
   void           GetPolarisation(TVector3 &v)  const;
   TParticlePDG*  GetPDG          (Int_t mode = 0) const;
   Int_t          Beauty          ()            const;
   Int_t          Charm           ()            const;
   Int_t          Strangeness     ()            const;
   void           Momentum(TLorentzVector &v)   const { v.SetPxPyPzE(fPx,fPy,fPz,fE);                           }
   void           ProductionVertex(TLorentzVector &v) const { v.SetXYZT(fVx,fVy,fVz,fVt);                       }

// ****** redefine several most oftenly used methods of LORENTZ_VECTOR

   Double_t       Vx              ()            const { return fVx;                                             }
   Double_t       Vy              ()            const { return fVy;                                             }
   Double_t       Vz              ()            const { return fVz;                                             }
   Double_t       T               ()            const { return fVt;                                             }
   Double_t       R               ()            const { return TMath::Sqrt(fVx*fVx+fVy*fVy);                    } //Radius of production vertex in cylindrical system
   Double_t       Rho             ()            const { return TMath::Sqrt(fVx*fVx+fVy*fVy+fVz*fVz);            } //Radius of production vertex in spherical system
   Double_t       Px              ()            const { return fPx;                                             }
   Double_t       Py              ()            const { return fPy;                                             }
   Double_t       Pz              ()            const { return fPz;                                             }
   Double_t       P               ()            const { return TMath::Sqrt(fPx*fPx+fPy*fPy+fPz*fPz);            }
   Double_t       Pt              ()            const { return TMath::Sqrt(fPx*fPx+fPy*fPy);                    }
   Double_t       Energy          ()            const { return fE;                                              }
   Double_t       Eta             ()            const
   {
      Double_t pmom = P();
      if (pmom != TMath::Abs(fPz)) return 0.5*TMath::Log((pmom+fPz)/(pmom-fPz));
      else                         return 1.e30;
   }
   Double_t       Y               ()            const
   {
      if (fE != TMath::Abs(fPz))   return 0.5*TMath::Log((fE+fPz)/(fE-fPz));
      else                         return 1.e30;
   }

   Double_t         Phi   () const { return TMath::Pi()+TMath::ATan2(-fPy,-fPx); }  // note that Phi() returns an angle between 0 and 2pi
   Double_t         Theta () const { return (fPz==0)?TMath::PiOver2():TMath::ACos(fPz/P()); }

              // setters

   void           SetFirstMother  (int code)                                               { fMother[0]   = code ; }
   void           SetMother  (int i, int code)                                             { fMother[i]   = code ; }
   void           SetLastMother  (int code)                                                { fMother[1]   = code ; }
   void           SetFirstDaughter(int code)                                               { fDaughter[0] = code ; }
   void           SetDaughter(int i, int code)                                             { fDaughter[i] = code ; }
   void           SetLastDaughter(int code)                                                { fDaughter[1] = code ; }
   void           SetCalcMass(Double_t mass)                                               { fCalcMass=mass;}
   void           SetPdgCode(Int_t pdg);
   void           SetPolarisation(Double_t polx, Double_t poly, Double_t polz);
   void           SetPolarisation(const TVector3& v)                                       {SetPolarisation(v.X(), v.Y(), v.Z());}
   void           SetStatusCode(int status)                                                {fStatusCode = status;}
   void           SetWeight(Float_t weight = 1)                                            {fWeight = weight; }
   void           SetMomentum(Double_t px, Double_t py, Double_t pz, Double_t e)           {fPx=px; fPy=py; fPz=pz; fE=e;}
   void           SetMomentum(const TLorentzVector& p)                                     {SetMomentum(p.Px(),p.Py(),p.Pz(),p.Energy());}
   void           SetProductionVertex(Double_t vx, Double_t vy, Double_t vz, Double_t t)   {fVx=vx; fVy=vy; fVz=vz; fVt=t;}
   void           SetProductionVertex(const TLorentzVector& v)                             {SetProductionVertex(v.X(),v.Y(),v.Z(),v.T());}

             // ****** overloaded functions of TObject

   virtual void      Paint(Option_t *option = "");
   virtual void      Print(Option_t *option = "") const;
   virtual void      Sizeof3D() const;
   virtual Int_t     DistancetoPrimitive(Int_t px, Int_t py);
   virtual void      ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual const     char *GetName() const;
   virtual const     char *GetTitle() const;


   ClassDef(TParticle,2)  // TParticle vertex particle information
};

#endif
