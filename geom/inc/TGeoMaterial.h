/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - date Wed 24 Oct 2001 03:41:14 PM CEST

/*************************************************************************
 * TGeoMaterial - package description
 *
 *
 *
 *************************************************************************/

#ifndef ROOT_TGeoMaterial
#define ROOT_TGeoMaterial

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

// forward declarations

//class TGeoShader;

/*************************************************************************
 * TGeoMaterial - base class describing materials 
 *
 *************************************************************************/

class TGeoMaterial : public TNamed,
                     public TAttFill
{
   enum EGeoMaterial {
      kMatUsed   =   BIT(17)
   };

protected:
   Int_t                    fId;         // unique Id
private :
// data members
   TObject                 *fShader;     // shader with optical properties
   Double_t                 fA;          // A of material
   Double_t                 fZ;          // Z of material
   Double_t                 fDensity;    // density of material
   Double_t                 fRadLen;     // radiation length
   Double_t                 fIntLen;     // interaction length
// methods

public:
   // constructors
   TGeoMaterial();
   TGeoMaterial(const char *name, const char *title);
   TGeoMaterial(const char *name, const char *title, Double_t a, Double_t z, 
                Double_t rho, Double_t radlen=0, Double_t intlen=0);
   // destructor
   virtual ~TGeoMaterial();
   // methods
   Int_t                    GetMedia()     {return fId;}
   virtual Int_t            GetByteCount() {return 32;}
   virtual Double_t         GetA()       {return fA;}
   virtual Double_t         GetZ()       {return fZ;}
   virtual Int_t            GetDefaultColor();
   virtual Double_t         GetDensity() {return fDensity;}
   virtual Double_t         GetRadLen()  {return fRadLen;}
   virtual Double_t         GetIntLen()  {return fIntLen;}
   virtual Bool_t           IsEq(TGeoMaterial *other);
   virtual void             Print(const Option_t *option="") const;
   void                     SetMedia(Int_t id) {fId = id;}


  ClassDef(TGeoMaterial, 1)              // base material class

//***** Need to add classes and globals to LinkDef.h *****
};

/*************************************************************************
 * TGeoMixture - mixtures of elements 
 *
 *************************************************************************/

class TGeoMixture : public TGeoMaterial
{
protected :
// data members
   Int_t                    fNelements;  // number of elements
   Double_t                *fZmixture;   // [fNelements] array of Z of the elements
   Double_t                *fAmixture;   // [fNelements] array of A of the elements
   Double_t                *fWeights;    // [fNelements] array of relative weights
// methods

public:
   // constructors
   TGeoMixture();
   TGeoMixture(const char *name, const char *title, Int_t nel);
   // destructor
   virtual ~TGeoMixture();
   // methods
   void                     DefineElement(Int_t i, Double_t a, Double_t z, Double_t weight);
   virtual Int_t            GetByteCount() {return 48+12*fNelements;}
   Int_t                    GetNelements() {return fNelements;}
   Double_t                *GetZmixt()     {return fZmixture;}
   Double_t                *GetAmixt()     {return fAmixture;}
   Double_t                *GetWmixt()     {return fWeights;}
   virtual Bool_t           IsEq(TGeoMaterial *other);
   virtual void             Print(const Option_t *option="") const;

  ClassDef(TGeoMixture, 1)              // material mixtures

//***** Need to add classes and globals to LinkDef.h *****
};

#endif

