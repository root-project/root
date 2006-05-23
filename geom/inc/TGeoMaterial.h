// @(#)root/geom:$Name:  $:$Id: TGeoMaterial.h,v 1.19 2005/11/18 16:07:58 brun Exp $
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#ifndef ROOT_TGeoElement
#include "TGeoElement.h"
#endif


// forward declarations


/*************************************************************************
 * TGeoMaterial - base class describing materials 
 *
 *************************************************************************/

class TGeoMaterial : public TNamed,
                     public TAttFill
{
public:
   enum EGeoMaterial {
      kMatUsed   =   BIT(17),
      kMatSavePrimitive = BIT(18)
   };

protected:
   Int_t                    fIndex;      // material index
   Double_t                 fA;          // A of material
   Double_t                 fZ;          // Z of material
   Double_t                 fDensity;    // density of material
   Double_t                 fRadLen;     // radiation length
   Double_t                 fIntLen;     // interaction length
   TObject                 *fShader;     // shader with optical properties
   TObject                 *fCerenkov;   // pointer to class with Cerenkov properties

// methods
   TGeoMaterial(const TGeoMaterial&);
   TGeoMaterial& operator=(const TGeoMaterial&);


public:
   // constructors
   TGeoMaterial();
   TGeoMaterial(const char *name);
   TGeoMaterial(const char *name, Double_t a, Double_t z, 
                Double_t rho, Double_t radlen=0, Double_t intlen=0);
   TGeoMaterial(const char *name, TGeoElement *elem, 
                Double_t rho);
   // destructor
   virtual ~TGeoMaterial();
   // methods
   static  Double_t         Coulomb(Double_t z);
   virtual Int_t            GetByteCount() const {return sizeof(this);}
   virtual Double_t         GetA() const       {return fA;}
   virtual Double_t         GetZ()  const      {return fZ;}
   virtual Int_t            GetDefaultColor() const;
   virtual Double_t         GetDensity() const {return fDensity;}
   virtual TGeoElement     *GetElement(Int_t i=0) const;
   char                    *GetPointerName() const;
   virtual Double_t         GetRadLen() const  {return fRadLen;}
   virtual Double_t         GetIntLen() const  {return fIntLen;}
   Int_t                    GetIndex();
   virtual TObject         *GetCerenkovProperties() const {return fCerenkov;}
   Char_t                   GetTransparency() const {return (fFillStyle<3000 || fFillStyle>3100)?0:Char_t(fFillStyle-3000);}
   virtual Bool_t           IsEq(const TGeoMaterial *other) const;
   Bool_t                   IsUsed() const {return TObject::TestBit(kMatUsed);}
   virtual Bool_t           IsMixture() const {return kFALSE;}
   virtual void             Print(const Option_t *option="") const;
   virtual void             SavePrimitive(ofstream &out, Option_t *option);
   void                     SetIndex(Int_t index) {fIndex=index;}
   virtual void             SetCerenkovProperties(TObject* cerenkov) {fCerenkov = cerenkov;}
   void                     SetRadLen(Double_t radlen, Double_t intlen=0.);
   void                     SetUsed(Bool_t flag=kTRUE) {TObject::SetBit(kMatUsed, flag);}
   void                     SetTransparency(Char_t transparency=0) {fFillStyle = 3000+transparency;}
   static  Double_t         ScreenFactor(Double_t z);

   

   ClassDef(TGeoMaterial, 3)              // base material class

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
   Double_t                *fWeights;    // [fNelements] array of relative proportions by mass

// methods
   TGeoMixture(const TGeoMixture&); // Not implemented
   TGeoMixture& operator=(const TGeoMixture&); // Not implemented

public:
   // constructors
   TGeoMixture();
   TGeoMixture(const char *name, Int_t nel, Double_t rho=-1);
   // destructor
   virtual ~TGeoMixture();
   // methods
   void                     DefineElement(Int_t iel, Double_t a, Double_t z, Double_t weight);
   void                     DefineElement(Int_t iel, TGeoElement *elem, Double_t weight);
   void                     DefineElement(Int_t iel, Int_t z, Int_t natoms);
   virtual Int_t            GetByteCount() const {return 48+12*fNelements;}
   virtual TGeoElement     *GetElement(Int_t i=0) const;
   Int_t                    GetNelements() const {return fNelements;}
   Double_t                *GetZmixt() const     {return fZmixture;}
   Double_t                *GetAmixt() const     {return fAmixture;}
   Double_t                *GetWmixt() const     {return fWeights;}
   virtual Bool_t           IsEq(const TGeoMaterial *other) const;
   virtual Bool_t           IsMixture() const {return kTRUE;}
   virtual void             Print(const Option_t *option="") const;
   virtual void             SavePrimitive(ofstream &out, Option_t *option);
   void                     SetA(Double_t a) {fA = a;}
   void                     SetZ(Double_t z) {fZ = z;}

   ClassDef(TGeoMixture, 1)              // material mixtures

//***** Need to add classes and globals to LinkDef.h *****
};

#endif

