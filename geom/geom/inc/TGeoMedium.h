// @(#)root/geom:$Id$
// Author: Rene Brun   26/12/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoMedium
#define ROOT_TGeoMedium

#ifndef ROOT_TGeoMaterial
#include "TGeoMaterial.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoMedium - base class describing tracking media                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoMedium : public TNamed
{
public:
   enum EGeoMedium {
      kMedSavePrimitive = BIT(18)
   };

protected:
   Int_t                    fId;         // unique Id
   Double_t                 fParams[20]; // parameters
   TGeoMaterial            *fMaterial;   // pointer to material
 
// methods
   TGeoMedium(const TGeoMedium&);
   TGeoMedium& operator=(const TGeoMedium&);

public:
   // constructors
   TGeoMedium();
   TGeoMedium(const char *name, Int_t numed, const TGeoMaterial *mat, Double_t *params=0);
   TGeoMedium(const char *name, Int_t numed, Int_t imat, Int_t isvol, Int_t ifield,
              Double_t fieldm, Double_t tmaxfd, Double_t stemax, Double_t deemax, Double_t epsil, Double_t stmin);
   virtual ~TGeoMedium();
   // methods
   virtual Int_t            GetByteCount() const {return sizeof(*this);}
   Int_t                    GetId()   const     {return fId;}
   Double_t                 GetParam(Int_t i) const {return fParams[i];}
   void                     SetParam(Int_t i, Double_t val)   {fParams[i] = val;}
   char                    *GetPointerName() const;
   TGeoMaterial            *GetMaterial() const {return fMaterial;}
   virtual void             SavePrimitive(std::ostream &out, Option_t *option = "");
   void                     SetId(Int_t id)     {fId = id;}
   void                     SetMaterial(TGeoMaterial *mat) {fMaterial = mat;}
   virtual void             SetCerenkovProperties(TObject* cerenkov) {fMaterial->SetCerenkovProperties(cerenkov);}   
   ClassDef(TGeoMedium, 1)              // tracking medium

};

#endif

