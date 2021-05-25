// @(#)root/geom:$Id$
// Author: Andrei Gheata   05/12/18

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/** \class TGeoMedium
\ingroup Geometry_classes

This is a wrapper class to G4OpticalSurface
*/

#include "TGeoOpticalSurface.h"

#include <string>

#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGDMLMatrix.h"

ClassImp(TGeoOpticalSurface)
ClassImp(TGeoSkinSurface)
ClassImp(TGeoBorderSurface)

//_____________________________________________________________________________
TGeoOpticalSurface::TGeoOpticalSurface(const char *name, ESurfaceModel model, ESurfaceFinish finish,
                                       ESurfaceType type, Double_t value)
   : TNamed(name, ""), fType(type), fModel(model), fFinish(finish), fValue(value)
{
   // Constructor
   fProperties.SetOwner();
   if (model == kMglisur) {
      fPolish = value;
      fSigmaAlpha = 0.0;
   } else if (model == kMunified || model == kMLUT || model == kMdichroic || model == kMDAVIS) {
      fSigmaAlpha = value;
      fPolish = 0.0;
   } else {
      Fatal("TGeoOpticalSurface::TGeoOpticalSurface()", "Constructor called with INVALID model.");
   }
}

//_____________________________________________________________________________
TGeoOpticalSurface::ESurfaceType TGeoOpticalSurface::StringToType(const char *name)
{
   // Convert string to optical surface type
   ESurfaceType type;
   TString stype(name);
   if ((stype == "dielectric_metal") || (stype == "0")) {
      type = kTdielectric_metal;
   } else if ((stype == "dielectric_dielectric") || (stype == "1")) {
      type = kTdielectric_dielectric;
   } else if ((stype == "dielectric_LUT") || (stype == "2")) {
      type = kTdielectric_LUT;
   } else if ((stype == "dielectric_dichroic") || (stype == "3")) {
      type = kTdielectric_dichroic;
   } else if ((stype == "firsov") || (stype == "4")) {
      type = kTfirsov;
   } else {
      type = kTx_ray;
   }
   return type;
}

//_____________________________________________________________________________
const char *TGeoOpticalSurface::TypeToString(ESurfaceType type)
{
   // Convert surface type to string
   switch (type) {
   case kTdielectric_metal: return "dielectric_metal";
   case kTdielectric_dielectric: return "dielectric_dielectric";
   case kTdielectric_LUT: return "dielectric_LUT";
   case kTdielectric_dichroic: return "dielectric_dichroic";
   case kTfirsov: return "firsov";
   case kTx_ray: return "x_ray";
   case kTdielectric_LUTDAVIS:;
   }

   return "unhandled surface type";
}

//_____________________________________________________________________________
TGeoOpticalSurface::ESurfaceModel TGeoOpticalSurface::StringToModel(const char *name)
{

   // Convert string to optical surface type
   TString smodel(name);
   ESurfaceModel model;
   if ((smodel == "glisur") || (smodel == "0")) {
      model = kMglisur;
   } else if ((smodel == "unified") || (smodel == "1")) {
      model = kMunified;
   } else if ((smodel == "LUT") || (smodel == "2")) {
      model = kMLUT;
   } else {
      model = kMdichroic;
   }
   return model;
}

//_____________________________________________________________________________
const char *TGeoOpticalSurface::ModelToString(ESurfaceModel model)
{
   // Convert optical surface model to string
   switch (model) {
   case kMglisur: return "glisur";
   case kMunified: return "unified";
   case kMLUT: return "LUT";
   case kMdichroic: return "dichoic";
   case kMDAVIS:;
   }

   return "unhandled model type";
}

//_____________________________________________________________________________
TGeoOpticalSurface::ESurfaceFinish TGeoOpticalSurface::StringToFinish(const char *name)
{
   // Convert surface finish to string
   TString sfinish(name);
   ESurfaceFinish finish;
   if ((sfinish == "polished") || (sfinish == "0")) {
      finish = kFpolished;
   } else if ((sfinish == "polishedfrontpainted") || (sfinish == "1")) {
      finish = kFpolishedfrontpainted;
   } else if ((sfinish == "polishedbackpainted") || (sfinish == "2")) {
      finish = kFpolishedbackpainted;
   } else if ((sfinish == "ground") || (sfinish == "3")) {
      finish = kFground;
   } else if ((sfinish == "groundfrontpainted") || (sfinish == "4")) {
      finish = kFgroundfrontpainted;
   } else if ((sfinish == "groundbackpainted") || (sfinish == "5")) {
      finish = kFgroundbackpainted;
   } else if ((sfinish == "polishedlumirrorair") || (sfinish == "6")) {
      finish = kFpolishedlumirrorair;
   } else if ((sfinish == "polishedlumirrorglue") || (sfinish == "7")) {
      finish = kFpolishedlumirrorglue;
   } else if ((sfinish == "polishedair") || (sfinish == "8")) {
      finish = kFpolishedair;
   } else if ((sfinish == "polishedteflonair") || (sfinish == "9")) {
      finish = kFpolishedteflonair;
   } else if ((sfinish == "polishedtioair") || (sfinish == "10")) {
      finish = kFpolishedtioair;
   } else if ((sfinish == "polishedtyvekair") || (sfinish == "11")) {
      finish = kFpolishedtyvekair;
   } else if ((sfinish == "polishedvm2000air") || (sfinish == "12")) {
      finish = kFpolishedvm2000air;
   } else if ((sfinish == "polishedvm2000glue") || (sfinish == "13")) {
      finish = kFpolishedvm2000glue;
   } else if ((sfinish == "etchedlumirrorair") || (sfinish == "14")) {
      finish = kFetchedlumirrorair;
   } else if ((sfinish == "etchedlumirrorglue") || (sfinish == "15")) {
      finish = kFetchedlumirrorglue;
   } else if ((sfinish == "etchedair") || (sfinish == "16")) {
      finish = kFetchedair;
   } else if ((sfinish == "etchedteflonair") || (sfinish == "17")) {
      finish = kFetchedteflonair;
   } else if ((sfinish == "etchedtioair") || (sfinish == "18")) {
      finish = kFetchedtioair;
   } else if ((sfinish == "etchedtyvekair") || (sfinish == "19")) {
      finish = kFetchedtyvekair;
   } else if ((sfinish == "etchedvm2000air") || (sfinish == "20")) {
      finish = kFetchedvm2000air;
   } else if ((sfinish == "etchedvm2000glue") || (sfinish == "21")) {
      finish = kFetchedvm2000glue;
   } else if ((sfinish == "groundlumirrorair") || (sfinish == "22")) {
      finish = kFgroundlumirrorair;
   } else if ((sfinish == "groundlumirrorglue") || (sfinish == "23")) {
      finish = kFgroundlumirrorglue;
   } else if ((sfinish == "groundair") || (sfinish == "24")) {
      finish = kFgroundair;
   } else if ((sfinish == "groundteflonair") || (sfinish == "25")) {
      finish = kFgroundteflonair;
   } else if ((sfinish == "groundtioair") || (sfinish == "26")) {
      finish = kFgroundtioair;
   } else if ((sfinish == "groundtyvekair") || (sfinish == "27")) {
      finish = kFgroundtyvekair;
   } else if ((sfinish == "groundvm2000air") || (sfinish == "28")) {
      finish = kFgroundvm2000air;
   } else {
      finish = kFgroundvm2000glue;
   }

   return finish;
}

//_____________________________________________________________________________
const char *TGeoOpticalSurface::FinishToString(ESurfaceFinish finish)
{
   switch (finish) {
   case kFpolished: return "polished";
   case kFpolishedfrontpainted: return "polishedfrontpainted";
   case kFpolishedbackpainted: return "polishedbackpainted";

   case kFground: return "ground";
   case kFgroundfrontpainted: return "groundfrontpainted";
   case kFgroundbackpainted: return "groundbackpainted";

   case kFpolishedlumirrorair: return "polishedlumirrorair";
   case kFpolishedlumirrorglue: return "polishedlumirrorglue";
   case kFpolishedair: return "polishedair";
   case kFpolishedteflonair: return "polishedteflonair";
   case kFpolishedtioair: return "polishedtioair";
   case kFpolishedtyvekair: return "polishedtyvekair";
   case kFpolishedvm2000air: return "polishedvm2000air";
   case kFpolishedvm2000glue: return "polishedvm2000glue";

   case kFetchedlumirrorair: return "etchedlumirrorair";
   case kFetchedlumirrorglue: return "etchedlumirrorglue";
   case kFetchedair: return "etchedair";
   case kFetchedteflonair: return "etchedteflonair";
   case kFetchedtioair: return "etchedtioair";
   case kFetchedtyvekair: return "etchedtyvekair";
   case kFetchedvm2000air: return "etchedvm2000air";
   case kFetchedvm2000glue: return "etchedvm2000glue";

   case kFgroundlumirrorair: return "groundlumirrorair";
   case kFgroundlumirrorglue: return "groundlumirrorglue";
   case kFgroundair: return "groundair";
   case kFgroundteflonair: return "groundteflonair";
   case kFgroundtioair: return "groundtioair";
   case kFgroundtyvekair: return "groundtyvekair";
   case kFgroundvm2000air: return "groundvm2000air";
   case kFgroundvm2000glue: return "groundvm2000glue";
   case kFRough_LUT:
   case kFRoughTeflon_LUT:
   case kFRoughESR_LUT:
   case kFRoughESRGrease_LUT:
   case kFPolished_LUT:
   case kFPolishedTeflon_LUT:
   case kFPolishedESR_LUT:
   case kFPolishedESRGrease_LUT:
   case kFDetector_LUT:;
   }

   return "unhandled model finish";
}

//_____________________________________________________________________________
const char *TGeoOpticalSurface::GetPropertyRef(const char *property)
{
   // Find reference for a given property
   TNamed *prop = (TNamed *)fProperties.FindObject(property);
   return (prop) ? prop->GetTitle() : nullptr;
}

//_____________________________________________________________________________
TGDMLMatrix *TGeoOpticalSurface::GetProperty(const char *property) const
{
   // Find reference for a given property
   TNamed *prop = (TNamed*)fProperties.FindObject(property);
   if ( !prop ) return nullptr;
   return gGeoManager->GetGDMLMatrix(prop->GetTitle());
}

//_____________________________________________________________________________
TGDMLMatrix *TGeoOpticalSurface::GetProperty(Int_t i) const
{
   // Find reference for a given property
   TNamed *prop = (TNamed*)fProperties.At(i);
   if ( !prop ) return nullptr;
   return gGeoManager->GetGDMLMatrix(prop->GetTitle());
}

//_____________________________________________________________________________
bool TGeoOpticalSurface::AddProperty(const char *property, const char *ref)
{
   fProperties.SetOwner();
   if (GetPropertyRef(property)) {
      Error("AddProperty", "Property %s already added to optical surface %s", property, GetName());
      return false;
   }
   fProperties.Add(new TNamed(property, ref));
   return true;
}

//_____________________________________________________________________________
void TGeoOpticalSurface::Print(Option_t *) const
{
   // Print info about this optical surface
   printf("*** opticalsurface: %s   type: %s   model: %s   finish: %s   value = %g\n", GetName(),
          TGeoOpticalSurface::TypeToString(fType), TGeoOpticalSurface::ModelToString(fModel),
          TGeoOpticalSurface::FinishToString(fFinish), fValue);
   if (fProperties.GetSize()) {
      TIter next(&fProperties);
      TNamed *property;
      while ((property = (TNamed *)next()))
         printf("   property: %s ref: %s\n", property->GetName(), property->GetTitle());
   }
}

//_____________________________________________________________________________
void TGeoSkinSurface::Print(Option_t *) const
{
   // Print info about this optical surface
   if (!fVolume) {
      Error("Print", "Skin surface %s: volume not set", GetName());
      return;
   }
   printf("*** skinsurface: %s   surfaceproperty: %s   volumeref: %s \n", GetName(), GetTitle(), fVolume->GetName());
}

//_____________________________________________________________________________
void TGeoBorderSurface::Print(Option_t *) const
{
   // Print info about this optical surface
   if (!fNode1 || !fNode2) {
      Error("Print", "Border surface %s: nodes not set", GetName());
      return;
   }
   printf("*** bordersurface: %s   surfaceproperty: %s   physvolref: %s  %s \n", GetName(), GetTitle(),
          fNode1->GetName(), fNode2->GetName());
}
