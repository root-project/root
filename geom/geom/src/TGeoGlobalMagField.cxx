// @(#)root/geom:$Id$

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TGeoGlobalMagField.h"

//______________________________________________________________________________
//                                                                        
//    TGeoGlobalMagField - Global magnetic field manager. Provides access to 
// and owns the actual magnetic field set via SetField(). The field is deleted
// upon destruction of the field manager at the end of ROOT session or
// by calling: TGeoGlobalMagField::Instance()->SetField(0). The previous
// global field is deleted upon replacement with notification.
//
// The global field manager provides access to the global field via:
//   TGeoGlobalMagField::Instance()->GetField()
// One can directly call the Field() method of a field via the global field manager:
//
//   TGeoGlobalMagField::Instance()->Field(x,B)
//                                                                     
//______________________________________________________________________________

ClassImp(TGeoGlobalMagField)

TGeoGlobalMagField *TGeoGlobalMagField::fgInstance = NULL;

//______________________________________________________________________________
TGeoGlobalMagField::TGeoGlobalMagField()
{
// Global field default constructor.
   fField = NULL;
   fLock = kFALSE;
   if (fgInstance) {
      TVirtualMagField *field = fgInstance->GetField();
      if (field)
         Fatal("TGeoGlobalMagField", "A global field manager already existing and containing a field. \
         \n If you want a new global field please set it via: \
         \n   TGeoGlobalMagField::Instance()->SetField(myField).");
      else
         Warning("TGeoGlobalMagField", "A global field manager already existing. Please access via: \
         \n   TGeoGlobalMagField::Instance().");
      delete fgInstance;
   }   
   gROOT->GetListOfGeometries()->Add(this); // list of cleanups not deleted
   fgInstance = this;
}

//______________________________________________________________________________
TGeoGlobalMagField::~TGeoGlobalMagField()
{
// Global field destructor.
   gROOT->GetListOfGeometries()->Remove(this);
   if (fField) {
      TVirtualMagField *field = fField;
      fField = NULL;
      delete field;
   }   
   fgInstance = NULL;
}
   
//______________________________________________________________________________
void TGeoGlobalMagField::SetField(TVirtualMagField *field)
{
// Field setter. Deletes previous field if any. Acts only if fLock=kFALSE.
   if (field==fField) return;
   // Check if we are allowed to change the old field.
   if (fField) {
      if (fLock) {
         Error("SetField", "Global field is already set to <%s> and locked", fField->GetName());
         return;
      }
      // We delete the old global field and notify user.   
      Info("SetField", "Previous magnetic field <%s> will be deleted", fField->GetName());
      TVirtualMagField *oldfield = fField;
      fField = NULL;
      delete oldfield;
   }   
   fField = field;
   if (fField) Info("SetField", "Global magnetic field set to <%s>", fField->GetName());
}

//______________________________________________________________________________
TGeoGlobalMagField *TGeoGlobalMagField::GetInstance()
{
// Static getter that does not create the object.
   return fgInstance;
}   

//______________________________________________________________________________
TGeoGlobalMagField *TGeoGlobalMagField::Instance()
{
// Returns always a valid static pointer to the field manager.
   if (fgInstance) return fgInstance;
   return new TGeoGlobalMagField();
}

//______________________________________________________________________________
void TGeoGlobalMagField::Lock()
{
// Locks the global magnetic field if this is set. Cannot be unlocked.
   if (!fField) {
      Warning("Lock", "Cannot lock global magnetic field since this was not set yet");
      return;
   }
   fLock = kTRUE;
   Info("Lock", "Global magnetic field <%s> is now locked", fField->GetName());
}   
