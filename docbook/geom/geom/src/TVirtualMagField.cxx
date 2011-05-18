// @(#)root/geom:$Id$

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoGlobalMagField.h"
#include "TVirtualMagField.h"

//______________________________________________________________________________
// TVirtualMagField - ABC for magnetic field. Derived classes are encouraged to
// use the TVirtualMagField named constructor and must implement the method:
//    Field(const Double_t *x, Double_t *B)
//
// A field object can be made global via:
//    TGlobalMagField::Instance()->SetField(field)         [1]
// A field which is made global is owned by the field manager. The used is not
// allowed to delete it directly anymore (otherwise a Fatal() is issued). Global
// field can be deleted by calling [1] with a different argument (which can be
// NULL). Otherwise the global field is deleted together with the field manager.
//
//______________________________________________________________________________

ClassImp(TVirtualMagField)

//______________________________________________________________________________
TVirtualMagField::~TVirtualMagField()
{
// Destructor. Unregisters the field.
   TVirtualMagField *global_field = TGeoGlobalMagField::Instance()->GetField();
   if (global_field == this)
      Fatal("~TVirtualMagField", "Not allowed to delete a field once set global. \
             \n To delete the field call: TGeoGlobalMagField::Instance()->SetField(NULL)");
}

//______________________________________________________________________________
// TGeoUniformMagField - Implementation for uniform magnetic field.
//______________________________________________________________________________

ClassImp(TGeoUniformMagField)

//______________________________________________________________________________
TGeoUniformMagField::TGeoUniformMagField()
                    :TVirtualMagField()
{
// Default constructor;
   fB[0] = 0.;
   fB[1] = 0.;
   fB[2] = 0.;
}

//______________________________________________________________________________
TGeoUniformMagField::TGeoUniformMagField(Double_t Bx, Double_t By, Double_t Bz)
                    :TVirtualMagField("Uniform magnetic field")
{
// Default constructor;
   fB[0] = Bx;
   fB[1] = By;
   fB[2] = Bz;
}
