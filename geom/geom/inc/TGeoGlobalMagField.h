// @(#)root/geom:$Id$

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoGlobalMagField
#define ROOT_TGeoGlobalMagField

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TVirtualMagField
#include "TVirtualMagField.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoGlobalMagField - Global magnetic field manager. A field derived    //
//   from TVirtualMagField becomes global if registered via SetField      //
//   method.
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoGlobalMagField : public TObject
{
private:
   static TGeoGlobalMagField *fgInstance;     // Static pointer to the field manager;
   TVirtualMagField       *fField;            // Magnetic field
   Bool_t                  fLock;             // Lock flag for global field.

protected:
   TGeoGlobalMagField(const TGeoGlobalMagField&);
   TGeoGlobalMagField& operator=(const TGeoGlobalMagField&);
   void                    Unlock() {fLock = kFALSE;}
   
public:
   TGeoGlobalMagField();
   virtual ~TGeoGlobalMagField();
   
   // Using SetField() makes a given field global. The field manager owns it from now on.
   TVirtualMagField       *GetField() const {return fField;}
   void                    SetField(TVirtualMagField *field);
   Bool_t                  IsLocked() {return fLock;}
   void                    Lock();
   
   // The field manager should be accessed via TGeoGlobalMagField::Instance()
   static TGeoGlobalMagField *Instance();

   // Inline access to Field() method
   void                    Field(const Double_t *x, Double_t *B) {if (fField) fField->Field(x,B);}
   
   ClassDef(TGeoGlobalMagField, 0)              // Global field manager
};

#endif
