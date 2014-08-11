/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Author: Andrei.Gheata@cern.ch  29/05/2013
// Following proposal by Markus Frank

#ifndef ROOT_TGeoExtension
#define ROOT_TGeoExtension

#include <cassert>

#ifndef ROOT_TObject
#include "TObject.h"
#endif

//______________________________________________________________________________
//   TGeoExtension - ABC for user objects attached to TGeoVolume or TGeoNode.
//                   Provides interface for getting a reference (grab) and
//                   releasing the extension object (release), allowing for
//                   derived classes to implement reference counted sharing.
//                   The user who should attach extensions to logical volumes
//                   or nodes BEFORE applying misalignment information so that
//                   these will be available to all copies.
//______________________________________________________________________________

class TGeoExtension : public TObject
{
protected:
   TGeoExtension() : TObject() {}
   virtual ~TGeoExtension() {}

public:
   // Method called whenever requiring a pointer to the extension
   // Equivalent to new()
   virtual TGeoExtension *Grab() = 0;
   // Method called always when the pointer to the extension is not needed
   // Equivalent to delete()
   virtual void           Release() const = 0;

   ClassDef(TGeoExtension, 1)       // User extension for volumes and nodes
};


//______________________________________________________________________________
//   TGeoRCExtension - Reference counted extension which has a pointer to and
//                   owns a user defined TObject. This class can be used as
//                   model for a reference counted derivation from TGeoExtension.
//                   The user object becomes owned by the extension.
//______________________________________________________________________________

class TGeoRCExtension : public TGeoExtension
{
protected:
   virtual ~TGeoRCExtension() {delete fUserObject;}
public:
   TGeoRCExtension() : TGeoExtension(), fRC(0), fUserObject(0) {fRC++;}
   TGeoRCExtension(TObject *obj) : TGeoExtension(), fRC(0), fUserObject(obj) {fRC++;}

   TGeoExtension       *Grab()                      {fRC++; return this;}
   void                 Release() const             {assert(fRC > 0); fRC--; if (fRC ==0) delete this;}

   void                 SetUserObject(TObject *obj) {fUserObject = obj;}
   TObject             *GetUserObject() const       {return fUserObject;}


private:
   // Copy constructor and assignment not allowed
   TGeoRCExtension(const TGeoRCExtension &); // Not implemented
   TGeoRCExtension &operator =(const TGeoRCExtension &); // Not implemented
   mutable Int_t        fRC;           // Reference counter
   TObject             *fUserObject;   // Attached user object

   ClassDef(TGeoRCExtension, 1)       // Reference counted extension for volumes and nodes
};

#endif
