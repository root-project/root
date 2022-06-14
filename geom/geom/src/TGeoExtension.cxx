// Author: Andrei.Gheata@cern.ch  29/05/2013
// Following proposal by Markus Frank

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoExtension.h"

#include "Rtypes.h"

ClassImp(TGeoExtension);

/** \class TGeoExtension
\ingroup Geometry_classes

ABC for user objects attached to TGeoVolume or TGeoNode.
Provides interface for getting a reference (grab) and
releasing the extension object (release), allowing for
derived classes to implement reference counted sharing.
The user who should attach extensions to logical volumes
or nodes BEFORE applying misalignment information so that
these will be available to all copies.
*/

ClassImp(TGeoRCExtension);

/** \class TGeoRCExtension
\ingroup Geometry_classes

Reference counted extension which has a pointer to and
owns a user defined TObject. This class can be used as
model for a reference counted derivation from TGeoExtension.

Note: Creating a TGeoRCExtension with new() automatically grabs it, but the
creator has to Release it before the pointer gets out of scope.
The following sequence is valid:

~~~ {.cpp}
  // producer:
  TGeoRCExtension *ext = new TGeoRCExtension();
  some_TGeoVolume->SetUserExtension(ext);
  ext->Release();
  // user:
  TGeoRCExtension *ext = dynamic_cast<TGeoRCExtension*>(some_TGeoVolume->GrabUserExtension());
  // ... use extension
  ext->Release();
~~~

The extension is going to be released by the TGeoVolume holder at the destruction
or when calling SetUserExtension(0).

The following usage is not correct:

~~~ {.cpp}
  some_TGeoVolume->SetUserExtension(new TGeoRCExtension())
~~~

since the producer code does not release the extension.
One cannot call directly "delete ext" nor allocate an extension on the stack,
since the destructor is protected. Use Release instead.
*/
