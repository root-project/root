// @(#)root/graf:$Id$
// Author: Advait Dhingra 13/04/21

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
//--------------------------------------------------------------------------

#include "TObject.h"

class TFeynmanEntry {
    public:
        TFeynmanEntry(const TObject* particle);
        virtual void SetObject(TObject *obj);
    protected:
        TObject      *fObject;   ///< pointer to object being represented by this entry
    private:
      ClassDef(TFeynmanEntry, 1);
};
