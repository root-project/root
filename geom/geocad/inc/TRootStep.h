// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootStep
#define ROOT_TRootStep

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoManager;
class OCCStep;


class TRootStep: public TObject {

protected:
   TGeoManager *fGeometry; //ROOT geometry pointer
   OCCStep *fCreate;       //OCC geometry build based on Root one

public:
   TRootStep(); 
   TRootStep(TGeoManager *geom);
   ~TRootStep();
   void *CreateGeometry();

   ClassDef(TRootStep,0)
};

#endif
