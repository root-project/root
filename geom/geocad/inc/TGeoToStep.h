// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoToStep
#define ROOT_TGeoToStep

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoManager;
class TOCCToStep;


class TGeoToStep: public TObject {

protected:
   TGeoManager *fGeometry; //ROOT geometry pointer
   TOCCToStep *fCreate;       //OCC geometry build based on Root one

public:
   TGeoToStep();
   TGeoToStep(TGeoManager *geom);
   ~TGeoToStep();
   void *CreateGeometry();

   ClassDef(TGeoToStep,0)
};

#endif
