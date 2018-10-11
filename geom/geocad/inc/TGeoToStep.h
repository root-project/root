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

#include "TObject.h"

#include <map>
#include <string>

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

   void CreateGeometry(const char* fname = "geometry.stp", int max_level = -1);
   void CreatePartialGeometry(const char* part_name, int max_level = -1,  const char* fname = "geometry.stp");
   void CreatePartialGeometry(std::map<std::string,int> part_name_levels,  const char* fname = "geometry.stp");

   ClassDef(TGeoToStep,0)
};

#endif
