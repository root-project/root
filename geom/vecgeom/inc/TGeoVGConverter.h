// Author: Mihaela Gheata   30/03/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGeoConverter
#define ROOT_TGeoConverter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGeoConverter                                                        //
//                                                                      //
// TGeo to VecGeom converter class.                                     //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualGeoConverter
#include "TVirtualGeoConverter.h"
#endif

#ifndef ROOT_TGeoManager
#include "TGeoManager.h"
#endif

class TGeoVGConverter : public TVirtualGeoConverter {
public:
   TGeoVGConverter(TGeoManager *manager);
   virtual ~TGeoVGConverter();

   virtual void       ConvertGeometry();

   ClassDef(TGeoVGConverter,0)  // VecGeom geometry converter
};

#endif
