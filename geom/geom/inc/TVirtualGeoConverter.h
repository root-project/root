// @(#)root/geom:$Id$
// Author: Mihaela Gheata   30/03/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualGeoConverter
#define ROOT_TVirtualGeoConverter

#include "TObject.h"

class TGeoManager;

class TVirtualGeoConverter : public TObject {

protected:
   static TVirtualGeoConverter   *fgGeoConverter; // Pointer to geometry converter
   TGeoManager                   *fGeom; // Pointer to geometry manager
public:
   TVirtualGeoConverter(TGeoManager *geom);
   virtual ~TVirtualGeoConverter();

   virtual void       ConvertGeometry() {}
   static  TVirtualGeoConverter *Instance(TGeoManager *geom=0);
   static void        SetConverter(const TVirtualGeoConverter *conv);
   void               SetGeometry(TGeoManager *geom) { fGeom = geom; }

   ClassDef(TVirtualGeoConverter,0)  // Abstract interface for geometry converters
};

#endif
