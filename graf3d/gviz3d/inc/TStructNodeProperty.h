// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TStructNodeProperty
#define ROOT_TStructNodeProperty

#include <TNamed.h>
#include <TColor.h>
#include <TGColorSelect.h>

class TStructNodeProperty : public TNamed {

private:
   TColor fColor;               // Color of a class

public:
   TStructNodeProperty(const char * name, Int_t color);
   TStructNodeProperty(const char * name, Pixel_t pixel);
   ~TStructNodeProperty();

   TColor   GetColor() const;
   Pixel_t  GetPixel() const;
   void     SetColor(const TColor & color);
   void     SetColor(Pixel_t pixel);
   void     SetColor(Int_t color);
   Int_t    Compare(const TObject* obj) const;
   Bool_t   IsSortable() const;

   ClassDef(TStructNodeProperty, 1); // Class with nodes color property
};

#endif
