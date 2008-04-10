// @(#)root/gui:$Id$
// Author: Fons Rademakers   27/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGObject                                                             //
//                                                                      //
// This class is the baseclass for all ROOT GUI widgets.                //
// The ROOT GUI components emulate the Win95 look and feel and the code //
// is based on the XClass'95 code (see Copyleft in source).             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGObject.h"
#include "TVirtualX.h"
#include "TImage.h"

ClassImp(TGObject)

void TGObject::SaveAs(const char* filename /*= ""*/, Option_t* option /*= ""*/) const
{
   // Write this TGObject to a file using TImage, if filename's extension signals 
   // a valid TImage::EImageFileType, as defined by TImage::GetImageFileTypeFromFilename().
   // Otherwise forward to TObject::SaveAs().

   TImage::EImageFileTypes type = TImage::GetImageFileTypeFromFilename(filename);
   if (type != TImage::kUnknown) {
      WindowAttributes_t wattr;
      gVirtualX->GetWindowAttributes(GetId(), wattr);
      TImage* img = TImage::Create();
      img->FromWindow(GetId(), 0, 0, wattr.fWidth, wattr.fHeight);
      img->WriteImage(filename, type);
      delete img;
      return;
   }

   TObject::SaveAs(filename, option);
}
