// @(#)root/gui:$Name:  $:$Id: TGIcon.cxx,v 1.5 2003/11/28 08:48:51 brun Exp $
// Author: Fons Rademakers   05/01/98

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
// TGIcon                                                               //
//                                                                      //
// This class handles GUI icons.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGIcon.h"
#include "TGPicture.h"
#include "TSystem.h"
#include "Riostream.h"

ClassImp(TGIcon)

//______________________________________________________________________________
TGIcon::~TGIcon()
{
   // Delete icon and free picture.

   if (fPic) fClient->FreePicture(fPic);
}

//______________________________________________________________________________
void TGIcon::SetPicture(const TGPicture *pic)
{
   // Set icon picture.

   fPic = pic;
   gVirtualX->ClearWindow(fId);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________
TGDimension TGIcon::GetDefaultSize() const
{
   // Return size of icon.

   return TGDimension((fPic) ? fPic->GetWidth()  : fWidth,
                      (fPic) ? fPic->GetHeight() : fHeight);
}

//______________________________________________________________________________
void TGIcon::DoRedraw()
{
   // Redraw picture.

   if (fPic) fPic->Draw(fId, GetBckgndGC()(), 0, 0);
}

//______________________________________________________________________________
void TGIcon::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save an icon widget as a C++ statement(s) on output stream out.

   char quote = '"';

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   if (!fPic) {
      Error("SavePrimitive()", "icon pixmap not found ");
      return;
   }

   char name[kMAXPATHLEN];
   int len = 0;
   const char *picname, *rootname, *pos;

   rootname = gSystem->WorkingDirectory();
#ifdef R__WIN32
   TString dirname = TString(rootname);
   dirname.ReplaceAll('\\','/');
   rootname = dirname.Data();
#endif
   len = strlen(rootname);
   picname = fPic->GetName();
#ifdef R__WIN32
   TString pname = TString(picname);
   pname.ReplaceAll('\\','/');
   picname = pname.Data();
#endif
   pos = strstr(picname, rootname);

   out <<"   TGIcon *";
   out << GetName() << " = new TGIcon(" << fParent->GetName()
       << ",gClient->GetPicture(" << quote;
   if (pos) {
      sprintf(name,"$ROOTSYS%s",pos+len);  // if absolute path
      out << name;
	  printf("name = %s\n",name);
   } else {
	  printf("picname = %s\n",picname);
	  out << picname;                      // if no path
   }
   out << quote << ")" << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
      if (!GetOptions()) {
         out <<");" << endl;
      } else {
         out << "," << GetOptionString() <<");" << endl;
      }
   } else {
      out << "," << GetOptionString() << ",ucolor);" << endl;
   }
}
