// @(#)root/gui:$Name:  $:$Id: TG3DLine.cxx,v 1.1 2000/09/29 08:51:57 rdm Exp $
// Author: Fons Rademakers   6/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGHorizontal3DLine and TGVertical3DLine                              //
//                                                                      //
// A horizontal 3D line is a line that typically separates a toolbar    //
// from the menubar.                                                    //
// A vertical 3D line is a line that can be used to separate groups of  //
// widgets.                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TG3DLine.h"
#include "Riostream.h"


ClassImp(TGHorizontal3DLine)
ClassImp(TGVertical3DLine)

//______________________________________________________________________________
void TGHorizontal3DLine::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save an vertical 3D line as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGHorizontal3DLine *";
   out << GetName() << " = new TGHorizontal3DLine(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
       if (!GetOptions()) {
            out << ");" << endl;
       } else {
         out << "," << GetOptionString() << ");" << endl;
       }
   } else {
     out << "," << GetOptionString() << ",ucolor);" << endl;
   }
}

//______________________________________________________________________________
void TGVertical3DLine::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save an vertical 3D line as a C++ statement(s) on output stream out.

   if (fBackground != GetDefaultFrameBackground()) SaveUserColor(out, option);

   out << "   TGVertical3DLine *";
   out << GetName() << " = new TGVertical3DLine(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight();

   if (fBackground == GetDefaultFrameBackground()) {
       if (!GetOptions()) {
            out << ");" << endl;
       } else {
         out << "," << GetOptionString() <<");" << endl;
       }
   } else {
     out << "," << GetOptionString() << ",ucolor);" << endl;
   }
}
