// @(#)root/base:$Name:  $:$Id: TNamed.cxx,v 1.6 2001/02/13 07:54:00 brun Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNamed                                                               //
//                                                                      //
// The TNamed class is the base class for all named ROOT classes        //
// A TNamed contains the essential elements (name, title)               //
// to identify a derived object in containers, directories and files.   //
// Most member functions defined in this base class are in general      //
// overridden by the derived classes.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream.h>

#include "Strlen.h"
#include "TNamed.h"
#include "TClass.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "Bytes.h"


ClassImp(TNamed)


//______________________________________________________________________________
TNamed::TNamed(const TNamed &named)
{
   // TNamed copy ctor.

   fName = named.fName;
   fTitle = named.fTitle;
}

//______________________________________________________________________________
TNamed& TNamed::operator=(const TNamed& rhs)
{
   // TNamed assignment operator.

   if (this != &rhs) {
      TObject::operator=(rhs);
      fName  = rhs.fName;
      fTitle = rhs.fTitle;
   }
   return *this;
}

//______________________________________________________________________________
TObject *TNamed::Clone(const char *newname) const
{
   // Make a clone of an object using the Streamer facility.
   // If newname is specified, this will be the name of the new object

   TNamed *named = (TNamed*)TObject::Clone(newname);
   if (newname && strlen(newname)) named->SetName(newname);
   return named;
}

//______________________________________________________________________________
Int_t TNamed::Compare(const TObject *obj) const
{
   // Compare two TNamed objects. Returns 0 when equal, -1 when this is
   // smaller and +1 when bigger (like strcmp).

   if (this == obj) return 0;
   return fName.CompareTo(obj->GetName());
}

//______________________________________________________________________________
void TNamed::Copy(TObject &obj)
{
   // Copy this to obj.

   TObject::Copy(obj);
   ((TNamed&)obj).fName  = fName;
   ((TNamed&)obj).fTitle = fTitle;
}

//______________________________________________________________________________
void TNamed::FillBuffer(char *&buffer)
{
   // Encode TNamed into output buffer.

   fName.FillBuffer(buffer);
   fTitle.FillBuffer(buffer);
}

//______________________________________________________________________________
void TNamed::ls(Option_t *) const
{
   // List TNamed name and title.

   TROOT::IndentLevel();
   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << " : "
        << Int_t(TestBit(kCanDelete)) << endl;
}

//______________________________________________________________________________
void TNamed::Print(Option_t *) const
{
   // Print TNamed name and title.

   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << endl;
}

//______________________________________________________________________________
void TNamed::SetName(const char *name)
{
   // Change (i.e. set) the name of the TNamed.
   // WARNING !!
   // If the object is a member of a THashTable, THashList container
   // The HashTable must be Rehashed after SetName
   // For example the list of objects in the current directory is a THashList

   fName = name;
   if (gPad && TestBit(kMustCleanup)) gPad->Modified();
}

//______________________________________________________________________________
void TNamed::SetNameTitle(const char *name, const char *title)
{
   // Change (i.e. set) all the TNamed parameters (name and title).
   // See also WARNING in SetName

   fName  = name;
   fTitle = title;
   if (gPad && TestBit(kMustCleanup)) gPad->Modified();
}

//______________________________________________________________________________
void TNamed::SetTitle(const char *title)
{
   // Change (i.e. set) the title of the TNamed.

   fTitle = title;
   if (gPad && TestBit(kMustCleanup)) gPad->Modified();
}

//______________________________________________________________________________
Int_t TNamed::Sizeof() const
{
   // Return size of the TNamed part of the TObject.

   Int_t nbytes = fName.Sizeof() + fTitle.Sizeof();
   return nbytes;
}
