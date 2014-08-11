// @(#)root/io:$Id$
// Author: Philippe Canal July, 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
  *************************************************************************/

#ifndef ROOT_TVirtualArray
#define ROOT_TVirtualArray


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualArray                                                       //
//                                                                      //
// Wrapper around an object and giving indirect access to its content    //
// even if the object is not of a class in the Cint/Reflex dictionary.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"

class TVirtualArray {
public:
   TClassRef  fClass;
   UInt_t     fCapacity;
   UInt_t     fSize;
   char      *fArray; //[fSize]

   TVirtualArray( TClass *cl, UInt_t size ) : fClass(cl), fCapacity(size), fSize(size), fArray( (char*)( cl ? cl->NewArray(size) : 0) ) {};
   ~TVirtualArray() { if (fClass) fClass->DeleteArray( fArray ); }

   TClass *GetClass() { return fClass; }
   char *operator[](UInt_t ind) const { return GetObjectAt(ind); }
   char *GetObjectAt(UInt_t ind) const { return fArray+fClass->Size()*ind; }

   void SetSize(UInt_t size) {
      // Set the used size of this array to 'size'.   If size is greater than the existing
      // capacity, reallocate the array BUT no data is preserved.
      fSize = size;
      if (fSize > fCapacity && fClass) {
         fClass->DeleteArray( fArray );
         fArray = (char*)fClass->NewArray(fSize);
         fCapacity = fSize;
      }
   }


};

#endif // ROOT_TVirtualArray
