// @(#)root/base:$Id$
// Author: Philippe Canal 03/09/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// std::string helper utilities                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfig.h"
#include <string>
#include "TBuffer.h"
#include "TString.h"

namespace std {} using namespace std;

void std_string_streamer(TBuffer &b, void *objadd) 
{
   // Streamer function for std::string object.
   string *obj = (string*)objadd;
   Int_t   nbig;
   UChar_t nwh;
   if (b.IsReading()) {
      b >> nwh;
      ;
      if (nwh == 0)  {
         obj->clear();
      } else {
         if( obj->size() ) {
            // Insure that the underlying data storage is not shared
            (*obj)[0] = '\0';
         }
         if (nwh == 255)  {
            b >> nbig;
            obj->resize(nbig,'\0');
            b.ReadFastArray((char*)obj->data(),nbig);
         }
         else  {
            obj->resize(nwh,'\0');
            b.ReadFastArray((char*)obj->data(),nwh);
         }
      }
   } else if ( obj ) {
      nbig = obj->length();
      if (nbig > 254) {
         nwh = 255;
         b << nwh;
         b << nbig;
      } else {
         nwh = UChar_t(nbig);
         b << nwh;
      }
      b.WriteFastArray(obj->data(),nbig);
   }
}

// Declare the streamer to the string TClass object
RootStreamer(string,std_string_streamer);

// Set a version number of the string TClass object
RootClassVersion(string,2);


