// @(#)root/base:$Name:  $:$Id: String.cxx,v 1.18 2002/12/05 15:31:03 rdm Exp $
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

#include <string>
#include "TBuffer.h"
#include "TString.h"

namespace std {} using namespace std;

void std_string_streamer(TBuffer &b, void *objadd) 
{
   // Streamer function for std::string object.

   string *obj = (string*)objadd;
   
   if (b.IsReading()) {

      TString helper;
      helper.Streamer(b);
      (*obj) = helper.Data();
      
   } else {

      TString helper( obj ? obj->c_str() : "" );
      helper.Streamer(b);

   }

}


// Declare the streamer to the string TClass object
RootStreamer(string,std_string_streamer);

// Set a version number of the string TClass object
RootClassVersion(string,2);


