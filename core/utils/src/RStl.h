// @(#)root/cont:$Id$
// Author: Philippe Canal 20/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef R__RSTL_H
#define R__RSTL_H

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RStl                                                                 //
//                                                                      //
// Use to manage the code that needs to be generated for the STL        //
// by rootcint.  This class is reserved for rootcint and is a           //
// singleton.                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <set>

namespace std {} using namespace std;

namespace ROOT {

   class RStl {
   private:
      set<string> fList;

   public:
      static RStl& Instance();
      ~RStl() {};
      
      static string DropDefaultArg(const string &classname);
      void GenerateTClassFor(const string& stlClassname);
      void Print();
      void WriteClassInit(FILE *file);
      void WriteStreamer(FILE *file, G__ClassInfo &stlcl);
      void WriteStreamer(FILE *file);
      
   private:
      RStl() : fList() {};
      RStl(const RStl&);
      RStl& operator=(const RStl&);
   };

}
#endif // R__RSTL_H
