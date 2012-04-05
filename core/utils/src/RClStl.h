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

#include "Scanner.h"

namespace clang {
   class CXXRecordDecl;
}

namespace ROOT {

   class RStl {
   private:
      typedef std::set<RScanner::AnnotatedRecordDecl> list_t;
      list_t fList;

   public:
      static RStl& Instance();
      ~RStl() {};
      
      static std::string DropDefaultArg(const std::string &classname);
      void GenerateTClassFor(const char *requestedName, const clang::CXXRecordDecl *stlClass);
      void Print();
      void WriteClassInit(FILE *file);
      void WriteStreamer(FILE *file,const clang::CXXRecordDecl *stlcl);
      void WriteStreamer(FILE *file);
      
   private:
      RStl() : fList() {};
      RStl(const RStl&);
      RStl& operator=(const RStl&);
   };

}
#endif // R__RSTL_H
