// @(#)root/meta:$Id$
// Author: Axel Naumann, 2011-10-19

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TCintWithCling
#define ROOT_TCintWithCling

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCintWithCling                                                       //
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto of HP Japan.                                        //
//                                                                      //
// CINT is an almost full ANSI compliant C/C++ interpreter.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TCint
#include "TCint.h"
#endif

namespace cling {
   class Interpreter;
   class MetaProcessor;
}

class TCintWithCling : public TCint {

private:
   cling::Interpreter* fInterpreter;// cling
   cling::MetaProcessor* fMetaProcessor;//cling's command processor

   TCintWithCling() :
      TCint("", ""),
      fInterpreter(0), fMetaProcessor(0)
   { }  //for Dictionary() only
   TCintWithCling(const TCintWithCling&);             // not implemented
   TCintWithCling &operator=(const TCintWithCling&);  // not implemented

public:
   TCintWithCling(const char *name, const char *title);
   virtual ~TCintWithCling();

   void    AddIncludePath(const char *path);
   Long_t  ProcessLine(const char *line, EErrorCode *error = 0);
   void    PrintIntro();
   void    InspectMembers(TMemberInspector&, void* obj, const char* clname);

   ClassDef(TCintWithCling,0)  //Interface to CINT C/C++ interpreter
};

#endif
