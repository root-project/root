/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

///////////////////////////////////////////////////////////////////////////////
// RooCintUtils is a namespace containing utility functions related 
// to CINT interfacing
//

#include "RooFit.h"

#include "RooCintUtils.h"

#ifdef R__BUILDING_CINT7
# include "cint7/Api.h"
#else
# include "Api.h"
#endif

#include "RooMsgService.h"
#include <string.h>
#include <string>
#include <iostream>

using namespace std ;


namespace RooCintUtils 
{

  const char* functionName(void* func) 
  {
    return G__p2f2funcname(func);
  }
  
  Bool_t matchFuncPtrArgs(void* func, const char* args) 
  {
    // Returns TRUE if given pointer to function takes true arguments as listed in string args
    
    // Retrieve CINT name of function
    const char* fname=G__p2f2funcname(func);
    if (!fname) {
      oocoutE((TObject*)0,InputArguments) << "bindFunction::ERROR CINT cannot resolve name of function " << func << endl ;
      return kFALSE ;
    }
    
    // Seperate namespace part from method name
    char buf[1024] ;
    strcpy(buf,fname) ;
    const char* methodName(0), *scopeName = buf ;
    for(int i=strlen(buf)-1 ; i>0 ; i--) {
      if (buf[i]==':' && buf[i-1]==':') {
	buf[i-1] = 0 ;
	methodName = buf+i+1 ;
	break ;
      }
    }
    
    // Get info on scope
    G__ClassInfo scope(scopeName);
    
    // Loop over all methods in scope
    G__MethodInfo method(scope);
    while(method.Next()) {
      // If method name matches, check argument list
      if (string(methodName)==method.Name()) {
	
	// Construct list of arguments
	string s ;
	G__MethodArgInfo arg(method);
      while(arg.Next()) {
	if (s.length()>0) s += "," ;
	s += arg.Type()->TrueName() ;
      }      
      
      if (s==args) {
	return kTRUE ;
      }
      }
    }
    
    // Fill s with comma separate list of methods true argument names
    return kFALSE ;
  }
  

}
