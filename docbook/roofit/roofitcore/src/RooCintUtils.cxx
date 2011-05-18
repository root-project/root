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

// #if ROOT_VERSION_CODE >= ROOT_VERSION(5,20,00)
// #include "cint/Api.h"
// #else
#include "Api.h"
// #endif

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

  pair<list<string>,unsigned int> ctorArgs(const char* classname, UInt_t nMinArg) 
  {
    // Utility function for RooFactoryWSTool. Return arguments of 'first' non-default, non-copy constructor of any RooAbsArg
    // derived class. Only constructors that start with two 'const char*' arguments (for name and title) are considered
    // The returned object contains 

    Int_t nreq(0) ;
    list<string> ret ;
    
    G__ClassInfo cls(classname);
    G__MethodInfo func(cls);
    while(func.Next()) {
      ret.clear() ;
      nreq=0 ;
      
      // Find 'the' constructor
      
      // Skip non-public methods
      if (!(func.Property()&G__BIT_ISPUBLIC)) {
	continue ;
      }	
      
      // Return type must be class name
      if (string(classname)!= func.Type()->Name()) {
	continue ;
      }
      
      // Skip default constructor
      if (func.NArg()==0 || func.NArg()==func.NDefaultArg()) {
	continue ;
      }
      
      // Skip copy constructor
      G__MethodArgInfo arg1(func);
      arg1.Next() ;	
      string tmp(Form("const %s&",classname)) ;
      if (tmp==arg1.Type()->Name()) {
	continue ;
      }
      
      // Examine definition of remaining ctor
      G__MethodArgInfo arg(func);

      
      // Require that first to arguments are of type const char*
      while(arg.Next()) {
	if (nreq<2 && string("const char*") != arg.Type()->Name()) {	  
	  continue ;
	}
	ret.push_back(arg.Type()->Name()) ;
	if(!arg.DefaultValue()) nreq++ ;
      }

      // Check that the number of required arguments is at least nMinArg
      if (ret.size()<nMinArg) {
	continue ;
      }

      return pair<list<string>,unsigned int>(ret,nreq) ;
      
    }
    return pair<list<string>,unsigned int>(ret,nreq) ;
  }


  Bool_t isEnum(const char* classname) 
  {
    // Returns true if given type is an enum
    G__ClassInfo cls(classname);
    long property = cls.Property();
    return (property&G__BIT_ISENUM) ;    
  }


  Bool_t isValidEnumValue(const char* typeName, const char* value) 
  {
    // Returns true if given type is an enum

    // Chop type name into class name and enum name
    char buf[256] ;
    strlcpy(buf,typeName,256) ;
    char* className = strtok(buf,":") ;

    // Chop any class name prefix from value
    if (strrchr(value,':')) {
      value = strrchr(value,':')+1 ;
    }

    G__ClassInfo cls(className);
    G__DataMemberInfo dm(cls);
    while (dm.Next()) {
      // Check if this data member represents an enum value
      if (string(Form("const %s",typeName))==dm.Type()->Name()) {
	if (string(value)==dm.Name()) {
	  return kTRUE ;
	}
      }
    }
    return kFALSE ;
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
    strlcpy(buf,fname,256) ;
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
      if (string(methodName?methodName:"")==method.Name()) {
	
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


Bool_t RooCintUtils::isTypeDef(const char* trueName, const char* aliasName)
{
  // Returns true if aliasName is a typedef for trueName
  G__TypedefInfo t;
  while(t.Next()) {
    if (string(trueName)==t.TrueName() && string(aliasName)==t.Name()) return kTRUE ;
  }
  return kFALSE ;
}


std::string RooCintUtils::trueName(const char* aliasName) 
{
  // Returns the true type for a given typedef name.
  G__TypedefInfo t;
  while(t.Next()) {
    if (string(aliasName)==t.Name()) {      
      return trueName(string(t.TrueName()).c_str()) ;
    }
  }
  return string(aliasName) ;
}

