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

#ifndef R__HAS_CLING
#include "Api.h"
#endif

#include "RooMsgService.h"
#include "TInterpreter.h"
#include <string.h>
#include <string>
#include <iostream>

using namespace std ;


namespace RooCintUtils 
{

#ifndef R__HAS_CLING
  const char* functionName(void* func) 
  {
    return G__p2f2funcname(func);
  }
#endif

  pair<list<string>,unsigned int> ctorArgs(const char* classname, UInt_t nMinArg) 
  {
    // Utility function for RooFactoryWSTool. Return arguments of 'first' non-default, non-copy constructor of any RooAbsArg
    // derived class. Only constructors that start with two 'const char*' arguments (for name and title) are considered
    // The returned object contains 

    Int_t nreq(0) ;
    list<string> ret ;
    
    ClassInfo_t* cls = gInterpreter->ClassInfo_Factory(classname);
    MethodInfo_t* func = gInterpreter->MethodInfo_Factory(cls);
    while(gInterpreter->MethodInfo_Next(func)) {
      ret.clear() ;
      nreq=0 ;
      
      // Find 'the' constructor
      
      // Skip non-public methods
      if (!(gInterpreter->MethodInfo_Property(func) & kIsPublic)) {
	continue ;
      }	
      
      // Return type must be class name
      if (string(classname) != gInterpreter->MethodInfo_TypeName(func)) {
	continue ;
      }
      
      // Skip default constructor
      int nargs = gInterpreter->MethodInfo_NArg(func);
      if (nargs==0 || nargs==gInterpreter->MethodInfo_NDefaultArg(func)) {
	continue ;
      }
      
      MethodArgInfo_t* arg = gInterpreter->MethodArgInfo_Factory(func);
      while (gInterpreter->MethodArgInfo_Next(arg)) {
        // Require that first to arguments are of type const char*
         const char* argTypeName = gInterpreter->MethodArgInfo_TypeName(arg);
        if (nreq<2 && string("const char *") != argTypeName) {
	  continue ;
	}
	ret.push_back(argTypeName) ;
	if(!gInterpreter->MethodArgInfo_DefaultValue(arg)) nreq++ ;
      }
      gInterpreter->MethodArgInfo_Delete(arg);

      // Check that the number of required arguments is at least nMinArg
      if (ret.size()<nMinArg) {
	continue ;
      }

      break;
    }
    gInterpreter->MethodInfo_Delete(func);
    gInterpreter->ClassInfo_Delete(cls);
    return pair<list<string>,unsigned int>(ret,nreq) ;
  }


  Bool_t isEnum(const char* classname) 
  {
    // Returns true if given type is an enum
    ClassInfo_t* cls = gInterpreter->ClassInfo_Factory(classname);
    long property = gInterpreter->ClassInfo_Property(cls);
    gInterpreter->ClassInfo_Delete(cls);
    return (property&kIsEnum) ;    
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

    ClassInfo_t* cls = gInterpreter->ClassInfo_Factory(className);
    DataMemberInfo_t* dm = gInterpreter->DataMemberInfo_Factory(cls);
    while (gInterpreter->DataMemberInfo_Next(dm)) {
      // Check if this data member represents an enum value
      if (string(Form("const %s",typeName))==gInterpreter->DataMemberInfo_TypeName(dm)) {
	if (string(value)==gInterpreter->DataMemberInfo_Name(dm)) {
          gInterpreter->DataMemberInfo_Delete(dm);
          gInterpreter->ClassInfo_Delete(cls);
	  return kTRUE ;
	}
      }
    }
    gInterpreter->DataMemberInfo_Delete(dm);
    gInterpreter->ClassInfo_Delete(cls);
    return kFALSE ;
  }
  
#ifndef R__HAS_CLING
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
    ClassInfo_t* scope = gInterpreter->ClassInfo_Factory(scopeName);
    
    // Loop over all methods in scope
    MethodInfo_t* method = gInterpreter->MethodInfo_Factory(scope);
    while(gInterpreter->MethodInfo_Next(method)) {
      // If method name matches, check argument list
      if (string(methodName?methodName:"")==gInterpreter->MethodInfo_Name(method)) {
	
	// Construct list of arguments
	string s ;
	MethodArgInfo_t* arg = gInterpreter->MethodArgInfo_Factory(method);
      while(gInterpreter->MethodArgInfo_Next(arg)) {
	if (s.length()>0) s += "," ;
        s += gInterpreter->MethodArgInfo_TrueTypeName(arg) ;
      }      
      
      gInterpreter->MethodArgInfo_Delete(arg);
      if (s==args) {
        gInterpreter->ClassInfo_Delete(scope);
        gInterpreter->MethodInfo_Delete(method);
	return kTRUE ;
      }
      }
    }
    
    gInterpreter->ClassInfo_Delete(scope);
    gInterpreter->MethodInfo_Delete(method);
    // Fill s with comma separate list of methods true argument names
    return kFALSE ;
  }
  
#endif

}

Bool_t RooCintUtils::isTypeDef(const char* trueName, const char* aliasName)
{
  // Returns true if aliasName is a typedef for trueName
  TypedefInfo_t* t = gInterpreter->TypedefInfo_Factory();
  while(gInterpreter->TypedefInfo_Next(t)) {
    if (string(trueName)==gInterpreter->TypedefInfo_TrueName(t)
        && string(aliasName)==gInterpreter->TypedefInfo_Name(t)) {
      gInterpreter->TypedefInfo_Delete(t);
      return kTRUE ;
    }
  }
  gInterpreter->TypedefInfo_Delete(t);
  return kFALSE ;
}


std::string RooCintUtils::trueName(const char* aliasName) 
{
  // Returns the true type for a given typedef name.
  TypedefInfo_t* t = gInterpreter->TypedefInfo_Factory();
  while(gInterpreter->TypedefInfo_Next(t)) {
    if (string(aliasName)==gInterpreter->TypedefInfo_Name(t)) {      
      std::string ret = trueName(string(gInterpreter->TypedefInfo_TrueName(t)).c_str()) ;
      gInterpreter->TypedefInfo_Delete(t);
      return ret;
    }
  }
  gInterpreter->TypedefInfo_Delete(t);
  return string(aliasName) ;
}

