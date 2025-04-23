//--------------------------------------------------------------------*- C++ -*-
// file:   Demangler.h
// date:   23.05.2008
// author: Lukasz Janyst <ljanyst@cern.ch>
//------------------------------------------------------------------------------

#ifndef DEMANGLER_H
#define DEMANGLER_H


#include <typeinfo>
#include <string>

#if !defined(_WIN32) && !defined(__CINT__)
#include <cxxabi.h>
#endif

//------------------------------------------------------------------------------
// Demangle the C++ name
//------------------------------------------------------------------------------
std::string getName( const std::type_info &t )
{
#ifndef _WIN32
   int   status    = 0;
   char* demangled = abi::__cxa_demangle(t.name(), 0, 0, &status);
   if( status )
      return "";
   std :: string tr = demangled;
   free( demangled );
   return tr;
#else
   return t.name();
#endif
}

#endif // DEMANGLER_H
