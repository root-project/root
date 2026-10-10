/*
 * Project: RooFit
 * Authors:
 *   WV, Wouter Verkerke, NIKHEF, verkerke@nikhef.nl
 *
 * Copyright (c) 2000-2008, NIKHEF, Regents of the University of California
 *                          and Stanford University. All rights reserved.
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_CFunctionBinding_h
#define RooFit_Detail_CFunctionBinding_h

// Implementation details shared by the RooCFunction[1-4]Binding classes.

#include <TString.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace RooFit {
namespace Detail {

/// Name <-> pointer mapping service for C functions of type `Func`, used to
/// persist the RooCFunction[1-4]Ref function pointer references.
template <class Func>
class CFunctionMap {
public:
   /// Register function with given name and argument names. Arguments without
   /// a name are called x, y, z, w.
   template <class... ArgNames>
   void add(const char *name, Func ptr, ArgNames... argNames)
   {
      _ptrmap[name] = ptr;
      _namemap[ptr] = name;
      (_argnamemap[ptr].emplace_back(argNames), ...);
   }

   /// Return name of function given by pointer.
   const char *lookupName(Func ptr) { return _namemap[ptr].c_str(); }

   /// Return pointer of function given by name.
   Func lookupPtr(const char *name) { return _ptrmap[name]; }

   /// Return name of i-th argument of function. If no name was registered,
   /// the arguments are called x, y, z, w.
   const char *lookupArgName(Func ptr, UInt_t iarg)
   {
      if (iarg < _argnamemap[ptr].size()) {
         return (_argnamemap[ptr])[iarg].c_str();
      }
      switch (iarg) {
      case 0: return "x";
      case 1: return "y";
      case 2: return "z";
      case 3: return "w";
      }
      return "v";
   }

   /// Return registered name of a function. If the function is not
   /// registered, return a string with the hex representation of the pointer.
   const char *nameOrAddress(Func ptr)
   {
      const char *result = lookupName(ptr);
      if (result && strlen(result)) {
         return result;
      }
      return Form("(%p)", toVoidPtr(ptr));
   }

   static void *toVoidPtr(Func ptr)
   {
      // This union is to avoid a warning message:
      union {
         void *_ptr;
         Func _funcptr;
      } temp;
      temp._funcptr = ptr;
      return temp._ptr;
   }

private:
   std::map<std::string, Func> _ptrmap;                  // Name-to-pointer map
   std::map<Func, std::string> _namemap;                 // Pointer-to-name map
   std::map<Func, std::vector<std::string>> _argnamemap; // Pointer-to-argnamelist map
};

} // namespace Detail
} // namespace RooFit

#endif
