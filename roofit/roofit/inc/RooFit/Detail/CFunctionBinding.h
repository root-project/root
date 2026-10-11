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

#include <RooAbsArg.h>
#include <RooAbsProxy.h>
#include <RooMsgService.h>

#include <TBuffer.h>
#include <TString.h>

#include <cstring>
#include <map>
#include <ostream>
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

/// Custom streamer for function pointer reference objects. When writing,
/// the function pointer is substituted by its registered name. When function
/// is unregistered name 'UNKNOWN' is written and a warning is issued. When
/// reading back, the embedded name is converted back to a function pointer
/// using the mapping service. When name UNKNOWN is encountered a warning is
/// issued and a dummy null function is substituted. When the registered function
/// name can not be mapped to a function pointer an ERROR is issued.
template <class Func>
void streamCFunctionRef(TBuffer &R__b, TObject const &self, Func &ptr, Func dummyFunction, CFunctionMap<Func> &fmap)
{
   if (R__b.IsReading()) {

      UInt_t R__s;
      UInt_t R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

      // Read name from file
      TString tmpName;
      tmpName.Streamer(R__b);

      if (tmpName == "UNKNOWN" && R__v > 0) {

         oocoutW(&self, ObjectHandling)
            << "WARNING: Objected embeds function pointer to unknown function, object will not be functional"
            << std::endl;
         ptr = dummyFunction;

      } else {

         // Lookup pointer to C function with given name
         ptr = fmap.lookupPtr(tmpName.Data());

         if (ptr == nullptr) {
            oocoutW(&self, ObjectHandling)
               << "ERROR: Objected embeds pointer to function named " << tmpName
               << " but no such function is registered, object will not be functional" << std::endl;
         }
      }

      R__b.CheckByteCount(R__s, R__c, self.IsA());

   } else {

      UInt_t R__c;
      R__c = R__b.WriteVersion(self.IsA(), true);

      // Lookup name of reference C function
      TString tmpName = fmap.lookupName(ptr);
      if (tmpName.Length() == 0) {
         oocoutW(&self, ObjectHandling) << "WARNING: Cannot persist unknown function pointer "
                                        << Form("%p", CFunctionMap<Func>::toVoidPtr(ptr))
                                        << " written object will not be functional when read back" << std::endl;
         tmpName = "UNKNOWN";
      }

      // Persist the name
      tmpName.Streamer(R__b);

      R__b.SetByteCount(R__c, true);
   }
}

/// Print arguments of a function binding and name/address of the function pointer.
inline void printCFunctionArgs(std::ostream &os, RooAbsArg const &arg, const char *funcName)
{
   os << "[ function=" << funcName << " ";
   for (Int_t i = 0; i < arg.numProxies(); i++) {
      RooAbsProxy *p = arg.getProxy(i);
      if (!TString(p->name()).BeginsWith("!")) {
         p->print(os);
         os << " ";
      }
   }
   os << "]";
}

} // namespace Detail
} // namespace RooFit

#endif
