// @(#)root/reflex:$Id$
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_PluginService
#define Reflex_PluginService

#include "Reflex/Builder/NamespaceBuilder.h"
#include "Reflex/Builder/FunctionBuilder.h"
#include "Reflex/ValueObject.h"
#include "Reflex/Tools.h"

#include <string>
#include <sstream>

#define PLUGINSVC_FACTORY_NS "__pf__"

namespace Reflex {
class PluginFactoryMap;

/**
 * @class PluginService PluginService.h PluginService/PluginService.h
 * @author Pere Mato
 * @date 01/09/2006
 * @ingroup Ref
 */
class RFLX_API PluginService {
public:
   template <typename R>
   static R
   Create(const std::string& name) {
      return (R) Create(name, GetType<R>(), std::vector<ValueObject>());
   }


   template <typename R, typename A0>
   static R
   Create(const std::string& name,
          const A0& a0) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0)));
   }


   template <typename R, typename A0, typename A1>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1)));
   }


   template <typename R, typename A0, typename A1, typename A2>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2)));
   }


   template <typename R, typename A0, typename A1, typename A2, typename A3>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2,
          const A3& a3) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2),
                                          ValueObject::Create(a3)));
   }


   template <typename R, typename A0, typename A1, typename A2, typename A3,
             typename A4>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2,
          const A3& a3,
          const A4& a4) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2),
                                          ValueObject::Create(a3),
                                          ValueObject::Create(a4)));
   }


   template <typename R, typename A0, typename A1, typename A2, typename A3,
             typename A4, typename A5>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2,
          const A3& a3,
          const A4& a4,
          const A5& a5) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2),
                                          ValueObject::Create(a3),
                                          ValueObject::Create(a4),
                                          ValueObject::Create(a5)));
   }


   template <typename R, typename A0, typename A1, typename A2, typename A3,
             typename A4, typename A5, typename A6>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2,
          const A3& a3,
          const A4& a4,
          const A5& a5,
          const A6& a6) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2),
                                          ValueObject::Create(a3),
                                          ValueObject::Create(a4),
                                          ValueObject::Create(a5),
                                          ValueObject::Create(a6)));
   }


   template <typename R, typename A0, typename A1, typename A2, typename A3,
             typename A4, typename A5, typename A6, typename A7>
   static R
   Create(const std::string& name,
          const A0& a0,
          const A1& a1,
          const A2& a2,
          const A3& a3,
          const A4& a4,
          const A5& a5,
          const A6& a6,
          const A7& a7) {
      return (R) Create(name, GetType<R>(),
                        Tools::MakeVector(ValueObject::Create(a0),
                                          ValueObject::Create(a1),
                                          ValueObject::Create(a2),
                                          ValueObject::Create(a3),
                                          ValueObject::Create(a4),
                                          ValueObject::Create(a5),
                                          ValueObject::Create(a6),
                                          ValueObject::Create(a7)));
   }


   static void* Create(const std::string& name,
                       const Type& ret,
                       const std::vector<ValueObject>& arg);


   template <typename T>
   static bool
   CompareId(const Any& id1,
             const Any& id2) {
      try { return id1.TypeInfo() == id2.TypeInfo() &&
                   any_cast<T>(id1) == any_cast<T>(id2); }
      catch (const BadAnyCast&) { return false; }
   }


   template <typename T> static std::string
   StringId(const Any& id) {
      std::stringstream s;
      try { s << any_cast<T>(id); }
      catch (const BadAnyCast&) {}
      return s.str();
   }


   template <typename R, typename T>
   static R
   CreateWithId(const T& id) {
      return (R) CreateWithId(id, StringId<T>, CompareId<T>, GetType<R>(),
                              std::vector<ValueObject>());
   }


   template <typename R, typename T, typename A0>
   static R
   CreateWithId(const T& id,
                const A0& a0) {
      return (R) CreateWithId(id, StringId<T>, CompareId<T>, GetType<R>(),
                              Tools::MakeVector(ValueObject::Create(a0)));
   }


   template <typename R, typename T, typename A0, typename A1>
   static R
   CreateWithId(const T& id,
                const A0& a0,
                const A1& a1) {
      return (R) CreateWithId(id, StringId<T>, CompareId<T>, GetType<R>(),
                              Tools::MakeVector(ValueObject::Create(a0), ValueObject::Create(a1)));
   }


   template <typename R, typename T, typename A0, typename A1, typename A2>
   static R
   CreateWithId(const T& id,
                const A0& a0,
                const A1& a1,
                const A2& a2) {
      return (R) CreateWithId(id, StringId<T>, CompareId<T>, GetType<R>(),
                              Tools::MakeVector(ValueObject::Create(a0), ValueObject::Create(a1),
                                                ValueObject::Create(a2)));
   }


   template <typename R, typename T, typename A0, typename A1, typename A2, typename A3>
   static R
   CreateWithId(const T& id,
                const A0& a0,
                const A1& a1,
                const A2& a2,
                const A3& a3) {
      return (R) CreateWithId(id, StringId<T>, CompareId<T>, GetType<R>(),
                              Tools::MakeVector(ValueObject::Create(a0), ValueObject::Create(a1),
                                                ValueObject::Create(a2), ValueObject::Create(a3)));
   }


   static void* CreateWithId(const Any &id,
                             std::string (* str)(const Any&),
                             bool (* cmp)(const Any&,
                                          const Any&),
                             const Type &ret,
                             const std::vector<ValueObject> &arg);


   static std::string FactoryName(const std::string& n);

   static int Debug();

   static void SetDebug(int);

private:
   /** Constructor */
   PluginService();

   /** Destructor */
   ~PluginService();

   /** Get single instance of PluginService */
   static PluginService& Instance();

   int LoadFactoryLib(const std::string& name);

   int ReadMaps();

   int fDebugLevel;

   Scope fFactories;

   PluginFactoryMap* fFactoryMap;

};    // class PluginService

}  // namespace Reflex

//--- Factory stub functions for the different number of parameters
namespace {
template <typename T> struct Arg {
   static T
   Cast(void* a) { return *(T*) a; }

};

template <typename T> struct Arg<T*> {
   static T*
   Cast(void* a) { return (T*) a; }

};

template <typename P, typename S> class Factory;

template <typename P, typename R> class Factory<P, R(void)> {
public:
   static void
   Func(void* retaddr,
        void*,
        const std::vector<void*>& /* arg */,
        void*) {
      *(R*) retaddr = (R) ::new P;
   }


};

template <typename P, typename R, typename A0> class Factory<P, R(A0)> {
public:
   static void
   Func(void* retaddr,
        void*,
        const std::vector<void*>& arg,
        void*) {
      *(R*) retaddr = (R) ::new P(Arg<A0>::Cast(arg[0]));
   }


};

template <typename P, typename R, typename A0, typename A1> class Factory<P, R(A0, A1)> {
public:
   static void
   Func(void* retaddr,
        void*,
        const std::vector<void*>& arg,
        void*) {
      *(R*) retaddr = (R) ::new P(Arg<A0>::Cast(arg[0]), Arg<A1>::Cast(arg[1]));
   }


};

template <typename P, typename R, typename A0, typename A1, typename A2> class Factory<P, R(A0, A1, A2)> {
public:
   static void
   Func(void* retaddr,
        void*,
        const std::vector<void*>& arg,
        void*) {
      *(R*) retaddr = (R) ::new P(Arg<A0>::Cast(arg[0]), Arg<A1>::Cast(arg[1]), Arg<A2>::Cast(arg[2]));
   }


};

template <typename P, typename R, typename A0, typename A1, typename A2, typename A3> class Factory<P, R(A0, A1, A2, A3)> {
public:
   static void
   Func(void* retaddr,
        void*,
        const std::vector<void*>& arg,
        void*) {
      *(R*) retaddr = (R) ::new P(Arg<A0>::Cast(arg[0]), Arg<A1>::Cast(arg[1]), Arg<A2>::Cast(arg[2]), Arg<A3>::Cast(arg[3]));
   }


};

} // unnamed namespace


#define PLUGINSVC_CNAME(name, serial) name ## _dict ## serial
#define PLUGINSVC_FACTORY(type, signature) _PLUGINSVC_FACTORY(type, signature, __LINE__)
#define PLUGINSVC_FACTORY_WITH_ID(type, id, signature) _PLUGINSVC_FACTORY_WITH_ID(type, id, signature, __LINE__)

#define _PLUGINSVC_FACTORY(type, signature, serial) \
   namespace { \
   struct PLUGINSVC_CNAME (type, serial) { \
      PLUGINSVC_CNAME(type, serial) () { \
         std::string name = Reflex::GetType < type > ().Name(Reflex::SCOPED); \
         Reflex::Type sig = Reflex::FunctionDistiller < signature > ::Get(); \
         std::string fname = (std::string(PLUGINSVC_FACTORY_NS "::") + Reflex::PluginService::FactoryName(name)); \
         Reflex::FunctionBuilder(sig, fname.c_str(), \
                                 Factory < type, signature > ::Func, 0, "", Reflex::PUBLIC) \
         .AddProperty("name", name); \
         if (Reflex::PluginService::Debug()) { std::cout << "PluginService: Declared factory for class " << name << " with signature " << sig.Name() << std::endl; } \
      } \
   }; \
   PLUGINSVC_CNAME(type, serial) PLUGINSVC_CNAME(s_ ## type, serial); \
   }

#define _PLUGINSVC_FACTORY_WITH_ID(type, id, signature, serial) \
   namespace { \
   struct PLUGINSVC_CNAME (type, serial) { \
      PLUGINSVC_CNAME(type, serial) () { \
         std::string name = Reflex::GetType < type > ().Name(Reflex::SCOPED); \
         Reflex::Type sig = Reflex::FunctionDistiller < signature > ::Get(); \
         std::stringstream s; s << id; \
         std::string fname = (std::string(PLUGINSVC_FACTORY_NS "::") + Reflex::PluginService::FactoryName(s.str())); \
         Reflex::FunctionBuilder(sig, fname.c_str(), \
                                 Factory < type, signature > ::Func, 0, "", Reflex::PUBLIC) \
         .AddProperty("name", name) \
         .AddProperty("id", id); \
         if (Reflex::PluginService::Debug()) { std::cout << "PluginService: Declared factory for id " << fname << " with signature " << sig.Name() << std::endl; } \
      } \
   }; \
   PLUGINSVC_CNAME(type, serial) PLUGINSVC_CNAME(s_ ## type, serial); \
   }

#endif // Reflex_PluginService
