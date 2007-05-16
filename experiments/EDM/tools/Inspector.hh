#ifndef INCLUDE_INSPECTOR_H
#define INCLUDE_INSPECTOR_H

#include "Reflex/Scope.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"
#include "TRandom3.h"
#include <map>
#include <limits>
#include <string>

namespace Inspect {
   typedef std::map<ROOT::Reflex::Member, void*> ResultMap_t;

   class Random {
   public:
      Random() {}

      void SetSeed(UInt_t seed = 0) {
         fgRndGen.SetSeed(seed);
         std::cout << "Inspect::Random: seed initialized to " 
                   << fgRndGen.GetSeed() << "." << std::endl;
	 fgInit = true;
      }

      template <typename T>
         T operator()(const ROOT::Reflex::Member&, T& v) {
         if (!fgInit) SetSeed(86914603225886822);
         const double r = fgRndGen.Rndm();
         v = (T) (r * (Max(v)) + (1. - r) * Min(v));
         return v;
      }

   private:
      template <typename T>
         T Min(const T&) const { return std::numeric_limits<T>::min(); }
      template <typename T>
         T Max(const T&) const { return std::numeric_limits<T>::max(); }
      
      static bool fgInit;
      static TRandom3 fgRndGen;
   };

   class InspectorBase {
   public:
      InspectorBase() {}
      InspectorBase(const ROOT::Reflex::Scope& scope): fScope(scope), fActual(0) {};
      InspectorBase(const ROOT::Reflex::Scope& scope,
                    const ResultMap_t& expected): 
         fScope(scope), fExpected(expected), fActual(0) {};
      virtual ~InspectorBase() {}

      const ROOT::Reflex::Scope& GetScope() const { return fScope; }

      void SetExpectedResults(const ResultMap_t& expected) 
      { fExpected = expected; }

      bool AsExpected() const;
      virtual void Inspect() {}
      const ResultMap_t* GetExpectedResults() const { return &fExpected; }
      const ResultMap_t* GetActualResults() const { return fActual; }

      virtual void InitDataMembers() {}
      //protected:
      //void CallFunctionMemberss();
      //void ReadDataMembers();
      bool IsEqual(const ROOT::Reflex::Type& type, const ROOT::Reflex::Member& member,
                   const void* lhs, const void* rhs) const;
      void Dump(std::ostream& out, const ROOT::Reflex::Type& type,
                const void* v, const char* indent = 0) const;
      void Dump(std::ostream& out, const ROOT::Reflex::Member& member,
                const void* v, const char* indent = 0) const;

   private:
      ROOT::Reflex::Scope fScope;
      ResultMap_t fExpected;
      ResultMap_t* fActual;
   };


   class InspectorGenSource: public InspectorBase {
   public:
      typedef InspectorBase* (*InspectorGeneratorFunc_t)(void*);

      InspectorGenSource(const ROOT::Reflex::Scope& scope, const char* header);
      ~InspectorGenSource() {};

      const std::string& GetName() const { return fName; }

      void WriteSource();
      static void RegisterInspector(const ROOT::Reflex::Scope& scope,
                                    InspectorGeneratorFunc_t initfunc) {
         fgInspectors[scope] = initfunc; }
      static InspectorBase* GetInspector(const ROOT::Reflex::Scope& scope, void* obj) {
         InspectorGeneratorFunc_t initfunc = fgInspectors[scope];
         if (!initfunc) return 0;
         return (*initfunc)(obj);
      }
   private:
      std::string fName;
      std::string fHeader;

      static std::map<ROOT::Reflex::Scope, InspectorGeneratorFunc_t> fgInspectors;
   };

}

#endif // INCLUDE_INSPECTOR_H
