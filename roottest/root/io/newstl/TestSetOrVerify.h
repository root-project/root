#ifndef TESTSETORVERIFY_H
#define TESTSETORVERIFY_H

#include "TestFill.h"
#include "TestEquiv.h"

#define VERIFY(X)                                  \
  bool Verify##X (Int_t entryNumber,               \
                  const std::string &testname,     \
                  Int_t splitlevel)                \
  {                                                \
     return SetOrVerify##X (entryNumber,false,     \
                            testname,splitlevel);  \
  }


namespace utility {
  template <class T> bool SetOrVerify(const char *dataname,
                                       T& datamember,
                                       Int_t seed,
                                       Int_t entryNumber, 
                                       bool reset, 
                                       const std::string &testname) {
      bool result = true;

      if (reset) {
         if (DebugTest()&TestDebug::kAddresses) {
            std::stringstream s;
            s << testname << " address of " << dataname << " is " << &datamember; // << std::ends;
            Debug(s.str());
         }
         fill(datamember, seed);
      } else {
         T build;
         fill(build, seed);
         std::stringstream s;
         s << testname << " verify " << dataname << " entry #" <<  entryNumber; // << std::ends;
         result = IsEquiv(s.str(), build, datamember);
      }
      return result;      
   }

  template <class T> bool SetOrVerifyDerived(const char *dataname,
                                             T& datamember,
                                             Int_t seed,
                                             Int_t entryNumber, 
                                             bool reset, 
                                             const std::string &testname) {
      bool result = true;

      if (reset) {
         if (DebugTest()&TestDebug::kAddresses) {
            std::stringstream s;
            s << testname << " address of " << dataname << " is " << &datamember; // << std::ends;
            Debug(s.str());
         }
         fillDerived(datamember, seed);
      } else {
         T build;
         fillDerived(build, seed);
         std::stringstream s;
         s << testname << " verify " << dataname << " entry #" <<  entryNumber; // << std::ends;
         result = IsEquiv(s.str(), build, datamember);
      }
      return result;      
   }

   template <class T> bool SetOrVerify(const char *dataname,
                                       T* &datamember,
                                       Int_t seed,
                                       Int_t entryNumber, 
                                       bool reset, 
                                       const std::string &testname) {
      bool result = true;

      if (reset) {
         delete datamember;
         datamember = new T;
         fill(*datamember, seed);

         if (DebugTest()&TestDebug::kAddresses) {
            std::stringstream s;
            s << testname << " address of " << dataname << " is " << &datamember << " and new value is " << datamember; // << std::ends;
            Debug(s.str());
         }
      } else {
         T build;
         fill(build, seed);
         std::stringstream s;
         s << testname << " verify " << dataname << " entry #" <<  entryNumber; // << std::ends;
         result = IsEquiv(s.str(), &build, datamember);
      }
      return result;      
   }

   template <class T> bool SetOrVerify(const char *dataname,
                                       T* const datamember,
                                       UInt_t arraysize,
                                       Int_t seed,
                                       Int_t entryNumber, 
                                       bool reset, 
                                       const std::string &testname) {
      bool result = true;

      if (reset && DebugTest()&TestDebug::kAddresses) {
         std::stringstream s;
         s << testname << " address of " << dataname << " value is " << datamember; // << std::ends;
         Debug(s.str());
      }

      for(UInt_t index=0; index<arraysize; index++) {
         // Int_t seed = 3 * (entryNumber+1);
         if (reset) {
            fill(datamember[index], seed);
         } else {
            T build;
            fill(build, seed);
            std::stringstream s;
            s << testname << " verify " << dataname << " entry #" << entryNumber << " index #" << index; // << std::ends;
            result = IsEquiv(s.str(), build, datamember[index]);
         }
      }
      return result;      
   }

   template <class T> bool SetOrVerify(const char *dataname,
                                       T** const datamember,
                                       UInt_t arraysize,
                                       Int_t seed,
                                       Int_t entryNumber, 
                                       bool reset, 
                                       const std::string &testname) {
      bool result = true;

      if (reset && DebugTest()&TestDebug::kAddresses) {
         std::stringstream s;
         s << testname << " address of " << dataname << " value is " << datamember; // << std::ends;
         Debug(s.str());
      }

      for(UInt_t index=0; index<arraysize; index++) {
         // Int_t seed = 3 * (entryNumber+1);
         if (reset) {
            delete datamember[index];
            datamember[index] = new T;
            fill(*(datamember[index]), seed);
         } else {
            T build;
            fill(build, seed);
            std::stringstream s;
            s << testname << " verify " << dataname << " entry #" << entryNumber << " index #" << index; // << std::ends;
            result = IsEquiv(s.str(), build, *(datamember[index]));
         }
      }
      return result;      
   }

   template <class T, class Q> bool SetOrVerifyArrVar(const char *dataname,
                                                      T* &datamember,
                                                      Q &arraySize,
                                                      Int_t seed,
                                                      Int_t entryNumber, 
                                                      bool reset, 
                                                      const std::string &testname) {
      Q calcSize = seed%5;
      if (reset) {
         arraySize = calcSize; 
         delete [] datamember;
         datamember = new T[arraySize];
      } else {
         TClass *cl = gROOT->GetClass(typeid(T));
         const char* classname = cl?cl->GetName():typeid(T).name();
         if ( !IsEquiv(testname, calcSize,arraySize) ) {
            TestError(testname,Form("For %s, wrong size for a variable array! Wrote %d and read %d\n",classname,calcSize,arraySize));
         }
      }

      return SetOrVerify(dataname, datamember, arraySize, seed, entryNumber, reset, testname);
   }

}

#endif
