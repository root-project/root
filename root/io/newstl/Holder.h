#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TClass.h"
#include "TError.h"
#include "TObject.h"
#include "TROOT.h"
#include "TMath.h"
#endif

#include <vector>

#include <iostream>
#include <sstream>

class Helper {
public:
   unsigned int val;
   Helper() : val(0) {};
   explicit Helper(int v) : val(v) {};
   //bool operator==(const Helper &rhs) const { return val==rhs.val; }
   //bool operator!=(const Helper &rhs) const { return !(*this==rhs); }
   bool IsEquiv(const Helper &rhs) const { return  val==rhs.val; }
};

#include "TestOutput.h"
#include "versions.h"

#define VERIFY(X)                                  \
  bool Verify##X (Int_t entryNumber,               \
                  const std::string &testname,     \
                  Int_t splitlevel)                \
  {                                                \
     return SetOrVerify##X (entryNumber,false,     \
                            testname,splitlevel);  \
  }

//void TestError(const char *msg);

template <class T> void fill(T& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      typename T::value_type val(seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> void fill(std::vector<T*>& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      T* val = new T(seed*10+i);
      filled.push_back(val);
  }  
}

bool IsEquiv(const std::string &, const Helper &orig, const Helper &copy) { return  orig.IsEquiv(copy); }

template <class T> bool IsEquiv(const std::string &test, T* orig, T* copy) {
   TClass *cl = gROOT->GetClass(typeid(T));
   const char* classname = cl?cl->GetName():typeid(T).name();

   if ( (orig==0 && copy) || (orig && copy==0) ) {
      TestError(test,Form("For %s, non-initialized pointer %p %p",classname,orig,copy));
      return false;
   }
   return IsEquiv(test, *orig, *copy);
}

bool IsEquiv(const std::string &, float orig, float copy) {
   float epsilon = 1e-6;
   float diff = orig-copy;
   return TMath::Abs( diff/copy ) < epsilon;
}

template <class T> bool IsEquiv(const std::string &test, const T& orig, const T& copy) {
   TClass *cl = gROOT->GetClass(typeid(T));
   const char* classname = cl?cl->GetName():typeid(T).name();

   if (orig.size() != copy.size()) {
      TestError(test,Form("For %s, wrong size! Wrote %d and read %d\n",classname,orig.size(),copy.size()));
      return false;
   }

   bool result = true;
   typename T::const_iterator iorig = orig.begin();
   typename T::const_iterator icopy = copy.begin();
   UInt_t i = 0;
   while ( iorig != orig.end() && icopy != copy.end() ) {
      if (!IsEquiv(test,*iorig,*icopy)) {
         TestError(test, Form("for %s:\nelem #%d are not equal",
                              classname,i));
         TestError(test,*iorig,*icopy);
         result = false;
      } else {
         //std::string notest("NOT a failure, ");
         //notest += test;
         //TestError(notest,*iorig,*icopy);
      }
      i++;
      iorig++;
      icopy++;
   }
   return result;
}


bool IsEquiv(const std::string &, int orig, int copy) {
   return orig==copy;
}

class vectorHolder : public TObject {

   template <class T> bool SetOrVerify(const char *dataname,
                                       T& datamember,
                                       Int_t seed,
                                       Int_t entryNumber, 
                                       bool reset, 
                                       const std::string &testname) {
      bool result = true;

      // Int_t seed = 3 * (entryNumber+1);
      if (reset) {
         fill(datamember, seed);
      } else {
         T build;
         fill(build, seed);
         std::stringstream s;
         s << testname << " verify " << dataname << " entry #" <<  entryNumber << std::ends;
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

      // Int_t seed = 3 * (entryNumber+1);
      if (reset) {
         delete datamember;
         datamember = new T;
         fill(*datamember, seed);
      } else {
         T build;
         fill(build, seed);
         std::stringstream s;
         s << testname << " verify " << dataname << " entry #" <<  entryNumber << std::ends;
         result = IsEquiv(s.str(), &build, datamember);
      }
      return result;      
   }

public:   

   vectorHolder() : TObject()
      ,fScalarPtr(0)
      ,fScalarArrVar(0)
      ,fObjectPtr(0)
      ,fPtrObjectPtr(0)
      {}

   explicit vectorHolder(Int_t entry) : TObject()
      ,fScalarPtr(0)
      ,fScalarArrVar(0)
      ,fObjectPtr(0)
      ,fPtrObjectPtr(0)
      {
         Reset(entry);
      }


   std::vector<int >    *fScalarPtr;
   std::vector<float >   fScalar;
   std::vector<short >   fScalarArr[2];
   Int_t                 fScalarArrVarSize;
   std::vector<char >   *fScalarArrVar; //[fScalarArrVarSize]
   std::vector<Helper >  fObject;
   std::vector<Helper > *fObjectPtr;
   std::vector<Helper* > fPtrObject;
   std::vector<Helper* >*fPtrObjectPtr;

   typedef std::vector<Helper > nested_t;
   typedef std::vector<nested_t > nesting_t;
#if defined(R__NO_NESTED_CONTAINER)
   nesting_t fNested;  //! this version of ROOT does not support nested container
#else
   nesting_t fNested;  //
#endif

   bool SetOrVerifyScalar(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      Int_t seed = 3 * (entryNumber+1);
      return SetOrVerify("fScalar",fScalar,seed,entryNumber,reset,testname);
   }
   VERIFY(Scalar);
   
   bool SetOrVerifyScalarPtr(Int_t entryNumber, bool reset, const std::string &testname, int splitlevel) {
      Int_t seed = 4 * (entryNumber+1);
      return SetOrVerify("fScalarPtr",fScalarPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarPtr);

   bool SetOrVerifyObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return SetOrVerify("fObject",fObject,seed,entryNumber,reset,testname);
   }
   VERIFY(Object);

   bool SetOrVerifyObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return SetOrVerify("fObjectPtr",fObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ObjectPtr);

   bool SetOrVerifyPtrObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 4 * (entryNumber+1);
      return SetOrVerify("fPtrObject",fPtrObject,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObject);

   bool SetOrVerifyPtrObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 5 * (entryNumber+1);
      return SetOrVerify("fPtrObjectPtr",fPtrObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObjectPtr);

   bool SetOrVerifyNested(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasNestedContainer(gFile)) {
         return true;
      }
      UInt_t seed = 1 * (entryNumber+1);
      return SetOrVerify("fNested",fNested,seed,entryNumber,reset,testname);
   }
   VERIFY(Nested)

protected:
   bool SetOrVerify(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      bool result = true;
      result &= SetOrVerifyScalar(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarPtr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyObjectPtr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrObjectPtr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyNested(entryNumber,reset,testname,splitlevel);
      if (reset) Assert(result);
      return result;
   }

public:
   
   void Reset(Int_t entryNumber) {
      SetOrVerify(entryNumber, true, "reseting", 0);
   }
   
   bool Verify(Int_t entryNumber, const std::string &testname, int splitlevel) {
      return SetOrVerify(entryNumber,false,testname,splitlevel);
   }

#if defined(R__NO_NESTED_CONTAINER)
   ClassDef(vectorHolder,1);
#else 
   ClassDef(vectorHolder,2);
#endif
};

