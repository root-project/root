#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TClass.h"
#include "TError.h"
#include "TObject.h"
#include "TROOT.h"
#include "TMath.h"
#include "versions.h"
#endif

#include <vector>

#include <iostream>
#include <sstream>

#define VERIFY(X)                                                 \
  bool Verify##X (Int_t entryNumber, const std::string &testname) \
  {                                                               \
     return SetOrVerify##X (entryNumber,false,testname);          \
  }

void TestError(const std::string &test, const char *msg) {
   std::cerr << "Error for '" << test << "' : " << msg << "\n";
}
void TestError(const std::string &test, const std::string &str) {
   TestError(test, str.c_str());
}
//void TestError(const char *msg);

class Helper {
public:
   unsigned int val;
   Helper() : val(0) {};
   explicit Helper(int v) : val(v) {};
   //bool operator==(const Helper &rhs) const { return val==rhs.val; }
   //bool operator!=(const Helper &rhs) const { return !(*this==rhs); }
   bool IsEquiv(const Helper &rhs) const { return  val==rhs.val; }
};

template <class T> void TestError(const std::string &test, const std::vector<T> &orig, const std::vector<T> &copy) {
   TestError(test,"Containers are not equivalent! See previous errors");
}

template <class T> void TestError(const std::string &test, const T &orig, const T &copy) {
   std::stringstream s;
   s << "We wrote: " << orig << " but read " << copy << std::ends;
   TestError(test, s.str());
}

void TestError(const std::string &test, const Helper &orig, const Helper &copy) {
     TestError(test, Form("Helper object wrote %d and read %d\n",
                          orig.val,copy.val));
}

void TestError(const std::string &test, const Helper* &orig, const Helper* &copy) {
   if (orig==0 || copy==0) {
      TestError(test,Form("For Helper, non-initialized pointer %p %p",orig,copy));
   } else {
      TestError(test, *orig, *copy); 
   }
}

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
      }
      i++;
      iorig++;
      icopy++;
   }
   return result;
}

template <class T> bool IsEquiv(const std::string &test, T* orig, T* copy) {
   TClass *cl = gROOT->GetClass(typeid(T));
   const char* classname = cl?cl->GetName():typeid(T).name();

   if ( (orig==0 && copy) || (orig && copy==0) ) {
      TestError(test,Form("For %s, non-initialized pointer %p %p",classname,orig,copy));
      return false;
   }
   return IsEquiv(test, *orig, *copy) && false;
}

bool IsEquiv(const std::string &, float orig, float copy) {
   float epsilon = 1e-6;
   float diff = orig-copy;
   return TMath::Abs( diff/copy ) < epsilon;
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

   vectorHolder() : TObject(),
      fScalarPtr(0),
      fObjectPtr(0)
      ,fPtrObjectPtr(0)
      {}

   explicit vectorHolder(Int_t entry) : TObject()
      ,fScalarPtr(0)
      ,fObjectPtr(0)
      ,fPtrObjectPtr(0)
      {
         Reset(entry);
      }


   std::vector<int >    *fScalarPtr;
   std::vector<float >   fScalar;
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

   bool SetOrVerifyScalar(Int_t entryNumber, bool reset, const std::string &testname) {
      Int_t seed = 3 * (entryNumber+1);
      return SetOrVerify("fScalar",fScalar,seed,entryNumber,reset,testname);
   }
   VERIFY(Scalar);
   
   bool SetOrVerifyScalarPtr(Int_t entryNumber, bool reset, const std::string &testname) {
      Int_t seed = 4 * (entryNumber+1);
      return SetOrVerify("fScalarPtr",fScalarPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarPtr);

   bool SetOrVerifyObject(Int_t entryNumber, bool reset, const std::string &testname) {
      UInt_t seed = 2 * (entryNumber+1);
      return SetOrVerify("fObject",fObject,seed,entryNumber,reset,testname);
   }
   VERIFY(Object);

   bool SetOrVerifyObjectPtr(Int_t entryNumber, bool reset, const std::string &testname) {
      UInt_t seed = 3 * (entryNumber+1);
      return SetOrVerify("fObjectPtr",fObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ObjectPtr);

   bool SetOrVerifyPtrObject(Int_t entryNumber, bool reset, const std::string &testname) {
      UInt_t seed = 4 * (entryNumber+1);
      return SetOrVerify("fPtrObject",fPtrObject,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObject);

   bool SetOrVerifyPtrObjectPtr(Int_t entryNumber, bool reset, const std::string &testname) {
      UInt_t seed = 5 * (entryNumber+1);
      return SetOrVerify("fPtrObjectPtr",fPtrObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObjectPtr);

   bool SetOrVerifyNested(Int_t entryNumber, bool reset, const std::string &testname) {
      if (gFile && !HasNestedContainer(gFile)) return true;

      UInt_t seed = 1 * (entryNumber+1);
      return SetOrVerify("fNested",fNested,seed,entryNumber,reset,testname);
   }
   VERIFY(Nested)

protected:
   bool SetOrVerify(Int_t entryNumber, bool reset, const std::string &testname) {
      bool result = true;
      result &= SetOrVerifyScalar(entryNumber,reset,testname);
      result &= SetOrVerifyScalarPtr(entryNumber,reset,testname);
      result &= SetOrVerifyObject(entryNumber,reset,testname);
      result &= SetOrVerifyObjectPtr(entryNumber,reset,testname);
      result &= SetOrVerifyPtrObject(entryNumber,reset,testname);
      result &= SetOrVerifyPtrObjectPtr(entryNumber,reset,testname);
      result &= SetOrVerifyNested(entryNumber,reset,testname);
      if (reset) Assert(result);
      return result;
   }

public:
   
   void Reset(Int_t entryNumber) {
      SetOrVerify(entryNumber, true, "reseting");
   }
   
   bool Verify(Int_t entryNumber, const std::string &testname) {
      return SetOrVerify(entryNumber,false,testname);
   }

#if defined(R__NO_NESTED_CONTAINER)
   ClassDef(vectorHolder,1);
#else 
   ClassDef(vectorHolder,2);
#endif
};

