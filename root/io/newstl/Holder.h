#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TClass.h"
#include "TError.h"
#include "TObject.h"
#include "TROOT.h"

#include "versions.h"
#endif

#include <vector>

#include <iostream>
void TestError(const char *msg) {
   std::cerr << msg << "\n";
}
//void TestError(const char *msg);

class Helper {
public:
   unsigned int val;
   Helper() : val(0) {};
   explicit Helper(int v) : val(v) {};
   bool operator==(const Helper &rhs) const { return val==rhs.val; }
   bool operator!=(const Helper &rhs) const { return !(*this==rhs); }
};

void TestError(const Helper &orig, const Helper &copy) {
     TestError(Form("Helper object wrote %d and read %d\n",
                    orig.val,copy.val));
}

template <class T> void fill(T& filled, UInt_t seed) {
   UInt_t size = seed%10;

   filled.clear();
   for(UInt_t i=0; i<size; i++) {
      typename T::value_type val(seed*10+i);
      filled.push_back(val);
  }  
}

template <class T> bool isEqual(const T& orig, const T& copy) {

    if (orig.size() != copy.size()) {
       TestError(Form("IsEqual for %s: Wrong size %d vs %d\n",typeid(T).name(),orig.size(),copy.size()));
       return false;
    }

    bool result = true;
    typename T::const_iterator iorig = orig.begin();
    typename T::const_iterator icopy = copy.begin();
    UInt_t i = 0;
    while ( iorig != orig.end() && icopy != copy.end() ) {
        if (!(*iorig == *icopy)) {
           TestError(Form("IsEqual for %s:\nelem #%d are not equal",
                     typeid(T).name(),i));
           TestError(*iorig,*icopy);
           result = false;
        }
        i++;
        iorig++;
        icopy++;
    }
    return result;
}




class vectorHolder : public TObject {

public:   
   std::vector<float >  *fScalarPtr; //!
   std::vector<float >   fScalar;
   std::vector<Helper >  fObject;

   typedef std::vector<Helper > nested_t;
   typedef std::vector<nested_t > nesting_t;
#if defined(R__NO_NESTED_CONTAINER)
   nesting_t fNested;  //! this version of ROOT does not support nested container
#else
   nesting_t fNested;  //
#endif

   vectorHolder() : TObject() {}

   explicit vectorHolder(Int_t entry) : TObject() {
      Reset(entry);
   }

   bool SetOrVerifyScalar(Int_t entryNumber, bool reset) {
      TClass *cl = gROOT->GetClass(typeid(fScalar));
      bool result = true;

      UInt_t size = 3 * (entryNumber+1);
      
      if (reset) {
         fScalar.clear();
      } else {
         if (size != fScalar.size()) {
            Error("VerifyScalar","At entry #%d wrong size (%d instead of %d) of fScalar of type %s",
                  entryNumber,fScalar.size(),size,cl?cl->GetName():typeid(fScalar).name());
            result = false;
         }
      }

     std::vector<float >::iterator iter = fScalar.begin();

      for(UInt_t i=0; i<size; i++) {
         float val = entryNumber*10000+i;

         if (reset) {
            fScalar.push_back(val);
         } else {
            if (iter == fScalar.end()) {
               Error("VerifyScalar","At entry #%d, index %d: premature end of container, only %d elements processed",
                     entryNumber, i);
               result = false;
               break;
            }
            if ( val != *iter) {
               Error("VerifyScalar","At entry #%d, index %d wrong value (%f instead of %f) for fScalar of type %s",
                     entryNumber,i,*iter,val,cl?cl->GetName():typeid(fScalar).name());
               result = false;
            }
            iter++;
         }

      }  
      return result;
   }
   
   bool SetOrVerifyObject(Int_t entryNumber, bool reset) {
      TClass *cl = gROOT->GetClass(typeid(fObject));
      bool result = true;
      UInt_t size = 2 * (entryNumber+1);
std::cerr << "verifying object " << fObject.size() << " " << size << std::endl;
      if (reset) {
         fObject.clear();
      } else {
         if (size != fObject.size()) {
            Error("VerifyObject","At entry #%d wrong size (%d instead of %d) of fObject of type %s",
                  entryNumber,fObject.size(),size,cl?cl->GetName():typeid(fObject).name());
            result = false;
         }
      }

      std::vector<Helper >::iterator iter(fObject.begin());

      for(UInt_t i=0; i<size; i++) {
         Helper val(entryNumber*20000+2*i);

         if (reset) {
            fObject.push_back(val);
         } else {
            if (iter == fObject.end()) {
               Error("VerifyObject","At entry #%d, index %d: premature end of container, only %d elements processed",
                     entryNumber, i);
               result = false;
               break;
            }
            if ( val != *iter) {
               Error("VerifyObject","At entry #%d, index %d wrong value (%f instead of %f) for fObject of type %s",
                     entryNumber,i,(*iter).val,val.val,cl?cl->GetName():typeid(fObject).name());
               result = false;
            }
            iter++;
         }

      }  
      return result;
   }

   bool SetOrVerifyNested(Int_t entryNumber, bool reset) {
      if (gFile && !HasNestedContainer(gFile)) return true;

      TClass *cl = gROOT->GetClass(typeid(fNested));
      bool result = true;
      UInt_t size = 1 * (entryNumber+1);
      if (reset) {
         fNested.clear();
      } else {
         if (size != fNested.size()) {
            Error("VerifyNested","At entry #%d wrong size (%d instead of %d) of fNested of type %s",
                  entryNumber,fNested.size(),size,cl?cl->GetName():typeid(fNested).name());
            result = false;
         }
      }

      nesting_t::iterator iter(fNested.begin());
      for(UInt_t i=0; i<size; i++) {
         nested_t subvec; // (entryNumber*20000+2*i);
         
         fill(subvec, entryNumber*1000+3*i);

         if (reset) {
            fNested.push_back(subvec);
         } else {
            if (iter == fNested.end()) {
               Error("VerifyNested","At entry #%d, index %d: premature end of container, only %d elements processed",
                     entryNumber, i);
               result = false;
               break;
            }
            if ( subvec.size() != (*iter).size() ) {
               Error("VerifyNested","At entry #%d, index %d wrong sub-size (%d instead of %d) for fNested of type %s",
                     entryNumber,i,(*iter).size(),subvec.size(),cl?cl->GetName():typeid(fNested).name());
               result = false;
            }
            isEqual(subvec, *iter);
            iter++;
         }

      }  
      return result;
   }

   void Reset(Int_t entryNumber) {
      bool result = true;
      result &= SetOrVerifyScalar(entryNumber,true);
      result &= SetOrVerifyObject(entryNumber,true);
      result &= SetOrVerifyNested(entryNumber,true);
      Assert(result);
   }
   
   bool Verify(Int_t entryNumber) {
      bool result = true;
      result &= VerifyScalar(entryNumber);
      result &= VerifyObject(entryNumber);
      result &= VerifyNested(entryNumber);
      return result;
   }
   bool VerifyScalar(Int_t entryNumber) { return SetOrVerifyScalar(entryNumber,false); }
   bool VerifyObject(Int_t entryNumber) { return SetOrVerifyObject(entryNumber,false); }
   bool VerifyNested(Int_t entryNumber) { return SetOrVerifyNested(entryNumber,false); }

#if defined(R__NO_NESTED_CONTAINER)
   ClassDef(vectorHolder,1);
#else 
   ClassDef(vectorHolder,2);
#endif
};

