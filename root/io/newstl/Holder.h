
#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TClass.h"
#include "TError.h"
#include "TObject.h"
#include "TROOT.h"
#endif

#include <vector>

class Helper {};

class vectorHolder : public TObject {

public:   
   std::vector<float >  fScalar;
   std::vector<Helper > fObject;
   std::vector<std::vector<Helper > > fNested;

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
      return true;
   }
   bool SetOrVerifyNested(Int_t entryNumber, bool reset) {
      return true;
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

   ClassDef(vectorHolder,1);
};

