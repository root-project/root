#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TClass.h"
#include "TError.h"
#include "TObject.h"
#include "TROOT.h"
#include "TMath.h"
#endif

#include <deque>
#include <vector>

#include <iostream>

#include "TestHelpers.h"
#include "versions.h"
#include "TestSetOrVerify.h"

class dequeHolder : public TObject {

public:   

   dequeHolder() : TObject()
      ,fScalarArrVar(0)
      ,fScalarPtr(0)
      ,fScalarPtrArrVar(0)
      ,fObjectPtr(0)
      ,fPtrObjectPtr(0)
      ,fTObjectPtr(0)
      ,fPtrTObjectPtr(0)
      {
         for(int index=0;index<4;index++) fScalarPtrArr[index]=0;
      }

   explicit dequeHolder(Int_t entry) : TObject()
      ,fScalarArrVar(0)
      ,fScalarPtr(0)
      ,fScalarPtrArrVar(0)
      ,fObjectPtr(0)
      ,fPtrObjectPtr(0)
      ,fTObjectPtr(0)
      ,fPtrTObjectPtr(0)
      {
         for(int index=0;index<4;index++) fScalarPtrArr[index]=0;
         Reset(entry);
      }
   
   std::deque<EHelper > fEnum;
   
   std::deque<float >   fScalar;
   std::deque<short >   fScalarArr[2];
   UInt_t                fScalarArrVarSize;
   std::deque<char >   *fScalarArrVar; //[fScalarArrVarSize]

   std::deque<int >    *fScalarPtr;
   std::deque<double > *fScalarPtrArr[4];
   UInt_t                fScalarPtrArrVarSize;
   std::deque<int >    *fScalarPtrArrVar; //[fScalarPtrArrVarSize]
   

   std::deque<Helper >  fObject;
   std::deque<Helper > *fObjectPtr;

   std::deque<Helper* > fPtrObject;
   std::deque<Helper* >*fPtrObjectPtr;


#if defined(R__CANNOT_SPLIT_VECTOR_OBJECT)
   std::deque<THelper >  fTObject;      //||
   std::deque<THelper > *fTObjectPtr;   //||

   std::deque<THelper* > fPtrTObject;   //||
   std::deque<THelper* >*fPtrTObjectPtr;//||
#else
   std::deque<THelper >  fTObject;
   std::deque<THelper > *fTObjectPtr;

   std::deque<THelper* > fPtrTObject;
   std::deque<THelper* >*fPtrTObjectPtr;
#endif

   std::deque<std::string>         fString;
   std::deque<std::string*>        fPtrString;
#if defined(R__NO_NESTED_CONST_STRING)
   std::deque<const std::string*>  fPtrConstString; //!  this version of ROOT does not support nested const string
#else
   std::deque<const std::string*>  fPtrConstString;
#endif

   std::deque<TString>  fTString;
   std::deque<TString*> fPtrTString;

#if defined(R__CANNOT_SPLIT_VECTOR_OBJECT)
   std::deque<TNamed>   fTNamed;    //||
   std::deque<TNamed*>  fPtrTNamed; //||
   std::deque<const TNamed*>  fPtrConstTNamed; //||
#endif


   typedef std::deque<Helper > nested_t;
   typedef std::deque<nested_t > nesting_t;
#if defined(R__NO_NESTED_CONTAINER)
   nesting_t fNested;  //! this version of ROOT does not support nested container
   std::deque<std::deque<Helper > > fNestedVV; //!
   std::deque<std::vector<Helper > > fNestedVD; //!
#else
   nesting_t fNested;  
   std::deque<std::deque<Helper > > fNestedVV; 
   std::deque<std::vector<Helper > > fNestedVD; 
#endif

   bool SetOrVerifyEnum(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      Int_t seed = 1 * (entryNumber+1);
      return utility::SetOrVerify("fEnum",fEnum,seed,entryNumber,reset,testname);
   }
   VERIFY(Enum);
   
   bool SetOrVerifyScalar(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      Int_t seed = 1 * (entryNumber+1);
      return utility::SetOrVerify("fScalar",fScalar,seed,entryNumber,reset,testname);
   }
   VERIFY(Scalar);
   
   bool SetOrVerifyScalarArr(Int_t entryNumber, bool reset, const std::string &testname, int /*splitlevel*/) {
      Int_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fScalarArr",&(fScalarArr[0]), 2 ,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarArr);

   bool SetOrVerifyScalarArrVar(Int_t entryNumber, bool reset, const std::string &testname, int /*splitlevel*/) {
      if (!reset && gFile && !HasVarArrayOfContainers(gFile)) {
         return true;
      }      
      Int_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerifyArrVar("fScalarArrVar",fScalarArrVar,fScalarArrVarSize,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarArrVar);


   bool SetOrVerifyScalarPtr(Int_t entryNumber, bool reset, const std::string &testname, int /*splitlevel*/) {
      Int_t seed = 4 * (entryNumber+1);
      return utility::SetOrVerify("fScalarPtr",fScalarPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarPtr);

   bool SetOrVerifyScalarPtrArr(Int_t entryNumber, bool reset, const std::string &testname, int /*splitlevel*/) {
      Int_t seed = 5 * (entryNumber+1);
      return utility::SetOrVerify("fScalarPtrArr",&(fScalarPtrArr[0]), 2 ,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarPtrArr);

   bool SetOrVerifyScalarPtrArrVar(Int_t entryNumber, bool reset, const std::string &testname, int /*splitlevel*/) {
      if (!reset && gFile && !HasVarArrayOfContainers(gFile)) {
         return true;
      }      
      Int_t seed = 6 * (entryNumber+1);
      return utility::SetOrVerifyArrVar("fScalarPtrArrVar",fScalarPtrArrVar,fScalarPtrArrVarSize,seed,entryNumber,reset,testname);
   }
   VERIFY(ScalarPtrArrVar);


   bool SetOrVerifyObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fObject",fObject,seed,entryNumber,reset,testname);
   }
   VERIFY(Object);

   bool SetOrVerifyObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fObjectPtr",fObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(ObjectPtr);

   bool SetOrVerifyPtrObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 4 * (entryNumber+1);
      return utility::SetOrVerify("fPtrObject",fPtrObject,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObject);

   bool SetOrVerifyPtrObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 5 * (entryNumber+1);
      return utility::SetOrVerify("fPtrObjectPtr",fPtrObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrObjectPtr);


   bool SetOrVerifyTObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fTObject",fTObject,seed,entryNumber,reset,testname);
   }
   VERIFY(TObject);

   bool SetOrVerifyTObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fTObjectPtr",fTObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(TObjectPtr);

   bool SetOrVerifyPtrTObject(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 4 * (entryNumber+1);
      return utility::SetOrVerify("fPtrTObject",fPtrTObject,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrTObject);

   bool SetOrVerifyPtrTObjectPtr(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 5 * (entryNumber+1);
      return utility::SetOrVerify("fPtrTObjectPtr",fPtrTObjectPtr,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrTObjectPtr);


   bool SetOrVerifyString(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fString",fString,seed,entryNumber,reset,testname);
   }
   VERIFY(String);

   bool SetOrVerifyPtrString(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fPtrString",fPtrString,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrString);

   bool SetOrVerifyPtrConstString(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      if (!reset && gFile && !HasNestedConstString(gFile)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fPtrConstString",fPtrConstString,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrConstString);

   bool SetOrVerifyTString(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fTString",fTString,seed,entryNumber,reset,testname);
   }
   VERIFY(TString);

   bool SetOrVerifyPtrTString(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fPtrTString",fPtrTString,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrTString);

   bool SetOrVerifyTNamed(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 2 * (entryNumber+1);
      return utility::SetOrVerify("fTNamed",fTNamed,seed,entryNumber,reset,testname);
   }
   VERIFY(TNamed);

   bool SetOrVerifyPtrTNamed(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fPtrTNamed",fPtrTNamed,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrTNamed);

   bool SetOrVerifyPtrConstTNamed(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      if (!reset && gFile && !HasSplitVectorObject(gFile,splitlevel)) {
         return true;
      }
      UInt_t seed = 3 * (entryNumber+1);
      return utility::SetOrVerify("fPtrConstTNamed",fPtrConstTNamed,seed,entryNumber,reset,testname);
   }
   VERIFY(PtrConstTNamed);

   bool SetOrVerifyNested(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      if (!reset && gFile && !HasNestedContainer(gFile)) {
         return true;
      }
      UInt_t seed = 1 * (entryNumber+1);
      return utility::SetOrVerify("fNested",fNested,seed,entryNumber,reset,testname);
   }
   VERIFY(Nested)

   bool SetOrVerifyNestedVV(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      if (!reset && gFile && !HasNestedContainer(gFile)) {
         return true;
      }
      UInt_t seed = 1 * (entryNumber+1);
      return utility::SetOrVerify("fNestedVV",fNestedVV,seed,entryNumber,reset,testname);
   }
   VERIFY(NestedVV)

   bool SetOrVerifyNestedVD(Int_t entryNumber, bool reset, const std::string &testname,int /*splitlevel*/) {
      if (!reset && gFile && !HasNestedContainer(gFile)) {
         return true;
      }
      UInt_t seed = 1 * (entryNumber+1);
      return utility::SetOrVerify("fNestedVD",fNestedVD,seed,entryNumber,reset,testname);
   }
   VERIFY(NestedVD)

protected:
   bool SetOrVerify(Int_t entryNumber, bool reset, const std::string &testname,int splitlevel) {
      bool result = true;
      result &= SetOrVerifyEnum(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyScalar(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarArr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarArrVar(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarPtr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarPtrArr(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyScalarPtrArrVar(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyObjectPtr(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyPtrObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrObjectPtr(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyTObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyTObjectPtr(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyPtrTObject(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrTObjectPtr(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyString        (entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrString     (entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrConstString(entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyTString     (entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrTString  (entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyTNamed     (entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrTNamed  (entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyPtrConstTNamed  (entryNumber,reset,testname,splitlevel);

      result &= SetOrVerifyNested(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyNestedVV(entryNumber,reset,testname,splitlevel);
      result &= SetOrVerifyNestedVD(entryNumber,reset,testname,splitlevel);
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
   ClassDef(dequeHolder,1);
#else 
   ClassDef(dequeHolder,2);
#endif
};

