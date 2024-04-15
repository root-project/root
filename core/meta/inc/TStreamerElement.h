// @(#)root/meta:$Id: e0eac11e63ad37390c9467c97c5c6849c4ab7d39 $
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStreamerElement
#define ROOT_TStreamerElement

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TStreamerElement                                                     //
//                                                                      //
// Describe one element (data member) to be Streamed                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"

#include "ESTLType.h"

class TMethodCall;
class TClass;
class TStreamerBasicType;
class TVirtualStreamerInfo;

class TStreamerElement : public TNamed {

private:
   TStreamerElement(const TStreamerElement &) = delete;
   TStreamerElement&operator=(const TStreamerElement&) = delete;

protected:
   Int_t            fType;            //element type
   Int_t            fSize;            //sizeof element
   Int_t            fArrayLength;     //cumulative size of all array dims
   Int_t            fArrayDim;        //number of array dimensions
   Int_t            fMaxIndex[5];     //Maximum array index for array dimension "dim"
   Int_t            fOffset;          //!element offset in class
   Int_t            fTObjectOffset;   //!base offset for TObject if the element inherits from it
   Int_t            fNewType;         //!new element type when reading
   TString          fTypeName;        //Data type name of data member
   TClass          *fClassObject;     //!pointer to class of object
   TClass          *fNewClass;        //!new element class when reading
   TMemberStreamer *fStreamer;        //!pointer to element Streamer
   Double_t         fXmin;            //!Minimum of data member if a range is specified  [xmin,xmax,nbits]
   Double_t         fXmax;            //!Maximum of data member if a range is specified  [xmin,xmax,nbits]
   Double_t         fFactor;          //!Conversion factor if a range is specified fFactor = (1<<nbits/(xmax-xmin)

public:

   enum ESTLtype {
      kSTL                  = ROOT::kSTLany,
      kSTLstring            = ROOT::kSTLstring,
      kSTLvector            = ROOT::kSTLvector,
      kSTLlist              = ROOT::kSTLlist,
      kSTLforwardlist       = ROOT::kSTLforwardlist,
      kSTLdeque             = ROOT::kSTLdeque,
      kSTLmap               = ROOT::kSTLmap,
      kSTLmultimap          = ROOT::kSTLmultimap,
      kSTLset               = ROOT::kSTLset,
      kSTLmultiset          = ROOT::kSTLmultiset,
      kSTLunorderedset      = ROOT::kSTLunorderedset,
      kSTLunorderedmultiset = ROOT::kSTLunorderedmultiset,
      kSTLunorderedmap      = ROOT::kSTLunorderedmap,
      kSTLunorderedmultimap = ROOT::kSTLunorderedmultimap,
      kSTLbitset            = ROOT::kSTLbitset
   };
   // TStreamerElement status bits
   enum EStatusBits {
      kHasRange     = BIT(6),
      kCache        = BIT(9),
      kRepeat       = BIT(10),
      kRead         = BIT(11),
      kWrite        = BIT(12),
      kDoNotDelete  = BIT(13),
      kWholeObject  = BIT(14),
      kWarned       = BIT(21)
   };

   enum class EStatusBitsDupExceptions {
      // This bit duplicates TObject::kInvalidObject. As the semantic of kDoNotDelete is a persistent,
      // we can not change its value without breaking forward compatibility.
      // Furthermore, TObject::kInvalidObject and its semantic is not (and should not be)
      // used in TStreamerElement
      kDoNotDelete = TStreamerElement::kDoNotDelete,

      // This bit duplicates TObject::kCannotPick. As the semantic of kHasRange is a persistent,
      // we can not change its value without breaking forward compatibility.
      // Furthermore, TObject::kCannotPick and its semantic is not (and should not be)
      // used in TStreamerElement
      kHasRange = TStreamerElement::kHasRange
   };


   TStreamerElement();
   TStreamerElement(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName);
   virtual         ~TStreamerElement();
   virtual Bool_t   CannotSplit() const;
   Int_t            GetArrayDim() const {return fArrayDim;}
   Int_t            GetArrayLength() const {return fArrayLength;}
   virtual TClass  *GetClassPointer() const;
           TClass  *GetClass()        const {return GetClassPointer();}
   virtual Int_t    GetExecID() const;
   virtual const char *GetFullName() const;
   virtual const char *GetInclude() const {return "";}
   Int_t            GetMaxIndex(Int_t i) const {return fMaxIndex[i];}
   virtual ULongptr_t GetMethod() const {return ULongptr_t(fStreamer);}
   TMemberStreamer *GetStreamer() const;
   virtual Int_t    GetSize() const;
   Int_t            GetNewType() const {return fNewType;}
   TClass*          GetNewClass() const { return fNewClass; }
   Int_t            GetType() const {return fType;}
   Int_t            GetOffset() const {return fOffset;}
   void             GetSequenceType(TString &type) const;
   Int_t            GetTObjectOffset() const { return fTObjectOffset; }
   const char      *GetTypeName() const {return fTypeName.Data();}
   const char      *GetTypeNameBasic() const;
   Double_t         GetFactor() const {return fFactor;}
   Double_t         GetXmin()   const {return fXmin;}
   Double_t         GetXmax()   const {return fXmax;}
   virtual void     Init(TVirtualStreamerInfo *obj = nullptr);
   virtual Bool_t   IsaPointer() const {return kFALSE;}
   virtual Bool_t   HasCounter() const {return kFALSE;}
   virtual Bool_t   IsOldFormat(const char *newTypeName);
   virtual Bool_t   IsBase() const;
   virtual Bool_t   IsTransient() const;
   void             ls(Option_t *option="") const override;
   virtual void     SetArrayDim(Int_t dim);
   virtual void     SetMaxIndex(Int_t dim, Int_t max);
   virtual void     SetOffset(Int_t offset) {fOffset=offset;}
   virtual void     SetTObjectOffset(Int_t tobjoffset) {fTObjectOffset=tobjoffset;}
   virtual void     SetStreamer(TMemberStreamer *streamer);
   virtual void     SetSize(Int_t dsize) {fSize = dsize;}
   virtual void     SetNewType(Int_t dtype) {fNewType = dtype;}
   virtual void     SetNewClass( TClass* cl ) { fNewClass= cl; }
   virtual void     SetType(Int_t dtype) {fType = dtype;}
   virtual void     SetTypeName(const char *name) {fTypeName = name; fClassObject = (TClass*)-1; }
   virtual void     Update(const TClass *oldClass, TClass *newClass);

   ClassDefOverride(TStreamerElement,4)  //Base class for one element (data member) to be Streamed
};

//________________________________________________________________________
class TStreamerBase : public TStreamerElement {

private:
   TStreamerBase(const TStreamerBase &) = delete;
   TStreamerBase&operator=(const TStreamerBase&) = delete;

protected:
   Int_t             fBaseVersion;    //version number of the base class (used during memberwise streaming)
   UInt_t           &fBaseCheckSum;   //!checksum of the base class (used during memberwise streaming)
   TClass           *fBaseClass;      //!pointer to base class
   TClass           *fNewBaseClass;   //!pointer to new base class if renamed
   ClassStreamerFunc_t     fStreamerFunc;     //!Pointer to a wrapper around a custom streamer member function.
   ClassConvStreamerFunc_t fConvStreamerFunc; //!Pointer to a wrapper around a custom convertion streamer member function.
   TVirtualStreamerInfo *fStreamerInfo; //!Pointer to the current StreamerInfo for the baset class.
   TString               fErrorMsg;     //!Error message in case of checksum/version mismatch.

   void InitStreaming(Bool_t isTransient);

public:

   TStreamerBase();
   TStreamerBase(const char *name, const char *title, Int_t offset, Bool_t isTransient = kFALSE);
   virtual         ~TStreamerBase();
   Int_t            GetBaseVersion() {return fBaseVersion;}
   UInt_t           GetBaseCheckSum() {return fBaseCheckSum;}
   TClass          *GetClassPointer() const override;
   const char      *GetErrorMessage() const { return fErrorMsg; }
   const char      *GetInclude() const override;
   TClass          *GetNewBaseClass() { return fNewBaseClass; }
   ULongptr_t       GetMethod() const override {return 0;}
   Int_t            GetSize() const override;
   TVirtualStreamerInfo *GetBaseStreamerInfo() const { return fStreamerInfo; }
   void             Init(TVirtualStreamerInfo *obj = nullptr) override;
   void             Init(Bool_t isTransient = kFALSE);
   Bool_t           IsBase() const override;
   void             ls(Option_t *option="") const override;
   Int_t            ReadBuffer (TBuffer &b, char *pointer);
   void             SetNewBaseClass( TClass* cl ) { fNewBaseClass = cl; InitStreaming(kFALSE); }
   void             SetBaseVersion(Int_t v) {fBaseVersion = v;}
   void             SetBaseCheckSum(UInt_t cs) {fBaseCheckSum = cs;}
   void             SetErrorMessage(const char *msg) { fErrorMsg = msg; }
   void             Update(const TClass *oldClass, TClass *newClass) override;
   Int_t            WriteBuffer(TBuffer &b, char *pointer);

   ClassDefOverride(TStreamerBase,3)  //Streamer element of type base class
};

//________________________________________________________________________
class TStreamerBasicPointer : public TStreamerElement {

private:
   TStreamerBasicPointer(const TStreamerBasicPointer &) = delete;
   TStreamerBasicPointer&operator=(const TStreamerBasicPointer&) = delete;

protected:
   Int_t               fCountVersion;   //version number of the class with the counter
   TString             fCountName;      //name of data member holding the array count
   TString             fCountClass;     //name of the class with the counter
   TStreamerBasicType *fCounter;        //!pointer to basic type counter

public:

   TStreamerBasicPointer();
   TStreamerBasicPointer(const char *name, const char *title, Int_t offset, Int_t dtype,
                         const char *countName, const char *countClass, Int_t version, const char *typeName);
   virtual       ~TStreamerBasicPointer();
   TClass        *GetClassPointer() const override { return nullptr; }
   const char    *GetCountClass()   const {return fCountClass.Data();}
   const char    *GetCountName()    const {return fCountName.Data();}
   Int_t          GetCountVersion() const {return fCountVersion;}
   ULongptr_t     GetMethod() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;
   Bool_t         HasCounter() const override { return fCounter != nullptr; }
   Bool_t         IsaPointer() const override { return kTRUE; }
   void           SetArrayDim(Int_t dim) override;
   void           SetCountClass(const char *clname) { fCountClass = clname; }
   void           SetCountName(const char *name)    { fCountName = name; }
   void           SetCountVersion(Int_t count)      { fCountVersion = count; }
   void           Update(const TClass * /* oldClass */, TClass * /*newClass*/ ) override {}

   ClassDefOverride(TStreamerBasicPointer,2)  //Streamer element for a pointer to a basic type
};

//________________________________________________________________________
class TStreamerLoop : public TStreamerElement {

private:
   TStreamerLoop(const TStreamerLoop&) = delete;
   TStreamerLoop&operator=(const TStreamerLoop&) = delete;

protected:
   Int_t               fCountVersion;   //version number of the class with the counter
   TString             fCountName;      //name of data member holding the array count
   TString             fCountClass;     //name of the class with the counter
   TStreamerBasicType *fCounter;        //!pointer to basic type counter

public:

   TStreamerLoop();
   TStreamerLoop(const char *name, const char *title, Int_t offset, const char *countName, const char *countClass, Int_t version, const char *typeName);
   virtual       ~TStreamerLoop();
   const char    *GetCountClass()   const {return fCountClass.Data();}
   const char    *GetCountName()    const {return fCountName.Data();}
   Int_t          GetCountVersion() const {return fCountVersion;}
   const char    *GetInclude() const override;
   ULongptr_t     GetMethod() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;
   Bool_t         IsaPointer() const override {return kTRUE; }
   Bool_t         HasCounter() const  override { return fCounter != nullptr; }
   void           SetCountClass(const char *clname) { fCountClass = clname; }
   void           SetCountName(const char *name) { fCountName = name; }
   void           SetCountVersion(Int_t count) { fCountVersion = count; }

   ClassDefOverride(TStreamerLoop,2)  //Streamer element for a pointer to an array of objects
};

//________________________________________________________________________
class TStreamerBasicType : public TStreamerElement {

private:
   TStreamerBasicType(const TStreamerBasicType&) = delete;
   TStreamerBasicType&operator=(const TStreamerBasicType&) = delete;

protected:
   Int_t             fCounter;     //!value of data member when referenced by an array

public:

   TStreamerBasicType();
   TStreamerBasicType(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName);
   virtual       ~TStreamerBasicType();
   TClass        *GetClassPointer() const override { return nullptr; }
   Int_t          GetCounter() const { return fCounter; }
   ULongptr_t     GetMethod() const override;
   Int_t          GetSize() const override;
   void           Update(const TClass * /* oldClass */, TClass * /* newClass */) override {}

   ClassDefOverride(TStreamerBasicType,2)  //Streamer element for a basic type
};

//________________________________________________________________________
class TStreamerObject : public TStreamerElement {

private:
   TStreamerObject(const TStreamerObject&) = delete;
   TStreamerObject&operator=(const TStreamerObject&) = delete;

public:

   TStreamerObject();
   TStreamerObject(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObject();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;

   ClassDefOverride(TStreamerObject,2)  //Streamer element of type object
};

//________________________________________________________________________
class TStreamerObjectAny : public TStreamerElement {

private:
   TStreamerObjectAny(const TStreamerObjectAny&) = delete;
   TStreamerObjectAny&operator=(const TStreamerObjectAny&) = delete;

public:

   TStreamerObjectAny();
   TStreamerObjectAny(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObjectAny();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;

   ClassDefOverride(TStreamerObjectAny,2)  //Streamer element of type object other than TObject
};

//________________________________________________________________________
class TStreamerObjectPointer : public TStreamerElement {

private:
   TStreamerObjectPointer(const TStreamerObjectPointer&) = delete;
   TStreamerObjectPointer&operator=(const TStreamerObjectPointer&) = delete;

public:

   TStreamerObjectPointer();
   TStreamerObjectPointer(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObjectPointer();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;
   Bool_t         IsaPointer() const override {return kTRUE;}
   void           SetArrayDim(Int_t dim) override;

   ClassDefOverride(TStreamerObjectPointer,2)  //Streamer element of type pointer to a TObject
};

//________________________________________________________________________
class TStreamerObjectAnyPointer : public TStreamerElement {

private:
   TStreamerObjectAnyPointer(const TStreamerObjectAnyPointer&) = delete;
   TStreamerObjectAnyPointer&operator=(const TStreamerObjectAnyPointer&) = delete;

public:

   TStreamerObjectAnyPointer();
   TStreamerObjectAnyPointer(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObjectAnyPointer();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;
   void           Init(TVirtualStreamerInfo *obj = nullptr) override;
   Bool_t         IsaPointer() const override { return kTRUE; }
   void           SetArrayDim(Int_t dim) override;

   ClassDefOverride(TStreamerObjectAnyPointer,1)  //Streamer element of type pointer to a non TObject
};

//________________________________________________________________________
class TStreamerString : public TStreamerElement {

private:
   TStreamerString(const TStreamerString&) = delete;
   TStreamerString&operator=(const TStreamerString&) = delete;

public:

   TStreamerString();
   TStreamerString(const char *name, const char *title, Int_t offset);
   virtual       ~TStreamerString();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;

   ClassDefOverride(TStreamerString,2)  //Streamer element of type TString
};

//________________________________________________________________________
class TStreamerSTL : public TStreamerElement {

private:
   TStreamerSTL(const TStreamerSTL&) = delete;
   TStreamerSTL&operator=(const TStreamerSTL&) = delete;

protected:
   Int_t       fSTLtype;       //type of STL vector
   Int_t       fCtype;         //STL contained type

public:

   TStreamerSTL();
   TStreamerSTL(const char *name, const char *title, Int_t offset,
                const char *typeName, const char *trueType, Bool_t dmPointer);
   TStreamerSTL(const char *name, const char *title, Int_t offset,
                const char *typeName, const TVirtualCollectionProxy &proxy , Bool_t dmPointer);
   virtual       ~TStreamerSTL();
   Bool_t         CannotSplit() const override;
   Bool_t         IsaPointer() const override;
   Bool_t         IsBase() const override;
   Int_t          GetSTLtype() const {return fSTLtype;}
   Int_t          GetCtype()   const {return fCtype;}
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;
   void           ls(Option_t *option="") const override;
   void           SetSTLtype(Int_t t) {fSTLtype = t;}
   void           SetCtype(Int_t t) {fCtype = t;}
   void           SetStreamer(TMemberStreamer *streamer) override;

   ClassDefOverride(TStreamerSTL,3)  //Streamer element of type STL container
};

//________________________________________________________________________
class TStreamerSTLstring : public TStreamerSTL {

private:
   TStreamerSTLstring(const TStreamerSTLstring&) = delete;
   TStreamerSTLstring&operator=(const TStreamerSTLstring&) = delete;

public:

   TStreamerSTLstring();
   TStreamerSTLstring(const char *name, const char *title, Int_t offset,
                      const char *typeName, Bool_t dmPointer);
   virtual       ~TStreamerSTLstring();
   const char    *GetInclude() const override;
   Int_t          GetSize() const override;

   ClassDefOverride(TStreamerSTLstring,2)  //Streamer element of type  C++ string
};

class TVirtualObject;
class TBuffer;

#include "TSchemaRule.h"

//________________________________________________________________________
class TStreamerArtificial : public TStreamerElement {
private:
   TStreamerArtificial(const TStreamerArtificial&) = delete;
   TStreamerArtificial&operator=(const TStreamerArtificial&) = delete;

protected:
   ROOT::TSchemaRule::ReadFuncPtr_t     fReadFunc;    //!
   ROOT::TSchemaRule::ReadRawFuncPtr_t  fReadRawFunc; //!

public:

   // TStreamerArtificial() : fReadFunc(0),fReadRawFunc(0) {}

   TStreamerArtificial(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName) : TStreamerElement(name,title,offset,dtype,typeName), fReadFunc(nullptr), fReadRawFunc(nullptr) {}

   void SetReadFunc( ROOT::TSchemaRule::ReadFuncPtr_t val ) { fReadFunc = val; };
   void SetReadRawFunc( ROOT::TSchemaRule::ReadRawFuncPtr_t val ) { fReadRawFunc = val; };

   ROOT::TSchemaRule::ReadFuncPtr_t     GetReadFunc();
   ROOT::TSchemaRule::ReadRawFuncPtr_t  GetReadRawFunc();

   ClassDefOverride(TStreamerArtificial, 0); // StreamerElement injected by a TSchemaRule. Transient only to preserve forward compatibility.
};

#endif
