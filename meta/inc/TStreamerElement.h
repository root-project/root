// @(#)root/meta:$Name:  $:$Id: TStreamerElement.h,v 1.11 2001/01/16 16:20:28 brun Exp $
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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TMethodCall;
class TClass;
class TStreamerBasicType;

class TStreamerElement : public TNamed {

protected:
   Int_t         fType;            //element type
   Int_t         fSize;            //sizeof element
   Int_t         fArrayLength;     //cumulative size of all array dims
   Int_t         fArrayDim;        //number of array dimensions
   Int_t         fMaxIndex[5];     //Maximum array index for array dimension "dim"
   Int_t         fOffset;          //!element offset in class
   Int_t         fNewType;         //!new element type when reading
   TString       fTypeName;        //Data type name of data member
   Streamer_t    fStreamer;        //!pointer to element Streamer      
   TMethodCall  *fMethod;          //!pointer to TMethodCall
public:

   enum ESTLtype { kSTL       = 300, kSTLstring  =365,   kSTLvector = 1,
                   kSTLlist   =  2,  kSTLdeque   =  3,   kSTLmap    = 4,
                   kSTLset    =  5,  kSTLmultimap=6,     kSTLmultiset=7};

   TStreamerElement();
   TStreamerElement(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName);
   virtual         ~TStreamerElement();
   Int_t            GetArrayDim() const {return fArrayDim;}
   Int_t            GetArrayLength() const {return fArrayLength;}
   virtual TClass  *GetClassPointer() const;
   virtual const char *GetInclude() const {return "";}
   Int_t            GetMaxIndex(Int_t i) const {return fMaxIndex[i];}
   virtual ULong_t  GetMethod() const {return ULong_t(fStreamer);}
   Streamer_t       GetStreamer() const {return fStreamer;}
   Int_t            GetSize() const {return fSize;}
   Int_t            GetNewType() const {return fNewType;}
   Int_t            GetType() const {return fType;}
   Int_t            GetOffset() const {return fOffset;}
   const char      *GetTypeName() const {return fTypeName.Data();}
   virtual void     Init(TObject *obj=0);
   virtual Bool_t   IsaPointer() const {return kFALSE;}
   virtual Bool_t   IsOldFormat(const char *newTypeName);
   virtual void     ls(Option_t *option="") const;
   virtual void     SetArrayDim(Int_t dim);
   virtual void     SetMaxIndex(Int_t dim, Int_t max);
   virtual void     SetOffset(Int_t offset) {fOffset=offset;}
   virtual void     SetStreamer(Streamer_t streamer);
   virtual void     SetSize(Int_t dsize) {fSize = dsize;}
   virtual void     SetNewType(Int_t dtype) {fNewType = dtype;}
   virtual void     SetType(Int_t dtype) {fType = dtype;}
         
   ClassDef(TStreamerElement,1)  //base class for one element (data member) to be Streamed
};

//________________________________________________________________________
class TStreamerBase : public TStreamerElement {

protected:
   TClass          *fBaseClass;           //pointer to base class

public:

   TStreamerBase();
   TStreamerBase(const char *name, const char *title, Int_t offset);
   virtual         ~TStreamerBase();
   virtual TClass  *GetClassPointer() const {return fBaseClass;}
   const char      *GetInclude() const;
   ULong_t          GetMethod() const {return ULong_t(fMethod);}
   virtual void     Init(TObject *obj=0);
   Int_t            ReadBuffer (TBuffer &b, char *pointer);
   Int_t            WriteBuffer(TBuffer &b, char *pointer);
   
   ClassDef(TStreamerBase,1)  //Streamer element of type base class
};

//________________________________________________________________________
class TStreamerBasicPointer : public TStreamerElement {

protected:
   Int_t               fCountVersion;   //version number of the class with the counter
   TString             fCountName;      //name of data member holding the array count
   TString             fCountClass;     //name of the class with the counter
   TStreamerBasicType *fCounter;        //pointer to basic type counter
           
public:

   TStreamerBasicPointer();
   TStreamerBasicPointer(const char *name, const char *title, Int_t offset, Int_t dtype, const char *countName, const char *countClass, Int_t version, const char *typeName);
   virtual       ~TStreamerBasicPointer();
   ULong_t        GetMethod() const;
   virtual void   Init(TObject *obj=0);
   virtual Bool_t IsaPointer() const {return kTRUE;}
   
   ClassDef(TStreamerBasicPointer,1)  //Streamer element for a pointer to a basic type
};

//________________________________________________________________________
class TStreamerLoop : public TStreamerElement {

protected:
   Int_t               fCountVersion;   //version number of the class with the counter
   TString             fCountName;      //name of data member holding the array count
   TString             fCountClass;     //name of the class with the counter
   TStreamerBasicType *fCounter;        //pointer to basic type counter
           
public:

   TStreamerLoop();
   TStreamerLoop(const char *name, const char *title, Int_t offset, const char *countName, const char *countClass, Int_t version, const char *typeName);
   virtual       ~TStreamerLoop();
   const char    *GetInclude() const;
   ULong_t        GetMethod() const;
   virtual void   Init(TObject *obj=0);
   virtual Bool_t IsaPointer() const {return kTRUE;}
   
   ClassDef(TStreamerLoop,1)  //Streamer element for a pointer to an array of objects
};

//________________________________________________________________________
class TStreamerBasicType : public TStreamerElement {

protected:
   Int_t             fCounter;     //!value of data member when referenced by an array
   
public:

   TStreamerBasicType();
   TStreamerBasicType(const char *name, const char *title, Int_t offset, Int_t dtype, const char *typeName);
   virtual       ~TStreamerBasicType();
   Int_t          GetCounter() const {return fCounter;}
   ULong_t        GetMethod() const;
   
   ClassDef(TStreamerBasicType,1)  //Streamer element for a basic type
};

//________________________________________________________________________
class TStreamerObject : public TStreamerElement {

protected:
   TClass           *fClassObject;    //!pointer to class of object
   
public:

   TStreamerObject();
   TStreamerObject(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObject();
   TClass        *GetClass() const {return fClassObject;}
   const char    *GetInclude() const;
   virtual void   Init(TObject *obj=0);
   
   ClassDef(TStreamerObject,1)  //Streamer element of type object
};

//________________________________________________________________________
class TStreamerObjectAny : public TStreamerElement {

protected:
   TClass           *fClassObject;    //!pointer to class of object
   
public:

   TStreamerObjectAny();
   TStreamerObjectAny(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObjectAny();
   const char    *GetInclude() const;
   virtual void   Init(TObject *obj=0);
   
   ClassDef(TStreamerObjectAny,1)  //Streamer element of type object other than TObject
};

//________________________________________________________________________
class TStreamerObjectPointer : public TStreamerElement {

protected:
   TClass           *fClassObject;    //!pointer to class of object
   
public:

   TStreamerObjectPointer();
   TStreamerObjectPointer(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerObjectPointer();
   TClass        *GetClass() const {return fClassObject;}
   const char    *GetInclude() const;
   virtual void   Init(TObject *obj=0);
   virtual Bool_t IsaPointer() const {return kTRUE;}
   
   ClassDef(TStreamerObjectPointer,1)  //Streamer element of type pointer to a TObject
};

//________________________________________________________________________
class TStreamerString : public TStreamerElement {
   
public:

   TStreamerString();
   TStreamerString(const char *name, const char *title, Int_t offset);
   virtual       ~TStreamerString();
   
   ClassDef(TStreamerString,1)  //Streamer element of type TString
};
 
//________________________________________________________________________
class TStreamerSTL : public TStreamerElement {
   
protected:
   Int_t       fSTLtype;       //type of STL vector
   Int_t       fCtype;         //STL contained type
      
public:

   TStreamerSTL();
   TStreamerSTL(const char *name, const char *title, Int_t offset, const char *typeName, Bool_t dmPointer);
   virtual       ~TStreamerSTL();
   Int_t          GetSTLtype() const {return fSTLtype;}
   Int_t          GetCtype()   const {return fCtype;}
   const char    *GetInclude() const;
   virtual void   ls(Option_t *option="") const;
     
   ClassDef(TStreamerSTL,1)  //Streamer element of type STL container
};

//________________________________________________________________________
class TStreamerSTLstring : public TStreamerSTL {
   
public:

   TStreamerSTLstring();
   TStreamerSTLstring(const char *name, const char *title, Int_t offset, const char *typeName);
   virtual       ~TStreamerSTLstring();
   const char    *GetInclude() const;
   
   ClassDef(TStreamerSTLstring,1)  //Streamer element of type  C++ string
};
  
#endif
