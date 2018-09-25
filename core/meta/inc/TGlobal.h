// @(#)root/meta:$Id$
// Author: Rene Brun   13/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGlobal
#define ROOT_TGlobal


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGlobal                                                              //
//                                                                      //
// Global variables class (global variables are obtained from CINT).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TDictionary.h"


class TGlobal : public TDictionary {

private:
   DataMemberInfo_t  *fInfo;      //!pointer to CINT data member info

public:
   TGlobal(DataMemberInfo_t *info = 0);
   TGlobal (const TGlobal &);
   TGlobal &operator=(const TGlobal &);

   virtual       ~TGlobal();
   virtual Int_t  GetArrayDim() const;
   virtual DeclId_t GetDeclId() const;
   virtual Int_t  GetMaxIndex(Int_t dim) const;
   virtual void  *GetAddress() const;
   virtual const char *GetTypeName() const;
   virtual const char *GetFullTypeName() const;
   virtual Bool_t IsValid();
   virtual Long_t Property() const;
   virtual bool   Update(DataMemberInfo_t *info);

   ClassDef(TGlobal,2)  //Global variable class
};

//Class to map the "funcky" globals and be able to add them to the list of globals.
class TGlobalMappedFunctionBase : public TGlobal {
public:
   TGlobalMappedFunctionBase(const char *name, const char *type) { SetNameTitle(name, type); }
   virtual ~TGlobalMappedFunctionBase() {}
   Int_t GetArrayDim() const override { return 0; }
   Int_t GetMaxIndex(Int_t /*dim*/) const override { return -1; }
   const char *GetTypeName() const override { return fTitle; }
   const char *GetFullTypeName() const override { return fTitle; }
   Long_t Property() const override { return 0; }
   virtual bool Update(DataMemberInfo_t * /*info*/) override { return false; }
   static void Add(TGlobalMappedFunctionBase *gmf);

private:
   TGlobalMappedFunctionBase &operator=(const TGlobal &); // not implemented.
   // Some of the special ones are created before the list is create e.g gFile
   // We need to buffer them.
   static TList &GetEarlyRegisteredGlobals();

   friend class TROOT;
};

/// Templated class to support any kind of function signature (with or without ref as ret value).
template <typename FuncRef>
class TGlobalMappedFunctionTempl : public TGlobalMappedFunctionBase {
public:
   TGlobalMappedFunctionTempl(const char *name, const char *type, FuncRef *func)
      : TGlobalMappedFunctionBase(name, type), fFuncPtr(func)
   {
   }
   virtual ~TGlobalMappedFunctionTempl() {}
   DeclId_t GetDeclId() const override { return (DeclId_t)(fFuncPtr); } // Used as DeclId because of uniqueness
   void *GetAddress() const override { return static_cast<void *>((*fFuncPtr)()); }

private:
   FuncRef *fFuncPtr; // Function to call to get the address
};


/// keep this class for backwards compatibility
class TGlobalMappedFunction : public TGlobalMappedFunctionBase {
public:
   typedef void *(*GlobalFunc_t)();

   TGlobalMappedFunction(const char *name, const char *type, GlobalFunc_t funcptr)
      : TGlobalMappedFunctionBase(name, type), fFuncPtr(funcptr)
   {
   }
   virtual ~TGlobalMappedFunction() {}

   DeclId_t GetDeclId() const override { return (DeclId_t)(fFuncPtr); } // Used as DeclId because of uniqueness
   void *GetAddress() const override { return static_cast<void *>((*fFuncPtr)()); }

   template <typename FuncRef>
   static void AddGlobal(const char *name, const char *type, FuncRef *func)
   {
      TGlobalMappedFunctionBase::Add(new TGlobalMappedFunctionTempl<FuncRef>(name, type, func));
   }

private:
   GlobalFunc_t fFuncPtr; // Function to call to get the address

};

#endif
