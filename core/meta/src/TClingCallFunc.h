// root/core/meta
// vim: sw=3
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CallFunc
#define ROOT_CallFunc

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingCallFunc                                                       //
//                                                                      //
// Emulation of the CINT CallFunc class.                                //
//                                                                      //
// The CINT C++ interpreter provides an interface for calling           //
// functions through the generated wrappers in dictionaries with        //
// the CallFunc class. This class provides the same functionality,      //
// using an interface as close as possible to CallFunc but the          //
// function metadata and calling service comes from the Cling           //
// C++ interpreter and the Clang C++ compiler, not CINT.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingMethodInfo.h"
#include "TClingClassInfo.h"

#include "llvm/ExecutionEngine/GenericValue.h"
#include "cling/Interpreter/StoredValueRef.h"

#include <vector>

namespace clang {
class Expr;
class FunctionDecl;
class CXXMethodDecl;
}

namespace cling {
class Interpreter;
class Value;
}

class TClingClassInfo;
class TInterpreterValue;

typedef void (*tcling_callfunc_Wrapper_t)(void*, int, void**, void*);
typedef void (*tcling_callfunc_ctor_Wrapper_t)(void**, void*, unsigned long);
typedef void (*tcling_callfunc_dtor_Wrapper_t)(void*, unsigned long, int);

class TClingCallFunc {

private:

   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Current method, we own.
   TClingMethodInfo* fMethod;
   /// Pointer to compiled wrapper, we do *not* own.
   tcling_callfunc_Wrapper_t fWrapper;
   /// Stored function arguments, we own.
   mutable std::vector<cling::StoredValueRef> fArgVals;
   /// If true, do not limit number of function arguments to declared number.
   bool fIgnoreExtraArgs;

private:
   void* compile_wrapper(const std::string& wrapper_name,
                         const std::string& wrapper,
                         bool withAccessControl = true);

   void collect_type_info(clang::QualType& QT, std::ostringstream& typedefbuf,
                          std::ostringstream& callbuf, std::string& type_name,
                          bool& isReference, int& ptrCnt, int indent_level,
                          bool forArgument);

   void make_narg_call(const unsigned N, std::ostringstream& typedefbuf,
                       std::ostringstream& callbuf,
                       const std::string& class_name, int indent_level);

   void make_narg_ctor(const unsigned N, std::ostringstream& typedefbuf,
                       std::ostringstream& callbuf,
                       const std::string& class_name, int indent_level);

   void make_narg_call_with_return(const unsigned N,
                                   const std::string& class_name,
                                   std::ostringstream& buf, int indent_level);

   void make_narg_ctor_with_return(const unsigned N,
                                   const std::string& class_name,
                                   std::ostringstream& buf, int indent_level);

   tcling_callfunc_Wrapper_t make_wrapper();

   tcling_callfunc_ctor_Wrapper_t
   make_ctor_wrapper(const TClingClassInfo* info);

   tcling_callfunc_dtor_Wrapper_t
   make_dtor_wrapper(const TClingClassInfo* info);

   void exec(void* address, void* ret) const;
   void exec_with_valref_return(void* address,
                                cling::StoredValueRef* ret) const;

   void EvaluateArgList(const std::string& ArgList);

public:

   ~TClingCallFunc() { 
      delete fMethod;
   }

   explicit TClingCallFunc(cling::Interpreter *interp)
      : fInterp(interp), fWrapper(0), fIgnoreExtraArgs(false)
   {
      fMethod = new TClingMethodInfo(interp);
   }

   TClingCallFunc(const TClingCallFunc &rhs)
      : fInterp(rhs.fInterp), fWrapper(rhs.fWrapper), fArgVals(rhs.fArgVals),
        fIgnoreExtraArgs(rhs.fIgnoreExtraArgs)
   {
      fMethod = new TClingMethodInfo(*rhs.fMethod);
   }

   TClingCallFunc &operator=(const TClingCallFunc &rhs)
   {
      if (this != &rhs) {
         fInterp = rhs.fInterp;
         delete fMethod;
         fMethod = new TClingMethodInfo(*rhs.fMethod);
         fWrapper = rhs.fWrapper;
         fArgVals = rhs.fArgVals;
         fIgnoreExtraArgs = rhs.fIgnoreExtraArgs;
      }
      return *this;
   }

   void* ExecDefaultConstructor(const TClingClassInfo* info, void* address = 0,
                                unsigned long nary = 0UL);
   void ExecDestructor(const TClingClassInfo* info, void* address = 0,
                       unsigned long nary = 0UL, bool withFree = true);
   void ExecWithReturn(void* address, void* ret = 0);
   void Exec(void* address, TInterpreterValue* interpVal = 0);
   long ExecInt(void* address);
   long long ExecInt64(void* address);
   double ExecDouble(void* address);
   TClingMethodInfo* FactoryMethod() const;
   void IgnoreExtraArgs(bool ignore) { fIgnoreExtraArgs = ignore; }
   void Init();
   void Init(TClingMethodInfo*);
   void* InterfaceMethod() const;
   bool IsValid() const;
   void ResetArg();
   void SetArg(long arg);
   void SetArg(double arg);
   void SetArg(long long arg);
   void SetArg(unsigned long long arg);
   void SetArgArray(long* argArr, int narg);
   void SetArgs(const char* args);
   void SetFunc(const TClingClassInfo* info, const char* method,
                const char* arglist, long* poffset);
   void SetFunc(const TClingClassInfo* info, const char* method,
                const char* arglist, bool objectIsConst, long* poffset);
   void SetFunc(const TClingMethodInfo* info);
   void SetFuncProto(const TClingClassInfo* info, const char* method,
                     const char* proto, long* poffset,
                     ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void SetFuncProto(const TClingClassInfo* info, const char* method,
                     const char* proto, bool objectIsConst, long* poffset,
                     ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void SetFuncProto(const TClingClassInfo* info, const char* method,
                     const llvm::SmallVector<clang::QualType, 4>& proto,
                     long* poffset,
                     ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
   void SetFuncProto(const TClingClassInfo* info, const char* method,
                     const llvm::SmallVector<clang::QualType, 4>& proto,
                     bool objectIsConst, long* poffset,
                     ROOT::EFunctionMatchMode mode = ROOT::kConversionMatch);
};

#endif // ROOT_CallFunc
