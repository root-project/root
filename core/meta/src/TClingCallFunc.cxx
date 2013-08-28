// root/core/meta
// vim: sw=3
// Author: Paul Russo   30/07/2012
// Author: Vassil Vassilev   9/02/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TClingCallFunc.h"

#include "TClingClassInfo.h"
#include "TClingMethodInfo.h"
#include "TInterpreterValue.h"
#include "TMetaUtils.h"
#include "TSystem.h"

#include "TError.h"
#include "TCling.h"

#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "clang/Sema/SemaInternal.h"

#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <vector>

using namespace ROOT;
using namespace llvm;
using namespace clang;
using namespace std;

// This ought to be declared by the implementer .. oh well...
extern void unresolvedSymbol();

static unsigned long long wrapper_serial = 0LL;
static const string indent_string("   ");

static map<const FunctionDecl*, void*> wrapper_store;

static
cling::StoredValueRef
EvaluateExpr(cling::Interpreter* interp, const Expr* E)
{
   // Evaluate an Expr* and return its cling::StoredValueRef
   ASTContext& C = interp->getCI()->getASTContext();
   APSInt res;
   if (E->EvaluateAsInt(res, C, /*AllowSideEffects*/Expr::SE_NoSideEffects)) {
      GenericValue gv;
      gv.IntVal = res;
      return cling::StoredValueRef::bitwiseCopy(C, cling::Value(gv, C.IntTy));
   }
   // TODO: Build a wrapper around the expression to avoid decompilation and
   // compilation and other string operations.
   PrintingPolicy Policy(C.getPrintingPolicy());
   Policy.SuppressTagKeyword = true;
   Policy.SuppressUnwrittenScope = false;
   Policy.SuppressInitializers = false;
   Policy.AnonymousTagLocations = false;
   string buf;
   raw_string_ostream out(buf);
   E->printPretty(out, /*Helper=*/0, Policy, /*Indentation=*/0);
   out << ';'; // no value printing
   out.flush();
   cling::StoredValueRef valref;
   cling::Interpreter::CompilationResult cr = interp->evaluate(buf, valref);
   if (cr != cling::Interpreter::kSuccess) {
      return cling::StoredValueRef::invalidValue();
   }
   return valref;
}

static
long long
sv_to_long_long(const cling::StoredValueRef& svref)
{
   const cling::Value& valref = svref.get();
   QualType QT = valref.getClangType();
   GenericValue gv = valref.getGV();
   if (QT->isMemberPointerType()) {
      const MemberPointerType* MPT =
         QT->getAs<MemberPointerType>();
      if (MPT->isMemberDataPointer()) {
         return (long long) (ptrdiff_t) gv.PointerVal;
      }
      return (long long) gv.PointerVal;
   }
   if (QT->isPointerType() || QT->isArrayType() || QT->isRecordType() ||
         QT->isReferenceType()) {
      return (long long) gv.PointerVal;
   }
   if (const EnumType* ET = dyn_cast<EnumType>(&*QT)) {
      // Note: We may need to worry about the underlying type
      //       of the enum here.
      (void) ET;
      return gv.IntVal.getSExtValue();
   }
   if (const BuiltinType* BT =
            dyn_cast<BuiltinType>(&*QT)) {
      //
      //  WARNING!!!
      //
      //  This switch is organized in order-of-declaration
      //  so that the produced assembly code is optimal.
      //  Do not reorder!
      //
      switch (BT->getKind()) {
            //
            //  Builtin Types
            //
         case BuiltinType::Void: {
               // void
            }
            break;
            //
            //  Unsigned Types
            //
         case BuiltinType::Bool: {
               // bool
               return (long long) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char_U: {
               // char on targets where it is unsigned
            }
            break;
         case BuiltinType::UChar: {
               // unsigned char
            }
            break;
         case BuiltinType::WChar_U: {
               // wchar_t on targets where it is unsigned
            }
            break;
         case BuiltinType::Char16: {
               // char16_t
            }
            break;
         case BuiltinType::Char32: {
               // char32_t
            }
            break;
         case BuiltinType::UShort: {
               // unsigned short
            }
            break;
         case BuiltinType::UInt: {
               // unsigned int
            }
            break;
         case BuiltinType::ULong: {
               // unsigned long
            }
            break;
         case BuiltinType::ULongLong: {
               // unsigned long long
            }
            break;
         case BuiltinType::UInt128: {
               // __uint128_t
            }
            break;
            //
            //  Signed Types
            //
         case BuiltinType::Char_S: {
               // char on targets where it is signed
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::SChar: {
               // signed char
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::WChar_S: {
               // wchar_t on targets where it is signed
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Short: {
               // short
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int: {
               // int
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Long: {
               // long
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::LongLong: {
               // long long
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int128: {
               // __int128_t
            }
            break;
         case BuiltinType::Half: {
               // half in OpenCL, __fp16 in ARM NEON
            }
            break;
         case BuiltinType::Float: {
               // float
               return (long long) gv.FloatVal;
            }
            break;
         case BuiltinType::Double: {
               // double
               return (long long) gv.DoubleVal;
            }
            break;
         case BuiltinType::LongDouble: {
               // long double
               // FIXME: Implement this!
            }
            break;
            //
            //  Language-Specific Types
            //
         case BuiltinType::NullPtr: {
               // C++11 nullptr
            }
            break;
         case BuiltinType::ObjCId: {
               // Objective C 'id' type
            }
            break;
         case BuiltinType::ObjCClass: {
               // Objective C 'Class' type
            }
            break;
         case BuiltinType::ObjCSel: {
               // Objective C 'SEL' type
            }
            break;
         case BuiltinType::OCLImage1d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dBuffer: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage3d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLSampler: {
               // OpenCL sampler_t
            }
            break;
         case BuiltinType::OCLEvent: {
               // OpenCL event_t
            }
            break;
            //
            //  Placeholder types.
            //
            //  These types are used during intermediate phases
            //  of semantic analysis.  They are eventually resolved
            //  to one of the preceeding types.
            //
         case BuiltinType::Dependent: {
               // dependent on a template argument
            }
            break;
         case BuiltinType::Overload: {
               // an unresolved function overload set
            }
            break;
         case BuiltinType::BoundMember: {
               // a bound C++ non-static member function
            }
            break;
         case BuiltinType::PseudoObject: {
               // Object C @property or VS.NET __property
            }
            break;
         case BuiltinType::UnknownAny: {
               // represents an unknown type
            }
            break;
         case BuiltinType::BuiltinFn: {
               // a compiler builtin function
            }
            break;
         case BuiltinType::ARCUnbridgedCast: {
               // Objective C Automatic Reference Counting cast
               // which would normally require __bridge, but which
               // may be ok because of the context.
            }
            break;
         default: {
               // There should be no others.  This is here in case
               // this changes in the future.
            }
            break;
      }
   }
   Error("TClingCallFunc::sv_to_long_long", "Invalid Type!");
   QT->dump();
   return 0LL;
}

static
unsigned long long
sv_to_ulong_long(const cling::StoredValueRef& svref)
{
   const cling::Value& valref = svref.get();
   QualType QT = valref.getClangType();
   GenericValue gv = valref.getGV();
   if (QT->isMemberPointerType()) {
      const MemberPointerType* MPT =
         QT->getAs<MemberPointerType>();
      if (MPT->isMemberDataPointer()) {
         return (unsigned long long) (ptrdiff_t) gv.PointerVal;
      }
      return (unsigned long long) gv.PointerVal;
   }
   if (QT->isPointerType() || QT->isArrayType() || QT->isRecordType() ||
         QT->isReferenceType()) {
      return (unsigned long long) gv.PointerVal;
   }
   if (const EnumType* ET = dyn_cast<EnumType>(&*QT)) {
      // Note: We may need to worry about the underlying type
      //       of the enum here.
      (void) ET;
      return gv.IntVal.getZExtValue();
   }
   if (const BuiltinType* BT =
            dyn_cast<BuiltinType>(&*QT)) {
      //
      //  WARNING!!!
      //
      //  This switch is organized in order-of-declaration
      //  so that the produced assembly code is optimal.
      //  Do not reorder!
      //
      switch (BT->getKind()) {
            //
            //  Builtin Types
            //
         case BuiltinType::Void: {
               // void
            }
            break;
            //
            //  Unsigned Types
            //
         case BuiltinType::Bool: {
               // bool
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char_U: {
               // char on targets where it is unsigned
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UChar: {
               // unsigned char
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::WChar_U: {
               // wchar_t on targets where it is unsigned
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char16: {
               // char16_t
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char32: {
               // char32_t
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UShort: {
               // unsigned short
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UInt: {
               // unsigned int
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::ULong: {
               // unsigned long
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::ULongLong: {
               // unsigned long long
               return gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UInt128: {
               // __uint128_t
            }
            break;
            //
            //  Signed Types
            //
         case BuiltinType::Char_S: {
               // char on targets where it is signed
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::SChar: {
               // signed char
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::WChar_S: {
               // wchar_t on targets where it is signed
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Short: {
               // short
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int: {
               // int
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Long: {
               // long
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::LongLong: {
               // long long
               return gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int128: {
               // __int128_t
            }
            break;
         case BuiltinType::Half: {
               // half in OpenCL, __fp16 in ARM NEON
            }
            break;
         case BuiltinType::Float: {
               // float
               return (unsigned long long) gv.FloatVal;
            }
            break;
         case BuiltinType::Double: {
               // double
               return (unsigned long long) gv.DoubleVal;
            }
            break;
         case BuiltinType::LongDouble: {
               // long double
               // FIXME: Implement this!
            }
            break;
            //
            //  Language-Specific Types
            //
         case BuiltinType::NullPtr: {
               // C++11 nullptr
            }
            break;
         case BuiltinType::ObjCId: {
               // Objective C 'id' type
            }
            break;
         case BuiltinType::ObjCClass: {
               // Objective C 'Class' type
            }
            break;
         case BuiltinType::ObjCSel: {
               // Objective C 'SEL' type
            }
            break;
         case BuiltinType::OCLImage1d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dBuffer: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage3d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLSampler: {
               // OpenCL sampler_t
            }
            break;
         case BuiltinType::OCLEvent: {
               // OpenCL event_t
            }
            break;
            //
            //  Placeholder types.
            //
            //  These types are used during intermediate phases
            //  of semantic analysis.  They are eventually resolved
            //  to one of the preceeding types.
            //
         case BuiltinType::Dependent: {
               // dependent on a template argument
            }
            break;
         case BuiltinType::Overload: {
               // an unresolved function overload set
            }
            break;
         case BuiltinType::BoundMember: {
               // a bound C++ non-static member function
            }
            break;
         case BuiltinType::PseudoObject: {
               // Object C @property or VS.NET __property
            }
            break;
         case BuiltinType::UnknownAny: {
               // represents an unknown type
            }
            break;
         case BuiltinType::BuiltinFn: {
               // a compiler builtin function
            }
            break;
         case BuiltinType::ARCUnbridgedCast: {
               // Objective C Automatic Reference Counting cast
               // which would normally require __bridge, but which
               // may be ok because of the context.
            }
            break;
         default: {
               // There should be no others.  This is here in case
               // this changes in the future.
            }
            break;
      }
   }
   Error("TClingCallFunc::sv_to_ulong_long", "Invalid Type!");
   QT->dump();
   return 0ULL;
}

static
double
sv_to_double(const cling::StoredValueRef& svref)
{
   const cling::Value& valref = svref.get();
   QualType QT = valref.getClangType();
   GenericValue gv = valref.getGV();
   if (QT->isMemberPointerType()) {
      const MemberPointerType* MPT =
         QT->getAs<MemberPointerType>();
      if (MPT->isMemberDataPointer()) {
         return (double) *(ptrdiff_t*)gv.PointerVal;
      }
      return (double) (long) gv.PointerVal;
   }
   if (QT->isPointerType() || QT->isArrayType() || QT->isRecordType() ||
         QT->isReferenceType()) {
      return (double) (long) gv.PointerVal;
   }
   if (const EnumType* ET = dyn_cast<EnumType>(&*QT)) {
      // Note: We may need to worry about the underlying type
      //       of the enum here.
      (void) ET;
      return (double) gv.IntVal.getSExtValue();
   }
   if (const BuiltinType* BT =
            dyn_cast<BuiltinType>(&*QT)) {
      //
      //  WARNING!!!
      //
      //  This switch is organized in order-of-declaration
      //  so that the produced assembly code is optimal.
      //  Do not reorder!
      //
      switch (BT->getKind()) {
            //
            //  Builtin Types
            //
         case BuiltinType::Void: {
               // void
            }
            break;
            //
            //  Unsigned Types
            //
         case BuiltinType::Bool: {
               // bool
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char_U: {
               // char on targets where it is unsigned
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UChar: {
               // unsigned char
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::WChar_U: {
               // wchar_t on targets where it is unsigned
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char16: {
               // char16_t
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::Char32: {
               // char32_t
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UShort: {
               // unsigned short
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UInt: {
               // unsigned int
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::ULong: {
               // unsigned long
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::ULongLong: {
               // unsigned long long
               return (double) gv.IntVal.getZExtValue();
            }
            break;
         case BuiltinType::UInt128: {
               // __uint128_t
            }
            break;
            //
            //  Signed Types
            //
         case BuiltinType::Char_S: {
               // char on targets where it is signed
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::SChar: {
               // signed char
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::WChar_S: {
               // wchar_t on targets where it is signed
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Short: {
               // short
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int: {
               // int
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Long: {
               // long
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::LongLong: {
               // long long
               return (double) gv.IntVal.getSExtValue();
            }
            break;
         case BuiltinType::Int128: {
               // __int128_t
            }
            break;
         case BuiltinType::Half: {
               // half in OpenCL, __fp16 in ARM NEON
            }
            break;
         case BuiltinType::Float: {
               // float
               return (double) gv.FloatVal;
            }
            break;
         case BuiltinType::Double: {
               // double
               return gv.DoubleVal;
            }
            break;
         case BuiltinType::LongDouble: {
               // long double
               // FIXME: Implement this!
            }
            break;
            //
            //  Language-Specific Types
            //
         case BuiltinType::NullPtr: {
               // C++11 nullptr
            }
            break;
         case BuiltinType::ObjCId: {
               // Objective C 'id' type
            }
            break;
         case BuiltinType::ObjCClass: {
               // Objective C 'Class' type
            }
            break;
         case BuiltinType::ObjCSel: {
               // Objective C 'SEL' type
            }
            break;
         case BuiltinType::OCLImage1d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage1dBuffer: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage2dArray: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLImage3d: {
               // OpenCL image type
            }
            break;
         case BuiltinType::OCLSampler: {
               // OpenCL sampler_t
            }
            break;
         case BuiltinType::OCLEvent: {
               // OpenCL event_t
            }
            break;
            //
            //  Placeholder types.
            //
            //  These types are used during intermediate phases
            //  of semantic analysis.  They are eventually resolved
            //  to one of the preceeding types.
            //
         case BuiltinType::Dependent: {
               // dependent on a template argument
            }
            break;
         case BuiltinType::Overload: {
               // an unresolved function overload set
            }
            break;
         case BuiltinType::BoundMember: {
               // a bound C++ non-static member function
            }
            break;
         case BuiltinType::PseudoObject: {
               // Object C @property or VS.NET __property
            }
            break;
         case BuiltinType::UnknownAny: {
               // represents an unknown type
            }
            break;
         case BuiltinType::BuiltinFn: {
               // a compiler builtin function
            }
            break;
         case BuiltinType::ARCUnbridgedCast: {
               // Objective C Automatic Reference Counting cast
               // which would normally require __bridge, but which
               // may be ok because of the context.
            }
            break;
         default: {
               // There should be no others.  This is here in case
               // this changes in the future.
            }
            break;
      }
   }
   Error("TClingCallFunc::sv_to_double", "Invalid Type!");
   QT->dump();
   return 0.0;
}

void
TClingCallFunc::
collect_type_info(QualType& QT, ostringstream& typedefbuf,
                  ostringstream& callbuf, string& type_name,
                  bool& isReference, int& ptrCnt, int indent_level)
{
   //
   //  Collect information about type type of a function parameter
   //  needed for building the wrapper function.
   //
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   PrintingPolicy Policy(FD->getASTContext().getPrintingPolicy());
   isReference = false;
   ptrCnt = 0;
   while (1) {
      if (QT->isArrayType()) {
         ++ptrCnt;
         QT = cast<clang::ArrayType>(QT)->getElementType();
         continue;
      }
      else if (QT->isFunctionPointerType()) {
         string fp_typedef_name;
         {
            ostringstream nm;
            nm << "FP" << wrapper_serial++;
            type_name = nm.str();
            raw_string_ostream OS(fp_typedef_name);
            QT.print(OS, Policy, type_name);
            OS.flush();
         }
         for (int i = 0; i < indent_level; ++i) {
            typedefbuf << indent_string;
         }
         typedefbuf << "typedef " << fp_typedef_name << ";\n";
         break;
      }
      else if (QT->isMemberPointerType()) {
         string mp_typedef_name;
         {
            ostringstream nm;
            nm << "MP" << wrapper_serial++;
            type_name = nm.str();
            raw_string_ostream OS(mp_typedef_name);
            QT.print(OS, Policy, type_name);
            OS.flush();
         }
         for (int i = 0; i < indent_level; ++i) {
            typedefbuf << indent_string;
         }
         typedefbuf << "typedef " << mp_typedef_name << ";\n";
         break;
      }
      else if (QT->isPointerType()) {
         ++ptrCnt;
         QT = cast<clang::PointerType>(QT)->getPointeeType();
         continue;
      }
      else if (QT->isReferenceType()) {
         isReference = true;
         QT = cast<ReferenceType>(QT)->getPointeeType();
         continue;
      }
      QT.getAsStringInternal(type_name, Policy);
      break;
   }
}

void
TClingCallFunc::
make_narg_ctor(const unsigned N, ostringstream& typedefbuf,
               ostringstream& callbuf, const string& class_name,
               int indent_level)
{
   // Make a code string that follows this pattern:
   //
   // new ClassName(args...)
   //
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   PrintingPolicy Policy(FD->getASTContext().getPrintingPolicy());
   callbuf << "new " << class_name << "(";
   for (unsigned i = 0U; i < N; ++i) {
      const ParmVarDecl* PVD = FD->getParamDecl(i);
      QualType Ty = PVD->getType();
      QualType QT = Ty.getCanonicalType();
      string type_name;
      bool isReference = false;
      int ptrCnt = 0;
      collect_type_info(QT, typedefbuf, callbuf, type_name,
                        isReference, ptrCnt, indent_level);
      if (i) {
         callbuf << ',';
         if (i % 2) {
            callbuf << ' ';
         }
         else {
            callbuf << "\n";
            for (int j = 0; j <= indent_level; ++j) {
               callbuf << indent_string;
            }
         }
      }
      if (isReference) {
         string stars;
         for (int j = 0; j < ptrCnt; ++j) {
            stars.push_back('*');
         }
         callbuf << "**(" << type_name.c_str() << stars << "**)args["
                 << i << "]";
      }
      else if (ptrCnt) {
         string stars;
         for (int j = 0; j < ptrCnt; ++j) {
            stars.push_back('*');
         }
         callbuf << "*(" << type_name.c_str() << stars << "*)args["
                 << i << "]";
      }
      else {
         callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
      }
   }
   callbuf << ")";
}

void
TClingCallFunc::
make_narg_call(const unsigned N, ostringstream& typedefbuf,
               ostringstream& callbuf, const string& class_name,
               int indent_level)
{
   //
   // Make a code string that follows this pattern:
   //
   // ((<class>*)obj)-><method>(*(<arg-i-type>*)args[i], ...)
   //
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   PrintingPolicy Policy(FD->getASTContext().getPrintingPolicy());
   if (const TypeDecl* TD =
            dyn_cast<TypeDecl>(FD->getDeclContext())) {
      // This is a class, struct, or union member.
      (void) TD;
      callbuf << "((" << class_name << "*)obj)->";
   }
   else if (const NamedDecl* ND =
               dyn_cast<NamedDecl>(FD->getDeclContext())) {
      // This is a namespace member.
      (void) ND;
      callbuf << class_name << "::";
   }
   callbuf << fMethod->Name() << "(";
   for (unsigned i = 0U; i < N; ++i) {
      const ParmVarDecl* PVD = FD->getParamDecl(i);
      QualType Ty = PVD->getType();
      QualType QT = Ty.getCanonicalType();
      string type_name;
      bool isReference = false;
      int ptrCnt = 0;
      collect_type_info(QT, typedefbuf, callbuf, type_name,
                        isReference, ptrCnt, indent_level);
      if (i) {
         callbuf << ',';
         if (i % 2) {
            callbuf << ' ';
         }
         else {
            callbuf << "\n";
            for (int j = 0; j <= indent_level; ++j) {
               callbuf << indent_string;
            }
         }
      }
      if (isReference) {
         string stars;
         for (int j = 0; j < ptrCnt; ++j) {
            stars.push_back('*');
         }
         callbuf << "**(" << type_name.c_str() << stars << "**)args["
                 << i << "]";
      }
      else if (ptrCnt) {
         string stars;
         for (int j = 0; j < ptrCnt; ++j) {
            stars.push_back('*');
         }
         callbuf << "*(" << type_name.c_str() << stars << "*)args["
                 << i << "]";
      }
      else {
         callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
      }
   }
   callbuf << ")";
}

void
TClingCallFunc::
make_narg_ctor_with_return(const unsigned N, const string& class_name,
                           ostringstream& buf, int indent_level)
{
   // Make a code string that follows this pattern:
   //
   // if (ret) {
   //    (*(ClassName**)ret) = new ClassName(args...);
   // }
   // else {
   //    new ClassName(args...);
   // }
   //
   for (int i = 0; i < indent_level; ++i) {
      buf << indent_string;
   }
   buf << "if (ret) {\n";
   ++indent_level;
   {
      ostringstream typedefbuf;
      ostringstream callbuf;
      //
      //  Write the return value assignment part.
      //
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      callbuf << "(*(" << class_name << "**)ret) = ";
      //
      //  Write the actual new expression.
      //
      make_narg_ctor(N, typedefbuf, callbuf, class_name, indent_level);
      //
      //  End the new expression statement.
      //
      callbuf << ";\n";
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      callbuf << "return;\n";
      //
      //  Output the whole new expression and return statement.
      //
      buf << typedefbuf.str() << callbuf.str();
   }
   --indent_level;
   for (int i = 0; i < indent_level; ++i) {
      buf << indent_string;
   }
   buf << "}\n";
   for (int i = 0; i < indent_level; ++i) {
      buf << indent_string;
   }
   buf << "else {\n";
   ++indent_level;
   {
      ostringstream typedefbuf;
      ostringstream callbuf;
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      make_narg_ctor(N, typedefbuf, callbuf, class_name, indent_level);
      callbuf << ";\n";
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      callbuf << "return;\n";
      buf << typedefbuf.str() << callbuf.str();
   }
   --indent_level;
   for (int i = 0; i < indent_level; ++i) {
      buf << indent_string;
   }
   buf << "}\n";
}

void
TClingCallFunc::
make_narg_call_with_return(const unsigned N, const string& class_name,
                           ostringstream& buf, int indent_level)
{
   // Make a code string that follows this pattern:
   //
   // if (ret) {
   //    new (ret) (return_type) ((class_name*)obj)->func(args...);
   // }
   // else {
   //    ((class_name*)obj)->func(args...);
   // }
   //
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   if (const CXXConstructorDecl* CD = dyn_cast<CXXConstructorDecl>(FD)) {
      (void) CD;
      make_narg_ctor_with_return(N, class_name, buf, indent_level);
      return;
   }
   QualType QT = FD->getResultType().getCanonicalType();
   if (QT->isVoidType()) {
      ostringstream typedefbuf;
      ostringstream callbuf;
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      make_narg_call(N, typedefbuf, callbuf, class_name, indent_level);
      callbuf << ";\n";
      for (int i = 0; i < indent_level; ++i) {
         callbuf << indent_string;
      }
      callbuf << "return;\n";
      buf << typedefbuf.str() << callbuf.str();
   }
   else {
      for (int i = 0; i < indent_level; ++i) {
         buf << indent_string;
      }
      buf << "if (ret) {\n";
      ++indent_level;
      {
         ostringstream typedefbuf;
         ostringstream callbuf;
         //
         //  Write the placement part of the placement new.
         //
         for (int i = 0; i < indent_level; ++i) {
            callbuf << indent_string;
         }
         callbuf << "new (ret) ";
         string type_name;
         bool isReference = false;
         int ptrCnt = 0;
         collect_type_info(QT, typedefbuf, callbuf, type_name,
                           isReference, ptrCnt, indent_level);
         //
         //  Write the type part of the placement new.
         //
         if (isReference) {
            string stars;
            for (int j = 0; j < ptrCnt; ++j) {
               stars.push_back('*');
            }
            callbuf << "(" << type_name.c_str() << stars << "*) (&";
         }
         else if (ptrCnt) {
            string stars;
            for (int j = 0; j < ptrCnt; ++j) {
               stars.push_back('*');
            }
            callbuf << "(" << type_name.c_str() << stars << ") (";
         }
         else {
            callbuf << "(" << type_name.c_str() << ") (";
         }
         //
         //  Write the actual function call.
         //
         make_narg_call(N, typedefbuf, callbuf, class_name, indent_level);
         //
         //  End the placement new.
         //
         callbuf << ");\n";
         for (int i = 0; i < indent_level; ++i) {
            callbuf << indent_string;
         }
         callbuf << "return;\n";
         //
         //  Output the whole placement new expression and return statement.
         //
         buf << typedefbuf.str() << callbuf.str();
      }
      --indent_level;
      for (int i = 0; i < indent_level; ++i) {
         buf << indent_string;
      }
      buf << "}\n";
      for (int i = 0; i < indent_level; ++i) {
         buf << indent_string;
      }
      buf << "else {\n";
      ++indent_level;
      {
         ostringstream typedefbuf;
         ostringstream callbuf;
         for (int i = 0; i < indent_level; ++i) {
            callbuf << indent_string;
         }
         make_narg_call(N, typedefbuf, callbuf, class_name, indent_level);
         callbuf << ";\n";
         for (int i = 0; i < indent_level; ++i) {
            callbuf << indent_string;
         }
         callbuf << "return;\n";
         buf << typedefbuf.str() << callbuf.str();
      }
      --indent_level;
      for (int i = 0; i < indent_level; ++i) {
         buf << indent_string;
      }
      buf << "}\n";
   }
}

tcling_callfunc_Wrapper_t
TClingCallFunc::
make_wrapper()
{
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   ASTContext& Context = FD->getASTContext();
   PrintingPolicy Policy(Context.getPrintingPolicy());
   //
   //  Get the class or namespace name.
   //
   string class_name;
   if (const TypeDecl* TD =
            dyn_cast<TypeDecl>(FD->getDeclContext())) {
      // This is a class, struct, or union member.
      QualType QT(TD->getTypeForDecl(), 0);
      ROOT::TMetaUtils::GetFullyQualifiedTypeName(class_name, QT, *fInterp);
   }
   else if (const NamedDecl* ND =
               dyn_cast<NamedDecl>(FD->getDeclContext())) {
      // This is a namespace member.
      raw_string_ostream stream(class_name);
      ND->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);
      stream.flush();
   }
   //
   //  Check to make sure that we can
   //  instantiate and codegen this function.
   //
   bool needInstantiation = false;
   const FunctionDecl* Definition = 0;
   if (!FD->isDefined(Definition)) {
      FunctionDecl::TemplatedKind TK = FD->getTemplatedKind();
      switch (TK) {
         case FunctionDecl::TK_NonTemplate: {
               // Ordinary function, not a template specialization.
               // Note: This might be ok, the body might be defined
               //       in a library, and all we have seen is the
               //       header file.
               //Error("TClingCallFunc::make_wrapper",
               //      "Cannot make wrapper for a function which is "
               //      "declared but not defined!");
               //return 0;
            }
            break;
         case FunctionDecl::TK_FunctionTemplate: {
               // This decl is actually a function template,
               // not a function at all.
               Error("TClingCallFunc::make_wrapper",
                     "Cannot make wrapper for a function template!");
               return 0;
            }
            break;
         case FunctionDecl::TK_MemberSpecialization: {
               // This function is the result of instantiating an ordinary
               // member function of a class template, or of instantiating
               // an ordinary member function of a class member of a class
               // template, or of specializing a member function template
               // of a class template, or of specializing a member function
               // template of a class member of a class template.
               if (!FD->isTemplateInstantiation()) {
                  // We are either TSK_Undeclared or
                  // TSK_ExplicitSpecialization.
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  //Error("TClingCallFunc::make_wrapper",
                  //      "Cannot make wrapper for a function template "
                  //      "explicit specialization which is declared "
                  //      "but not defined!");
                  //return 0;
                  break;
               }
               const FunctionDecl* Pattern =
                  FD->getTemplateInstantiationPattern();
               if (!Pattern) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a member function "
                        "instantiation with no pattern!");
                  return 0;
               }
               FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
               TemplateSpecializationKind PTSK =
                  Pattern->getTemplateSpecializationKind();
               if (
                     // The pattern is an ordinary member function.
                     (PTK == FunctionDecl::TK_NonTemplate) || 
                     // The pattern is an explicit specialization, and
                     // so is not a template.
                     ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                      ((PTSK == TSK_Undeclared) ||
                       (PTSK == TSK_ExplicitSpecialization)))
               ) {
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  break;
               }
               else if (!Pattern->hasBody()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a member function "
                        "instantiation with no body!");
                  return 0;
               }
               if (FD->isImplicitlyInstantiable()) {
                  needInstantiation = true;
               }
            }
            break;
         case FunctionDecl::TK_FunctionTemplateSpecialization: {
               // This function is the result of instantiating a function
               // template or possibly an explicit specialization of a
               // function template.  Could be a namespace scope function or a
               // member function.
               if (!FD->isTemplateInstantiation()) {
                  // We are either TSK_Undeclared or
                  // TSK_ExplicitSpecialization.
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  //Error("TClingCallFunc::make_wrapper",
                  //      "Cannot make wrapper for a function template "
                  //      "explicit specialization which is declared "
                  //      "but not defined!");
                  //return 0;
                  break;
               }
               const FunctionDecl* Pattern =
                  FD->getTemplateInstantiationPattern();
               if (!Pattern) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a function template"
                        "instantiation with no pattern!");
                  return 0;
               }
               FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
               TemplateSpecializationKind PTSK =
                  Pattern->getTemplateSpecializationKind();
               if (
                     // The pattern is an ordinary member function.
                     (PTK == FunctionDecl::TK_NonTemplate) || 
                     // The pattern is an explicit specialization, and
                     // so is not a template.
                     ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                      ((PTSK == TSK_Undeclared) ||
                       (PTSK == TSK_ExplicitSpecialization)))
               ) {
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  break;
               }
               if (!Pattern->hasBody()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a function template"
                        "instantiation with no body!");
                  return 0;
               }
               if (FD->isImplicitlyInstantiable()) {
                  needInstantiation = true;
               }
            }
            break;
         case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
               // This function is the result of instantiating or
               // specializing a  member function of a class template,
               // or a member function of a class member of a class template,
               // or a member function template of a class template, or a
               // member function template of a class member of a class
               // template where at least some part of the function is
               // dependent on a template argument.
               if (!FD->isTemplateInstantiation()) {
                  // We are either TSK_Undeclared or
                  // TSK_ExplicitSpecialization.
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  //Error("TClingCallFunc::make_wrapper",
                  //      "Cannot make wrapper for a dependent function "
                  //      "template explicit specialization which is declared "
                  //      "but not defined!");
                  //return 0;
                  break;
               }
               const FunctionDecl* Pattern =
                  FD->getTemplateInstantiationPattern();
               if (!Pattern) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a dependent function template"
                        "instantiation with no pattern!");
                  return 0;
               }
               FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
               TemplateSpecializationKind PTSK =
                  Pattern->getTemplateSpecializationKind();
               if (
                     // The pattern is an ordinary member function.
                     (PTK == FunctionDecl::TK_NonTemplate) || 
                     // The pattern is an explicit specialization, and
                     // so is not a template.
                     ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                      ((PTSK == TSK_Undeclared) ||
                       (PTSK == TSK_ExplicitSpecialization)))
               ) {
                  // Note: This might be ok, the body might be defined
                  //       in a library, and all we have seen is the
                  //       header file.
                  break;
               }
               if (!Pattern->hasBody()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a dependent function template"
                        "instantiation with no body!");
                  return 0;
               }
               if (FD->isImplicitlyInstantiable()) {
                  needInstantiation = true;
               }
            }
            break;
         default: {
               // Will only happen if clang implementation changes.
               // Protect ourselves in case that happens.
               Error("TClingCallFunc::make_wrapper",
                     "Unhandled template kind!");
               return 0;
            }
            break;
      }
      // We do not set needInstantiation to true in these cases:
      //
      // isInvalidDecl()
      // TSK_Undeclared
      // TSK_ExplicitInstantiationDefinition
      // TSK_ExplicitSpecialization && !getClassScopeSpecializationPattern()
      // TSK_ExplicitInstantiationDeclaration &&
      //    getTemplateInstantiationPattern() &&
      //    PatternDecl->hasBody() &&
      //    !PatternDecl->isInlined()
      //
      // Set it true in these cases:
      //
      // TSK_ImplicitInstantiation
      // TSK_ExplicitInstantiationDeclaration && (!getPatternDecl() ||
      //    !PatternDecl->hasBody() || PatternDecl->isInlined())
      //
   }
   if (needInstantiation) {
      clang::FunctionDecl* FDmod = const_cast<clang::FunctionDecl*>(FD);
      clang::Sema& S = fInterp->getSema();
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(fInterp);
      S.InstantiateFunctionDefinition(SourceLocation(), FDmod,
                                      /*Recursive=*/ true,
                                      /*DefinitionRequired=*/ true);
      if (!FD->isDefined(Definition)) {
         Error("TClingCallFunc::make_wrapper",
               "Failed to force template instantiation!");
         return 0;
      }
   }
   if (Definition) {
      FunctionDecl::TemplatedKind TK = Definition->getTemplatedKind();
      switch (TK) {
         case FunctionDecl::TK_NonTemplate: {
               // Ordinary function, not a template specialization.
               if (Definition->isDeleted()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a deleted function!");
                  return 0;
               }
               else if (Definition->isLateTemplateParsed()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a late template parsed "
                        "function!");
                  return 0;
               }
               //else if (Definition->isDefaulted()) {
               //   // Might not have a body, but we can still use it.
               //}
               //else {
               //   // Has a body.
               //}
            }
            break;
         case FunctionDecl::TK_FunctionTemplate: {
               // This decl is actually a function template,
               // not a function at all.
               Error("TClingCallFunc::make_wrapper",
                     "Cannot make wrapper for a function template!");
               return 0;
            }
            break;
         case FunctionDecl::TK_MemberSpecialization: {
               // This function is the result of instantiating an ordinary
               // member function of a class template or of a member class
               // of a class template.
               if (Definition->isDeleted()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a deleted member function "
                        "of a specialization!");
                  return 0;
               }
               else if (Definition->isLateTemplateParsed()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a late template parsed "
                        "member function of a specialization!");
                  return 0;
               }
               //else if (Definition->isDefaulted()) {
               //   // Might not have a body, but we can still use it.
               //}
               //else {
               //   // Has a body.
               //}
            }
            break;
         case FunctionDecl::TK_FunctionTemplateSpecialization: {
               // This function is the result of instantiating a function
               // template or possibly an explicit specialization of a
               // function template.  Could be a namespace scope function or a
               // member function.
               if (Definition->isDeleted()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a deleted function "
                        "template specialization!");
                  return 0;
               }
               else if (Definition->isLateTemplateParsed()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a late template parsed "
                        "function template specialization!");
                  return 0;
               }
               //else if (Definition->isDefaulted()) {
               //   // Might not have a body, but we can still use it.
               //}
               //else {
               //   // Has a body.
               //}
            }
            break;
         case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
               // This function is the result of instantiating or
               // specializing a  member function of a class template,
               // or a member function of a class member of a class template,
               // or a member function template of a class template, or a
               // member function template of a class member of a class
               // template where at least some part of the function is
               // dependent on a template argument.
               if (Definition->isDeleted()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a deleted dependent function "
                        "template specialization!");
                  return 0;
               }
               else if (Definition->isLateTemplateParsed()) {
                  Error("TClingCallFunc::make_wrapper",
                        "Cannot make wrapper for a late template parsed "
                        "dependent function template specialization!");
                  return 0;
               }
               //else if (Definition->isDefaulted()) {
               //   // Might not have a body, but we can still use it.
               //}
               //else {
               //   // Has a body.
               //}
            }
            break;
         default: {
               // Will only happen if clang implementation changes.
               // Protect ourselves in case that happens.
               Error("TClingCallFunc::make_wrapper",
                     "Unhandled template kind!");
               return 0;
            }
            break;
      }
   }
   unsigned min_args = FD->getMinRequiredArguments();
   unsigned num_params = FD->getNumParams();
   //
   //  Make the wrapper name.
   //
   string wrapper_name;
   {
      ostringstream buf;
      buf << cling::utils::Synthesize::CFUniquePrefix;
      //const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
      //string mn;
      //fInterp->maybeMangleDeclName(ND, mn);
      //buf << '_' << mn;
      buf << '_' << wrapper_serial++;
      wrapper_name = buf.str();
   }
   //
   //  Write the wrapper code.
   //
   int indent_level = 0;
   string wrapperFwdDecl = "void " + wrapper_name + "(void* obj, int nargs, void** args, void* ret)";
   ostringstream buf;
   buf << wrapperFwdDecl << "{\n";
   ++indent_level;
   if (min_args == num_params) {
      // No parameters with defaults.
      make_narg_call_with_return(num_params, class_name, buf, indent_level);
   }
   else {
      // We need one function call clause compiled for every
      // possible number of arguments per call.
      for (unsigned N = min_args; N <= num_params; ++N) {
         for (int i = 0; i < indent_level; ++i) {
            buf << indent_string;
         }
         buf << "if (nargs == " << N << ") {\n";
         ++indent_level;
         make_narg_call_with_return(N, class_name, buf, indent_level);
         --indent_level;
         for (int i = 0; i < indent_level; ++i) {
            buf << indent_string;
         }
         buf << "}\n";
      }
   }
   --indent_level;
   buf << "}\n";
   string wrapper = wrapperFwdDecl + "__attribute__((used));\n"
      + buf.str();
   //
   //  Compile the wrapper code.
   //
   const FunctionDecl* WFD = 0;
   {
      cling::Transaction* Tp = 0;
      cling::Interpreter::CompilationResult CR = fInterp->declare(wrapper, &Tp);
      if (CR != cling::Interpreter::kSuccess) {
         Error("TClingCallFunc::make_wrapper", "Wrapper compile failed!");
         return 0;
      }
      WFD = Tp->getCFWrapperFD();
      if (!WFD) {
         Error("TClingCallFunc::make_wrapper", 
               "Wrapper compile did not return a function decl!");
         return 0;
      }
   }
   //
   //  Lookup the new wrapper declaration.
   //
   string mn;
   fInterp->maybeMangleDeclName(WFD, mn);
   const NamedDecl* WND = dyn_cast<NamedDecl>(WFD);
   if (!WND) {
      Error("TClingCallFunc::make_wrapper", "Wrapper named decl is null!");
      return 0;
   }
   {
      //WND->dump();
      //fInterp->getModule()->dump();
      //gROOT->ProcessLine(".printIR");
   }
   //
   //  Get the wrapper function pointer
   //  from the ExecutionEngine (the JIT).
   //
   const GlobalValue* GV = fInterp->getModule()->getNamedValue(mn);
   if (!GV) {
      Error("TClingCallFunc::make_wrapper",
            "Wrapper function name not found in Module!");
      return 0;
   }
   ExecutionEngine* EE = fInterp->getExecutionEngine();
   void* f = EE->getPointerToGlobalIfAvailable(GV);
   if (!f) {
      //
      //  Wrapper function not yet codegened by the JIT,
      //  force this to happen now.
      //
      f = EE->getPointerToGlobal(GV);
      if (!f) {
         Error("TClingCallFunc::make_wrapper", "Wrapper function has no "
               "mapping in Module after forced codegen!");
         return 0;
      }
   }
   wrapper_store.insert(make_pair(FD, f));
   return (tcling_callfunc_Wrapper_t) f;
}

class ValHolder {
public:
   union {
      long double ldbl;
      double dbl;
      float flt;
      //__uint128_t ui128;
      //__int128_t i128;
      unsigned long long ull;
      long long ll;
      unsigned long ul;
      long l;
      unsigned int ui;
      int i;
      unsigned short us;
      short s;
      //char32_t c32;
      //char16_t c16;
      unsigned wchar_t uwc;
      wchar_t wc;
      unsigned char uc;
      signed char sc;
      char c;
      bool b;
      void* vp;
   } u;
};

void
TClingCallFunc::
exec(void* address, void* ret) const
{
   vector<ValHolder> vh_ary;
   vector<void*> vp_ary;
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   PrintingPolicy Policy(FD->getASTContext().getPrintingPolicy());
   //
   //  Convert the arguments from cling::StoredValueRef to their
   //  actual type and store them in a holder for passing to the
   //  wrapper function by pointer to value.
   //
   unsigned num_params = FD->getNumParams();
   unsigned num_args = fArgVals.size();
   vh_ary.reserve(num_args);
   vp_ary.reserve(num_args);
   for (unsigned i = 0U; i < num_args; ++i) {
      QualType Ty;
      if (i < num_params) {
         const ParmVarDecl* PVD = FD->getParamDecl(i);
         Ty = PVD->getType();
      }
      else {
         Ty = fArgVals[i].get().getClangType();
      }
      QualType QT = Ty.getCanonicalType();
      if (QT->isReferenceType()) {
         ValHolder vh;
         vh.u.vp = (void*) sv_to_ulong_long(fArgVals[i]);
         vh_ary.push_back(vh);
         vp_ary.push_back(&vh_ary.back());
      }
      else if (QT->isMemberPointerType()) {
         ValHolder vh;
         vh.u.vp = (void*) sv_to_ulong_long(fArgVals[i]);
         vh_ary.push_back(vh);
         vp_ary.push_back(&vh_ary.back());
      }
      else if (QT->isPointerType() || QT->isArrayType()) {
         ValHolder vh;
         vh.u.vp = (void*) sv_to_ulong_long(fArgVals[i]);
         vh_ary.push_back(vh);
         vp_ary.push_back(&vh_ary.back());
      }
      else if (QT->isRecordType()) {
         ValHolder vh;
         vh.u.vp = (void*) sv_to_ulong_long(fArgVals[i]);
         vh_ary.push_back(vh);
         vp_ary.push_back(&vh_ary.back());
      }
      else if (const EnumType* ET =
                  dyn_cast<EnumType>(&*QT)) {
         // Note: We may need to worry about the underlying type
         //       of the enum here.
         (void) ET;
         ValHolder vh;
         vh.u.i = (int) sv_to_long_long(fArgVals[i]);
         vh_ary.push_back(vh);
         vp_ary.push_back(&vh_ary.back());
      }
      else if (const BuiltinType* BT =
                  dyn_cast<BuiltinType>(&*QT)) {
         //
         //  WARNING!!!
         //
         //  This switch is organized in order-of-declaration
         //  so that the produced assembly code is optimal.
         //  Do not reorder!
         //
         switch (BT->getKind()) {
               //
               //  Builtin Types
               //
            case BuiltinType::Void: {
                  // void
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'Void'!");
                  return;
               }
               break;
               //
               //  Unsigned Types
               //
            case BuiltinType::Bool: {
                  // bool
                  ValHolder vh;
                  vh.u.b = (bool) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Char_U: {
                  // char on targets where it is unsigned
                  ValHolder vh;
                  vh.u.c = (char) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::UChar: {
                  // unsigned char
                  ValHolder vh;
                  vh.u.uc = (unsigned char) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::WChar_U: {
                  // wchar_t on targets where it is unsigned
                  ValHolder vh;
                  vh.u.wc = (wchar_t) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Char16: {
                  // char16_t
                  //ValHolder vh;
                  //vh.u.c16 = (char16_t) sv_to_ulong_long(fArgVals[i]);
                  //vh_ary.push_back(vh);
                  //vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Char32: {
                  // char32_t
                  //ValHolder vh;
                  //vh.u.c32 = (char32_t) sv_to_ulong_long(fArgVals[i]);
                  //vh_ary.push_back(vh);
                  //vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::UShort: {
                  // unsigned short
                  ValHolder vh;
                  vh.u.us = (unsigned short) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::UInt: {
                  // unsigned int
                  ValHolder vh;
                  vh.u.ui = (unsigned int) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::ULong: {
                  // unsigned long
                  ValHolder vh;
                  vh.u.ul = (unsigned long) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::ULongLong: {
                  // unsigned long long
                  ValHolder vh;
                  vh.u.ull = (unsigned long long) sv_to_ulong_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::UInt128: {
                  // __uint128_t
               }
               break;
               //
               //  Signed Types
               //
               //
               //  Signed Types
               //
            case BuiltinType::Char_S: {
                  // char on targets where it is signed
                  ValHolder vh;
                  vh.u.c = (char) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::SChar: {
                  // signed char
                  ValHolder vh;
                  vh.u.sc = (signed char) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::WChar_S: {
                  // wchar_t on targets where it is signed
                  ValHolder vh;
                  vh.u.wc = (wchar_t) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Short: {
                  // short
                  ValHolder vh;
                  vh.u.s = (short) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Int: {
                  // int
                  ValHolder vh;
                  vh.u.i = (int) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Long: {
                  // long
                  ValHolder vh;
                  vh.u.l = (long) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::LongLong: {
                  // long long
                  ValHolder vh;
                  vh.u.ll = (long long) sv_to_long_long(fArgVals[i]);
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Int128: {
                  // __int128_t
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'Int128'!");
                  return;
               }
               break;
            case BuiltinType::Half: {
                  // half in OpenCL, __fp16 in ARM NEON
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'Half'!");
                  return;
               }
               break;
            case BuiltinType::Float: {
                  // float
                  ValHolder vh;
                  vh.u.flt = (float) fArgVals[i].get().getGV().FloatVal;
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::Double: {
                  // double
                  ValHolder vh;
                  vh.u.dbl = fArgVals[i].get().getGV().DoubleVal;
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::LongDouble: {
                  // long double
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'LongDouble'!");
                  return;
               }
               break;
               //
               //  Language-Specific Types
               //
            case BuiltinType::NullPtr: {
                  // C++11 nullptr
                  ValHolder vh;
                  vh.u.vp = (void*) fArgVals[i].get().getGV().PointerVal;
                  vh_ary.push_back(vh);
                  vp_ary.push_back(&vh_ary.back());
               }
               break;
            case BuiltinType::ObjCId: {
                  // Objective C 'id' type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'ObjCId'!");
                  return;
               }
               break;
            case BuiltinType::ObjCClass: {
                  // Objective C 'Class' type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'ObjCClass'!");
                  return;
               }
               break;
            case BuiltinType::ObjCSel: {
                  // Objective C 'SEL' type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'ObjCSel'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage1d: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage1d'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage1dArray: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage1dArray'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage1dBuffer: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage1dBuffer'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage2d: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage2d'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage2dArray: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage2dArray'!");
                  return;
               }
               break;
            case BuiltinType::OCLImage3d: {
                  // OpenCL image type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLImage3d'!");
                  return;
               }
               break;
            case BuiltinType::OCLSampler: {
                  // OpenCL sampler_t
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLSampler'!");
                  return;
               }
               break;
            case BuiltinType::OCLEvent: {
                  // OpenCL event_t
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'OCLEvent'!");
                  return;
               }
               break;
               //
               //  Placeholder types.
               //
               //  These types are used during intermediate phases
               //  of semantic analysis.  They are eventually resolved
               //  to one of the preceeding types.
               //
            case BuiltinType::Dependent: {
                  // dependent on a template argument
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'Dependent'!");
                  return;
               }
               break;
            case BuiltinType::Overload: {
                  // an unresolved function overload set
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'Overload'!");
                  return;
               }
               break;
            case BuiltinType::BoundMember: {
                  // a bound C++ non-static member function
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'BoundMember'!");
                  return;
               }
               break;
            case BuiltinType::PseudoObject: {
                  // Object C @property or VS.NET __property
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'PseudoObject'!");
                  return;
               }
               break;
            case BuiltinType::UnknownAny: {
                  // represents an unknown type
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'UnknownAny'!");
                  return;
               }
               break;
            case BuiltinType::BuiltinFn: {
                  // a compiler builtin function
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'BuiltinFn'!");
                  return;
               }
               break;
            case BuiltinType::ARCUnbridgedCast: {
                  // Objective C Automatic Reference Counting cast
                  // which would normally require __bridge, but which
                  // may be ok because of the context.
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid type 'ARCUnbridgedCast'!");
                  return;
               }
               break;
            default: {
                  // There should be no others.  This is here in case
                  // this changes in the future.
                  Error("TClingCallFunc::exec(void*)",
                        "Invalid builtin type (unrecognized)!");
                  QT->dump();
                  return;
               }
               break;
         }
      }
      else {
         Error("TClingCallFunc::exec(void*)",
               "Invalid type (unrecognized)!");
         QT->dump();
         return;
      }
   }
   (*fWrapper)(address, (int)num_args, (void**)vp_ary.data(), ret);
}

void
TClingCallFunc::
exec_with_valref_return(void* address, cling::StoredValueRef* ret) const
{
   if (!ret) {
      exec(address, 0);
      return;
   }
   const FunctionDecl* FD = fMethod->GetMethodDecl();
   ASTContext& Context = FD->getASTContext();
   PrintingPolicy Policy(Context.getPrintingPolicy());
   if (const CXXConstructorDecl* CD = dyn_cast<CXXConstructorDecl>(FD)) {
      const TypeDecl* TD = dyn_cast<TypeDecl>(CD->getDeclContext());
      QualType ClassTy(TD->getTypeForDecl(), 0);
      QualType QT = Context.getLValueReferenceType(ClassTy);
      GenericValue gv;
      exec(address, &gv.PointerVal);
      *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                cling::Value(gv, QT));
      return;
   }
   QualType QT = FD->getResultType().getCanonicalType();
   if (QT->isReferenceType()) {
      GenericValue gv;
      exec(address, &gv.PointerVal);
      *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                cling::Value(gv, QT));
      return;
   }
   else if (QT->isMemberPointerType()) {
      const MemberPointerType* MPT =
         QT->getAs<MemberPointerType>();
      if (MPT->isMemberDataPointer()) {
         // A member data pointer is a actually a struct with one
         // member of ptrdiff_t, the offset from the base of the object
         // storage to the storage for the designated data member.
         GenericValue gv;
         exec(address, &gv.PointerVal);
         *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                   cling::Value(gv, QT));
         return;
      }
      // We are a function member pointer.
      GenericValue gv;
      exec(address, &gv.PointerVal);
      *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                cling::Value(gv, QT));
      return;
   }
   else if (QT->isPointerType() || QT->isArrayType()) {
      // Note: ArrayType is an illegal function return value type.
      GenericValue gv;
      exec(address, &gv.PointerVal);
      *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                cling::Value(gv, QT));
      return;
   }
   else if (QT->isRecordType()) {
      uint64_t size = Context.getTypeSizeInChars(QT).getQuantity();
      void* p = ::operator new(size);
      exec(address, p);
      *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                cling::Value(PTOGV(p),
                                                QT));
      return;
   }
   else if (const EnumType* ET =
               dyn_cast<EnumType>(&*QT)) {
      // Note: We may need to worry about the underlying type
      //       of the enum here.
      (void) ET;
      uint64_t numBits = Context.getTypeSize(QT);
      int retVal = 0;
      exec(address, &retVal);
      GenericValue gv;
      gv.IntVal = APInt(numBits, (uint64_t) retVal,
                              true/*isSigned*/);
      *ret =  cling::StoredValueRef::bitwiseCopy(Context,
                                                 cling::Value(gv, QT));
      return;
   }
   else if (const BuiltinType* BT =
         dyn_cast<BuiltinType>(&*QT)) {
      uint64_t numBits = Context.getTypeSize(QT);
      switch (BT->getKind()) {
            //
            //  Builtin Types
            //
         case BuiltinType::Void: {
               // void
               exec(address, 0);
               return;
            }
            break;
            //
            //  Unsigned Types
            //
         case BuiltinType::Bool: {
               // bool
               bool retVal = false;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Char_U: {
               // char on targets where it is unsigned
               char retVal = '\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::UChar: {
               // unsigned char
               unsigned char retVal = '\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::WChar_U: {
               // wchar_t on targets where it is unsigned
               wchar_t retVal = L'\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Char16: {
               // char16_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'char16_t'!");
               return;
               //char16_t retVal = u'\0';
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                  false/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                         cling::Value(gv, QT));
               //return;
            }
            break;
         case BuiltinType::Char32: {
               // char32_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'char32_t'!");
               return;
               //char32_t retVal = U'\0';
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                  false/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                         cling::Value(gv, QT));
               //return;
            }
            break;
         case BuiltinType::UShort: {
               // unsigned short
               unsigned short retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::UInt: {
               // unsigned int
               unsigned int retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::ULong: {
               // unsigned long
               unsigned long retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::ULongLong: {
               // unsigned long long
               unsigned long long retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, false/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::UInt128: {
               // __uint128_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type '__uint128_t'!");
               return;
               //__uint128_t retVal = 0;
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                  false/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                          cling::Value(gv, QT));
               //return;
            }
            break;
            //
            //  Signed Types
            //
            //
            //  Signed Types
            //
         case BuiltinType::Char_S: {
               // char on targets where it is signed
               char retVal = '\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::SChar: {
               // signed char
               signed char retVal = '\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::WChar_S: {
               // wchar_t on targets where it is signed
               wchar_t retVal = L'\0';
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Short: {
               // short
               short retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Int: {
               // int
               int retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Long: {
               // long
               long retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::LongLong: {
               // long long
               long long retVal = 0;
               exec(address, &retVal);
               GenericValue gv;
               gv.IntVal = APInt(numBits, (uint64_t) retVal, true/*isSigned*/);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Int128: {
               // __int128_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type '__int128_t'!");
               return;
               //__int128_t retVal = 0;
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                  true/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                         cling::Value(gv, QT));
               //return;
            }
            break;
         case BuiltinType::Half: {
               // half in OpenCL, __fp16 in ARM NEON
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'Half'!");
               return;
               //half retVal = 0;
               //__fp16 retVal = 0;
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                  true/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                         cling::Value(gv, QT));
               //return;
            }
            break;
         case BuiltinType::Float: {
               // float
               GenericValue gv;
               exec(address, &gv.FloatVal);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::Double: {
               // double
               GenericValue gv;
               exec(address, &gv.DoubleVal);
               *ret = cling::StoredValueRef::bitwiseCopy(Context,
                                                         cling::Value(gv, QT));
               return;
            }
            break;
         case BuiltinType::LongDouble: {
               // long double
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'LongDouble'!");
               return;
               //long double retVal = 0.0L;
               //exec(address, &retVal);
               //GenericValue gv;
               //gv.IntVal = APInt(numBits, (uint64_t) retVal,
               //                        false/*isSigned*/);
               //*ret = cling::StoredValueRef::bitwiseCopy(Context,
               //                                         cling::Value(gv, QT));
               //return;
            }
            break;
            //
            //  Language-Specific Types
            //
         case BuiltinType::NullPtr: {
               // C++11 nullptr
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'nullptr'!");
               return;
            }
            break;
         case BuiltinType::ObjCId: {
               // Objective C 'id' type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'ObjCId'!");
               return;
            }
            break;
         case BuiltinType::ObjCClass: {
               // Objective C 'Class' type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'ObjCClass'!");
               return;
            }
            break;
         case BuiltinType::ObjCSel: {
               // Objective C 'SEL' type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'ObjCSel'!");
               return;
            }
            break;
         case BuiltinType::OCLImage1d: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage1d'!");
               return;
            }
            break;
         case BuiltinType::OCLImage1dArray: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage1dArray'!");
               return;
            }
            break;
         case BuiltinType::OCLImage1dBuffer: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage1dBuffer'!");
               return;
            }
            break;
         case BuiltinType::OCLImage2d: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage2d'!");
               return;
            }
            break;
         case BuiltinType::OCLImage2dArray: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage2dArray'!");
               return;
            }
            break;
         case BuiltinType::OCLImage3d: {
               // OpenCL image type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLImage3d'!");
               return;
            }
            break;
         case BuiltinType::OCLSampler: {
               // OpenCL sampler_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLSampler'!");
               return;
            }
            break;
         case BuiltinType::OCLEvent: {
               // OpenCL event_t
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'OCLEvent'!");
               return;
            }
            break;
            //
            //  Placeholder types.
            //
            //  These types are used during intermediate phases
            //  of semantic analysis.  They are eventually resolved
            //  to one of the preceeding types.
            //
         case BuiltinType::Dependent: {
               // dependent on a template argument
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'Dependent'!");
               return;
            }
            break;
         case BuiltinType::Overload: {
               // an unresolved function overload set
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'Overload'!");
               return;
            }
            break;
         case BuiltinType::BoundMember: {
               // a bound C++ non-static member function
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'BoundMember'!");
               return;
            }
            break;
         case BuiltinType::PseudoObject: {
               // Object C @property or VS.NET __property
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'PseudoObject'!");
               return;
            }
            break;
         case BuiltinType::UnknownAny: {
               // represents an unknown type
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'UnknownAny'!");
               return;
            }
            break;
         case BuiltinType::BuiltinFn: {
               // a compiler builtin function
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'BuiltinFn'!");
               return;
            }
            break;
         case BuiltinType::ARCUnbridgedCast: {
               // Objective C Automatic Reference Counting cast
               // which would normally require __bridge, but which
               // may be ok because of the context.
               Error("TClingCallFunc::exec_with_valref_return",
                     "Invalid type 'ARCUnbridgedCast'!");
               return;
            }
            break;
         default: {
               // There should be no others.  This is here in case
               // this changes in the future.
               Error("TClingCallFunc::exec_with_valref_return",
                     "Unrecognized builtin type!");
               return;
            }
            break;
      }
   }
   Error("TClingCallFunc::exec_with_valref_return",
         "Unrecognized return type!");
   return;
}

void
TClingCallFunc::
EvaluateArgList(const string& ArgList)
{
   SmallVector<Expr*, 4> exprs;
   fInterp->getLookupHelper().findArgList(ArgList, exprs);
   for (SmallVector<Expr*, 4>::const_iterator I = exprs.begin(),
         E = exprs.end(); I != E; ++I) {
      cling::StoredValueRef val = EvaluateExpr(fInterp, *I);
      if (!val.isValid()) {
         // Bad expression, all done.
         return;
      }
      fArgVals.push_back(val);
   }
}

void
TClingCallFunc::
Exec(void* address, TInterpreterValue* interpVal/*=0*/)
{
   if (!IsValid()) {
      Error("TClingCallFunc::Exec(address, interpVal)",
            "Attempt to execute while invalid.");
      return;
   }
   const FunctionDecl* decl = fMethod->GetMethodDecl();
   map<const FunctionDecl*, void*>::iterator I =
      wrapper_store.find(decl);
   if (I != wrapper_store.end()) {
      fWrapper = (tcling_callfunc_Wrapper_t) I->second;
   }
   else {
      fWrapper = make_wrapper();
   }
   if (!fWrapper) {
      Error("TClingCallFunc::Exec(address, interpVal)",
            "Called with no wrapper, not implemented!");
      return;
   }
   if (!interpVal) {
      exec(address, 0);
      return;
   }
   cling::StoredValueRef valref;
   exec_with_valref_return(address, &valref);
   reinterpret_cast<cling::StoredValueRef&>(interpVal->Get()) = valref;
   return;
}

Long_t
TClingCallFunc::
ExecInt(void* address)
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecInt", "Attempt to execute while invalid.");
      return 0L;
   }
   const FunctionDecl* decl = fMethod->GetMethodDecl();
   map<const FunctionDecl*, void*>::iterator I =
      wrapper_store.find(decl);
   if (I != wrapper_store.end()) {
      fWrapper = (tcling_callfunc_Wrapper_t) I->second;
   }
   else {
      fWrapper = make_wrapper();
   }
   if (!fWrapper) {
      Error("TClingCallFunc::ExecInt",
            "Called with no wrapper, not implemented!");
      return 0L;
   }
   cling::StoredValueRef ret;
   exec_with_valref_return(address, &ret);
   if (!ret.isValid()) {
      // Sometimes we are called on a function returning void!
      return 0L;
   }
   return static_cast<Long_t>(sv_to_long_long(ret));
}

long long
TClingCallFunc::
ExecInt64(void* address)
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecInt64", "Attempt to execute while invalid.");
      return 0LL;
   }
   const FunctionDecl* decl = fMethod->GetMethodDecl();
   map<const FunctionDecl*, void*>::iterator I =
      wrapper_store.find(decl);
   if (I != wrapper_store.end()) {
      fWrapper = (tcling_callfunc_Wrapper_t) I->second;
   }
   else {
      fWrapper = make_wrapper();
   }
   if (!fWrapper) {
      Error("TClingCallFunc::ExecInt64",
            "Called with no wrapper, not implemented!");
      return 0LL;
   }
   cling::StoredValueRef ret;
   exec_with_valref_return(address, &ret);
   if (!ret.isValid()) {
      // Sometimes we are called on a function returning void!
      return 0LL;
   }
   return sv_to_long_long(ret);
}

double
TClingCallFunc::
ExecDouble(void* address)
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecDouble", "Attempt to execute while invalid.");
      return 0.0;
   }
   const FunctionDecl* decl = fMethod->GetMethodDecl();
   map<const FunctionDecl*, void*>::iterator I =
      wrapper_store.find(decl);
   if (I != wrapper_store.end()) {
      fWrapper = (tcling_callfunc_Wrapper_t) I->second;
   }
   else {
      fWrapper = make_wrapper();
   }
   if (!fWrapper) {
      Error("TClingCallFunc::ExecDouble",
            "Called with no wrapper, not implemented!");
      return 0.0;
   }
   cling::StoredValueRef ret;
   exec_with_valref_return(address, &ret);
   if (!ret.isValid()) {
      // Sometimes we are called on a function returning void!
      return 0.0;
   }
   return sv_to_double(ret);
}

void
TClingCallFunc::
ExecWithReturn(void* address, void* ret/*= 0*/)
{
   if (!IsValid()) {
      Error("TClingCallFunc::ExecWithReturn(address, ret)",
            "Attempt to execute while invalid.");
      return;
   }
   const FunctionDecl* decl = fMethod->GetMethodDecl();
   map<const FunctionDecl*, void*>::iterator I =
      wrapper_store.find(decl);
   if (I != wrapper_store.end()) {
      fWrapper = (tcling_callfunc_Wrapper_t) I->second;
   }
   else {
      fWrapper = make_wrapper();
   }
   if (!fWrapper) {
      Error("TClingCallFunc::ExecWithReturn(address, ret)",
            "Called with no wrapper, not implemented!");
      return;
   }
   exec(address, ret);
   return;
}

TClingMethodInfo*
TClingCallFunc::
FactoryMethod() const
{
   return new TClingMethodInfo(*fMethod);
}

void
TClingCallFunc::
Init()
{
   delete fMethod;
   fMethod = 0;
   fWrapper = 0;
   ResetArg();
}

void
TClingCallFunc::
Init(TClingMethodInfo* minfo)
{
   delete fMethod;
   fMethod = new TClingMethodInfo(*minfo);
   fWrapper = 0;
   ResetArg();
}

void*
TClingCallFunc::
InterfaceMethod() const
{
   if (!IsValid()) {
      return 0;
   }
   return (void*) const_cast<FunctionDecl*>(fMethod->GetMethodDecl());
}

bool
TClingCallFunc::
IsValid() const
{
   if (!fMethod) {
      return false;
   }
   return fMethod->IsValid();
}

void
TClingCallFunc::
ResetArg()
{
   fArgVals.clear();
}

void
TClingCallFunc::
SetArg(long param)
{
   ASTContext& C = fInterp->getCI()->getASTContext();
   GenericValue gv;
   QualType QT = C.LongTy;
   gv.IntVal = APInt(C.getTypeSize(QT), param);
   fArgVals.push_back(cling::StoredValueRef::bitwiseCopy(C,
                                                         cling::Value(gv, QT)));
}

void
TClingCallFunc::
SetArg(double param)
{
   ASTContext& C = fInterp->getCI()->getASTContext();
   GenericValue gv;
   QualType QT = C.DoubleTy;
   gv.DoubleVal = param;
   fArgVals.push_back(cling::StoredValueRef::bitwiseCopy(C,
                                                         cling::Value(gv, QT)));
}

void
TClingCallFunc::
SetArg(long long param)
{
   ASTContext& C = fInterp->getCI()->getASTContext();
   GenericValue gv;
   QualType QT = C.LongLongTy;
   gv.IntVal = APInt(C.getTypeSize(QT), param);
   fArgVals.push_back(cling::StoredValueRef::bitwiseCopy(C,
                                                         cling::Value(gv, QT)));
}

void
TClingCallFunc::
SetArg(unsigned long long param)
{
   ASTContext& C = fInterp->getCI()->getASTContext();
   GenericValue gv;
   QualType QT = C.UnsignedLongLongTy;
   gv.IntVal = APInt(C.getTypeSize(QT), param);
   fArgVals.push_back(cling::StoredValueRef::bitwiseCopy(C,
                                                         cling::Value(gv, QT)));
}

void
TClingCallFunc::
SetArgArray(long* paramArr, int nparam)
{
   ResetArg();
   for (int i = 0; i < nparam; ++i) {
      SetArg(paramArr[i]);
   }
}

void
TClingCallFunc::
SetArgs(const char* params)
{
   ResetArg();
   EvaluateArgList(params);
}

void
TClingCallFunc::
SetFunc(const TClingClassInfo* info, const char* method, const char* arglist,
        long* poffset)
{
   SetFunc(info, method, arglist, false, poffset);
}

void
TClingCallFunc::
SetFunc(const TClingClassInfo* info, const char* method, const char* arglist,
        bool objectIsConst, long* poffset)
{
   fWrapper = 0;
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFunc", "Class info is invalid!");
      return;
   }
   if (!strcmp(arglist, ")")) {
      // CINT accepted a single right paren as meaning no arguments.
      arglist = "";
   }
   *fMethod = info->GetMethodWithArgs(method, arglist, objectIsConst, poffset);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFunc", "Could not find method %s(%s)", method,
      //      arglist);
      return;
   }
   // FIXME: The arglist was already parsed by the lookup, we should
   //        enhance the lookup to return the resulting expression
   //        list so we do not need to parse it again here.
   EvaluateArgList(arglist);
}

void
TClingCallFunc::
SetFunc(const TClingMethodInfo* info)
{
   fWrapper = 0;
   delete fMethod;
   fMethod = new TClingMethodInfo(*info);
   ResetArg();
   if (!fMethod->IsValid()) {
      return;
   }
}

void
TClingCallFunc::
SetFuncProto(const TClingClassInfo* info, const char* method,
             const char* proto, long* poffset,
             EFunctionMatchMode mode/*=kConversionMatch*/)
{
   SetFuncProto(info, method, proto, false, poffset, mode);
}

void
TClingCallFunc::
SetFuncProto(const TClingClassInfo* info, const char* method,
             const char* proto, bool objectIsConst, long* poffset,
             EFunctionMatchMode mode/*=kConversionMatch*/)
{
   fWrapper = 0;
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
}

void
TClingCallFunc::
SetFuncProto(const TClingClassInfo* info, const char* method,
             const llvm::SmallVector<clang::QualType, 4>& proto, long* poffset,
             EFunctionMatchMode mode/*=kConversionMatch*/)
{
   SetFuncProto(info, method, proto, false, poffset, mode);
}

void
TClingCallFunc::
SetFuncProto(const TClingClassInfo* info, const char* method,
             const llvm::SmallVector<clang::QualType, 4>& proto,
             bool objectIsConst, long* poffset,
             EFunctionMatchMode mode/*=kConversionMatch*/)
{
   delete fMethod;
   fMethod = new TClingMethodInfo(fInterp);
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
}

