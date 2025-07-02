// root/core/meta
// vim: sw=3
// Author: Paul Russo   30/07/2012
// Author: Vassil Vassilev   9/02/2013

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TClingCallFunc
Emulation of the CINT CallFunc class.

The CINT C++ interpreter provides an interface for calling
functions through the generated wrappers in dictionaries with
the CallFunc class. This class provides the same functionality,
using an interface as close as possible to CallFunc but the
function metadata and calling service comes from the Cling
C++ interpreter and the Clang C++ compiler, not CINT.
*/

#include "TClingCallFunc.h"

#include "TClingClassInfo.h"
#include "TClingMethodInfo.h"
#include "TClingUtils.h"

#include "TError.h"
#include "TCling.h"

#include "TInterpreter.h"

#include "cling/Interpreter/CompilationOptions.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/Interpreter/Value.h"
#include "cling/Utils/AST.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/QualTypeNames.h"
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

#include <map>
#include <string>
#include <sstream>

using namespace ROOT;
using namespace clang;
using llvm::APSInt, llvm::raw_string_ostream;
using std::string, std::map, std::ostringstream, std::make_pair;

static unsigned long long gWrapperSerial = 0LL;
static const string kIndentString("   ");

static
inline
void
indent(ostringstream &buf, int indent_level)
{
   for (int i = 0; i < indent_level; ++i) {
      buf << kIndentString;
   }
}

static
void
EvaluateExpr(cling::Interpreter &interp, const Expr *E, cling::Value &V)
{
   // Evaluate an Expr* and return its cling::Value
   ASTContext &C = interp.getCI()->getASTContext();
   clang::Expr::EvalResult evalRes;
   if (E->EvaluateAsInt(evalRes, C, /*AllowSideEffects*/Expr::SE_NoSideEffects)) {
      // FIXME: Find the right type or make sure we have an interface to update
      // the clang::Type in the cling::Value
      APSInt res = evalRes.Val.getInt();
      // IntTy or maybe better E->getType()?
      V = cling::Value(C.IntTy, interp);
      // We must use the correct signedness otherwise the zero extension
      // fails if the actual type is strictly less than long long.
      if (res.isSigned())
        V.setLongLong(res.getSExtValue());
      else
        V.setULongLong(res.getZExtValue());
      return;
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
   E->printPretty(out, /*Helper=*/nullptr, Policy, /*Indentation=*/0);
   out << ';'; // no value printing
   out.flush();
   // Evaluate() will set V to invalid if evaluation fails.
   interp.evaluate(buf, V);
}

size_t TClingCallFunc::CalculateMinRequiredArguments()
{
   // This function is non-const to use caching overload of GetDecl()!
   return GetDecl()->getMinRequiredArguments();
}

void *TClingCallFunc::compile_wrapper(const string &wrapper_name, const string &wrapper,
                                      bool withAccessControl/*=true*/)
{
   return fInterp->compileFunction(wrapper_name, wrapper, false /*ifUnique*/,
                                   withAccessControl);
}

static void GetTypeAsString(QualType QT, string& type_name, ASTContext &C,
                            PrintingPolicy Policy) {

   // FIXME: Take the code here https://github.com/root-project/root/blob/550fb2644f3c07d1db72b9b4ddc4eba5a99ddc12/interpreter/cling/lib/Utils/AST.cpp#L316-L350
   // to make hist/histdrawv7/test/histhistdrawv7testUnit work into
   // QualTypeNames.h in clang
   //type_name = clang::TypeName::getFullyQualifiedName(QT, C, Policy);
   cling::utils::Transform::Config Config;
   QT = cling::utils::Transform::GetPartiallyDesugaredType(C, QT, Config, /*fullyQualify=*/true);
   QT.getAsStringInternal(type_name, Policy);
}

static void GetDeclName(const clang::Decl *D, ASTContext &Context, std::string &name)
{
   // Helper to extract a fully qualified name from a Decl

   PrintingPolicy Policy(Context.getPrintingPolicy());
   Policy.SuppressTagKeyword = true;
   Policy.SuppressUnwrittenScope = true;
   if (const TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
      // This is a class, struct, or union member.
      QualType QT(TD->getTypeForDecl(), 0);
      GetTypeAsString(QT, name, Context, Policy);
   } else if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
      // This is a namespace member.
      raw_string_ostream stream(name);
      ND->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);
      stream.flush();
   }
}

void TClingCallFunc::collect_type_info(QualType &QT, ostringstream &typedefbuf, std::ostringstream &callbuf,
                                       string &type_name, EReferenceType &refType, bool &isPointer, int indent_level,
                                       bool forArgument)
{
   //
   //  Collect information about type type of a function parameter
   //  needed for building the wrapper function.
   //
   const FunctionDecl *FD = GetDecl();
   ASTContext &C = FD->getASTContext();
   PrintingPolicy Policy(C.getPrintingPolicy());
   refType = kNotReference;
   if (QT->isRecordType() && forArgument) {
      GetTypeAsString(QT, type_name, C, Policy);
      return;
   }
   if (QT->isFunctionPointerType()) {
      string fp_typedef_name;
      {
         ostringstream nm;
         nm << "FP" << gWrapperSerial++;
         type_name = nm.str();
         raw_string_ostream OS(fp_typedef_name);
         QT.print(OS, Policy, type_name);
         OS.flush();
      }
      indent(typedefbuf, indent_level);
      typedefbuf << "typedef " << fp_typedef_name << ";\n";
      return;
   } else if (QT->isMemberPointerType()) {
      string mp_typedef_name;
      {
         ostringstream nm;
         nm << "MP" << gWrapperSerial++;
         type_name = nm.str();
         raw_string_ostream OS(mp_typedef_name);
         QT.print(OS, Policy, type_name);
         OS.flush();
      }
      indent(typedefbuf, indent_level);
      typedefbuf << "typedef " << mp_typedef_name << ";\n";
      return;
   } else if (QT->isPointerType()) {
      isPointer = true;
      QT = cast<clang::PointerType>(QT)->getPointeeType();
   } else if (QT->isReferenceType()) {
      if (QT->isRValueReferenceType()) refType = kRValueReference;
      else refType = kLValueReference;
      QT = cast<ReferenceType>(QT)->getPointeeType();
   }
   // Fall through for the array type to deal with reference/pointer ro array type.
   if (QT->isArrayType()) {
      string ar_typedef_name;
      {
         ostringstream ar;
         ar << "AR" << gWrapperSerial++;
         type_name = ar.str();
         raw_string_ostream OS(ar_typedef_name);
         QT.print(OS, Policy, type_name);
         OS.flush();
      }
      indent(typedefbuf, indent_level);
      typedefbuf << "typedef " << ar_typedef_name << ";\n";
      return;
   }
   GetTypeAsString(QT, type_name, C, Policy);
}

static bool IsCopyConstructorDeleted(QualType QT)
{
   CXXRecordDecl *RD = QT->getAsCXXRecordDecl();
   if (!RD) {
      // For types that are not C++ records (such as PODs), we assume that they are copyable, ie their copy constructor
      // is not deleted.
      return false;
   }

   RD = RD->getDefinition();
   assert(RD && "expecting a definition");

   if (RD->hasSimpleCopyConstructor())
      return false;

   for (auto *Ctor : RD->ctors()) {
      if (Ctor->isCopyConstructor()) {
         return Ctor->isDeleted();
      }
   }

   assert(0 && "did not find a copy constructor?");
   // Should never happen and the return value is somewhat arbitrary, but we did not see a deleted copy ctor. The user
   // will be told if the generated code doesn't compile.
   return false;
}

void TClingCallFunc::make_narg_ctor(const unsigned N, ostringstream &typedefbuf,
                                    ostringstream &callbuf, const string &class_name,
                                    int indent_level)
{
   // Make a code string that follows this pattern:
   //
   // new ClassName(args...)
   //
   const FunctionDecl *FD = GetDecl();

   callbuf << "new " << class_name << "(";

   // IsCopyConstructorDeleted could trigger deserialization of decls.
   cling::Interpreter::PushTransactionRAII RAII(fInterp);

   for (unsigned i = 0U; i < N; ++i) {
      const ParmVarDecl *PVD = FD->getParamDecl(i);
      QualType Ty = PVD->getType();
      QualType QT = Ty.getCanonicalType();
      string type_name;
      EReferenceType refType = kNotReference;
      bool isPointer = false;
      collect_type_info(QT, typedefbuf, callbuf, type_name,
                        refType, isPointer, indent_level, true);
      if (i) {
         callbuf << ',';
         if (i % 2) {
            callbuf << ' ';
         } else {
            callbuf << "\n";
            indent(callbuf, indent_level + 1);
         }
      }
      if (refType != kNotReference) {
         callbuf << "(" << type_name.c_str() <<
                 (refType == kLValueReference ? "&" : "&&") << ")*(" << type_name.c_str() << "*)args["
                 << i << "]";
      } else if (isPointer) {
         callbuf << "*(" << type_name.c_str() << "**)args["
                 << i << "]";
      } else {
         // By-value construction: Figure out if the type can be copy-constructed. This is tricky and cannot be done in
         // a fully reliable way, also because std::vector<T> always defines a copy constructor, even if the type T is
         // only moveable. As a heuristic, we only check if the copy constructor is deleted, or would be if implicit.
         bool Move = IsCopyConstructorDeleted(QT);
         if (Move) {
            callbuf << "static_cast<" << type_name << "&&>(";
         }
         callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
         if (Move) {
            callbuf << ")";
         }
      }
   }
   callbuf << ")";
}

void TClingCallFunc::make_narg_call(const std::string &return_type, const unsigned N, ostringstream &typedefbuf,
                                    ostringstream &callbuf, const string &class_name, int indent_level)
{
   //
   // Make a code string that follows this pattern:
   //
   // ((<class>*)obj)-><method>(*(<arg-i-type>*)args[i], ...)
   //
   const FunctionDecl *FD = GetDecl();

   // Sometimes it's necessary that we cast the function we want to call first
   // to its explicit function type before calling it. This is supposed to prevent
   // that we accidentially ending up in a function that is not the one we're
   // supposed to call here (e.g. because the C++ function lookup decides to take
   // another function that better fits).
   // This method has some problems, e.g. when we call a function with default
   // arguments and we don't provide all arguments, we would fail with this pattern.
   // Same applies with member methods which seem to cause parse failures even when
   // we supply the object parameter.
   // Therefore we only use it in cases where we know it works and set this variable
   // to true when we do.
   bool ShouldCastFunction = !isa<CXXMethodDecl>(FD) && N == FD->getNumParams();
   if (ShouldCastFunction) {
      callbuf << "(";
      callbuf << "(";
      callbuf << return_type << " (&)";
      {
         callbuf << "(";
         for (unsigned i = 0U; i < N; ++i) {
            if (i) {
               callbuf << ',';
               if (i % 2) {
                  callbuf << ' ';
               } else {
                  callbuf << "\n";
                  indent(callbuf, indent_level + 1);
               }
            }
            const ParmVarDecl *PVD = FD->getParamDecl(i);
            QualType Ty = PVD->getType();
            QualType QT = Ty.getCanonicalType();
            std::string arg_type;
            ASTContext &C = FD->getASTContext();
            GetTypeAsString(QT, arg_type, C, C.getPrintingPolicy());
            callbuf << arg_type;
         }
         if (FD->isVariadic())
            callbuf << ", ...";
         callbuf << ")";
      }

      callbuf << ")";
   }

   if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
      // This is a class, struct, or union member.
      if (MD->isConst())
         callbuf << "((const " << class_name << "*)obj)->";
      else
         callbuf << "((" << class_name << "*)obj)->";
   } else if (const NamedDecl *ND =
                 dyn_cast<NamedDecl>(GetDeclContext())) {
      // This is a namespace member.
      (void) ND;
      callbuf << class_name << "::";
   }
   //   callbuf << fMethod->Name() << "(";
   {
      std::string name;
      {
         llvm::raw_string_ostream stream(name);
         FD->getNameForDiagnostic(stream, FD->getASTContext().getPrintingPolicy(), /*Qualified=*/false);
      }
      callbuf << name;
   }
   if (ShouldCastFunction) callbuf << ")";

   callbuf << "(";

   // IsCopyConstructorDeleted could trigger deserialization of decls.
   cling::Interpreter::PushTransactionRAII RAII(fInterp);

   for (unsigned i = 0U; i < N; ++i) {
      const ParmVarDecl *PVD = FD->getParamDecl(i);
      QualType Ty = PVD->getType();
      QualType QT = Ty.getCanonicalType();
      string type_name;
      EReferenceType refType = kNotReference;
      bool isPointer = false;
      collect_type_info(QT, typedefbuf, callbuf, type_name, refType, isPointer, indent_level, true);

      if (i) {
         callbuf << ',';
         if (i % 2) {
            callbuf << ' ';
         } else {
            callbuf << "\n";
            indent(callbuf, indent_level + 1);
         }
      }

      if (refType != kNotReference) {
         callbuf << "(" << type_name.c_str() <<
                 (refType == kLValueReference ? "&" : "&&") << ")*(" << type_name.c_str() << "*)args["
                 << i << "]";
      } else if (isPointer) {
         callbuf << "*(" << type_name.c_str() << "**)args["
                 << i << "]";
      } else {
         // By-value construction: Figure out if the type can be copy-constructed. This is tricky and cannot be done in
         // a fully reliable way, also because std::vector<T> always defines a copy constructor, even if the type T is
         // only moveable. As a heuristic, we only check if the copy constructor is deleted, or would be if implicit.
         bool Move = IsCopyConstructorDeleted(QT);
         if (Move) {
            callbuf << "static_cast<" << type_name << "&&>(";
         }
         callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
         if (Move) {
            callbuf << ")";
         }
      }
   }
   callbuf << ")";
}

void TClingCallFunc::make_narg_ctor_with_return(const unsigned N, const string &class_name,
      ostringstream &buf, int indent_level)
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
   indent(buf, indent_level);
   buf << "if (ret) {\n";
   ++indent_level;
   {
      ostringstream typedefbuf;
      ostringstream callbuf;
      //
      //  Write the return value assignment part.
      //
      indent(callbuf, indent_level);
      callbuf << "(*(" << class_name << "**)ret) = ";
      //
      //  Write the actual new expression.
      //
      make_narg_ctor(N, typedefbuf, callbuf, class_name, indent_level);
      //
      //  End the new expression statement.
      //
      callbuf << ";\n";
      indent(callbuf, indent_level);
      callbuf << "return;\n";
      //
      //  Output the whole new expression and return statement.
      //
      buf << typedefbuf.str() << callbuf.str();
   }
   --indent_level;
   for (int i = 0; i < indent_level; ++i) {
      buf << kIndentString;
   }
   buf << "}\n";
   for (int i = 0; i < indent_level; ++i) {
      buf << kIndentString;
   }
   buf << "else {\n";
   ++indent_level;
   {
      ostringstream typedefbuf;
      ostringstream callbuf;
      for (int i = 0; i < indent_level; ++i) {
         callbuf << kIndentString;
      }
      make_narg_ctor(N, typedefbuf, callbuf, class_name, indent_level);
      callbuf << ";\n";
      for (int i = 0; i < indent_level; ++i) {
         callbuf << kIndentString;
      }
      callbuf << "return;\n";
      buf << typedefbuf.str() << callbuf.str();
   }
   --indent_level;
   for (int i = 0; i < indent_level; ++i) {
      buf << kIndentString;
   }
   buf << "}\n";
}

///////////////////////////////////////////////////////////////////////////////
// Returns the DeclContext corresponding to fMethod's Decl.
// \Note that this might be a FunctionDecl or a UsingShadowDecl; we use the
// DeclContext of the UsingShadowDecl e.g. for constructing a derived class
// object, even if invoking a function made available by a using declaration
// of a constructor of a base class (ROOT-11010).

const clang::DeclContext *TClingCallFunc::GetDeclContext() const {
   return fMethod->GetDecl()->getDeclContext();
}

int TClingCallFunc::get_wrapper_code(std::string &wrapper_name, std::string &wrapper)
{
   const FunctionDecl *FD = GetDecl();
   assert(FD && "generate_wrapper called without a function decl!");
   //
   //  Get the class or namespace name.
   //
   string class_name;
   ASTContext &Context = FD->getASTContext();
   const clang::DeclContext *DC = GetDeclContext();
   GetDeclName(cast<Decl>(DC), Context, class_name);
   //
   //  Check to make sure that we can
   //  instantiate and codegen this function.
   //
   bool needInstantiation = false;
   const FunctionDecl *Definition = nullptr;
   if (!FD->isDefined(Definition)) {
      FunctionDecl::TemplatedKind TK = FD->getTemplatedKind();
      switch (TK) {
      case FunctionDecl::TK_NonTemplate: {
         // Ordinary function, not a template specialization.
         // Note: This might be ok, the body might be defined
         //       in a library, and all we have seen is the
         //       header file.
         //::Error("TClingCallFunc::make_wrapper",
         //      "Cannot make wrapper for a function which is "
         //      "declared but not defined!");
         // return 0;
      } break;
      case FunctionDecl::TK_FunctionTemplate: {
         // This decl is actually a function template,
         // not a function at all.
         ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a function template!");
         return 0;
      } break;
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
            //::Error("TClingCallFunc::make_wrapper",
            //      "Cannot make wrapper for a function template "
            //      "explicit specialization which is declared "
            //      "but not defined!");
            // return 0;
            break;
         }
         const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
         if (!Pattern) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a member function "
                                                    "instantiation with no pattern!");
            return 0;
         }
         FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
         TemplateSpecializationKind PTSK = Pattern->getTemplateSpecializationKind();
         if (
            // The pattern is an ordinary member function.
            (PTK == FunctionDecl::TK_NonTemplate) ||
            // The pattern is an explicit specialization, and
            // so is not a template.
            ((PTK != FunctionDecl::TK_FunctionTemplate) &&
             ((PTSK == TSK_Undeclared) || (PTSK == TSK_ExplicitSpecialization)))) {
            // Note: This might be ok, the body might be defined
            //       in a library, and all we have seen is the
            //       header file.
            break;
         } else if (!Pattern->hasBody()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a member function "
                                                    "instantiation with no body!");
            return 0;
         }
         if (FD->isImplicitlyInstantiable()) {
            needInstantiation = true;
         }
      } break;
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
            //::Error("TClingCallFunc::make_wrapper",
            //      "Cannot make wrapper for a function template "
            //      "explicit specialization which is declared "
            //      "but not defined!");
            // return 0;
            break;
         }
         const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
         if (!Pattern) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a function template"
                                                    "instantiation with no pattern!");
            return 0;
         }
         FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
         TemplateSpecializationKind PTSK = Pattern->getTemplateSpecializationKind();
         if (
            // The pattern is an ordinary member function.
            (PTK == FunctionDecl::TK_NonTemplate) ||
            // The pattern is an explicit specialization, and
            // so is not a template.
            ((PTK != FunctionDecl::TK_FunctionTemplate) &&
             ((PTSK == TSK_Undeclared) || (PTSK == TSK_ExplicitSpecialization)))) {
            // Note: This might be ok, the body might be defined
            //       in a library, and all we have seen is the
            //       header file.
            break;
         }
         if (!Pattern->hasBody()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a function template"
                                                    "instantiation with no body!");
            return 0;
         }
         if (FD->isImplicitlyInstantiable()) {
            needInstantiation = true;
         }
      } break;
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
            //::Error("TClingCallFunc::make_wrapper",
            //      "Cannot make wrapper for a dependent function "
            //      "template explicit specialization which is declared "
            //      "but not defined!");
            // return 0;
            break;
         }
         const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern();
         if (!Pattern) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a dependent function template"
                                                    "instantiation with no pattern!");
            return 0;
         }
         FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
         TemplateSpecializationKind PTSK = Pattern->getTemplateSpecializationKind();
         if (
            // The pattern is an ordinary member function.
            (PTK == FunctionDecl::TK_NonTemplate) ||
            // The pattern is an explicit specialization, and
            // so is not a template.
            ((PTK != FunctionDecl::TK_FunctionTemplate) &&
             ((PTSK == TSK_Undeclared) || (PTSK == TSK_ExplicitSpecialization)))) {
            // Note: This might be ok, the body might be defined
            //       in a library, and all we have seen is the
            //       header file.
            break;
         }
         if (!Pattern->hasBody()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a dependent function template"
                                                    "instantiation with no body!");
            return 0;
         }
         if (FD->isImplicitlyInstantiable()) {
            needInstantiation = true;
         }
      } break;
      default: {
         // Will only happen if clang implementation changes.
         // Protect ourselves in case that happens.
         ::Error("TClingCallFunc::make_wrapper", "Unhandled template kind!");
         return 0;
      } break;
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
      clang::FunctionDecl *FDmod = const_cast<clang::FunctionDecl *>(FD);
      clang::Sema &S = fInterp->getSema();
      // Could trigger deserialization of decls.
      cling::Interpreter::PushTransactionRAII RAII(fInterp);
      S.InstantiateFunctionDefinition(SourceLocation(), FDmod,
                                      /*Recursive=*/true,
                                      /*DefinitionRequired=*/true);
      if (!FD->isDefined(Definition)) {
         ::Error("TClingCallFunc::make_wrapper", "Failed to force template instantiation!");
         return 0;
      }
   }
   if (Definition) {
      FunctionDecl::TemplatedKind TK = Definition->getTemplatedKind();
      switch (TK) {
      case FunctionDecl::TK_NonTemplate: {
         // Ordinary function, not a template specialization.
         if (Definition->isDeleted()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a deleted function!");
            return 0;
         } else if (Definition->isLateTemplateParsed()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a late template parsed "
                                                    "function!");
            return 0;
         }
         // else if (Definition->isDefaulted()) {
         //   // Might not have a body, but we can still use it.
         //}
         // else {
         //   // Has a body.
         //}
      } break;
      case FunctionDecl::TK_FunctionTemplate: {
         // This decl is actually a function template,
         // not a function at all.
         ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a function template!");
         return 0;
      } break;
      case FunctionDecl::TK_MemberSpecialization: {
         // This function is the result of instantiating an ordinary
         // member function of a class template or of a member class
         // of a class template.
         if (Definition->isDeleted()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a deleted member function "
                                                    "of a specialization!");
            return 0;
         } else if (Definition->isLateTemplateParsed()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a late template parsed "
                                                    "member function of a specialization!");
            return 0;
         }
         // else if (Definition->isDefaulted()) {
         //   // Might not have a body, but we can still use it.
         //}
         // else {
         //   // Has a body.
         //}
      } break;
      case FunctionDecl::TK_FunctionTemplateSpecialization: {
         // This function is the result of instantiating a function
         // template or possibly an explicit specialization of a
         // function template.  Could be a namespace scope function or a
         // member function.
         if (Definition->isDeleted()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a deleted function "
                                                    "template specialization!");
            return 0;
         } else if (Definition->isLateTemplateParsed()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a late template parsed "
                                                    "function template specialization!");
            return 0;
         }
         // else if (Definition->isDefaulted()) {
         //   // Might not have a body, but we can still use it.
         //}
         // else {
         //   // Has a body.
         //}
      } break;
      case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
         // This function is the result of instantiating or
         // specializing a  member function of a class template,
         // or a member function of a class member of a class template,
         // or a member function template of a class template, or a
         // member function template of a class member of a class
         // template where at least some part of the function is
         // dependent on a template argument.
         if (Definition->isDeleted()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a deleted dependent function "
                                                    "template specialization!");
            return 0;
         } else if (Definition->isLateTemplateParsed()) {
            ::Error("TClingCallFunc::make_wrapper", "Cannot make wrapper for a late template parsed "
                                                    "dependent function template specialization!");
            return 0;
         }
         // else if (Definition->isDefaulted()) {
         //   // Might not have a body, but we can still use it.
         //}
         // else {
         //   // Has a body.
         //}
      } break;
      default: {
         // Will only happen if clang implementation changes.
         // Protect ourselves in case that happens.
         ::Error("TClingCallFunc::make_wrapper", "Unhandled template kind!");
         return 0;
      } break;
      }
   }
   unsigned min_args = GetMinRequiredArguments();
   unsigned num_params = FD->getNumParams();
   //
   //  Make the wrapper name.
   //
   {
      ostringstream buf;
      buf << "__cf";
      // const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
      // string mn;
      // fInterp->maybeMangleDeclName(ND, mn);
      // buf << '_' << mn;
      buf << '_' << gWrapperSerial++;
      wrapper_name = buf.str();
   }
   //
   //  Write the wrapper code.
   // FIXME: this should be synthesized into the AST!
   //
   int indent_level = 0;
   ostringstream buf;
   buf << "#pragma clang diagnostic push\n"
          "#pragma clang diagnostic ignored \"-Wformat-security\"\n"
          "__attribute__((used)) "
          "__attribute__((annotate(\"__cling__ptrcheck(off)\")))\n"
          "extern \"C\" void ";
   buf << wrapper_name;
   buf << "(void* obj, int nargs, void** args, void* ret)\n"
          "{\n";
   ++indent_level;
   if (min_args == num_params) {
      // No parameters with defaults.
      make_narg_call_with_return(num_params, class_name, buf, indent_level);
   } else {
      // We need one function call clause compiled for every
      // possible number of arguments per call.
      for (unsigned N = min_args; N <= num_params; ++N) {
         indent(buf, indent_level);
         buf << "if (nargs == " << N << ") {\n";
         ++indent_level;
         make_narg_call_with_return(N, class_name, buf, indent_level);
         --indent_level;
         indent(buf, indent_level);
         buf << "}\n";
      }
   }
   --indent_level;
   buf << "}\n"
          "#pragma clang diagnostic pop";
   wrapper = buf.str();
   return 1;
}

void TClingCallFunc::make_narg_call_with_return(const unsigned N, const string &class_name,
      ostringstream &buf, int indent_level)
{
   // Make a code string that follows this pattern:
   //
   // if (ret) {
   //    new (ret) (return_type) ((class_name*)obj)->func(args...);
   // }
   // else {
   //    (void)(((class_name*)obj)->func(args...));
   // }
   //
   const FunctionDecl *FD = GetDecl();
   if (const CXXConstructorDecl *CD = dyn_cast<CXXConstructorDecl>(FD)) {
      if (N <= 1 && llvm::isa<UsingShadowDecl>(GetFunctionOrShadowDecl())) {
         auto SpecMemKind = fInterp->getSema().getSpecialMember(CD);
         if ((N == 0 && SpecMemKind == clang::Sema::CXXDefaultConstructor) ||
             (N == 1 &&
              (SpecMemKind == clang::Sema::CXXCopyConstructor || SpecMemKind == clang::Sema::CXXMoveConstructor))) {
            // Using declarations cannot inject special members; do not call them
            // as such. This might happen by using `Base(Base&, int = 12)`, which
            // is fine to be called as `Derived d(someBase, 42)` but not as
            // copy constructor of `Derived`.
            return;
         }
      }
      make_narg_ctor_with_return(N, class_name, buf, indent_level);
      return;
   }
   QualType QT = FD->getReturnType().getCanonicalType();
   if (QT->isVoidType()) {
      ostringstream typedefbuf;
      ostringstream callbuf;
      indent(callbuf, indent_level);
      make_narg_call("void", N, typedefbuf, callbuf, class_name, indent_level);
      callbuf << ";\n";
      indent(callbuf, indent_level);
      callbuf << "return;\n";
      buf << typedefbuf.str() << callbuf.str();
   } else {
      indent(buf, indent_level);

      string type_name;
      EReferenceType refType = kNotReference;
      bool isPointer = false;

      buf << "if (ret) {\n";
      ++indent_level;
      {
         ostringstream typedefbuf;
         ostringstream callbuf;
         //
         //  Write the placement part of the placement new.
         //
         indent(callbuf, indent_level);
         callbuf << "new (ret) ";
         collect_type_info(QT, typedefbuf, callbuf, type_name,
                           refType, isPointer, indent_level, false);
         //
         //  Write the type part of the placement new.
         //
         callbuf << "(" << type_name.c_str();
         if (refType != kNotReference) {
            callbuf << "*) (&";
            type_name += "&";
         } else if (isPointer) {
            callbuf << "*) (";
            type_name += "*";
         } else {
            callbuf << ") (";
         }
         //
         //  Write the actual function call.
         //
         make_narg_call(type_name, N, typedefbuf, callbuf, class_name, indent_level);
         //
         //  End the placement new.
         //
         callbuf << ");\n";
         indent(callbuf, indent_level);
         callbuf << "return;\n";
         //
         //  Output the whole placement new expression and return statement.
         //
         buf << typedefbuf.str() << callbuf.str();
      }
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      indent(buf, indent_level);
      buf << "else {\n";
      ++indent_level;
      {
         ostringstream typedefbuf;
         ostringstream callbuf;
         indent(callbuf, indent_level);
         callbuf << "(void)(";
         make_narg_call(type_name, N, typedefbuf, callbuf, class_name, indent_level);
         callbuf << ");\n";
         indent(callbuf, indent_level);
         callbuf << "return;\n";
         buf << typedefbuf.str() << callbuf.str();
      }
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
   }
}

tcling_callfunc_Wrapper_t TClingCallFunc::make_wrapper()
{
   static map<const Decl *, void *> gWrapperStore;

   R__LOCKGUARD_CLING(gInterpreterMutex);

   const Decl *D = GetFunctionOrShadowDecl();

   auto I = gWrapperStore.find(D);
   if (I != gWrapperStore.end())
      return (tcling_callfunc_Wrapper_t)I->second;

   string wrapper_name;
   string wrapper;

   if (get_wrapper_code(wrapper_name, wrapper) == 0) return nullptr;

   //fprintf(stderr, "%s\n", wrapper.c_str());
   //
   //  Compile the wrapper code.
   //
   void *F = compile_wrapper(wrapper_name, wrapper);
   if (F) {
      gWrapperStore.insert(make_pair(D, F));
   } else {
      ::Error("TClingCallFunc::make_wrapper",
            "Failed to compile\n  ==== SOURCE BEGIN ====\n%s\n  ==== SOURCE END ====",
            wrapper.c_str());
   }
   return (tcling_callfunc_Wrapper_t)F;
}

void TClingCallFunc::exec(void *address, void *ret)
{
   SmallVector<void *, 8> vp_ary;
   const unsigned num_args = fArgVals.size();
   {
      R__LOCKGUARD_CLING(gInterpreterMutex);
      const FunctionDecl *FD = GetDecl();

      // FIXME: Consider the implicit this which is sometimes not passed.
      if (num_args < GetMinRequiredArguments()) {
         ::Error("TClingCallFunc::exec",
                 "Not enough arguments provided for %s (%d instead of the minimum %d)",
                 fMethod->Name(),
                 num_args, (int)GetMinRequiredArguments());
         return;
      } else if (!isa<CXXMethodDecl>(FD) && num_args > FD->getNumParams()) {
         ::Error("TClingCallFunc::exec",
                 "Too many arguments provided for %s (%d instead of the minimum %d)",
                 fMethod->Name(),
                 num_args, (int)GetMinRequiredArguments());
         return;
      }
      if (auto CXXMD = dyn_cast<CXXMethodDecl>(FD))
         if (!address && CXXMD && !CXXMD->isStatic() && !isa<CXXConstructorDecl>(FD)) {
            ::Error("TClingCallFunc::exec",
                    "The method %s is called without an object.",
                    fMethod->Name());
            return;
         }

      vp_ary.reserve(num_args);
      for (unsigned i = 0; i < num_args; ++i) {
         QualType QT;
         // Check if we provided a this parameter.
         // FIXME: Currently we do not provide consistently the this pointer at
         // index 0 of the call arguments passed to the wrapper.
         // In C++ we can still call member functions which do not use it. Eg:
         // struct S {int Print() { return printf("a");} }; auto r1 = ((S*)0)->Print();
         // This works just fine even though it might be UB...
         bool implicitThisPassed = i == 0 && isa<CXXMethodDecl>(FD) && num_args - FD->getNumParams() == 1;
         if (implicitThisPassed)
           QT = cast<CXXMethodDecl>(FD)->getThisType();
         else
           QT = FD->getParamDecl(i)->getType();
         QT = QT.getCanonicalType();
         if (QT->isReferenceType() || QT->isRecordType()) {
            // the argument is already a pointer value (points to the same thing
            // as the reference or pointing to object passed by value.
            vp_ary.push_back(fArgVals[i].getPtr());
         } else {
            // Check if arguments need readjusting. This can happen if we called
            // cling::Value::Create which instantiates to say double but the
            // function signature requires a float.
            ASTContext &C = FD->getASTContext();
            if (QT->isBuiltinType() && !C.hasSameType(QT, fArgVals[i].getType())) {
               switch(QT->getAs<BuiltinType>()->getKind()) {
               default:
                  ROOT::TMetaUtils::Error("TClingCallFunc::exec", "Unknown builtin type!");
#ifndef NDEBUG
                  QT->dump();
#endif // NDEBUG
                  break;
#define X(type, name)                                                   \
                  case BuiltinType::name: fArgVals[i] = cling::Value::Create(*fInterp, fArgVals[i].castAs<type>()); break;
                  CLING_VALUE_BUILTIN_TYPES
#undef X
              }
            }
            vp_ary.push_back(fArgVals[i].getPtrAddress());
         }
      }
   } // End of scope holding the lock
   (*fWrapper)(address, (int)num_args, (void **)vp_ary.data(), ret);
}

void TClingCallFunc::exec_with_valref_return(void *address, cling::Value &ret)
{
   const FunctionDecl *FD = GetDecl();

   QualType QT;
   if (llvm::isa<CXXConstructorDecl>(FD)) {
     R__LOCKGUARD_CLING(gInterpreterMutex);
     ASTContext &Context = FD->getASTContext();
     const TypeDecl *TD = dyn_cast<TypeDecl>(GetDeclContext());
     QualType ClassTy(TD->getTypeForDecl(), 0);
     QT = Context.getLValueReferenceType(ClassTy);
     ret = cling::Value(QT, *fInterp);
   } else {
     QT = FD->getReturnType().getCanonicalType();
     ret = cling::Value(QT, *fInterp);

     if (QT->isRecordType() || QT->isMemberDataPointerType())
       return exec(address, ret.getPtr());
   }
   exec(address, ret.getPtrAddress());
}

void TClingCallFunc::EvaluateArgList(const string &ArgList)
{
   R__LOCKGUARD_CLING(gInterpreterMutex);

   SmallVector<Expr *, 4> exprs;
   fInterp->getLookupHelper().findArgList(ArgList, exprs,
                                          gDebug > 5 ? cling::LookupHelper::WithDiagnostics
                                          : cling::LookupHelper::NoDiagnostics);
   for (SmallVectorImpl<Expr *>::const_iterator I = exprs.begin(),
         E = exprs.end(); I != E; ++I) {
      cling::Value val;
      EvaluateExpr(*fInterp, *I, val);
      if (!val.isValid()) {
         // Bad expression, all done.
         ::Error("TClingCallFunc::EvaluateArgList",
               "Bad expression in parameter %d of '%s'!",
               (int)(I - exprs.begin()),
               ArgList.c_str());
         return;
      }
      fArgVals.push_back(val);
   }
}

void TClingCallFunc::Exec(void *address, TInterpreterValue *interpVal/*=0*/)
{
   IFacePtr();
   if (!fWrapper) {
      ::Error("TClingCallFunc::Exec(address, interpVal)",
            "Called with no wrapper, not implemented!");
      return;
   }
   if (!interpVal || !interpVal->GetValAddr()) {
      exec(address, nullptr);
      return;
   }
   cling::Value *val = reinterpret_cast<cling::Value *>(interpVal->GetValAddr());
   exec_with_valref_return(address, *val);
}

template <typename T>
T TClingCallFunc::ExecT(void *address)
{
   IFacePtr();
   if (!fWrapper) {
      ::Error("TClingCallFunc::ExecT",
            "Called with no wrapper, not implemented!");
      return 0;
   }
   cling::Value ret;
   exec_with_valref_return(address, ret);
   if (ret.isVoid()) {
      return 0;
   }

   if (ret.needsManagedAllocation())
      ((TCling *)gCling)->RegisterTemporary(ret);

   return ret.castAs<T>();
}

Longptr_t TClingCallFunc::ExecInt(void *address)
{
   return ExecT<Longptr_t>(address);
}

long long TClingCallFunc::ExecInt64(void *address)
{
   return ExecT<long long>(address);
}

double TClingCallFunc::ExecDouble(void *address)
{
   return ExecT<double>(address);
}

void TClingCallFunc::ExecWithArgsAndReturn(void *address, const void *args[] /*= 0*/,
      int nargs /*= 0*/, void *ret/*= 0*/)
{
   IFacePtr();
   if (!fWrapper) {
      ::Error("TClingCallFunc::ExecWithArgsAndReturn(address, args, ret)",
            "Called with no wrapper, not implemented!");
      return;
   }
   (*fWrapper)(address, nargs, const_cast<void **>(args), ret);
}

void TClingCallFunc::ExecWithReturn(void *address, void *ret/*= 0*/)
{
   IFacePtr();
   if (!fWrapper) {
      ::Error("TClingCallFunc::ExecWithReturn(address, ret)",
            "Called with no wrapper, not implemented!");
      return;
   }
   exec(address, ret);
}

void *TClingCallFunc::ExecDefaultConstructor(const TClingClassInfo *info,
                                             ROOT::TMetaUtils::EIOCtorCategory kind,
                                             const std::string &type_name,
                                             void *address /*=0*/, unsigned long nary /*= 0UL*/)
{
   if (!info->IsValid()) {
      ::Error("TClingCallFunc::ExecDefaultConstructor", "Invalid class info!");
      return nullptr;
   }
   // this function's clients assume nary = 0 for operator new and nary > 0 for array new. JitCall expects
   // the number of objects you want to construct; nary = 1 constructs a single object
   // This handles this difference in semantics
   if (nary == 0)
      nary = 1;

   clang::Decl *D = const_cast<clang::Decl *>(info->GetDecl());

   if (Cpp::IsClass(D) || Cpp::IsConstructor(D)) {
      R__LOCKGUARD_CLING(gInterpreterMutex);
      return Cpp::Construct(D, address, nary);
   }

   ::Error("TClingCallFunc::ExecDefaultConstructor", "ClassInfo missing a valid Scope/Constructor");
   return nullptr;
}

void TClingCallFunc::ExecDestructor(const TClingClassInfo *info, void *address /*=0*/,
                                    unsigned long nary /*= 0UL*/, bool withFree /*= true*/)
{
   if (!info->IsValid()) {
      ::Error("TClingCallFunc::ExecDestructor", "Invalid class info!");
      return;
   }

   R__LOCKGUARD_CLING(gInterpreterMutex);

   if (Cpp::Destruct(address, info->GetDecl(), nary, withFree))
      return;

   ::Error("TClingCallFunc::ExecDestructor", "Called with no wrapper, not implemented!");
}

TClingMethodInfo *
TClingCallFunc::FactoryMethod() const
{
   return new TClingMethodInfo(*fMethod);
}

void TClingCallFunc::Init()
{
   fMethod.reset();
   fWrapper = nullptr;
   fDecl = nullptr;
   fMinRequiredArguments = -1;
   ResetArg();
}

void TClingCallFunc::Init(const TClingMethodInfo &minfo)
{
   Init();
   fMethod = std::unique_ptr<TClingMethodInfo>(new TClingMethodInfo(minfo));
}

void TClingCallFunc::Init(std::unique_ptr<TClingMethodInfo> minfo)
{
   Init();
   fMethod = std::move(minfo);
}

void *TClingCallFunc::InterfaceMethod()
{
   if (!IsValid()) {
      return nullptr;
   }

   if (!fWrapper)
      fWrapper = make_wrapper();

   return (void *)fWrapper.load();
}

bool TClingCallFunc::IsValid() const
{
   if (!fMethod) {
      return false;
   }
   return fMethod->IsValid();
}

TInterpreter::CallFuncIFacePtr_t TClingCallFunc::IFacePtr()
{
   if (!IsValid()) {
      ::Error("TClingCallFunc::IFacePtr(kind)",
            "Attempt to get interface while invalid.");
      return TInterpreter::CallFuncIFacePtr_t();
   }

   if (!fWrapper)
      fWrapper = make_wrapper();

   return TInterpreter::CallFuncIFacePtr_t(fWrapper);
}


void TClingCallFunc::ResetArg()
{
   fArgVals.clear();
}

void TClingCallFunc::SetArgArray(Longptr_t *paramArr, int nparam)
{
   ResetArg();
   for (int i = 0; i < nparam; ++i) {
      SetArg(paramArr[i]);
   }
}

void TClingCallFunc::SetArgs(const char *params)
{
   ResetArg();
   EvaluateArgList(params);
}

void TClingCallFunc::SetFunc(const TClingClassInfo *info, const char *method, const char *arglist,
                             Longptr_t *poffset)
{
   SetFunc(info, method, arglist, false, poffset);
}

void TClingCallFunc::SetFunc(const TClingClassInfo *info, const char *method, const char *arglist,
                             bool objectIsConst, Longptr_t *poffset)
{
   Init(std::unique_ptr<TClingMethodInfo>(new TClingMethodInfo(fInterp)));
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      ::Error("TClingCallFunc::SetFunc", "Class info is invalid!");
      return;
   }
   if (!strcmp(arglist, ")")) {
      // CINT accepted a single right paren as meaning no arguments.
      arglist = "";
   }
   *fMethod = info->GetMethodWithArgs(method, arglist, objectIsConst, poffset);
   if (!fMethod->IsValid()) {
      //::Error("TClingCallFunc::SetFunc", "Could not find method %s(%s)", method,
      //      arglist);
      return;
   }
   // FIXME: The arglist was already parsed by the lookup, we should
   //        enhance the lookup to return the resulting expression
   //        list so we do not need to parse it again here.
   EvaluateArgList(arglist);
}

void TClingCallFunc::SetFunc(const TClingMethodInfo *info)
{
   Init(std::unique_ptr<TClingMethodInfo>(new TClingMethodInfo(*info)));
   ResetArg();
   if (!fMethod->IsValid()) {
      return;
   }
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info, const char *method,
                                  const char *proto, Longptr_t *poffset,
                                  EFunctionMatchMode mode/*=kConversionMatch*/)
{
   SetFuncProto(info, method, proto, false, poffset, mode);
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info, const char *method,
                                  const char *proto, bool objectIsConst, Longptr_t *poffset,
                                  EFunctionMatchMode mode/*=kConversionMatch*/)
{
   Init(std::unique_ptr<TClingMethodInfo>(new TClingMethodInfo(fInterp)));
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      ::Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //::Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info, const char *method,
                                  const llvm::SmallVectorImpl<clang::QualType> &proto, Longptr_t *poffset,
                                  EFunctionMatchMode mode/*=kConversionMatch*/)
{
   SetFuncProto(info, method, proto, false, poffset, mode);
}

void TClingCallFunc::SetFuncProto(const TClingClassInfo *info, const char *method,
                                  const llvm::SmallVectorImpl<clang::QualType> &proto,
                                  bool objectIsConst, Longptr_t *poffset,
                                  EFunctionMatchMode mode/*=kConversionMatch*/)
{
   Init(std::unique_ptr<TClingMethodInfo>(new TClingMethodInfo(fInterp)));
   if (poffset) {
      *poffset = 0L;
   }
   ResetArg();
   if (!info->IsValid()) {
      ::Error("TClingCallFunc::SetFuncProto", "Class info is invalid!");
      return;
   }
   *fMethod = info->GetMethod(method, proto, objectIsConst, poffset, mode);
   if (!fMethod->IsValid()) {
      //::Error("TClingCallFunc::SetFuncProto", "Could not find method %s(%s)",
      //      method, proto);
      return;
   }
}

