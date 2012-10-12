#include <cassert>
#include <string>
#include <set>

//CLANG
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Specifiers.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"

//CLING
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Interpreter.h"

//CLUNG??
#include "TClingDisplayClass.h"
#include "TSystem.h"
#include "TString.h"
#include "TError.h"

namespace TClingDisplayClass {

namespace {

typedef clang::DeclContext::decl_iterator decl_iterator;
typedef clang::CXXRecordDecl::base_class_const_iterator base_decl_iterator;

//______________________________________________________________________________
void AppendLocation(const clang::CompilerInstance *compiler, const clang::CXXRecordDecl *classDecl, std::string &textLine, bool verbose)
{
   //Location has a fixed format - from G__display_class.

   assert(compiler != 0 && "AppendLocation, 'compiler' parameter is null");
   assert(classDecl != 0 && "AppendLocation, 'classDecl' parameter is null");

   TString formatted("");
   if (compiler->hasSourceManager()) {
      const clang::SourceManager &sourceManager = compiler->getSourceManager();
      clang::PresumedLoc loc(sourceManager.getPresumedLoc(classDecl->getLocation()));
      if (loc.isValid()) {
         if (!verbose)
            formatted.Form("%-25s%5d", gSystem->BaseName(loc.getFilename()), int(loc.getLine()));
         else
            formatted.Form("FILE: %s LINE: %d", gSystem->BaseName(loc.getFilename()), int(loc.getLine()));
      }
   }

   //No source manager or location is invalid(?)
   if (!formatted.Length()) {
      if (!verbose)
         formatted.Form("%-30s", " ");
   }

   if (formatted.Length())
      textLine += formatted.Data();
}

//______________________________________________________________________________
void AppendClassKeyword(const clang::CXXRecordDecl *classDecl, std::string &name)
{
   assert(classDecl != 0 && "AppendClassKeyword, 'classDecl' parameter is null");

   name += classDecl->getKindName();
   name += ' ';
}

//______________________________________________________________________________
void AppendClassName(const clang::CXXRecordDecl *classDecl, std::string &name)
{
   assert(classDecl != 0 && "AppendClassName, 'classDecl' parameter is null");

   if (const clang::NamedDecl * const namedClassDecl = llvm::dyn_cast<clang::NamedDecl>(classDecl)) {
      const clang::LangOptions langOpts;
      const clang::PrintingPolicy printingPolicy(langOpts);
      std::string tmp;
      //Name for diagnostic will include template arguments if any.
      namedClassDecl->getNameForDiagnostic(tmp, printingPolicy, true);//true == qualified name.
      name += tmp;
   } else
      name += "<unknown class>";
}

//______________________________________________________________________________
void AppendBaseClassSpecifiers(base_decl_iterator base, std::string &textLine)
{
   if (base->isVirtual())
      textLine += "virtual ";

   switch (base->getAccessSpecifier()) {
   case clang::AS_private:
      textLine += "private";
      break;
   case clang::AS_protected:
      textLine += "protected";
      break;
   case clang::AS_public:
   case clang::AS_none://TODO - check this.
      textLine += "public";
   }
}

//______________________________________________________________________________
void AppendClassSize(const clang::CompilerInstance *compiler, const clang::RecordDecl *decl, std::string &textLine)
{
   assert(compiler != 0 && "AppendClassSize, 'compiler' parameter is null");
   assert(decl != 0 && "AppendClassSize, 'decl' parameter is null");

   const clang::ASTRecordLayout &layout = compiler->getASTContext().getASTRecordLayout(decl);

   TString formatted(TString::Format("SIZE: %d", int(layout.getSize().getQuantity())));
   textLine += formatted.Data();
}

//______________________________________________________________________________
void AppendBaseClassOffset(const clang::CompilerInstance *compiler,
                           const clang::CXXRecordDecl *completeClass,
                           const clang::CXXRecordDecl *baseClass,
                           bool isVirtual,
                           std::string &textLine)
{
   assert(compiler != 0 && "AppendBaseClassOffset, 'compiler' parameter is null");
   assert(completeClass != 0 && "AppendBaseClassOffset, 'completeClass' parameter is null");
   assert(baseClass != 0 && "AppendBaseClassOffset, 'baseClass' parameter is null");

   const clang::ASTRecordLayout &layout = compiler->getASTContext().getASTRecordLayout(completeClass);

   TString formatted;
   if (isVirtual)//format is from G__display_classinheritance.
      formatted.Form("0x%-8x", int(layout.getVBaseClassOffset(baseClass).getQuantity()));
   else
      formatted.Form("0x%-8x", int(layout.getBaseClassOffset(baseClass).getQuantity()));

   textLine += formatted.Data();
}

//
//This is a primitive class which does nothing except fprintf for the moment,
//but this can change later.
class FILEPrintHelper {
public:
   FILEPrintHelper(FILE *out);

   void Print(const char *msg)const;

private:
   FILE *fOut;
};

//______________________________________________________________________________
FILEPrintHelper::FILEPrintHelper(FILE *out)
                   : fOut(out)
{
   assert(out != 0 && "FILEPrintHelper, 'out' parameter is null");
}

//______________________________________________________________________________
void FILEPrintHelper::Print(const char *msg)const
{
   assert(fOut != 0 && "Print, fOut is null");
   assert(msg != 0 && "Print, 'msg' parameter is null");

   fprintf(fOut, "%s", msg);
}

//
//Aux. class to traverse translation-unit-declaration/class-declaration.
//

class ClassPrinter {
private:
   enum {
      kBaseTreeShift = 3
   };
public:
   ClassPrinter(FILE * out, const class cling::Interpreter *interpreter);

   void DisplayAllClasses()const;
   void DisplayClass(const std::string &className)const;

   void SetVerbose(bool verbose);
private:

   //These are declarations, which can contain nested class declarations,
   //I have separate function for the case I want to treat them in different ways.
   //Can be only one processDecl actually.

   void ProcessDecl(decl_iterator decl)const;
   void ProcessBlockDecl(decl_iterator decl)const;
   void ProcessFunctionDecl(decl_iterator decl)const;
   void ProcessNamespaceDecl(decl_iterator decl)const;
   void ProcessLinkageSpecDecl(decl_iterator decl)const;
   void ProcessClassDecl(decl_iterator decl)const;

   void DisplayClassDecl(const clang::CXXRecordDecl *classDecl)const;
   void DisplayBasesAsList(const clang::CXXRecordDecl *classDecl)const;
   void DisplayBasesAsTree(const clang::CXXRecordDecl *classDecl, unsigned nSpaces)const;
   void DisplayBases(const clang::CXXRecordDecl *classDecl)const;

   FILEPrintHelper fOut;
   const cling::Interpreter *fInterpreter;
   bool fVerbose;

   mutable std::set<const clang::Decl *> fSeenDecls;
};

//______________________________________________________________________________
ClassPrinter::ClassPrinter(FILE *out, const cling::Interpreter *interpreter)
                : fOut(out),
                  fInterpreter(interpreter),
                  fVerbose(false)
{
   assert(interpreter != 0 && "ClassPrinter, 'compiler' parameter is null");
}


//______________________________________________________________________________
void ClassPrinter::DisplayAllClasses()const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "DisplayAllClasses, fCompiler is null");

   const clang::CompilerInstance * const compiler = fInterpreter->getCI();
   assert(compiler != 0 && "DisplayAllClasses, compiler instance is null");

   const clang::TranslationUnitDecl * const tuDecl = compiler->getASTContext().getTranslationUnitDecl();
   assert(tuDecl != 0 && "DisplayAllClasses, translation unit is empty");

   fSeenDecls.clear();

   for (decl_iterator decl = tuDecl->decls_begin(); decl != tuDecl->decls_end(); ++decl)
      ProcessDecl(decl);
}

//______________________________________________________________________________
void ClassPrinter::DisplayClass(const std::string &className)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "DisplayClass, fCompiler is null");

   fSeenDecls.clear();

   const cling::LookupHelper &lookupHelper = fInterpreter->getLookupHelper();
   if (const clang::Decl *const decl = lookupHelper.findScope(className)) {
      if (const clang::CXXRecordDecl * const classDecl = llvm::dyn_cast<clang::CXXRecordDecl>(decl)) {
         if (classDecl->hasDefinition())
            DisplayClassDecl(classDecl);
      } else {
         if (gDebug > 0)
            ::Info("ClassPrinter::DisplayClass", "entity %s is not a class/struct/union", className.c_str());
      }
   } else {
      if (gDebug > 0)
         ::Info("ClassPrinter::DisplayClass", "cling class not found, name: %s\n", className.c_str());
   }
}

//______________________________________________________________________________
void ClassPrinter::SetVerbose(bool verbose)
{
   fVerbose = verbose;
}

//______________________________________________________________________________
void ClassPrinter::ProcessDecl(decl_iterator decl)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "ProcessDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessDecl, 'decl' parameter is not a valid iterator");

   switch (decl->getKind()) {
   case clang::Decl::Namespace:
      ProcessNamespaceDecl(decl);
      break;
   case clang::Decl::Block:
      ProcessBlockDecl(decl);
      break;
   case clang::Decl::Function:
   case clang::Decl::CXXMethod:
   case clang::Decl::CXXConstructor:
   case clang::Decl::CXXConversion:
   case clang::Decl::CXXDestructor:
      ProcessFunctionDecl(decl);
      break;
   case clang::Decl::LinkageSpec:
      ProcessLinkageSpecDecl(decl);
      break;
   case clang::Decl::CXXRecord:
   case clang::Decl::ClassTemplateSpecialization:
   case clang::Decl::ClassTemplatePartialSpecialization:
      ProcessClassDecl(decl);
      break;
   default:
      if (llvm::dyn_cast<clang::FunctionDecl>(*decl))
         ProcessFunctionDecl(decl);//decl->getKind() != clang::Decl::Function.
      break;
   }
}

//______________________________________________________________________________
void ClassPrinter::ProcessBlockDecl(decl_iterator decl)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "ProcessBlockDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessBlockDecl, 'decl' parameter is not a valid iterator");
   assert(decl->getKind() == clang::Decl::Block && "ProcessBlockDecl, decl->getKind() != BlockDecl");

   //Block can contain nested (arbitrary deep) class declarations.
   //Though, I'm not sure if have block in our code.
   const clang::BlockDecl *blockDecl = llvm::dyn_cast<clang::BlockDecl>(*decl);
   assert(blockDecl != 0 && "ProcessBlockDecl, internal error - decl is not a BlockDecl");

   for (decl_iterator it = blockDecl->decls_begin(); it != blockDecl->decls_end(); ++it)
      ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessFunctionDecl(decl_iterator decl)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "ProcessFunctionDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessFunctionDecl, 'decl' parameter is not a valid iterator");

   //Function can contain class declarations, we have to check this.
   const clang::FunctionDecl *functionDecl = llvm::dyn_cast<clang::FunctionDecl>(*decl);
   assert(functionDecl != 0 && "ProcessFunctionDecl, internal error - decl is not a FunctionDecl");

   for (decl_iterator it = functionDecl->decls_begin(); it != functionDecl->decls_end(); ++it)
      ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessNamespaceDecl(decl_iterator decl)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "ProcessNamespaceDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessNamespaceDecl, 'decl' parameter is not a valid iterator");
   assert(decl->getKind() == clang::Decl::Namespace && "ProcessNamespaceDecl, decl->getKind() != Namespace");

   //Namespace can contain nested (arbitrary deep) class declarations.
   const clang::NamespaceDecl *namespaceDecl = llvm::dyn_cast<clang::NamespaceDecl>(*decl);
   assert(namespaceDecl != 0 && "ProcessNamespaceDecl, 'decl' parameter is not a NamespaceDecl");

   for (decl_iterator it = namespaceDecl->decls_begin(); it != namespaceDecl->decls_end(); ++it)
      ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessLinkageSpecDecl(decl_iterator decl)const
{
   //Just in case asserts were deleted from ctor:
   assert(fInterpreter != 0 && "ProcessLinkageSpecDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessLinkageSpecDecl, 'decl' parameter is not a valid iterator");

   const clang::LinkageSpecDecl *linkageSpec = llvm::dyn_cast<clang::LinkageSpecDecl>(*decl);
   assert(linkageSpec != 0 && "ProcessLinkageSpecDecl, internal error - decl is not a LinkageSpecDecl");

   for (decl_iterator it = linkageSpec->decls_begin(); it != linkageSpec->decls_end(); ++it)
      ProcessDecl(it);
}

//______________________________________________________________________________
void ClassPrinter::ProcessClassDecl(decl_iterator decl)const
{
   assert(fInterpreter != 0 && "ProcessClassDecl, fInterpreter is null");
   assert(*decl != 0 && "ProcessClassDecl, 'decl' parameter is not a valid iterator");

   const clang::CXXRecordDecl *classDecl = llvm::dyn_cast<clang::CXXRecordDecl>(*decl);
   assert(classDecl != 0 && "ProcessClassDecl, internal error, declaration is not a CXXRecordDecl");

   if (!classDecl->hasDefinition())
      return;

   DisplayClassDecl(classDecl);

   //Now we have to check nested scopes for class declarations.
   for (decl_iterator decl = classDecl->decls_begin(); decl != classDecl->decls_end(); ++decl)
      ProcessDecl(decl);
}

//______________________________________________________________________________
void ClassPrinter::DisplayClassDecl(const clang::CXXRecordDecl *classDecl)const
{
   assert(classDecl != 0 && "DisplayClassDecl, 'classDecl' parameter is null");
   assert(fInterpreter != 0 && "DisplayClassDecl, fInterpreter is null");

   classDecl = classDecl->getDefinition();
   assert(classDecl != 0 && "DisplayClassDecl, invalid decl - no definition");

   if (fSeenDecls.find(classDecl) != fSeenDecls.end())
      return;
   else
      fSeenDecls.insert(classDecl);

   if (!fVerbose) {
      //Print: source file, line number, class-keyword, qualifies class name, base classes.
      std::string classInfo;

      AppendLocation(fInterpreter->getCI(), classDecl, classInfo, false);
      classInfo += ' ';
      AppendClassKeyword(classDecl, classInfo);
      classInfo += ' ';
      AppendClassName(classDecl, classInfo);
      classInfo += ' ';
      //
      fOut.Print(classInfo.c_str());

      DisplayBasesAsList(classDecl);

      fOut.Print("\n");
   } else {
      fOut.Print("===========================================================================\n");//Hehe, this line was stolen from CINT.

      std::string classInfo;
      AppendClassName(classDecl, classInfo);
      fOut.Print(classInfo.c_str());
      fOut.Print("\n");

      classInfo.clear();
      AppendClassSize(fInterpreter->getCI(), classDecl, classInfo);
      classInfo += ' ';
      AppendLocation(fInterpreter->getCI(), classDecl, classInfo, true);
      fOut.Print(classInfo.c_str());
      fOut.Print("\n");

      if (classDecl->bases_begin() != classDecl->bases_end())
         fOut.Print("Base classes: --------------------------------------------------------\n");

      DisplayBasesAsTree(classDecl, 0);
      //now list all members.
   }
}

//______________________________________________________________________________
void ClassPrinter::DisplayBasesAsList(const clang::CXXRecordDecl *classDecl)const
{
   assert(fInterpreter != 0 && "DisplayBasesAsList, fInterpreter is null");
   assert(classDecl != 0 && "DisplayBasesAsList, 'classDecl' parameter is 0");
   assert(classDecl->hasDefinition() == true && "DisplayBasesAsList, 'classDecl' parameter points to an invalid declaration");
   assert(fVerbose == false && "DisplayBasesAsList, called in a verbose output");

   //we print a list of base classes as one line, with access specifiers and 'virtual' if needed.
   std::string bases(": ");
   for (base_decl_iterator baseIt = classDecl->bases_begin(); baseIt != classDecl->bases_end(); ++baseIt) {
      if (baseIt != classDecl->bases_begin())
         bases += ", ";

      const clang::RecordType * const type = baseIt->getType()->getAs<clang::RecordType>();
      if (type) {
         const clang::CXXRecordDecl * const baseDecl = llvm::cast<clang::CXXRecordDecl>(type->getDecl()->getDefinition());
         if (baseDecl) {
            AppendBaseClassSpecifiers(baseIt, bases);
            bases += ' ';
            AppendClassName(baseDecl, bases);
         } else
            return;
      } else
         return;
   }

   if (bases.length() > 2) //initial ": "
      fOut.Print(bases.c_str());
}

//______________________________________________________________________________
void ClassPrinter::DisplayBasesAsTree(const clang::CXXRecordDecl *classDecl, unsigned nSpaces)const
{
   assert(classDecl != 0 && "DisplayBasesAsTree, 'classDecl' parameter is null");
   assert(classDecl->hasDefinition() == true && "DisplayBasesAsTree, 'classDecl' parameter points to an invalid declaration");

   assert(fInterpreter != 0 && "DisplayBasesAsTree, fInterpreter is null");
   assert(fVerbose == true && "DisplayBasesAsTree, call in a simplified output");

   std::string textLine;
   for (base_decl_iterator baseIt = classDecl->bases_begin(); baseIt != classDecl->bases_end(); ++baseIt) {
      textLine.assign(nSpaces, ' ');
      const clang::RecordType * const type = baseIt->getType()->getAs<clang::RecordType>();
      if (type) {
         const clang::CXXRecordDecl * const baseDecl = llvm::cast<clang::CXXRecordDecl>(type->getDecl()->getDefinition());
         if (baseDecl) {
            AppendBaseClassOffset(fInterpreter->getCI(), classDecl, baseDecl, baseIt->isVirtual(), textLine);
            textLine += ' ';
            AppendBaseClassSpecifiers(baseIt, textLine);
            textLine += ' ';
            AppendClassName(baseDecl, textLine);
            textLine += '\n';

            fOut.Print(textLine.c_str());

            DisplayBasesAsTree(baseDecl, nSpaces + kBaseTreeShift);

            continue;
         }
      }

      textLine += "<no type info for a base found>\n";
      fOut.Print(textLine.c_str());
   }
}

}//unnamed namespace

//______________________________________________________________________________
void DisplayAllClasses(FILE *out, const cling::Interpreter *interpreter, bool verbose)
{
   assert(out != 0 && "DisplayAllClasses, 'out' parameter is null");
   assert(interpreter != 0 && "DisplayAllClasses, 'interpreter' parameter is null");

   ClassPrinter printer(out, interpreter);
   printer.SetVerbose(verbose);
   printer.DisplayAllClasses();
}

//______________________________________________________________________________
void DisplayClass(FILE *out, const cling::Interpreter *interpreter, const char *className, bool verbose)
{
   assert(out != 0 && "DisplayClass, 'out' parameter is null");
   assert(interpreter != 0 && "DisplayClass, 'interpreter' parameter is null");
   assert(className != 0 && "DisplayClass, 'className' parameter is null");

   ClassPrinter printer(out, interpreter);
   printer.SetVerbose(verbose);
   printer.DisplayClass(className);

}


}//namespace cling
