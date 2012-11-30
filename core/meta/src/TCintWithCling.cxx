// @(#)root/meta:$Id$
// vim: sw=3 ts=3 expandtab foldmethod=indent

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class defines an interface to cling as interpreter backend.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "TCintWithCling.h"

#include "TClingBaseClassInfo.h"
#include "TClingCallFunc.h"
#include "TClingClassInfo.h"
#include "TClingDataMemberInfo.h"
#include "TClingMethodArgInfo.h"
#include "TClingMethodInfo.h"
#include "TClingTypedefInfo.h"
#include "TClingTypeInfo.h"

#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TClassTable.h"
#include "TClingCallbacks.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TMemberInspector.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "THashList.h"
#include "TOrdCollection.h"
#include "TVirtualPad.h"
#include "TSystem.h"
#include "TVirtualMutex.h"
#include "TError.h"
#include "TEnv.h"
#include "THashTable.h"
#include "RConfigure.h"
#include "compiledata.h"
#include "TMetaUtils.h"
#include "TVirtualCollectionProxy.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/StoredValueRef.h"
#include "cling/Interpreter/Transaction.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "cling/Utils/AST.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <cassert>
#include <map>
#include <set>
#include <stdint.h>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include <cxxabi.h>
#include <limits.h>
#include <stdio.h>

#ifdef __APPLE__
#include <dlfcn.h>
#include <mach-o/dyld.h>
#endif // __APPLE__

#ifdef R__LINUX
#include <dlfcn.h>
#endif

using namespace std;
using namespace clang;

R__EXTERN int optind;

//______________________________________________________________________________
namespace {
   // A module and its headers. Intentionally not a copy:
   // If these strings end up in this struct they are
   // long lived by definition because they get passed in
   // before initialization of TCintWithCling.
   struct ModuleHeaderInfo_t {
      ModuleHeaderInfo_t(const char* moduleName,
                         const char** headers,
                         const char** includePaths,
                         const char** macroDefines,
                         const char** macroUndefines):
         fModuleName(moduleName), fHeaders(headers),
         fIncludePaths(includePaths), fMacroDefines(macroDefines),
         fMacroUndefines(macroUndefines) {}
      const char* fModuleName; // module name
      const char** fHeaders; // 0-terminated array of header files
      const char** fIncludePaths; // 0-terminated array of header files
      const char** fMacroDefines; // 0-terminated array of header files
      const char** fMacroUndefines; // 0-terminated array of header files
   };

   llvm::SmallVector<ModuleHeaderInfo_t, 10> *gModuleHeaderInfoBuffer;
}

//______________________________________________________________________________
extern "C"
void TCintWithCling__RegisterModule(const char* modulename,
                                    const char** headers,
                                    const char** includePaths,
                                    const char** macroDefines,
                                    const char** macroUndefines)
{
   // Called by static dictionary initialization to register clang modules
   // for headers. Calls TCintWithCling::RegisterModule() unless gCling
   // is NULL, i.e. during startup, where the information is buffered in
   // the static moduleHeaderInfoBuffer which then later can be accessed
   // via the global *gModuleHeaderInfoBuffer after a call to this function
   // with modulename=0.

   static llvm::SmallVector<ModuleHeaderInfo_t, 10> moduleHeaderInfoBuffer;
   if (!modulename) {
      gModuleHeaderInfoBuffer = &moduleHeaderInfoBuffer;
      return;
   }

   if (gCint) {
      ((TCintWithCling*)gCint)->RegisterModule(modulename,
                                               headers,
                                               includePaths,
                                               macroDefines,
                                               macroUndefines);
   } else {
      moduleHeaderInfoBuffer.push_back(ModuleHeaderInfo_t(modulename,
                                                          headers,
                                                          includePaths,
                                                          macroDefines,
                                                          macroUndefines));
   }
}

// The functions are used to bridge cling/clang/llvm compiled with no-rtti and
// ROOT (which uses rtti)

// Class extracting recursively every typedef defined somewhere.
class TypedefVisitor : public RecursiveASTVisitor<TypedefVisitor> {
private:
   llvm::SmallVector<TypedefDecl*,128> &fTypedefs;
public:
   TypedefVisitor(llvm::SmallVector<TypedefDecl*,128> &defs) : fTypedefs(defs)
   {}

   bool TraverseStmt(Stmt*) {
      // Don't descend into function bodies.
      return true;
   }
   bool TraverseClassTemplateDecl(ClassTemplateDecl*) {
      // Don't descend into templates (but only instances thereof).
      return true;
   }
   bool TraverseClassTemplatePartialSpecializationDecl(ClassTemplatePartialSpecializationDecl*) {
      // Don't descend into templates partial specialization (but only instances thereof).
      return true;
   }

   bool VisitTypedefDecl(TypedefDecl *TdefD) {
      fTypedefs.push_back(TdefD);
      return true; // returning false will abort the in-depth traversal.
   }
};

//______________________________________________________________________________
static void TCintWithCling__UpdateClassInfo(TagDecl* TD)
{
   // Update TClingClassInfo for a class (e.g. upon seeing a definition).
   static Bool_t entered = kFALSE;
   static vector<TagDecl*> updateList;
   Bool_t topLevel;

   if (entered) topLevel = kFALSE;
   else {
      entered = kTRUE;
      topLevel = kTRUE;
   }
   if (topLevel) {
      ((TCintWithCling*)gInterpreter)->UpdateClassInfoWithDecl(TD);
   } else {
      // If we are called indirectly from within another call to
      // TCint::UpdateClassInfo, we delay the update until the dictionary loading
      // is finished (i.e. when we return to the top level TCint::UpdateClassInfo).
      // This allows for the dictionary to be fully populated when we actually
      // update the TClass object.   The updating of the TClass sometimes
      // (STL containers and when there is an emulated class) forces the building
      // of the TClass object's real data (which needs the dictionary info).
      updateList.push_back(TD);
   }
   if (topLevel) {
      while (!updateList.empty()) {
         ((TCintWithCling*)gInterpreter)
            ->UpdateClassInfoWithDecl(updateList.back());
         updateList.pop_back();
      }
      entered = kFALSE;
   }
}



extern "C" 
void TCintWithCling__UpdateListsOnCommitted(const cling::Transaction &T) {
   TCollection *listOfSmth = 0;
   cling::Interpreter* interp = ((TCintWithCling*)gCint)->GetInterpreter();
   for(cling::Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
       I != E; ++I)
      for (DeclGroupRef::const_iterator DI = I->begin(), DE = I->end(); 
           DI != DE; ++DI) {
         if (isa<DeclContext>(*DI) && !isa<EnumDecl>(*DI)) {
            // We have to find all the typedefs contained in that decl context
            // and add it to the list of types.
            listOfSmth = gROOT->GetListOfTypes();
            llvm::SmallVector<TypedefDecl*, 128> Defs;
            TypedefVisitor V(Defs);
            V.TraverseDecl(*DI);
            for (size_t i = 0; i < Defs.size(); ++i)
            if (!listOfSmth->FindObject(Defs[i]->getNameAsString().c_str())) {
               listOfSmth->Add(new TDataType(new TClingTypedefInfo(interp, Defs[i])));
            }

         }
         if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*DI)) {
            listOfSmth = gROOT->GetListOfGlobalFunctions();
            if (!isa<TranslationUnitDecl>(FD->getDeclContext()))
               continue;
            if(FD->isOverloadedOperator() 
               || cling::utils::Analyze::IsWrapper(FD))
               continue;

            if (!listOfSmth->FindObject(FD->getNameAsString().c_str())) {  
               listOfSmth->Add(new TFunction(new TClingMethodInfo(interp, FD)));
            }            
         }
         else if (TagDecl *TD = dyn_cast<TagDecl>(*DI)) {
            TCintWithCling__UpdateClassInfo(TD);
         }
         else if (const TypedefDecl* TdefD = dyn_cast<TypedefDecl>(*DI)) {
            listOfSmth = gROOT->GetListOfTypes();
            if (!listOfSmth->FindObject(TdefD->getNameAsString().c_str())) {
               listOfSmth->Add(new TDataType(new TClingTypedefInfo(interp, TdefD)));
            }
         }
         else if (const NamedDecl *ND = dyn_cast<NamedDecl>(*DI)) {
            // We care about declarations on the global scope.
            if (!isa<TranslationUnitDecl>(ND->getDeclContext()))
               continue;

            // ROOT says that global is enum/var/field declared on the global
            // scope.

            listOfSmth = gROOT->GetListOfGlobals();

            if (!(isa<EnumDecl>(ND) || isa<VarDecl>(ND)))
               continue;

            // Skip if already in the list.
            if (listOfSmth->FindObject(ND->getNameAsString().c_str()))
               continue;

            if (EnumDecl *ED = dyn_cast<EnumDecl>(*DI)) {
               for(EnumDecl::enumerator_iterator EDI = ED->enumerator_begin(),
                      EDE = ED->enumerator_end(); EDI != EDE; ++EDI) {
                  if (!listOfSmth->FindObject((*EDI)->getNameAsString().c_str())) {  
                     listOfSmth->Add(new TGlobal(new TClingDataMemberInfo(interp, *EDI)));
                  }
               }
            }
            else
               listOfSmth->Add(new TGlobal(new TClingDataMemberInfo(interp, 
                                                                    cast<ValueDecl>(ND))));
         }
      }
}

extern "C" 
void TCintWithCling__UpdateListsOnUnloaded(const cling::Transaction &T) {
   TGlobal *global = 0;
   TCollection* globals = gROOT->GetListOfGlobals();
   for(cling::Transaction::const_iterator I = T.decls_begin(), E = T.decls_end();
       I != E; ++I)
      for (DeclGroupRef::const_iterator
              DI = I->begin(), DE = I->end(); DI != DE; ++DI) {

         if (const VarDecl* VD = dyn_cast<VarDecl>(*DI)) {
            global = (TGlobal*)globals->FindObject(VD->getNameAsString().c_str());
            if (global) {
               globals->Remove(global);
               if (!globals->IsOwner())
                  delete global;
            }
         }
      }
}

extern "C"  
TObject* TCintWithCling__GetObjectAddress(const char *Name, void *&LookupCtx) {
   return gROOT->FindSpecialObject(Name, LookupCtx);
}

extern "C" const Decl* TCintWithCling__GetObjectDecl(TObject *obj) {
   return ((TClingClassInfo*)obj->IsA()->GetClassInfo())->GetDecl();
}

// Load library containing specified class. Returns 0 in case of error
// and 1 in case if success.
extern "C" int TCintWithCling__AutoLoadCallback(const char* className)
{
   string cls(className);
   return gCint->AutoLoad(cls.c_str());
}

//______________________________________________________________________________
//
//
//

void* autoloadCallback(const std::string& mangled_name)
{
   // Autoload a library. Given a mangled function name find the
   // library which provides the function and load it.
   //--
   //
   //  Use the C++ ABI provided function to demangle the function name.
   //
   int err = 0;
   char* demangled_name = abi::__cxa_demangle(mangled_name.c_str(), 0, 0, &err);
   if (err) {
      return 0;
   }
   //fprintf(stderr, "demangled name: '%s'\n", demangled_name);
   //
   //  Separate out the class or namespace part of the
   //  function name.
   //
   std::string name(demangled_name);
   // Remove the function arguments.
   std::string::size_type pos = name.rfind('(');
   if (pos != std::string::npos) {
      name.erase(pos);
   }
   // Remove the function name.
   pos = name.rfind(':');
   if (pos != std::string::npos) {
      if ((pos != 0) && (name[pos-1] == ':')) {
         name.erase(pos-1);
      }
   }
   //fprintf(stderr, "name: '%s'\n", name.c_str());
   // Now we have the class or namespace name, so do the lookup.
   TString libs = gCint->GetClassSharedLibs(name.c_str());
   if (libs.IsNull()) {
      // Not found in the map, all done.
      return 0;
   }
   //fprintf(stderr, "library: %s\n", iter->second.c_str());
   // Now we have the name of the libraries to load, so load them.
   
   TString lib;
   Ssiz_t posLib = 0;
   while (libs.Tokenize(lib, posLib)) {
      if (gInterpreter->Load(lib, kFALSE /*system*/) < 0) {
         // The library load failed, all done.
         //fprintf(stderr, "load failed: %s\n", errmsg.c_str());
         return 0;
      }
   }
   //fprintf(stderr, "load succeeded.\n");
   // Get the address of the function being called.
   void* addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(mangled_name.c_str());
   //fprintf(stderr, "addr: %016lx\n", reinterpret_cast<unsigned long>(addr));
   return addr;
}

//______________________________________________________________________________
//
//
//

extern "C" int ScriptCompiler(const char* filename, const char* opt)
{
   return gSystem->CompileMacro(filename, opt);
}

extern "C" int IgnoreInclude(const char* fname, const char* expandedfname)
{
   return gROOT->IgnoreInclude(fname, expandedfname);
}

//______________________________________________________________________________
//
//
//

int TCint_GenerateDictionary(const std::vector<std::string> &classes,
                             const std::vector<std::string> &headers,
                             const std::vector<std::string> &fwdDecls,
                             const std::vector<std::string> &unknown)
{
   //This function automatically creates the "LinkDef.h" file for templated
   //classes then executes CompileMacro on it.
   //The name of the file depends on the class name, and it's not generated again
   //if the file exist.
   if (classes.empty()) {
      return 0;
   }
   // Use the name of the first class as the main name.
   const std::string& className = classes[0];
   //(0) prepare file name
   TString fileName = "AutoDict_";
   std::string::const_iterator sIt;
   for (sIt = className.begin(); sIt != className.end(); sIt++) {
      if (*sIt == '<' || *sIt == '>' ||
            *sIt == ' ' || *sIt == '*' ||
            *sIt == ',' || *sIt == '&' ||
            *sIt == ':') {
         fileName += '_';
      }
      else {
         fileName += *sIt;
      }
   }
   if (classes.size() > 1) {
      Int_t chk = 0;
      std::vector<std::string>::const_iterator it = classes.begin();
      while ((++it) != classes.end()) {
         for (UInt_t cursor = 0; cursor != it->length(); ++cursor) {
            chk = chk * 3 + it->at(cursor);
         }
      }
      fileName += TString::Format("_%u", chk);
   }
   fileName += ".cxx";
   if (gSystem->AccessPathName(fileName) != 0) {
      //file does not exist
      //(1) prepare file data
      // If STL, also request iterators' operators.
      // vector is special: we need to check whether
      // vector::iterator is a typedef to pointer or a
      // class.
      static std::set<std::string> sSTLTypes;
      if (sSTLTypes.empty()) {
         sSTLTypes.insert("vector");
         sSTLTypes.insert("list");
         sSTLTypes.insert("deque");
         sSTLTypes.insert("map");
         sSTLTypes.insert("multimap");
         sSTLTypes.insert("set");
         sSTLTypes.insert("multiset");
         sSTLTypes.insert("queue");
         sSTLTypes.insert("priority_queue");
         sSTLTypes.insert("stack");
         sSTLTypes.insert("iterator");
      }
      std::vector<std::string>::const_iterator it;
      std::string fileContent("");
      for (it = headers.begin(); it != headers.end(); ++it) {
         fileContent += "#include \"" + *it + "\"\n";
      }
      for (it = unknown.begin(); it != unknown.end(); ++it) {
         TClass* cl = TClass::GetClass(it->c_str());
         if (cl && cl->GetDeclFileName()) {
            TString header(gSystem->BaseName(cl->GetDeclFileName()));
            TString dir(gSystem->DirName(cl->GetDeclFileName()));
            TString dirbase(gSystem->BaseName(dir));
            while (dirbase.Length() && dirbase != "."
                   && dirbase != "include" && dirbase != "inc"
                   && dirbase != "prec_stl") {
               gSystem->PrependPathName(dirbase, header);
               dir = gSystem->DirName(dir);
            }
            fileContent += TString("#include \"") + header + "\"\n";
         }
      }
      for (it = fwdDecls.begin(); it != fwdDecls.end(); ++it) {
         fileContent += "class " + *it + ";\n";
      }
      fileContent += "#ifdef __CINT__ \n";
      fileContent += "#pragma link C++ nestedclasses;\n";
      fileContent += "#pragma link C++ nestedtypedefs;\n";
      for (it = classes.begin(); it != classes.end(); ++it) {
         std::string n(*it);
         size_t posTemplate = n.find('<');
         std::set<std::string>::const_iterator iSTLType = sSTLTypes.end();
         if (posTemplate != std::string::npos) {
            n.erase(posTemplate, std::string::npos);
            if (n.compare(0, 5, "std::") == 0) {
               n.erase(0, 5);
            }
            iSTLType = sSTLTypes.find(n);
         }
         fileContent += "#pragma link C++ class ";
         fileContent +=    *it + "+;\n" ;
         fileContent += "#pragma link C++ class ";
         if (iSTLType != sSTLTypes.end()) {
            // STL class; we cannot (and don't need to) store iterators;
            // their shadow and the compiler's version don't agree. So
            // don't ask for the '+'
            fileContent +=    *it + "::*;\n" ;
         }
         else {
            // Not an STL class; we need to allow the I/O of contained
            // classes (now that we have a dictionary for them).
            fileContent +=    *it + "::*+;\n" ;
         }
         std::string oprLink("#pragma link C++ operators ");
         oprLink += *it;
         // Don't! Requests e.g. op<(const vector<T>&, const vector<T>&):
         // fileContent += oprLink + ";\n";
         if (iSTLType != sSTLTypes.end()) {
            if (n == "vector") {
               fileContent += "#ifdef G__VECTOR_HAS_CLASS_ITERATOR\n";
            }
            fileContent += oprLink + "::iterator;\n";
            fileContent += oprLink + "::const_iterator;\n";
            fileContent += oprLink + "::reverse_iterator;\n";
            if (n == "vector") {
               fileContent += "#endif\n";
            }
         }
      }
      fileContent += "#endif\n";
      //end(1)
      //(2) prepare the file
      FILE* filePointer;
      filePointer = fopen(fileName, "w");
      if (filePointer == NULL) {
         //can't open a file
         return 1;
      }
      //end(2)
      //write data into the file
      fprintf(filePointer, "%s", fileContent.c_str());
      fclose(filePointer);
   }
   //(3) checking if we can compile a macro, if not then cleaning
   Int_t oldErrorIgnoreLevel = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kWarning; // no "Info: creating library..."
   Int_t ret = gSystem->CompileMacro(fileName, "k");
   gErrorIgnoreLevel = oldErrorIgnoreLevel;
   if (ret == 0) { //can't compile a macro
      return 2;
   }
   //end(3)
   return 0;
}

int TCint_GenerateDictionary(const std::string& className,
                             const std::vector<std::string> &headers,
                             const std::vector<std::string> &fwdDecls,
                             const std::vector<std::string> &unknown)
{
   //This function automatically creates the "LinkDef.h" file for templated
   //classes then executes CompileMacro on it.
   //The name of the file depends on the class name, and it's not generated again
   //if the file exist.
   std::vector<std::string> classes;
   classes.push_back(className);
   return TCint_GenerateDictionary(classes, headers, fwdDecls, unknown);
}

//______________________________________________________________________________
//
//
//

// It is a "fantom" method to synchronize user keyboard input
// and ROOT prompt line (for WIN32)
const char* fantomline = "TRint::EndOfLineAction();";

//______________________________________________________________________________
//
//
//

void* TCintWithCling::fgSetOfSpecials = 0;

//______________________________________________________________________________
//
//
//

ClassImp(TCintWithCling)

//______________________________________________________________________________
TCintWithCling::TCintWithCling(const char *name, const char *title)
: TInterpreter(name, title), fGlobalsListSerial(-1), fInterpreter(0), 
   fMetaProcessor(0), fNormalizedCtxt(0), fPrevLoadedDynLibInfo(0),
   fClingCallbacks(0)
{
   // Initialize the CINT+cling interpreter interface.

   fTemporaries = new std::vector<cling::StoredValueRef>();
   std::string interpInclude = ROOT::TMetaUtils::GetInterpreterExtraIncludePath(false);
   const char* interpArgs[]
      = {"cling4root", interpInclude.c_str(), "-Xclang", "-fmodules"};

   fInterpreter = new cling::Interpreter(sizeof(interpArgs) / sizeof(char*),
                                         interpArgs,
                                         ROOT::TMetaUtils::GetLLVMResourceDir(false).c_str());
   fInterpreter->installLazyFunctionCreator(autoloadCallback);

   // Add the current path to the include path
   TCintWithCling::AddIncludePath(".");

   // Add the root include directory and etc/ to list searched by default.
   // Use explicit TCintWithCling::AddIncludePath() to avoid vtable: we're in the c'tor!
   TCintWithCling::AddIncludePath(ROOT::TMetaUtils::GetROOTIncludeDir(false).c_str());

   // Don't check whether modules' files exist.
   fInterpreter->getCI()->getPreprocessorOpts().DisablePCHValidation = true;

   // Set the patch for searching for modules
#ifndef ROOTINCDIR
   TString dictDir = getenv("ROOTSYS");
   dictDir += "/lib";
#else // ROOTINCDIR
   TString dictDir = ROOTLIBDIR;
#endif // ROOTINCDIR
   clang::HeaderSearch& HS = fInterpreter->getCI()->getPreprocessor().getHeaderSearchInfo();
   HS.setModuleCachePath(dictDir.Data());

   fMetaProcessor = new cling::MetaProcessor(*fInterpreter);

   fInterpreter->declare("namespace std {} using namespace std;");

   // For the list to also include string, we have to include it now.
   fInterpreter->declare("#include \"Rtypes.h\"\n#include <string>");

   // During the loading of the first modules, RegisterModule which can calls Info
   // which needs the TClass for TCintWithCling, which in turns need the 'dictionary'
   // information to be loaded:
   fInterpreter->declare("#include \"TCintWithCling.h\"");
  
   // We are now ready (enough is loaded) to init the list of opaque typedefs.
   fNormalizedCtxt = new ROOT::TMetaUtils::TNormalizedCtxt(fInterpreter->getLookupHelper());

   TClassEdit::Init(*fInterpreter,*fNormalizedCtxt);

   // set the gModuleHeaderInfoBuffer pointer
   TCintWithCling__RegisterModule(0, 0, 0, 0, 0);

   TCintWithCling::InitializeDictionaries();
   
   // Initialize the CINT interpreter interface.
   fMore      = 0;
   fPrompt[0] = 0;
   fMapfile   = 0;
   fRootmapFiles = 0;
   fLockProcessLine = kTRUE;
   // Disable the autoloader until it is explicitly enabled.
   SetClassAutoloading(false);
   //G__RegisterScriptCompiler(&ScriptCompiler);
   G__set_ignoreinclude(&IgnoreInclude);
   // check whether the compiler is available:
   //char* path = gSystem->Which(gSystem->Getenv("PATH"), gSystem->BaseName(COMPILER));
   //if (path && path[0]) {
   //   G__InitGenerateDictionary(&TCint_GenerateDictionary);
   //}
   //delete[] path;
   ResetAll();
#ifndef R__WIN32
   optind = 1;  // make sure getopt() works in the main program
#endif // R__WIN32
   // Initialize for ROOT:
   TString include;
   // Add the root include directory to list searched by default
#ifndef ROOTINCDIR
   include = gSystem->Getenv("ROOTSYS");
   include.Append("/include");
#else // ROOTINCDIR
   include = ROOTINCDIR;
#endif // ROOTINCDIR
   TCintWithCling::AddIncludePath(include);

   // Attach cling callbacks
   fClingCallbacks = new TClingCallbacks(fInterpreter);
   fInterpreter->setCallbacks(fClingCallbacks);
}

//______________________________________________________________________________
TCintWithCling::~TCintWithCling()
{
   // Destroy the interpreter interface.
   delete fMapfile;
   delete fRootmapFiles;
   delete fMetaProcessor;
   delete fTemporaries;
   delete fNormalizedCtxt;
   delete fInterpreter;
   gCint = 0;
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("~TCintWithCling", "Interface not available yet.");
#ifdef R__COMPLETE_MEM_TERMINATION
   // remove all cling objects
#endif
#endif
#endif
   //--
}

//______________________________________________________________________________
void TCintWithCling::RegisterModule(const char* modulename,
                                    const char** headers,
                                    const char** includePaths,
                                    const char** macroDefines,
                                    const char** macroUndefines)
{
   // Inject the module named "modulename" into cling; load all headers.
   // headers is a 0-terminated array of header files to #include after
   // loading the module. The module is searched for in all $LD_LIBRARY_PATH
   // entries (or %PATH% on Windows).
   // This function gets called by the static initialization of dictionary
   // libraries.

   TString pcmFileName(ROOT::TMetaUtils::GetModuleFileName(modulename).c_str());

   for (const char** inclPath = includePaths; *inclPath; ++inclPath) {
      TCintWithCling::AddIncludePath(*inclPath);
   }
   for (const char** macroD = macroDefines; *macroD; ++macroD) {
      TString macroPP("#define ");
      macroPP += *macroD;
      // comes in as "A=B" from "-DA=B", need "#define A B":
      Ssiz_t posAssign = macroPP.Index('=');
      if (posAssign != kNPOS) {
         macroPP[posAssign] = ' ';
      }
      fInterpreter->parse(macroPP.Data());
   }
   for (const char** macroU = macroUndefines; *macroU; ++macroU) {
      TString macroPP("#undef ");
      macroPP += *macroU;
      // comes in as "A=B" from "-DA=B", need "#define A B":
      Ssiz_t posAssign = macroPP.Index('=');
      if (posAssign != kNPOS) {
         macroPP[posAssign] = ' ';
      }
      fInterpreter->parse(macroPP.Data());
   }

   // Assemble search path:   
#ifdef R__WIN32
   TString searchPath = "$(PATH);";
#else
   TString searchPath = "$(LD_LIBRARY_PATH):";
#endif
#ifndef ROOTLIBDIR
   TString rootsys = gSystem->Getenv("ROOTSYS");
# ifdef R__WIN32
   searchPath += rootsys + "/bin";
# else
   searchPath += rootsys + "/lib";
# endif
#else // ROOTLIBDIR
# ifdef R__WIN32
   searchPath += ROOTBINDIR;
# else
   searchPath += ROOTLIBDIR;
# endif
#endif // ROOTLIBDIR
   gSystem->ExpandPathName(searchPath);

   if (!gSystem->FindFile(searchPath, pcmFileName)) {
      Error("RegisterModule", "cannot find dictionary module %s in %s",
            ROOT::TMetaUtils::GetModuleFileName(modulename).c_str(), searchPath.Data());
   } else {
      if (gDebug > 5) Info("RegisterModule", "Loading PCM %s", pcmFileName.Data());
      clang::CompilerInstance* CI = fInterpreter->getCI();
      ROOT::TMetaUtils::declareModuleMap(CI, pcmFileName, headers);
   }

   bool oldValue = false;
   if (fClingCallbacks)
     oldValue = SetClassAutoloading(false);

   for (const char** hdr = headers; *hdr; ++hdr) {
      if (gDebug > 5) Info("RegisterModule", "   #including %s...", *hdr);
      fInterpreter->parse(TString::Format("#include \"%s\"", *hdr).Data());
   }

   if (fClingCallbacks)
     SetClassAutoloading(oldValue);
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLine(const char* line, EErrorCode* error/*=0*/)
{
   // Let cling process a command line.
   //
   // If the command is executed and the error is 0, then the return value
   // is the int value corresponding to the result of the executed command
   // (float and double return values will be truncated).
   //

   // Copy the passed line, it comes from a static buffer in TApplication
   // which can be reentered through the Cling evaluation routines,
   // which would overwrite the static buffer and we would forget what we
   // were doing.
   //
   TString sLine(line);
   if (strstr(line,fantomline)) {
      // End-Of-Line action
      // See the comment (copied from above):
      // It is a "fantom" method to synchronize user keyboard input
      // and ROOT prompt line (for WIN32)
      // and is implemented by 
      if (gApplication) {
         if (gApplication->IsCmdThread()) {
            if (gGlobalMutex && !gCINTMutex && fLockProcessLine) {
               gGlobalMutex->Lock();
               if (!gCINTMutex)
                  gCINTMutex = gGlobalMutex->Factory(kTRUE);
               gGlobalMutex->UnLock();
            }
            R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);
            gROOT->SetLineIsProcessing();

            UpdateAllCanvases();

            gROOT->SetLineHasBeenProcessed();
         }
      }
      return 0;
   }
      
   if (gGlobalMutex && !gCINTMutex && fLockProcessLine) {
      gGlobalMutex->Lock();
      if (!gCINTMutex)
         gCINTMutex = gGlobalMutex->Factory(kTRUE);
      gGlobalMutex->UnLock();
   }
   R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);
   gROOT->SetLineIsProcessing();

   // A non-zero returned value means the given line was
   // not a complete statement.
   int indent = 0;
   // This will hold the resulting value of the evaluation the given line.
   cling::StoredValueRef result;
   cling::Interpreter::CompilationResult compRes = cling::Interpreter::kSuccess;
   if (!strncmp(sLine.Data(), ".L", 2) || !strncmp(sLine.Data(), ".x", 2) ||
       !strncmp(sLine.Data(), ".X", 2)) {
      // If there was a trailing "+", then CINT compiled the code above,
      // and we will need to strip the "+" before passing the line to cling.
      TString mod_line(sLine);
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname = gSystem->SplitAclicMode(sLine.Data() + 3,
         aclicMode, arguments, io);
      if (aclicMode.Length()) {
         // Remove the leading '+'
         R__ASSERT(aclicMode[0]=='+' && "ACLiC mode must start with a +");
         aclicMode[0]='k';    // We always want to keep the .so around.
         if (aclicMode[1]=='+') {
            // We have a 2nd +
            aclicMode[1]='f'; // We want to force the recompilation.
         }
         gSystem->CompileMacro(fname,aclicMode);

         if (strncmp(sLine.Data(), ".L", 2) != 0) {
            // if execution was requested.
            
            if (arguments.Length()==0) {
               arguments = "()";
            }
            // We need to remove the extension.
            Ssiz_t ext = fname.Last('.');
            if (ext != kNPOS) {
               fname.Remove(ext);
            }
            const char *function = gSystem->BaseName(fname);
            mod_line = function + arguments + io;
            indent = fMetaProcessor->process(mod_line, &result, &compRes);
         }
      } else {
         // not ACLiC
         bool unnamedMacro = false;
         {
            std::string line;
            std::ifstream in(fname);
            static const char whitespace[] = " \t\r\n";
            while (in) {
               std::getline(in, line);
               std::string::size_type posNonWS = line.find_first_not_of(whitespace);
               if (posNonWS == std::string::npos) continue;
               if (line[posNonWS] == '/' && line[posNonWS + 1] == '/')
                  // Too bad, we only suppose C++ comments here.
                  continue;
               unnamedMacro = (line[posNonWS] == '{');
               break;
            }
         }
         if (unnamedMacro) {
            compRes = fMetaProcessor->readInputFromFile(fname.Data(), &result,
                                                        true /*ignoreOutmostBlock*/);
         } else {
            indent = fMetaProcessor->process(mod_line, &result, &compRes);
         }
      }
   } // .L / .X / .x
   else {
      if (0!=strncmp(sLine.Data(), ".autodict ",10)) {
         // explicitly ignore .autodict without having to support it
         // in cling.

         indent = fMetaProcessor->process(sLine, &result, &compRes);
      }
   }
   if (result.isValid() && result.needsManagedAllocation())
      fTemporaries->push_back(result);
   if (indent) {
      // incomplete expression, needs something like:
      /// fMetaProcessor->abortEvaluation();
      gROOT->SetLineHasBeenProcessed();
      return 0;
   }
   if (error) {
      switch (compRes) {
      case cling::Interpreter::kSuccess: *error = kNoError; break;
      case cling::Interpreter::kFailure: *error = kRecoverable; break;
      case cling::Interpreter::kMoreInputExpected: *error = kProcessing; break;
      }
   }
   if (compRes == cling::Interpreter::kSuccess
       && result.isValid()
       && !result.get().isVoid(fInterpreter->getCI()->getASTContext())) 
   {
      gROOT->SetLineHasBeenProcessed();
      return result.get().simplisticCastAs<long>();
   }
   gROOT->SetLineHasBeenProcessed();
   return 0;
}

//______________________________________________________________________________
void TCintWithCling::PrintIntro()
{
   // Print CINT introduction and help message.

   Printf("cling C/C++ Interpreter: type .? for help.");
}

//______________________________________________________________________________
void TCintWithCling::AddIncludePath(const char *path)
{
   // Add the given path to the list of directories in which the interpreter
   // looks for include files. Only one path item can be specified at a
   // time, i.e. "path1:path2" is not supported.

   R__LOCKGUARD(gCINTMutex);
   fInterpreter->AddIncludePath(path);
}

//______________________________________________________________________________
void TCintWithCling::InspectMembers(TMemberInspector& insp, void* obj,
                                    const TClass* cl)
{
   // Visit all members over members, recursing over base classes.
   
   if (!cl || cl->GetCollectionProxy()) {
      // We do not need to investigate the content of the STL
      // collection, they are opaque to us (and details are
      // uninteresting).
      return;
   }

   char* cobj = (char*) obj; // for ptr arithmetics

   static clang::PrintingPolicy
      printPol(fInterpreter->getCI()->getLangOpts());
   if (printPol.Indentation) {
      // not yet inialized
      printPol.Indentation = 0;
      printPol.SuppressInitializers = true;
   }
   
   const char* clname = cl->GetName();
   // Printf("Inspecting class %s\n", clname);

   const clang::ASTContext& astContext = fInterpreter->getCI()->getASTContext();
   const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
   const clang::Decl *scopeDecl = lh.findScope(clname);
   if (!scopeDecl) {
      Error("InspectMembers", "Cannot find Decl for class %s", clname);
      return;
   }
   const clang::CXXRecordDecl* recordDecl 
     = llvm::dyn_cast<const clang::CXXRecordDecl>(scopeDecl);
   if (!recordDecl) {
      Error("InspectMembers", "Cannot find Decl for class %s is not a CXXRecordDecl.", clname);
      return;
   }
   
   const clang::ASTRecordLayout& recLayout
      = astContext.getASTRecordLayout(recordDecl);

   // TVirtualCollectionProxy *proxy = cl->GetCollectionProxy();
   // if (proxy && ( proxy->GetProperties() & TVirtualCollectionProxy::kIsEmulated ) ) {
   //    Error("InspectMembers","The TClass for %s has an emulated proxy but we are looking at a compiled version of the collection!\n",
   //          cl->GetName());
   // }
   if (cl->Size() != recLayout.getSize().getQuantity()) {
      Error("InspectMembers","TClass and cling disagree on the size of the class %s, respectively %d %lld\n",
            cl->GetName(),cl->Size(),(Long64_t)recLayout.getSize().getQuantity());
   }

   unsigned iNField = 0;
   // iterate over fields
   // FieldDecls are non-static, else it would be a VarDecl.
   for (clang::RecordDecl::field_iterator iField = recordDecl->field_begin(),
        eField = recordDecl->field_end(); iField != eField;
        ++iField, ++iNField) {

      clang::QualType memberQT = cling::utils::Transform::GetPartiallyDesugaredType(astContext, iField->getType(), fNormalizedCtxt->GetTypeToSkip(), false /* fully qualify */);
      if (memberQT.isNull()) {
         std::string memberName;
         iField->getNameForDiagnostic(memberName, printPol, true /*fqi*/);
         Error("InspectMembers",
               "Cannot retrieve QualType for member %s while inspecting class %s",
               memberName.c_str(), clname);
         continue; // skip member
      }
      const clang::Type* memType = memberQT.getTypePtr();
      if (!memType) {
         std::string memberName;
         iField->getNameForDiagnostic(memberName, printPol, true /*fqi*/);
         Error("InspectMembers",
               "Cannot retrieve Type for member %s while inspecting class %s",
               memberName.c_str(), clname);
         continue; // skip member
      }
      
      const clang::Type* memNonPtrType = memType;
      Bool_t ispointer = false;
      if (memNonPtrType->isPointerType()) {
         ispointer = true;
         clang::QualType ptrQT
            = memNonPtrType->getAs<clang::PointerType>()->getPointeeType();
         ptrQT = cling::utils::Transform::GetPartiallyDesugaredType(astContext, ptrQT, fNormalizedCtxt->GetTypeToSkip(), false /* fully qualify */);
         if (ptrQT.isNull()) {
            std::string memberName;
            iField->getNameForDiagnostic(memberName, printPol, true /*fqi*/);
            Error("InspectMembers",
                  "Cannot retrieve pointee Type for member %s while inspecting class %s",
                  memberName.c_str(), clname);
            continue; // skip member
         }
         memNonPtrType = ptrQT.getTypePtr();
      }

      // assemble array size(s): "[12][4][]"
      llvm::SmallString<8> arraySize;
      const clang::ArrayType* arrType = memNonPtrType->getAsArrayTypeUnsafe();
      unsigned arrLevel = 0;
      bool haveErrorDueToArray = false;
      while (arrType) {
         ++arrLevel;
         arraySize += '[';
         const clang::ConstantArrayType* constArrType =
         clang::dyn_cast<clang::ConstantArrayType>(arrType);
         if (constArrType) {
            constArrType->getSize().toStringUnsigned(arraySize);
         }
         arraySize += ']';
         clang::QualType subArrQT = arrType->getElementType();
         if (subArrQT.isNull()) {
            std::string memberName;
            iField->getNameForDiagnostic(memberName, printPol, true /*fqi*/);
            Error("InspectMembers",
                  "Cannot retrieve QualType for array level %d (i.e. element type of %s) for member %s while inspecting class %s",
                  arrLevel, subArrQT.getAsString(printPol).c_str(),
                  memberName.c_str(), clname);
            haveErrorDueToArray = true;
            break;
         }
         arrType = subArrQT.getTypePtr()->getAsArrayTypeUnsafe();
      }
      if (haveErrorDueToArray) {
         continue; // skip member
      }

      // construct member name
      std::string fieldName;
      if (memType->isPointerType()) {
         fieldName = "*";
      }
      fieldName += iField->getName();
      fieldName += arraySize;
      
      // get member offset
      // NOTE currently we do not support bitfield and do not support
      // member that are not aligned on 'bit' boundaries.
      clang::CharUnits offset(astContext.toCharUnitsFromBits(recLayout.getFieldOffset(iNField)));
      ptrdiff_t fieldOffset = offset.getQuantity();

      // R__insp.Inspect(R__cl, R__insp.GetParent(), "fBits[2]", fBits);
      // R__insp.Inspect(R__cl, R__insp.GetParent(), "fName", &fName);
      // R__insp.InspectMember(fName, "fName.");
      // R__insp.Inspect(R__cl, R__insp.GetParent(), "*fClass", &fClass);
      insp.Inspect(const_cast<TClass*>(cl), insp.GetParent(), fieldName.c_str(), cobj + fieldOffset);

      if (!ispointer) {
         const clang::CXXRecordDecl* fieldRecDecl = memNonPtrType->getAsCXXRecordDecl();
         if (fieldRecDecl) {
            // nested objects get an extra call to InspectMember
            // R__insp.InspectMember("FileStat_t", (void*)&fFileStat, "fFileStat.", false);
            std::string sFieldRecName;
            ROOT::TMetaUtils::GetNormalizedName(sFieldRecName, clang::QualType(memNonPtrType,0), *fInterpreter, *fNormalizedCtxt);
            llvm::StringRef comment = ROOT::TMetaUtils::GetComment(* (*iField), 0);
            // NOTE, we have to change this to support selection XML!
            bool transient = !comment.empty() && comment[0] == '!';
            
            insp.InspectMember(sFieldRecName.c_str(), cobj + fieldOffset,
                               (fieldName + '.').c_str(), transient);
         }
      }
   } // loop over fields
   
   // inspect bases
   // TNamed::ShowMembers(R__insp);
   unsigned iNBase = 0;
   for (clang::CXXRecordDecl::base_class_const_iterator iBase
        = recordDecl->bases_begin(), eBase = recordDecl->bases_end();
        iBase != eBase; ++iBase, ++iNBase) {
      clang::QualType baseQT = iBase->getType();
      if (baseQT.isNull()) {
         Error("InspectMembers",
               "Cannot find QualType for base number %d while inspecting class %s",
               iNBase, clname);
         continue;
      }
      const clang::CXXRecordDecl* baseDecl
         = baseQT->getAsCXXRecordDecl();
      if (!baseDecl) {
         Error("InspectMembers",
               "Cannot find CXXRecordDecl for base number %d while inspecting class %s",
               iNBase, clname);
         continue;
      }
      std::string sBaseName;
      baseDecl->getNameForDiagnostic(sBaseName, printPol, true /*fqi*/);
      TClass* baseCl = TClass::GetClass(sBaseName.c_str());
      if (!baseCl) {
         Error("InspectMembers",
               "Cannot find TClass for base %s while inspecting class %s",
               sBaseName.c_str(), clname);
         continue;
      }
      int64_t baseOffset = recLayout.getBaseClassOffset(baseDecl).getQuantity();      
      if (baseCl->IsLoaded()) {
         // For loaded class, CallShowMember will (especially for TObject)
         // call the virtual ShowMember rather than the class specific version
         // resulting in an infinite recursion.
         InspectMembers(insp, cobj + baseOffset, baseCl);
      } else {
         baseCl->CallShowMembers(cobj + baseOffset,
                                 insp, 0);
      }
   } // loop over bases
}

//______________________________________________________________________________
void TCintWithCling::ClearFileBusy()
{
   // Reset the interpreter internal state in case a previous action was not correctly
   // terminated.

   // No-op there is not equivalent state (to be cleared) in Cling.
}

//______________________________________________________________________________
void TCintWithCling::ClearStack()
{
   // Delete existing temporary values.

   // No-op for cling due to StoredValueRef.
}

//______________________________________________________________________________
Int_t TCintWithCling::InitializeDictionaries()
{
   // Initialize all registered dictionaries.

   R__LOCKGUARD(gCINTMutex);

   SmallVectorImpl<ModuleHeaderInfo_t>::iterator
      li = gModuleHeaderInfoBuffer->begin(),
      le = gModuleHeaderInfoBuffer->end();
   for (; li != le; ++li) {
      // process buffered module registrations
      ((TCintWithCling*)gCint)->RegisterModule(li->fModuleName,
                                               li->fHeaders,
                                               li->fIncludePaths,
                                               li->fMacroDefines,
                                               li->fMacroUndefines);
   }
   gModuleHeaderInfoBuffer->clear();

   return 0;
}

//______________________________________________________________________________
void TCintWithCling::EnableAutoLoading()
{
   // Enable the automatic loading of shared libraries when a class
   // is used that is stored in a not yet loaded library. Uses the
   // information stored in the class/library map (typically
   // $ROOTSYS/etc/system.rootmap).
   LoadLibraryMap();
   SetClassAutoloading(true);
}

//______________________________________________________________________________
void TCintWithCling::EndOfLineAction()
{
   // It calls a "fantom" method to synchronize user keyboard input
   // and ROOT prompt line.
   ProcessLineSynch(fantomline);
}

//______________________________________________________________________________
Bool_t TCintWithCling::IsLoaded(const char* filename) const
{
   // Return true if the file has already been loaded by cint.
   // We will try in this order:
   //   actual filename
   //   filename as a path relative to
   //            the include path
   //            the shared library path
   R__LOCKGUARD(gCINTMutex);

   typedef cling::Interpreter::LoadedFileInfo FileInfo_t;
   typedef llvm::SmallVectorImpl<FileInfo_t*> AllFileInfos_t;

   llvm::StringMap<const FileInfo_t*> fileMap;

   // Fill fileMap; return early on exact match.
   const AllFileInfos_t& allFiles = fInterpreter->getLoadedFiles();
   for (AllFileInfos_t::const_iterator iF = allFiles.begin(), iE = allFiles.end();
        iF != iE; ++iF) {
      if ((*iF)->getName() == filename) return kTRUE; // exact match
      fileMap[(*iF)->getName()] = *iF;
   }

   if (fileMap.empty()) return kFALSE;

   // Check MacroPath.
   TString sFilename(filename);
   if (gSystem->FindFile(TROOT::GetMacroPath(), sFilename, kReadPermission)
       && fileMap.count(sFilename.Data())) {
      return kTRUE;
   }

   // Check IncludePath.
   TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
   incPath.Append(":").Prepend(" "); // to match " -I" (note leading ' ')
   incPath.ReplaceAll(" -I", ":");      // of form :dir1 :dir2:dir3
   while (incPath.Index(" :") != -1) {
      incPath.ReplaceAll(" :", ":");
   }
   incPath.Prepend(".:");
   sFilename = filename;
   if (gSystem->FindFile(incPath, sFilename, kReadPermission)
       && fileMap.count(sFilename.Data())) {
      return kTRUE;
   }

   // Check shared library.
   sFilename = filename;
   if (gSystem->FindDynamicLibrary(sFilename, kTRUE)
       && fileMap.count(sFilename.Data())) {
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfLoadedSharedLibraries()
{
#ifdef R_WIN32
   // need to call RegisterLoadedSharedLibrary() here
   // by calling Win32's EnumerateLoadedModules().
   Error("TCintWithCling::UpdateListOfLoadedSharedLibraries",
         "Platform not supported!");   
#elif defined(R__MACOSX)
   // fPrevLoadedDynLibInfo stores the *next* image index to look at
   uint32_t imageIndex = (uint32_t) (size_t) fPrevLoadedDynLibInfo;
   const char* imageName = 0;
   while ((imageName = _dyld_get_image_name(imageIndex))) {
      // Skip binary
      if (imageIndex > 0)
         RegisterLoadedSharedLibrary(imageName);
      ++imageIndex;
   }
   fPrevLoadedDynLibInfo = (void*)(size_t)imageIndex;
#elif defined(R__LINUX)
   struct PointerNo4_t {
      void* fSkip[3];
      void* fPtr;
   };
   struct LinkMap_t {
      void* fAddr;
      const char* fName;
      void* fLd;
      LinkMap_t* fNext;
      LinkMap_t* fPrev;
   };
   if (!fPrevLoadedDynLibInfo || fPrevLoadedDynLibInfo == (void*)(size_t)-1) {
      PointerNo4_t* procLinkMap = (PointerNo4_t*)dlopen(0,  RTLD_LAZY | RTLD_GLOBAL);
      // 4th pointer of 4th pointer is the linkmap.
      // See http://syprog.blogspot.fr/2011/12/listing-loaded-shared-objects-in-linux.html
      LinkMap_t* linkMap = (LinkMap_t*) ((PointerNo4_t*)procLinkMap->fPtr)->fPtr;
      RegisterLoadedSharedLibrary(linkMap->fName);
      fPrevLoadedDynLibInfo = linkMap;
   }
   
   LinkMap_t* iDyLib = (LinkMap_t*)fPrevLoadedDynLibInfo;
   while (iDyLib->fNext) {
      iDyLib = iDyLib->fNext;
      RegisterLoadedSharedLibrary(iDyLib->fName);
   }
   fPrevLoadedDynLibInfo = iDyLib;
#else
   Error("TCintWithCling::UpdateListOfLoadedSharedLibraries",
         "Platform not supported!");
#endif
}

//______________________________________________________________________________
void TCintWithCling::RegisterLoadedSharedLibrary(const char* filename)
{
   // Register a new shared library name with the interpreter; add it to
   // fSharedLibs.

   // Ignore NULL filenames, aka "the process".
   if (!filename) return;

   // Tell the interpreter that this library is available; all libraries can be
   // used to resolve symbols.
   fInterpreter->loadLibrary(filename, true /*permanent*/);

#if defined(R__MACOSX)
   // Check that this is not a system library
   if (!strncmp(filename, "/usr/lib/system/", 16)
       || !strncmp(filename, "/usr/lib/libc++", 15)
       || !strncmp(filename, "/System/Library/Frameworks/", 27)
       || !strncmp(filename, "/System/Library/PrivateFrameworks/", 34))
      return;
#endif
   // Update string of available libraries.
   if (!fSharedLibs.IsNull()) {
      fSharedLibs.Append(" ");
   }
   fSharedLibs.Append(filename);
}

//______________________________________________________________________________
Int_t TCintWithCling::Load(const char* filename, Bool_t system)
{
   // Load a library file in CINT's memory.
   // if 'system' is true, the library is never unloaded.
   // Return 0 on success, -1 on failure.

   // Used to return 0 on success, 1 on duplicate, -1 on failure, -2 on "fatal".
   R__LOCKGUARD2(gCINTMutex);
   cling::Interpreter::LoadLibResult res
      = fInterpreter->loadLibrary(filename, system);
   if (res == cling::Interpreter::kLoadLibSuccess) {
      UpdateListOfLoadedSharedLibraries();
   }
   switch (res) {
   case cling::Interpreter::kLoadLibSuccess: return 0;
   case cling::Interpreter::kLoadLibExists:  return 1;
   default: break;
   };
   return -1;
}

//______________________________________________________________________________
void TCintWithCling::LoadMacro(const char* filename, EErrorCode* error)
{
   // Load a macro file in CINT's memory.
   ProcessLine(Form(".L %s", filename), error);
   UpdateListOfTypes();
   UpdateListOfGlobals();
   UpdateListOfGlobalFunctions();
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLineAsynch(const char* line, EErrorCode* error)
{
   // Let CINT process a command line asynch.
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLineSynch(const char* line, EErrorCode* error)
{
   // Let CINT process a command line synchronously, i.e we are waiting
   // it will be finished.
   R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);
   if (gApplication) {
      if (gApplication->IsCmdThread()) {
         return ProcessLine(line, error);
      }
      return 0;
   }
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCintWithCling::Calc(const char* line, EErrorCode* error)
{
   // Directly execute an executable statement (e.g. "func()", "3+5", etc.
   // however not declarations, like "Int_t x;").
   Long_t result;
#ifdef R__WIN32
   // Test on ApplicationImp not being 0 is needed because only at end of
   // TApplication ctor the IsLineProcessing flag is set to 0, so before
   // we can not use it.
   if (gApplication && gApplication->GetApplicationImp()) {
      while (gROOT->IsLineProcessing() && !gApplication) {
         Warning("Calc", "waiting for CINT thread to free");
         gSystem->Sleep(500);
      }
      gROOT->SetLineIsProcessing();
   }
#endif
   R__LOCKGUARD2(gCINTMutex);
   result = (Long_t) G__int_cast(G__calc(const_cast<char*>(line)));
   if (error) {
      *error = (EErrorCode)G__lasterror();
   }
#ifdef R__WIN32
   if (gApplication && gApplication->GetApplicationImp()) {
      gROOT->SetLineHasBeenProcessed();
   }
#endif
   return result;
}

//______________________________________________________________________________
void TCintWithCling::SetGetline(const char * (*getlineFunc)(const char* prompt),
                                void (*histaddFunc)(const char* line))
{
   // Set a getline function to call when input is needed.

   // If cling offers a replacement for G__pause(), it would need to 
   // also offer a way to customize at least the history recording.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetGetline","Cling should support the equivalent of SetGetlineFunc(getlineFunc, histaddFunc)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::RecursiveRemove(TObject* obj)
{
   // Delete object from CINT symbol table so it can not be used anymore.
   // CINT objects are always on the heap.
   R__LOCKGUARD(gCINTMutex);
   if (obj->IsOnHeap() && fgSetOfSpecials && !((std::set<TObject*>*)fgSetOfSpecials)->empty()) {
      std::set<TObject*>::iterator iSpecial = ((std::set<TObject*>*)fgSetOfSpecials)->find(obj);
      if (iSpecial != ((std::set<TObject*>*)fgSetOfSpecials)->end()) {
         DeleteGlobal(obj);
         ((std::set<TObject*>*)fgSetOfSpecials)->erase(iSpecial);
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::Reset()
{
   // Reset the Cling state to the state saved by the last call to
   // TCintWithCling::SaveContext().

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("Reset","Cling should support the equivalent of scratch_upto(&fDictPos)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::ResetAll()
{
   // Reset the Cling state to its initial state.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("ResetAll","Cling should support the equivalent of complete reset (unload everything but the startup decls.");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::ResetGlobals()
{
   // Reset in Cling the list of global variables to the state saved by the last
   // call to TCintWithCling::SaveGlobalsContext().

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("ResetGlobals","Cling should support the equivalent of scratch_globals_upto(&fDictPosGlobals)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::ResetGlobalVar(void* obj)
{
   // Reset the Cling 'user' global objects/variables state to the state saved by the last
   // call to TCintWithCling::SaveGlobalsContext().

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("ResetGlobalVar","Cling should support the equivalent of resetglobalvar(obj)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::RewindDictionary()
{
   // Rewind Cling dictionary to the point where it was before executing
   // the current macro. This function is typically called after SEGV or
   // ctlr-C after doing a longjmp back to the prompt.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("RewindDictionary","Cling should provide a way to revert transaction similar to rewinddictionary()");
#endif
#endif
}

//______________________________________________________________________________
Int_t TCintWithCling::DeleteGlobal(void* obj)
{
   // Delete obj from Cling symbol table so it cannot be accessed anymore.
   // Returns 1 in case of success and 0 in case object was not in table.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("DeleteGlobal","Cling should provide the equivalent of deleteglobal(obj), see also DeleteVariable.");
#endif
#endif
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::DeleteVariable(const char* name)
{
   // Undeclare obj called name.
   // Returns 1 in case of success, 0 for failure.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("DeleteVariable","should do more that just reseting the value to zero");
#endif
#endif

   R__LOCKGUARD(gCINTMutex);
   llvm::StringRef srName(name);
   const char* unscopedName = name;
   llvm::StringRef::size_type posScope = srName.rfind("::");
   const clang::DeclContext* declCtx = 0;
   if (posScope != llvm::StringRef::npos) {
      const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
      const clang::Decl* scopeDecl = lh.findScope(srName.substr(0, posScope));
      if (!scopeDecl) {
         Error("DeleteVariable", "Cannot find enclosing scope for variable %s",
               name);
         return 0;
      }
      declCtx = llvm::dyn_cast<clang::DeclContext>(scopeDecl);
      if (!declCtx) {
         Error("DeleteVariable",
               "Enclosing scope for variable %s is not a declaration context",
               name);
         return 0;
      }
      unscopedName += posScope + 2;
   }
   clang::NamedDecl* nVarDecl
      = cling::utils::Lookup::Named(&fInterpreter->getSema(), unscopedName, declCtx);
   if (!nVarDecl) {
      Error("DeleteVariable", "Unknown variable %s", name);
      return 0;
   }
   clang::VarDecl* varDecl = llvm::dyn_cast<clang::VarDecl>(nVarDecl);
   if (!varDecl) {
      Error("DeleteVariable", "Entity %s is not a variable", name);
      return 0;
   }

   clang::QualType qType = varDecl->getType();
   const clang::Type* type = qType->getUnqualifiedDesugaredType();
   if (type->isPointerType() || type->isReferenceType()) {
      int** ppInt = (int**)fInterpreter->getAddressOfGlobal(varDecl);
      // set pointer / reference to invalid.
      if (ppInt) *ppInt = 0;
   }
   return 1;
}

//______________________________________________________________________________
void TCintWithCling::SaveContext()
{
   // Save the current Cling state.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("SaveContext","Cling should provide a way to record a state watermark similar to store_dictposition(&fDictPos)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::SaveGlobalsContext()
{
   // Save the current Cling state of global objects.

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   R__LOCKGUARD(gCINTMutex);
   Warning("SaveGlobalsContext","Cling should provide a way to record a watermark for the list of global variable similar to store_dictposition(&fDictPosGlobals)");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfGlobals()
{
   // No op: see TClingCallbacks (used to update the list of globals)
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfGlobalFunctions()
{
   // No op: see TClingCallbacks (used to update the list of global functions)
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfTypes()
{
   // No op: see TClingCallbacks (used to update the list of types)
}

//______________________________________________________________________________
void TCintWithCling::SetClassInfo(TClass* cl, Bool_t reload)
{
   // Set pointer to the TClingClassInfo in TClass.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fClassInfo && !reload) {
      return;
   }
   delete (TClingClassInfo*) cl->fClassInfo;
   cl->fClassInfo = 0;
   std::string name(cl->GetName());
   TClingClassInfo* info = new TClingClassInfo(fInterpreter, name.c_str());
   if (!info->IsValid()) {
      bool cint_class_exists = CheckClassInfo(name.c_str());
      if (!cint_class_exists) {
         // Try resolving all the typedefs (even Float_t and Long64_t).
         name = TClassEdit::ResolveTypedef(name.c_str(), kTRUE);
         if (name == cl->GetName()) {
            // No typedefs found, all done.
            return;
         }
         // Try the new name.
         cint_class_exists = CheckClassInfo(name.c_str());
         if (!cint_class_exists) {
            // Nothing found, nothing to do.
            return;
         }
      }
      info = new TClingClassInfo(fInterpreter, name.c_str());
      if (!info->IsValid()) {
         // Failed, done.
         return;
      }
   }
   cl->fClassInfo = info; // Note: We are transfering ownership here.
   // In case a class contains an external enum, the enum will be seen as a
   // class. We must detect this special case and make the class a Zombie.
   // Here we assume that a class has at least one method.
   // We can NOT call TClass::Property from here, because this method
   // assumes that the TClass is well formed to do a lot of information
   // caching. The method SetClassInfo (i.e. here) is usually called during
   // the building phase of the TClass, hence it is NOT well formed yet.
   Bool_t zombieCandidate = kFALSE;
   if (
      info->IsValid() &&
      !(info->Property() & (kIsClass | kIsStruct | kIsNamespace))
   ) {
      zombieCandidate = kTRUE;
   }
   if (!info->IsLoaded()) {
      if (info->Property() & (kIsNamespace)) {
         // Namespaces can have info but no corresponding CINT dictionary
         // because they are auto-created if one of their contained
         // classes has a dictionary.
         zombieCandidate = kTRUE;
      }
      // this happens when no CINT dictionary is available
      delete info;
      cl->fClassInfo = 0;
   }
   if (zombieCandidate && !TClassEdit::IsSTLCont(cl->GetName())) {
      cl->MakeZombie();
   }
}

//______________________________________________________________________________
Bool_t TCintWithCling::CheckClassInfo(const char* name, Bool_t autoload /*= kTRUE*/)
{
   // Checks if a class with the specified name is defined in Cling.
   // Returns kFALSE is class is not defined.
   // In the case where the class is not loaded and belongs to a namespace
   // or is nested, looking for the full class name is outputing a lots of
   // (expected) error messages.  Currently the only way to avoid this is to
   // specifically check that each level of nesting is already loaded.
   // In case of templates the idea is that everything between the outer
   // '<' and '>' has to be skipped, e.g.: aap<pipo<noot>::klaas>::a_class

   R__LOCKGUARD(gCINTMutex);
   static const char *anonEnum = "anonymous enum ";
   static int cmplen = strlen(anonEnum);

   if (0 == strncmp(name,anonEnum,cmplen)) {
      return kFALSE;
   }

   Int_t nch = strlen(name) * 2;
   char* classname = new char[nch];
   strlcpy(classname, name, nch);
   char* current = classname;
   while (*current) {
      while (*current && *current != ':' && *current != '<') {
         current++;
      }
      if (!*current) {
         break;
      }
      if (*current == '<') {
         int level = 1;
         current++;
         while (*current && level > 0) {
            if (*current == '<') {
               level++;
            }
            if (*current == '>') {
               level--;
            }
            current++;
         }
         continue;
      }
      // *current == ':', must be a "::"
      if (*(current + 1) != ':') {
         Error("CheckClassInfo", "unexpected token : in %s", classname);
         delete[] classname;
         return kFALSE;
      }
      *current = '\0';
      if (0 == strncmp(name,anonEnum,cmplen)) {
         delete[] classname;
         return kFALSE;
      }
      TClingClassInfo info(fInterpreter, classname);
      if (!info.IsValid()) {
         delete[] classname;
         return kFALSE;
      }
      *current = ':';
      current += 2;
   }
   strlcpy(classname, name, nch);

   int storeAutoload = SetClassAutoloading(false);

   // Note that when using CINT we explicitly requested
   // for template to *not* be instantiated as a
   // consequence to the equivalent call.  
   // We need to review whether we want to re-add this
   // distinction ...
   TClingClassInfo tci(fInterpreter, classname);
   if (!tci.IsValid()) {
      delete[] classname;
      SetClassAutoloading(storeAutoload);
      return kFALSE;
   }
   if (tci.Property() & (G__BIT_ISENUM | G__BIT_ISCLASS | G__BIT_ISSTRUCT | G__BIT_ISUNION | G__BIT_ISNAMESPACE)) {
      // We are now sure that the entry is not in fact an autoload entry.
      delete[] classname;
      SetClassAutoloading(storeAutoload);
      return kTRUE;
   }

   // Setting up iterator part of TClingTypedefInfo is too slow.
   // Copy the lookup code instead:
   /*
   TClingTypedefInfo t(fInterpreter, name);
   if (t.IsValid() && !(t.Property() & G__BIT_ISFUNDAMENTAL)) {
      delete[] classname;
      SetClassAutoloading(storeAutoload);
      return kTRUE;
   }
   */

   const cling::LookupHelper& lh = fInterpreter->getLookupHelper();
   const clang::Decl *decl = lh.findScope(name);
   if (!decl) {
      if (gDebug > 0) {
         Info("TClingClassInfo(name)", "cling class not found name: %s\n",
              name);
      }
      std::string buf = TClassEdit::InsertStd(name);
      decl = lh.findScope(buf);
      if (!decl) {
         if (gDebug > 0) {
            Info("TClingClassInfo(name)", "cling class not found name: %s\n",
                 buf.c_str());
         }
      }
      else {
         if (gDebug > 0) {
            Info("TClingClassInfo(name)", "found cling class name: %s  "
                 "decl: 0x%lx\n", buf.c_str(), (long) decl);
         }
      }
   }

   delete[] classname;
   SetClassAutoloading(storeAutoload);
   return (decl);
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfBaseClasses(TClass* cl)
{
   // Create list of pointers to base class(es) for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fBase) {
      return;
   }
   cl->fBase = new TList;
   TClingClassInfo tci(fInterpreter, cl->GetName());
   TClingBaseClassInfo t(fInterpreter, &tci);
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         TClingBaseClassInfo* a = new TClingBaseClassInfo(t);
         cl->fBase->Add(new TBaseClass(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfDataMembers(TClass* cl)
{
   // Create list of pointers to data members for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fData) {
      return;
   }
   cl->fData = new TList;
   TClingDataMemberInfo t(fInterpreter, (TClingClassInfo*)cl->GetClassInfo());
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         TClingDataMemberInfo* a = new TClingDataMemberInfo(t);
         cl->fData->Add(new TDataMember(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfMethods(TClass* cl)
{
   // Create list of pointers to methods for TClass cl.
   R__LOCKGUARD2(gCINTMutex);
   if (cl->fMethod) {
      return;
   }
   cl->fMethod = new THashList;
   TClingMethodInfo t(fInterpreter, (TClingClassInfo*)cl->GetClassInfo());
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         TClingMethodInfo* a = new TClingMethodInfo(t);
         cl->fMethod->Add(new TMethod(a, cl));
      }
   }
}

//______________________________________________________________________________
void TCintWithCling::UpdateListOfMethods(TClass* cl)
{
   // Update the list of pointers to method for TClass cl, if necessary
   delete cl->fMethod;
   cl->fMethod = 0;
   CreateListOfMethods(cl);
}

//______________________________________________________________________________
void TCintWithCling::CreateListOfMethodArgs(TFunction* m)
{
   // Create list of pointers to method arguments for TMethod m.
   R__LOCKGUARD2(gCINTMutex);
   if (m->fMethodArgs) {
      return;
   }
   m->fMethodArgs = new TList;
   TClingMethodArgInfo t(fInterpreter, (TClingMethodInfo*)m->fInfo);
   while (t.Next()) {
      if (t.IsValid()) {
         TClingMethodArgInfo* a = new TClingMethodArgInfo(t);
         m->fMethodArgs->Add(new TMethodArg(a, m));
      }
   }
}

//______________________________________________________________________________
TClass *TCintWithCling::GenerateTClass(const char *classname, Bool_t silent /* = kFALSE */)
{
   // Generate a TClass for the given class.

   TClingClassInfo tci(fInterpreter, classname);
   if (1 || !tci.IsValid()) {  
      int version = 1;
      if (TClassEdit::IsSTLCont(classname)) {
         version = TClass::GetClass("TVirtualStreamerInfo")->GetClassVersion();
      }
      TClass *cl = new TClass(classname, version, 0, 0, -1, -1, silent);
      cl->SetBit(TClass::kIsEmulation);

      return cl;
   } else {
      return GenerateTClass(&tci,silent);
   }
}

//______________________________________________________________________________
static void GenerateTClass_GatherInnerIncludes(cling::Interpreter *interp, TString &includes,TClingClassInfo *info)
{
   includes += info->FileName();

   const clang::ClassTemplateSpecializationDecl *templateCl 
      = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(info->GetDecl());
   if (templateCl) {
      for(unsigned int i=0; i <  templateCl->getTemplateArgs().size(); ++i) {
          const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(i) );
          if (arg.getKind() == clang::TemplateArgument::Type) {
             const clang::Type *uType = ROOT::TMetaUtils::GetUnderlyingType( arg.getAsType() );

            if (!uType->isFundamentalType() && !uType->isEnumeralType()) {
               // We really need a header file.
               const clang::CXXRecordDecl *argdecl = uType->getAsCXXRecordDecl();            
               if (argdecl) {
                  includes += ";";
                  TClingClassInfo subinfo(interp,*(argdecl->getASTContext().getRecordType(argdecl).getTypePtr()));
                  GenerateTClass_GatherInnerIncludes(interp, includes, &subinfo);
               } else {
                  std::string Result;
                  llvm::raw_string_ostream OS(Result);
                  arg.print(argdecl->getASTContext().getPrintingPolicy(),OS);
                  Warning("TCintWithCling::GenerateTClass","Missing header file for %s",OS.str().c_str());
               }
            }
          }
      }
   }
}

//______________________________________________________________________________
TClass *TCintWithCling::GenerateTClass(ClassInfo_t *classinfo, Bool_t silent /* = kFALSE */)
{
   // Generate a TClass for the given class.

   TClingClassInfo *info = (TClingClassInfo*)classinfo;
   if (!info && !info->IsValid()) {
      Fatal("GenerateTClass","Requires a valid ClassInfo object");
      return 0;
   }
   // We are in the case where we have AST nodes for this class.
   TClass *cl = 0;
   TString classname = info->FullName(*fNormalizedCtxt); // Could we use Name()?
   if (TClassEdit::IsSTLCont(classname)) {
#if 0
      Info("GenerateTClass","Will (try to) generate the compiled TClass for %s.",classname.Data());
      // We need to build up the list of required headers, by 
      // looking at each template arguments.
      TString includes;
      GenerateTClass_GatherInnerIncludes(fInterpreter,includes,info);

      if (0 == GenerateDictionary(classname,includes)) {
         // 0 means success.
         cl = gROOT->LoadClass(classname, silent);
         if (cl == 0) {
            Error("GenerateTClass","Even though the dictionary generation for %s seemed successfull we can't find the TClass bootstrap!",classname.Data());
         }
      }
#endif
      if (cl == 0) {
         int version = TClass::GetClass("TVirtualStreamerInfo")->GetClassVersion();
         cl = new TClass(classname, version, 0, 0, -1, -1, silent);
         cl->SetBit(TClass::kIsEmulation);
      }
   } else {
      // For regular class, just create a TClass on the fly ...
      // Not quite useful yet, but that what CINT used to do anyway.
      cl = new TClass(classname, 1, 0, 0, -1, -1, silent);
   }
   return cl;
}

//______________________________________________________________________________
Int_t TCintWithCling::GenerateDictionary(const char* classes, const char* includes /* = 0 */, const char* /* options  = 0 */)
{
   // Generate the dictionary for the C++ classes listed in the first
   // argmument (in a semi-colon separated list).
   // 'includes' contains a semi-colon separated list of file to
   // #include in the dictionary.
   // For example:
   //    gInterpreter->GenerateDictionary("vector<vector<float> >;list<vector<float> >","list;vector");
   // or
   //    gInterpreter->GenerateDictionary("myclass","myclass.h;myhelper.h");
   if (classes == 0 || classes[0] == 0) {
      return 0;
   }
   // Split the input list
   std::vector<std::string> listClasses;
   for (
      const char* current = classes, *prev = classes;
      *current != 0;
      ++current
   ) {
      if (*current == ';') {
         listClasses.push_back(std::string(prev, current - prev));
         prev = current + 1;
      }
      else if (*(current + 1) == 0) {
         listClasses.push_back(std::string(prev, current + 1 - prev));
         prev = current + 1;
      }
   }
   std::vector<std::string> listIncludes;
   for (
      const char* current = includes, *prev = includes;
      *current != 0;
      ++current
   ) {
      if (*current == ';') {
         listIncludes.push_back(std::string(prev, current - prev));
         prev = current + 1;
      }
      else if (*(current + 1) == 0) {
         listIncludes.push_back(std::string(prev, current + 1 - prev));
         prev = current + 1;
      }
   }
   // Generate the temporary dictionary file
   return TCint_GenerateDictionary(listClasses, listIncludes,
      std::vector<std::string>(), std::vector<std::string>());
}


//______________________________________________________________________________
TString TCintWithCling::GetMangledName(TClass* cl, const char* method,
                              const char* params)
{
   // Return the CINT mangled name for a method of a class with parameters
   // params (params is a string of actual arguments, not formal ones). If the
   // class is 0 the global function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   TClingCallFunc func(fInterpreter);
   if (cl) {
      Long_t offset;
      func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params,
         &offset);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   TClingMethodInfo* mi = (TClingMethodInfo*) func.FactoryMethod();
   const char* mangled_name = mi->GetMangledName();
   delete mi;
   mi = 0;
   return mangled_name;
}

//______________________________________________________________________________
TString TCintWithCling::GetMangledNameWithPrototype(TClass* cl, const char* method,
      const char* proto)
{
   // Return the CINT mangled name for a method of a class with a certain
   // prototype, i.e. "char*,int,float". If the class is 0 the global function
   // list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   Long_t offset;
   if (cl) {
      return ((TClingClassInfo*)cl->GetClassInfo())->
             GetMethod(method, proto, &offset).GetMangledName();
   }
   TClingClassInfo gcl(fInterpreter);
   return gcl.GetMethod(method, proto, &offset).GetMangledName();
}

//______________________________________________________________________________
void* TCintWithCling::GetInterfaceMethod(TClass* cl, const char* method,
                                         const char* params)
{
   // Return pointer to CINT interface function for a method of a class with
   // parameters params (params is a string of actual arguments, not formal
   // ones). If the class is 0 the global function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   TClingCallFunc func(fInterpreter);
   if (cl) {
      Long_t offset;
      func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params,
         &offset);
   }
   else {
      TClingClassInfo gcl(fInterpreter);
      Long_t offset;
      func.SetFunc(&gcl, method, params, &offset);
   }
   return (void*) func.InterfaceMethod();
}

//______________________________________________________________________________
void* TCintWithCling::GetInterfaceMethodWithPrototype(TClass* cl, const char* method,
      const char* proto)
{
   // Return pointer to CINT interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.
   R__LOCKGUARD2(gCINTMutex);
   void* f;
   if (cl) {
      Long_t offset;
      f = ((TClingClassInfo*)cl->GetClassInfo())->
          GetMethod(method, proto, &offset).InterfaceMethod();
   }
   else {
      Long_t offset;
      TClingClassInfo gcl(fInterpreter);
      f = gcl.GetMethod(method, proto, &offset).InterfaceMethod();
   }
   return f;
}

//______________________________________________________________________________
const char* TCintWithCling::GetInterpreterTypeName(const char* name, Bool_t full)
{
   // The 'name' is known to the interpreter, this function returns
   // the internal version of this name (usually just resolving typedefs)
   // This is used in particular to synchronize between the name used
   // by rootcint and by the run-time enviroment (TClass)
   // Return 0 if the name is not known.

   R__LOCKGUARD(gCINTMutex);

   // This first step is likely redundant if
   // the next step never issue any warnings.
   if (!CheckClassInfo(name)) {
      return 0;
   }
   TClingClassInfo cl(fInterpreter, name);
   if (!cl.IsValid()) {
      return 0;
   }
   if (full) {
      return cl.FullName(*fNormalizedCtxt);
   }
   // Well well well, for backward compatibility we need to act a bit too
   // much like CINT.
   TClassEdit::TSplitType splitname( cl.Name(), TClassEdit::kDropStd );
   static std::string result;
   splitname.ShortType(result, TClassEdit::kDropStd );
   return result.c_str();
}

//______________________________________________________________________________
void TCintWithCling::Execute(const char* function, const char* params, int* error)
{
   // Execute a global function with arguments params.
   R__LOCKGUARD2(gCINTMutex);
   TClingClassInfo cl(fInterpreter);
   Long_t offset;
   TClingCallFunc func(fInterpreter);
   func.SetFunc(&cl, function, params, &offset);
   func.Exec(0);
   if (error) {
      *error = G__lasterror();
   }
}

//______________________________________________________________________________
void TCintWithCling::Execute(TObject* obj, TClass* cl, const char* method,
                    const char* params, int* error)
{
   // Execute a method from class cl with arguments params.
   R__LOCKGUARD2(gCINTMutex);
   // If the actual class of this object inherits 2nd (or more) from TObject,
   // 'obj' is unlikely to be the start of the object (as described by IsA()),
   // hence gInterpreter->Execute will improperly correct the offset.
   void* addr = cl->DynamicCast(TObject::Class(), obj, kFALSE);
   Long_t offset = 0L;
   TClingCallFunc func(fInterpreter);
   func.SetFunc((TClingClassInfo*)cl->GetClassInfo(), method, params, &offset);
   void* address = (void*)((Long_t)addr + offset);
   func.Exec(address);
   if (error) {
      *error = G__lasterror();
   }
}

//______________________________________________________________________________
void TCintWithCling::Execute(TObject* obj, TClass* cl, TMethod* method,
      TObjArray* params, int* error)
{
   // Execute a method from class cl with the arguments in array params
   // (params[0] ... params[n] = array of TObjString parameters).
   // Convert the TObjArray array of TObjString parameters to a character
   // string of comma separated parameters.
   // The parameters of type 'char' are enclosed in double quotes and all
   // internal quotes are escaped.
   if (!method) {
      Error("Execute", "No method was defined");
      return;
   }
   TList* argList = method->GetListOfMethodArgs();
   // Check number of actual parameters against of expected formal ones
   Int_t nparms = argList->LastIndex() + 1;
   Int_t argc   = params ? params->LastIndex() + 1 : 0;
   if (nparms != argc) {
      Error("Execute", "Wrong number of the parameters");
      return;
   }
   const char* listpar = "";
   TString complete(10);
   if (params) {
      // Create a character string of parameters from TObjArray
      TIter next(params);
      for (Int_t i = 0; i < argc; i ++) {
         TMethodArg* arg = (TMethodArg*) argList->At(i);
         TClingTypeInfo type(fInterpreter, arg->GetFullTypeName());
         TObjString* nxtpar = (TObjString*) next();
         if (i) {
            complete += ',';
         }
         if (strstr(type.TrueName(*fNormalizedCtxt), "char")) {
            TString chpar('\"');
            chpar += (nxtpar->String()).ReplaceAll("\"", "\\\"");
            // At this point we have to check if string contains \\"
            // and apply some more sophisticated parser. Not implemented yet!
            complete += chpar;
            complete += '\"';
         }
         else {
            complete += nxtpar->String();
         }
      }
      listpar = complete.Data();
   }
   Execute(obj, cl, const_cast<char*>(method->GetName()), const_cast<char*>(listpar), error);
}

//______________________________________________________________________________
Long_t TCintWithCling::ExecuteMacro(const char* filename, EErrorCode* error)
{
   // Execute a CINT macro.
   R__LOCKGUARD(gCINTMutex);
   return TApplication::ExecuteFile(filename, (int*)error);
}

//______________________________________________________________________________
const char* TCintWithCling::GetTopLevelMacroName() const
{
   // Return the file name of the current un-included interpreted file.
   // See the documentation for GetCurrentMacroName().

   Warning("GetTopLevelMacroName", "Must change return type!");
   static std::string sMacroName;
   sMacroName = fMetaProcessor->getTopExecutingFile();
   return sMacroName.c_str();
}

//______________________________________________________________________________
const char* TCintWithCling::GetCurrentMacroName() const
{
   // Return the file name of the currently interpreted file,
   // included or not. Example to illustrate the difference between
   // GetCurrentMacroName() and GetTopLevelMacroName():
   // BEGIN_HTML <!--
   /* -->
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">inclfile.C</span>
      <!--div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;"-->
      <div class="code">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   void inclfile() {
   std::cout &lt;&lt; "In inclfile.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCintWithCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCintWithCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   }</pre></div>
      <div style="clear:both"></div>
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">mymacro.C</span>
      <div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;">
      <pre style="margin:0pt">#include &lt;iostream&gt;
   void mymacro() {
   std::cout &lt;&lt; "In mymacro.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetCurrentMacroName() returns  " &lt;&lt;
      TCintWithCling::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCintWithCling::GetTopLevelMacroName() returns " &lt;&lt;
      TCintWithCling::GetTopLevelMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  Now calling inclfile..." &lt;&lt; std::endl;
   gInterpreter->ProcessLine(".x inclfile.C");;
   }</pre></div>
   <div style="clear:both"></div>
   <!-- */
   // --> END_HTML
   // Running mymacro.C will print:
   //
   // root [0] .x mymacro.C
   // In mymacro.C
   //   TCintWithCling::GetCurrentMacroName() returns  ./mymacro.C
   //   TCintWithCling::GetTopLevelMacroName() returns ./mymacro.C
   //   Now calling inclfile...
   // In inclfile.h
   //   TCintWithCling::GetCurrentMacroName() returns  inclfile.C
   //   TCintWithCling::GetTopLevelMacroName() returns ./mymacro.C

#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,0)
   Warning("GetCurrentMacroName", "Must change return type!");
#endif
#endif
   static std::string sMacroName;
   sMacroName = fMetaProcessor->getCurrentlyExecutingFile();
   return sMacroName.c_str();
}

//______________________________________________________________________________
const char* TCintWithCling::TypeName(const char* typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // You need to use the result immediately before it is being overwritten.
   static char* t = 0;
   static unsigned int tlen = 0;
   R__LOCKGUARD(gCINTMutex); // Because of the static array.
   unsigned int dlen = strlen(typeDesc);
   if (dlen > tlen) {
      delete[] t;
      t = new char[dlen + 1];
      tlen = dlen;
   }
   char* s, *template_start;
   if (!strstr(typeDesc, "(*)(")) {
      s = const_cast<char*>(strchr(typeDesc, ' '));
      template_start = const_cast<char*>(strchr(typeDesc, '<'));
      if (!strcmp(typeDesc, "long long")) {
         strlcpy(t, typeDesc, dlen + 1);
      }
      else if (!strncmp(typeDesc, "unsigned ", s + 1 - typeDesc)) {
         strlcpy(t, typeDesc, dlen + 1);
      }
      // s is the position of the second 'word' (if any)
      // except in the case of templates where there will be a space
      // just before any closing '>': eg.
      //    TObj<std::vector<UShort_t,__malloc_alloc_template<0> > >*
      else if (s && (template_start == 0 || (s < template_start))) {
         strlcpy(t, s + 1, dlen + 1);
      }
      else {
         strlcpy(t, typeDesc, dlen + 1);
      }
   }
   else {
      strlcpy(t, typeDesc, dlen + 1);
   }
   int l = strlen(t);
   while (l > 0 && (t[l - 1] == '*' || t[l - 1] == '&')) {
      t[--l] = 0;
   }
   return t;
}

//______________________________________________________________________________
Int_t TCintWithCling::LoadLibraryMap(const char* rootmapfile)
{
   // Load map between class and library. If rootmapfile is specified a
   // specific rootmap file can be added (typically used by ACLiC).
   // In case of error -1 is returned, 0 otherwise.
   // The interpreter uses this information to automatically load the shared 
   // library for a class (autoload mechanism).
   // See also the AutoLoadCallback() method below.
   R__LOCKGUARD(gCINTMutex);
   // open the [system].rootmap files
   if (!fMapfile) {
      fMapfile = new TEnv(".rootmap");
      fMapfile->IgnoreDuplicates(kTRUE);
      fRootmapFiles = new TObjArray;
      fRootmapFiles->SetOwner();
   }
   // Load all rootmap files in the dynamic load path ((DY)LD_LIBRARY_PATH, etc.).
   // A rootmap file must end with the string ".rootmap".
   TString ldpath = gSystem->GetDynamicPath();
   if (ldpath != fRootmapLoadPath) {
      fRootmapLoadPath = ldpath;
#ifdef WIN32
      TObjArray* paths = ldpath.Tokenize(";");
#else
      TObjArray* paths = ldpath.Tokenize(":");
#endif
      TString d;
      for (Int_t i = 0; i < paths->GetEntriesFast(); i++) {
         d = ((TObjString*)paths->At(i))->GetString();
         // check if directory already scanned
         Int_t skip = 0;
         for (Int_t j = 0; j < i; j++) {
            TString pd = ((TObjString*)paths->At(j))->GetString();
            if (pd == d) {
               skip++;
               break;
            }
         }
         if (!skip) {
            void* dirp = gSystem->OpenDirectory(d);
            if (dirp) {
               if (gDebug > 3) {
                  Info("LoadLibraryMap", "%s", d.Data());
               }
               const char* f1;
               while ((f1 = gSystem->GetDirEntry(dirp))) {
                  TString f = f1;
                  if (f.EndsWith(".rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     if (!gSystem->AccessPathName(p, kReadPermission)) {
                        if (!fRootmapFiles->FindObject(f) && f != ".rootmap") {
                           if (gDebug > 4) {
                              Info("LoadLibraryMap", "   rootmap file: %s", p.Data());
                           }
                           fMapfile->ReadFile(p, kEnvGlobal);
                           fRootmapFiles->Add(new TNamed(f, p));
                        }
                        // else {
                        //    fprintf(stderr,"Reject %s because %s is already there\n",p.Data(),f.Data());
                        //    fRootmapFiles->FindObject(f)->ls();
                        // }
                     }
                  }
                  if (f.BeginsWith("rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     FileStat_t stat;
                     if (gSystem->GetPathInfo(p, stat) == 0 && R_ISREG(stat.fMode)) {
                        Warning("LoadLibraryMap", "please rename %s to end with \".rootmap\"", p.Data());
                     }
                  }
               }
            }
            gSystem->FreeDirectory(dirp);
         }
      }
      delete paths;
      if (!fMapfile->GetTable()->GetEntries()) {
         return -1;
      }
   }
   if (rootmapfile && *rootmapfile) {
      // Add content of a specific rootmap file
      Bool_t ignore = fMapfile->IgnoreDuplicates(kFALSE);
      fMapfile->ReadFile(rootmapfile, kEnvGlobal);
      fRootmapFiles->Add(new TNamed(gSystem->BaseName(rootmapfile), rootmapfile));
      fMapfile->IgnoreDuplicates(ignore);
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {
         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") {
            continue;
         }
         TString delim(" ");
         TObjArray* tokens = libs.Tokenize(delim);
         const char* lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0, 8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (gDebug > 6) {
            const char* wlib = gSystem->DynamicPathName(lib, kTRUE);
            if (wlib) {
               Info("LoadLibraryMap", "class %s in %s", cls.Data(), wlib);
            }
            else {
               Info("LoadLibraryMap", "class %s in %s (library does not exist)", cls.Data(), lib);
            }
            delete[] wlib;
         }
         delete tokens;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::RescanLibraryMap()
{
   // Scan again along the dynamic path for library maps. Entries for the loaded
   // shared libraries are unloaded first. This can be useful after reseting
   // the dynamic path through TSystem::SetDynamicPath()
   // In case of error -1 is returned, 0 otherwise.
   UnloadAllSharedLibraryMaps();
   LoadLibraryMap();
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::ReloadAllSharedLibraryMaps()
{
   // Reload the library map entries coming from all the loaded shared libraries,
   // after first unloading the current ones.
   // In case of error -1 is returned, 0 otherwise.
   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray* sharedLibL = sharedLibLStr.Tokenize(" ");
   const Int_t nrSharedLibs = sharedLibL->GetEntriesFast();
   for (Int_t ilib = 0; ilib < nrSharedLibs; ilib++) {
      const TString sharedLibStr = ((TObjString*)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      const Int_t ret = UnloadLibraryMap(sharedLibBaseStr);
      if (ret < 0) {
         continue;
      }
      TString rootMapBaseStr = sharedLibBaseStr;
      if (sharedLibBaseStr.EndsWith(".dll")) {
         rootMapBaseStr.ReplaceAll(".dll", "");
      }
      else if (sharedLibBaseStr.EndsWith(".DLL")) {
         rootMapBaseStr.ReplaceAll(".DLL", "");
      }
      else if (sharedLibBaseStr.EndsWith(".so")) {
         rootMapBaseStr.ReplaceAll(".so", "");
      }
      else if (sharedLibBaseStr.EndsWith(".sl")) {
         rootMapBaseStr.ReplaceAll(".sl", "");
      }
      else if (sharedLibBaseStr.EndsWith(".dl")) {
         rootMapBaseStr.ReplaceAll(".dl", "");
      }
      else if (sharedLibBaseStr.EndsWith(".a")) {
         rootMapBaseStr.ReplaceAll(".a", "");
      }
      else {
         Error("ReloadAllSharedLibraryMaps", "Unknown library type %s", sharedLibBaseStr.Data());
         delete sharedLibL;
         return -1;
      }
      rootMapBaseStr += ".rootmap";
      const char* rootMap = gSystem->Which(gSystem->GetDynamicPath(), rootMapBaseStr);
      if (!rootMap) {
         Error("ReloadAllSharedLibraryMaps", "Could not find rootmap %s in path", rootMap);
         delete[] rootMap;
         delete sharedLibL;
         return -1;
      }
      const Int_t status = LoadLibraryMap(rootMap);
      if (status < 0) {
         Error("ReloadAllSharedLibraryMaps", "Error loading map %s", rootMap);
         delete[] rootMap;
         delete sharedLibL;
         return -1;
      }
      delete[] rootMap;
   }
   delete sharedLibL;
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::UnloadAllSharedLibraryMaps()
{
   // Unload the library map entries coming from all the loaded shared libraries.
   // Returns 0 if succesful
   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray* sharedLibL = sharedLibLStr.Tokenize(" ");
   for (Int_t ilib = 0; ilib < sharedLibL->GetEntriesFast(); ilib++) {
      const TString sharedLibStr = ((TObjString*)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      UnloadLibraryMap(sharedLibBaseStr);
   }
   delete sharedLibL;
   return 0;
}

//______________________________________________________________________________
Int_t TCintWithCling::UnloadLibraryMap(const char* library)
{
   // Unload library map entries coming from the specified library.
   // Returns -1 in case no entries for the specified library were found,
   // 0 otherwise.
   if (!fMapfile || !library || !*library) {
      return 0;
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   R__LOCKGUARD(gCINTMutex);
   Int_t ret = 0;
   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {
         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") {
            continue;
         }
         TString delim(" ");
         TObjArray* tokens = libs.Tokenize(delim);
         const char* lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0, 8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (!strcmp(library, lib)) {
            if (fMapfile->GetTable()->Remove(rec) == 0) {
               Error("UnloadLibraryMap", "entry for <%s,%s> not found in library map table", cls.Data(), lib);
               ret = -1;
            }
         }
         delete tokens;
      }
   }
   if (ret >= 0) {
      TString library_rootmap(library);
      library_rootmap.Append(".rootmap");
      TNamed* mfile = 0;
      while ((mfile = (TNamed*)fRootmapFiles->FindObject(library_rootmap))) {
         fRootmapFiles->Remove(mfile);
         delete mfile;
      }
      fRootmapFiles->Compress();
   }
   return ret;
}

//______________________________________________________________________________
Int_t TCintWithCling::AutoLoad(const char* cls)
{
   // Load library containing the specified class. Returns 0 in case of error
   // and 1 in case if success.
   R__LOCKGUARD(gCINTMutex);
   Int_t status = 0;
   if (!gROOT || !gInterpreter || gROOT->TestBit(TObject::kInvalidObject)) {
      return status;
   }
   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldvalue = SetClassAutoloading(false);
   // lookup class to find list of dependent libraries
   TString deplibs = GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray* tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntriesFast() - 1; i > 0; i--) {
         const char* deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 0)
               ::Info("TCintWithCling::AutoLoad",
                      "loaded dependent library %s for class %s", deplib, cls);
         }
         else
            ::Error("TCintWithCling::AutoLoad",
                    "failure loading dependent library %s for class %s",
                    deplib, cls);
      }
      const char* lib = ((TObjString*)tokens->At(0))->GetName();
      if (lib[0]) {
         if (gROOT->LoadClass(cls, lib) == 0) {
            if (gDebug > 0)
               ::Info("TCintWithCling::AutoLoad",
                      "loaded library %s for class %s", lib, cls);
            status = 1;
         }
         else
            ::Error("TCintWithCling::AutoLoad",
                    "failure loading library %s for class %s", lib, cls);
      }
      delete tokens;
   }
   SetClassAutoloading(oldvalue);
   return status;
}

//______________________________________________________________________________
//FIXME: Use of G__ClassInfo in the interface!
void* TCintWithCling::FindSpecialObject(const char* item, G__ClassInfo* type,
                                        void** prevObj, void** assocPtr)
{
   // Static function called by CINT when it finds an un-indentified object.
   // This function tries to find the UO in the ROOT files, directories, etc.
   // This functions has been registered by the TCint ctor.
   if (!*prevObj || *assocPtr != gDirectory) {
      *prevObj = gROOT->FindSpecialObject(item, *assocPtr);
      if (!fgSetOfSpecials) {
         fgSetOfSpecials = new std::set<TObject*>;
      }
      if (*prevObj) {
         ((std::set<TObject*>*)fgSetOfSpecials)->insert((TObject*)*prevObj);
      }
   }
   if (*prevObj) {
      type->Init(((TObject*)*prevObj)->ClassName());
   }
   return *prevObj;
}

//______________________________________________________________________________
void TCintWithCling::UpdateClassInfoWithDecl(void* vTD)
{
   // Internal function. Inform a TClass about its new TagDecl.
   TagDecl* td = (TagDecl*)vTD;
   TagDecl* tdDef = td->getDefinition();
   if (tdDef) td = tdDef;
   std::string name = td->getName();
   
   // Supposedly we are being called being something is being
   // loaded ... let's now tell the autoloader to do the work
   // yet another time.
   int storedAutoloading = SetClassAutoloading(false);
   TClass* cl = TClass::GetClassOrAlias(name.c_str());
   if (cl) {
      TClingClassInfo* cci = ((TClingClassInfo*)cl->fClassInfo);
      if (cci) {
         const TagDecl* tdOld = llvm::dyn_cast_or_null<TagDecl>(cci->GetDecl());
         if (!tdOld || tdDef) {
            cl->ResetCaches();
            cci->Init(*cci->GetType());
         }
      } else {
         cl->ResetCaches();
         // yes, this is alsmost a waste of time, but we do need to lookup
         // the 'type' corresponding to the TClass anyway in order to
         // preserver the opaque typedefs (Double32_t)
         cl->fClassInfo = new TClingClassInfo(fInterpreter, cl->GetName());
      }
   }
   SetClassAutoloading(storedAutoloading);
}

//______________________________________________________________________________
void TCintWithCling::UpdateClassInfo(char* item, Long_t tagnum)
{
   // No op: see TClingCallbacks
}

//______________________________________________________________________________
//FIXME: Factor out that function in TClass, because TClass does it already twice
void TCintWithCling::UpdateClassInfoWork(const char* item)
{
   // This is a no-op as part of the API.
   // TCintWithCling uses UpdateClassInfoWithDecl() instead.
}

//______________________________________________________________________________
void TCintWithCling::UpdateAllCanvases()
{
   // Update all canvases at end the terminal input command.
   TIter next(gROOT->GetListOfCanvases());
   TVirtualPad* canvas;
   while ((canvas = (TVirtualPad*)next())) {
      canvas->Update();
   }
}

//______________________________________________________________________________
const char* TCintWithCling::GetSharedLibs()
{
   // Return the list of shared libraries loaded into the process.
   if (!fPrevLoadedDynLibInfo && fSharedLibs.IsNull())
      UpdateListOfLoadedSharedLibraries();
   return fSharedLibs;
}

//______________________________________________________________________________
const char* TCintWithCling::GetClassSharedLibs(const char* cls)
{
   // Get the list of shared libraries containing the code for class cls.
   // The first library in the list is the one containing the class, the
   // others are the libraries the first one depends on. Returns 0
   // in case the library is not found.
   if (!cls || !*cls) {
      return 0;
   }
   // lookup class to find list of libraries
   if (fMapfile) {
      TString c = TString("Library.") + cls;
      // convert "::" to "@@", we used "@@" because TEnv
      // considers "::" a terminator
      c.ReplaceAll("::", "@@");
      // convert "-" to " ", since class names may have
      // blanks and TEnv considers a blank a terminator
      c.ReplaceAll(" ", "-");
      // Use TEnv::Lookup here as the rootmap file must start with Library.
      // and do not support using any stars (so we do not need to waste time
      // with the search made by TEnv::GetValue).
      TEnvRec* libs_record = fMapfile->Lookup(c);
      if (libs_record) {
         const char* libs = libs_record->GetValue();
         return (*libs) ? libs : 0;
      }
   }
   return 0;
}

//______________________________________________________________________________
const char* TCintWithCling::GetSharedLibDeps(const char* lib)
{
   // Get the list a libraries on which the specified lib depends. The
   // returned string contains as first element the lib itself.
   // Returns 0 in case the lib does not exist or does not have
   // any dependencies.
   if (!fMapfile || !lib || !lib[0]) {
      return 0;
   }
   TString libname(lib);
   Ssiz_t idx = libname.Last('.');
   if (idx != kNPOS) {
      libname.Remove(idx);
   }
   TEnvRec* rec;
   TIter next(fMapfile->GetTable());
   size_t len = libname.Length();
   while ((rec = (TEnvRec*) next())) {
      const char* libs = rec->GetValue();
      if (!strncmp(libs, libname.Data(), len) && strlen(libs) >= len
            && (!libs[len] || libs[len] == ' ' || libs[len] == '.')) {
         return libs;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TCintWithCling::IsErrorMessagesEnabled() const
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("IsErrorMessagesEnabled", "Interface not available yet.");
#endif
#endif
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TCintWithCling::SetErrorMessages(Bool_t enable)
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout. Return the previous state.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetErrorMessages", "Interface not available yet.");
#endif
#endif
   return TCintWithCling::IsErrorMessagesEnabled();
}

//______________________________________________________________________________
const char* TCintWithCling::GetIncludePath()
{
   // Refresh the list of include paths known to the interpreter and return it
   // with -I prepended.

   R__LOCKGUARD(gCINTMutex);
   
   fIncludePath = "";

   llvm::SmallVector<std::string, 10> includePaths;//Why 10? Hell if I know.
   //false - no system header, true - with flags.
   fInterpreter->GetIncludePaths(includePaths, false, true);
   if (const size_t nPaths = includePaths.size()) {
      assert(!(nPaths & 1) && "GetIncludePath, number of paths and options is not equal");

      for (size_t i = 0; i < nPaths; i += 2) {
         if (i)
            fIncludePath.Append(' ');
         fIncludePath.Append(includePaths[i].c_str());

         if (includePaths[i] != "-I")
            fIncludePath.Append(' ');
         fIncludePath.Append(includePaths[i + 1], includePaths[i + 1].length());
      }
   }

   return fIncludePath;
}

//______________________________________________________________________________
const char* TCintWithCling::GetSTLIncludePath() const
{
   // Return the directory containing CINT's stl cintdlls.
   static TString stldir;
   if (!stldir.Length()) {
#ifdef CINTINCDIR
      stldir = CINTINCDIR;
#else
      stldir = gRootDir;
      stldir += "/cint";
#endif
      if (!stldir.EndsWith("/")) {
         stldir += '/';
      }
      stldir += "cint/stl";
   }
   return stldir;
}

//______________________________________________________________________________
//                      M I S C
//______________________________________________________________________________

int TCintWithCling::DisplayClass(FILE* /*fout*/, const char* /*name*/, int /*base*/, int /*start*/) const
{
   // Interface to CINT function
   return 0;
}

//______________________________________________________________________________
int TCintWithCling::DisplayIncludePath(FILE *fout) const
{
   // Interface to CINT function
   assert(fout != 0 && "DisplayIncludePath, 'fout' parameter is null");

   llvm::SmallVector<std::string, 10> includePaths;//Why 10? Hell if I know.
   //false - no system header, true - with flags.
   fInterpreter->GetIncludePaths(includePaths, false, true);
   if (const size_t nPaths = includePaths.size()) {
      assert(!(nPaths & 1) && "DisplayIncludePath, number of paths and options is not equal");

      std::string allIncludes("include path:");
      for (size_t i = 0; i < nPaths; i += 2) {
         allIncludes += ' ';
         allIncludes += includePaths[i];

         if (includePaths[i] != "-I")
            allIncludes += ' ';
         allIncludes += includePaths[i + 1];
      }

      fprintf(fout, "%s\n", allIncludes.c_str());
   }

   return 0;
}

//______________________________________________________________________________
void* TCintWithCling::FindSym(const char* entry) const
{
   // Interface to CINT function
   return fInterpreter->getAddressOfGlobal(entry);
}

//______________________________________________________________________________
void TCintWithCling::GenericError(const char* error) const
{
   // Let the interpreter issue a generic error, and set its error state.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("GenericError","Interface not available yet.");
#endif
#endif
}

//______________________________________________________________________________
Long_t TCintWithCling::GetExecByteCode() const
{
   // This routines used to return the address of the internal wrapper
   // function (of the interpreter) that was used to call *all* the
   // interpreted functions that were bytecode compiled (no longer 
   // interpreted line by line).  In Cling, there is no such
   // wrapper function.
   // In practice this routines was use to decipher whether the
   // pointer returns by InterfaceMethod could be used to uniquely
   // represent the function.  In Cling if the function is in a 
   // useable state (its compiled version is available), this is
   // always the case.  
   // See TClass::GetMethod.

   return 0;
}

//______________________________________________________________________________
Long_t TCintWithCling::Getgvp() const
{
   // Interface to the CINT global object pointer which was controlling the
   // behavior of the wrapper around the calls to operator new and the constructor
   // and operator delete and the destructor.

   Error("Getgvp","This was controlling the behavior of the wrappers for object construction and destruction.\nThis is now a nop and likely change the behavior of the calling routines.");
   return 0;
}

//______________________________________________________________________________
const char* TCintWithCling::Getp2f2funcname(void*) const
{
   Error("Getp2f2funcname", "Will not be implemented: "
         "all function pointers are compiled!");
   return NULL;
}

//______________________________________________________________________________
int TCintWithCling::GetSecurityError() const
{
   // Interface to CINT function
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("GetSecurityError", "Interface not available yet.");
#endif
#endif
   return 0;
}

//______________________________________________________________________________
int TCintWithCling::LoadFile(const char* path) const
{
   // Load a source file or library called path into the interpreter.
   return fInterpreter->loadFile(path);
}

//______________________________________________________________________________
void TCintWithCling::LoadText(const char* text) const
{
   // Load the declarations from text into the interpreter.
   // Note that this cannot be (top level) statements; text must contain
   // top level declarations.
   fInterpreter->declare(text);
}

//______________________________________________________________________________
const char* TCintWithCling::MapCppName(const char* name) const
{
   // Interface to CINT function
   static std::string buffer;
   ROOT::TMetaUtils::GetCppName(buffer,name);
   return buffer.c_str();
}

//______________________________________________________________________________
void TCintWithCling::SetAlloclockfunc(void (* /* p */ )()) const
{
   // [Place holder for Mutex Lock] 
   // Provide the interpreter with a way to
   // acquire a lock used to protect critical section 
   // of its code (non-thread safe parts).

   // nothing to do for now.
}

//______________________________________________________________________________
void TCintWithCling::SetAllocunlockfunc(void (* /* p */ )()) const
{
   // [Place holder for Mutex Unlock] Provide the interpreter with a way to
   // release a lock used to protect critical section 
   // of its code (non-thread safe parts).

   // nothing to do for now.
}

//______________________________________________________________________________
int TCintWithCling::SetClassAutoloading(int autoload) const
{
   // Enable/Disable the Autoloading of libraries.
   // Returns the old value, i.e whether it was enabled or not.
   if (!autoload && !fClingCallbacks) return false;

   assert(fClingCallbacks && "We must have callbacks!");
   bool oldVal =  fClingCallbacks->IsAutoloadingEnabled();
   fClingCallbacks->SetAutoloadingEnabled(autoload);
   return oldVal;
}

//______________________________________________________________________________
void TCintWithCling::SetErrmsgcallback(void* p) const
{
   // Set a callback to receive error messages.
#if defined(R__MUST_REVISIT)
#if R__MUST_REVISIT(6,2)
   Warning("SetErrmsgcallback", "Interface not available yet.");
#endif
#endif
}

//______________________________________________________________________________
void TCintWithCling::Setgvp(Long_t gvp) const
{
   // Interface to the CINT global object pointer which was controlling the
   // behavior of the wrapper around the calls to operator new and the constructor
   // and operator delete and the destructor.

   Error("Setgvp","This was controlling the behavior of the wrappers for object construction and destruction.\nThis is now a nop and likely change the behavior of the calling routines.");

}

//______________________________________________________________________________
void TCintWithCling::SetRTLD_NOW() const
{
   Error("SetRTLD_NOW()", "Will never be implemented! Don't use!");
}

//______________________________________________________________________________
void TCintWithCling::SetRTLD_LAZY() const
{
   Error("SetRTLD_LAZY()", "Will never be implemented! Don't use!");
}

//______________________________________________________________________________
void TCintWithCling::SetTempLevel(int val) const
{
   // Create / close a scope for temporaries. No-op for cling; use
   // StoredValueRef instead.
}

//______________________________________________________________________________
int TCintWithCling::UnloadFile(const char* path) const
{
   // Unload a shared library or a source file.

   // Check fInterpreter->getLoadedFiles() to determine whether this is a shared
   // library or code. If it's not in there complain.
   typedef llvm::SmallVectorImpl<cling::Interpreter::LoadedFileInfo*> LoadedFiles_t;
   const LoadedFiles_t& loadedFiles = fInterpreter->getLoadedFiles();
   const cling::Interpreter::LoadedFileInfo* fileInfo = 0;
   for (LoadedFiles_t::const_iterator iF = loadedFiles.begin(),
           eF = loadedFiles.end(); iF != eF; ++iF) {
      if ((*iF)->getName() == path) {
         fileInfo = *iF;
      }
   }
   if (!fileInfo) {
      Error("UnloadFile", "File %s has not been loaded!", path);
      return -1;
   }

   if (fileInfo->getType() == cling::Interpreter::LoadedFileInfo::kDynamicLibrary) {
      // Signal that the list of shared libs needs to be updated.
      const_cast<TCintWithCling*>(this)->fPrevLoadedDynLibInfo = 0;
      const_cast<TCintWithCling*>(this)->fSharedLibs = "";

      Error("UnloadFile", "Unloading of shared libraries not yet implemented!\n"
            "Not unloading file %s!", path);

      return -1;
   } else if (fileInfo->getType() == cling::Interpreter::LoadedFileInfo::kSource) {
      Error("UnloadFile", "Unloading of source files not yet implemented!\n"
            "Not unloading file %s!", path);
      return -1;
   } else {
      Error("UnloadFile", "Unloading of files of type %d not yet implemented!\n"
            "Not unloading file %s!", (int)fileInfo->getType(), path);
      return -1;
   }

   return -1;
}



//______________________________________________________________________________
//
//  CallFunc interface
//

//______________________________________________________________________________
void TCintWithCling::CallFunc_Delete(CallFunc_t* func) const
{
   delete (TClingCallFunc*) func;
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_Exec(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->Exec(address);
}

//______________________________________________________________________________
Long_t TCintWithCling::CallFunc_ExecInt(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecInt(address);
}

//______________________________________________________________________________
Long_t TCintWithCling::CallFunc_ExecInt64(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecInt64(address);
}

//______________________________________________________________________________
Double_t TCintWithCling::CallFunc_ExecDouble(CallFunc_t* func, void* address) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->ExecDouble(address);
}

//______________________________________________________________________________
CallFunc_t* TCintWithCling::CallFunc_Factory() const
{
   return (CallFunc_t*) new TClingCallFunc(fInterpreter);
}

//______________________________________________________________________________
CallFunc_t* TCintWithCling::CallFunc_FactoryCopy(CallFunc_t* func) const
{
   return (CallFunc_t*) new TClingCallFunc(*(TClingCallFunc*)func);
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::CallFunc_FactoryMethod(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return (MethodInfo_t*) f->FactoryMethod();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_Init(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->Init();
}

//______________________________________________________________________________
bool TCintWithCling::CallFunc_IsValid(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   return f->IsValid();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_ResetArg(CallFunc_t* func) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->ResetArg();
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Long_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Double_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, Long64_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArg(CallFunc_t* func, ULong64_t param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArg(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArgArray(CallFunc_t* func, Long_t* paramArr, Int_t nparam) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArgArray(paramArr, nparam);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetArgs(CallFunc_t* func, const char* param) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   f->SetArgs(param);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFunc(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* params, Long_t* offset) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFunc(ci, method, params, offset);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFunc(CallFunc_t* func, MethodInfo_t* info) const
{
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingMethodInfo* minfo = (TClingMethodInfo*) info;
   f->SetFunc(minfo);
}

//______________________________________________________________________________
void TCintWithCling::CallFunc_SetFuncProto(CallFunc_t* func, ClassInfo_t* info, const char* method, const char* proto, Long_t* offset) const
{
   // Interface to CINT function
   TClingCallFunc* f = (TClingCallFunc*) func;
   TClingClassInfo* ci = (TClingClassInfo*) info;
   f->SetFuncProto(ci, method, proto, offset);
}


//______________________________________________________________________________
//
//  ClassInfo interface
//

Long_t TCintWithCling::ClassInfo_ClassProperty(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->ClassProperty();
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Delete(ClassInfo_t* cinfo) const
{
   delete (TClingClassInfo*) cinfo;
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Delete(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Delete(arena);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_DeleteArray(ClassInfo_t* cinfo, void* arena, bool dtorOnly) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->DeleteArray(arena, dtorOnly);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Destruct(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Destruct(arena);
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory() const
{
   return (ClassInfo_t*) new TClingClassInfo(fInterpreter);
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory(ClassInfo_t* cinfo) const
{
   return (ClassInfo_t*) new TClingClassInfo(*(TClingClassInfo*)cinfo);
}

//______________________________________________________________________________
ClassInfo_t* TCintWithCling::ClassInfo_Factory(const char* name) const
{
   return (ClassInfo_t*) new TClingClassInfo(fInterpreter, name);
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_GetMethodNArg(ClassInfo_t* cinfo, const char* method, const char* proto) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->GetMethodNArg(method, proto);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_HasDefaultConstructor(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->HasDefaultConstructor();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_HasMethod(ClassInfo_t* cinfo, const char* name) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->HasMethod(name);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Init(ClassInfo_t* cinfo, const char* name) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
void TCintWithCling::ClassInfo_Init(ClassInfo_t* cinfo, int tagnum) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   TClinginfo->Init(tagnum);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsBase(ClassInfo_t* cinfo, const char* name) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsBase(name);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsEnum(const char* name) const
{
   return TClingClassInfo::IsEnum(fInterpreter, name);
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsLoaded(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsLoaded();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsValid(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
bool TCintWithCling::ClassInfo_IsValidMethod(ClassInfo_t* cinfo, const char* method, const char* proto, Long_t* offset) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->IsValidMethod(method, proto, offset);
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_Next(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New();
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, int n) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(n);
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, int n, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(n, arena);
}

//______________________________________________________________________________
void* TCintWithCling::ClassInfo_New(ClassInfo_t* cinfo, void* arena) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->New(arena);
}

//______________________________________________________________________________
Long_t TCintWithCling::ClassInfo_Property(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCintWithCling::ClassInfo_Size(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
Long_t TCintWithCling::ClassInfo_Tagnum(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Tagnum();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_FileName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->FileName();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_FullName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->FullName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_Name(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_Title(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->Title();
}

//______________________________________________________________________________
const char* TCintWithCling::ClassInfo_TmpltName(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return TClinginfo->TmpltName();
}



//______________________________________________________________________________
//
//  BaseClassInfo interface
//

//______________________________________________________________________________
void TCintWithCling::BaseClassInfo_Delete(BaseClassInfo_t* bcinfo) const
{
   delete(TClingBaseClassInfo*) bcinfo;
}

//______________________________________________________________________________
BaseClassInfo_t* TCintWithCling::BaseClassInfo_Factory(ClassInfo_t* cinfo) const
{
   TClingClassInfo* TClinginfo = (TClingClassInfo*) cinfo;
   return (BaseClassInfo_t*) new TClingBaseClassInfo(fInterpreter, TClinginfo);
}

//______________________________________________________________________________
int TCintWithCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
int TCintWithCling::BaseClassInfo_Next(BaseClassInfo_t* bcinfo, int onlyDirect) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Next(onlyDirect);
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Offset(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Offset();
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Property(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
Long_t TCintWithCling::BaseClassInfo_Tagnum(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Tagnum();
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_FullName(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->FullName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_Name(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::BaseClassInfo_TmpltName(BaseClassInfo_t* bcinfo) const
{
   TClingBaseClassInfo* TClinginfo = (TClingBaseClassInfo*) bcinfo;
   return TClinginfo->TmpltName();
}

//______________________________________________________________________________
//
//  DataMemberInfo interface
//

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_ArrayDim(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->ArrayDim();
}

//______________________________________________________________________________
void TCintWithCling::DataMemberInfo_Delete(DataMemberInfo_t* dminfo) const
{
   delete(TClingDataMemberInfo*) dminfo;
}

//______________________________________________________________________________
DataMemberInfo_t* TCintWithCling::DataMemberInfo_Factory(ClassInfo_t* clinfo /*= 0*/) const
{
   TClingClassInfo* TClingclass_info = (TClingClassInfo*) clinfo;
   return (DataMemberInfo_t*) new TClingDataMemberInfo(fInterpreter, TClingclass_info);
}

//______________________________________________________________________________
DataMemberInfo_t* TCintWithCling::DataMemberInfo_FactoryCopy(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return (DataMemberInfo_t*) new TClingDataMemberInfo(*TClinginfo);
}

//______________________________________________________________________________
bool TCintWithCling::DataMemberInfo_IsValid(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_MaxIndex(DataMemberInfo_t* dminfo, Int_t dim) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->MaxIndex(dim);
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_Next(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_Offset(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Offset();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_Property(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
Long_t TCintWithCling::DataMemberInfo_TypeProperty(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeProperty();
}

//______________________________________________________________________________
int TCintWithCling::DataMemberInfo_TypeSize(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeSize();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_TypeName(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeName();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_TypeTrueName(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->TypeTrueName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_Name(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_Title(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->Title();
}

//______________________________________________________________________________
const char* TCintWithCling::DataMemberInfo_ValidArrayIndex(DataMemberInfo_t* dminfo) const
{
   TClingDataMemberInfo* TClinginfo = (TClingDataMemberInfo*) dminfo;
   return TClinginfo->ValidArrayIndex();
}



//______________________________________________________________________________
//
//  MethodInfo interface
//

//______________________________________________________________________________
void TCintWithCling::MethodInfo_Delete(MethodInfo_t* minfo) const
{
   // Interface to CINT function
   delete(TClingMethodInfo*) minfo;
}

//______________________________________________________________________________
void TCintWithCling::MethodInfo_CreateSignature(MethodInfo_t* minfo, TString& signature) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   info->CreateSignature(signature);
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::MethodInfo_Factory() const
{
   return (MethodInfo_t*) new TClingMethodInfo(fInterpreter);
}

//______________________________________________________________________________
MethodInfo_t* TCintWithCling::MethodInfo_Factory(ClassInfo_t* clinfo) const
{
   return (MethodInfo_t*) new TClingMethodInfo(fInterpreter, (TClingClassInfo*)clinfo);
}
//______________________________________________________________________________
MethodInfo_t* TCintWithCling::MethodInfo_FactoryCopy(MethodInfo_t* minfo) const
{
   return (MethodInfo_t*) new TClingMethodInfo(*(TClingMethodInfo*)minfo);
}

//______________________________________________________________________________
void* TCintWithCling::MethodInfo_InterfaceMethod(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return (void*) info->InterfaceMethod();
}

//______________________________________________________________________________
bool TCintWithCling::MethodInfo_IsValid(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_NArg(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->NArg();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_NDefaultArg(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->NDefaultArg();
}

//______________________________________________________________________________
int TCintWithCling::MethodInfo_Next(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::MethodInfo_Property(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Property();
}

//______________________________________________________________________________
void* TCintWithCling::MethodInfo_Type(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Type();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_GetMangledName(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->GetMangledName();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_GetPrototype(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->GetPrototype();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_Name(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_TypeName(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->TypeName();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodInfo_Title(MethodInfo_t* minfo) const
{
   TClingMethodInfo* info = (TClingMethodInfo*) minfo;
   return info->Title();
}

//______________________________________________________________________________
//
//  MethodArgInfo interface
//

//______________________________________________________________________________
void TCintWithCling::MethodArgInfo_Delete(MethodArgInfo_t* marginfo) const
{
   delete(TClingMethodArgInfo*) marginfo;
}

//______________________________________________________________________________
MethodArgInfo_t* TCintWithCling::MethodArgInfo_Factory() const
{
   return (MethodArgInfo_t*) new TClingMethodArgInfo(fInterpreter);
}

//______________________________________________________________________________
MethodArgInfo_t* TCintWithCling::MethodArgInfo_Factory(MethodInfo_t *minfo) const
{
   return (MethodArgInfo_t*) new TClingMethodArgInfo(fInterpreter, (TClingMethodInfo*)minfo);
}

//______________________________________________________________________________
MethodArgInfo_t* TCintWithCling::MethodArgInfo_FactoryCopy(MethodArgInfo_t* marginfo) const
{
   return (MethodArgInfo_t*)
          new TClingMethodArgInfo(*(TClingMethodArgInfo*)marginfo);
}

//______________________________________________________________________________
bool TCintWithCling::MethodArgInfo_IsValid(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->IsValid();
}

//______________________________________________________________________________
int TCintWithCling::MethodArgInfo_Next(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::MethodArgInfo_Property(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Property();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_DefaultValue(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->DefaultValue();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_Name(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_TypeName(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->TypeName();
}

//______________________________________________________________________________
const char* TCintWithCling::MethodArgInfo_TrueTypeName(MethodArgInfo_t* marginfo) const
{
   TClingMethodArgInfo* info = (TClingMethodArgInfo*) marginfo;
   return info->Type()->TrueName(*fNormalizedCtxt);
}


//______________________________________________________________________________
//
//  TypeInfo interface
//

//______________________________________________________________________________
void TCintWithCling::TypeInfo_Delete(TypeInfo_t* tinfo) const
{
   delete (TClingTypeInfo*) tinfo;
}

//______________________________________________________________________________
TypeInfo_t* TCintWithCling::TypeInfo_Factory() const
{
   return (TypeInfo_t*) new TClingTypeInfo(fInterpreter);
}

//______________________________________________________________________________
TypeInfo_t* TCintWithCling::TypeInfo_FactoryCopy(TypeInfo_t* tinfo) const
{
   return (TypeInfo_t*) new TClingTypeInfo(*(TClingTypeInfo*)tinfo);
}

//______________________________________________________________________________
void TCintWithCling::TypeInfo_Init(TypeInfo_t* tinfo, const char* name) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
bool TCintWithCling::TypeInfo_IsValid(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
const char* TCintWithCling::TypeInfo_Name(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
Long_t TCintWithCling::TypeInfo_Property(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCintWithCling::TypeInfo_RefType(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->RefType();
}

//______________________________________________________________________________
int TCintWithCling::TypeInfo_Size(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
const char* TCintWithCling::TypeInfo_TrueName(TypeInfo_t* tinfo) const
{
   TClingTypeInfo* TClinginfo = (TClingTypeInfo*) tinfo;
   return TClinginfo->TrueName(*fNormalizedCtxt);
}


//______________________________________________________________________________
//
//  TypedefInfo interface
//

//______________________________________________________________________________
void TCintWithCling::TypedefInfo_Delete(TypedefInfo_t* tinfo) const
{
   delete(TClingTypedefInfo*) tinfo;
}

//______________________________________________________________________________
TypedefInfo_t* TCintWithCling::TypedefInfo_Factory() const
{
   return (TypedefInfo_t*) new TClingTypedefInfo(fInterpreter);
}

//______________________________________________________________________________
TypedefInfo_t* TCintWithCling::TypedefInfo_FactoryCopy(TypedefInfo_t* tinfo) const
{
   return (TypedefInfo_t*) new TClingTypedefInfo(*(TClingTypedefInfo*)tinfo);
}

//______________________________________________________________________________
TypedefInfo_t TCintWithCling::TypedefInfo_Init(TypedefInfo_t* tinfo,
                                      const char* name) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   TClinginfo->Init(name);
}

//______________________________________________________________________________
bool TCintWithCling::TypedefInfo_IsValid(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->IsValid();
}

//______________________________________________________________________________
Int_t TCintWithCling::TypedefInfo_Next(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Next();
}

//______________________________________________________________________________
Long_t TCintWithCling::TypedefInfo_Property(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Property();
}

//______________________________________________________________________________
int TCintWithCling::TypedefInfo_Size(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Size();
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_TrueName(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->TrueName(*fNormalizedCtxt);
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_Name(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Name();
}

//______________________________________________________________________________
const char* TCintWithCling::TypedefInfo_Title(TypedefInfo_t* tinfo) const
{
   TClingTypedefInfo* TClinginfo = (TClingTypedefInfo*) tinfo;
   return TClinginfo->Title();
}
