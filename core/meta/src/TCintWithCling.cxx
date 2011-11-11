// @(#)root/meta:$Id$
// Author: Paul Russo, 2009-10-06

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto from HP Japan, using cling as interpreter backend.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCintWithCling.h"
#include "RConfigure.h"
#include "compiledata.h"
#include "TSystem.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Serialization/ASTReader.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Support/DynamicLibrary.h"

#include <map>
#include <vector>
#include <set>
#include <string>
#include <cxxabi.h>

using namespace std;

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
      std::string errmsg;
      bool load_failed = llvm::sys::DynamicLibrary::LoadLibraryPermanently(lib, &errmsg);
      if (load_failed) {
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

ClassImp(TCintWithCling)

//______________________________________________________________________________
TCintWithCling::TCintWithCling(const char *name, const char *title) :
   TCint(name, title),
   fInterpreter(0),
   fMetaProcessor(0)
{
   // Initialize the CINT+cling interpreter interface.

   TString interpInclude;
#ifndef ROOTINCDIR
   TString rootsys = gSystem->Getenv("ROOTSYS");
   interpInclude = rootsys + "/etc";
#else
   interpInclude = ROOTETCDIR;
#endif
   interpInclude.Prepend("-I");
   const char* interpArgs[] = {interpInclude.Data(), 0};

   TString llvmDir;
   if (gSystem->Getenv("$(LLVMDIR)")) {
      llvmDir = gSystem->ExpandPathName("$(LLVMDIR)");
   }
#ifdef R__LLVMDIR
   if (llvmDir.IsNull()) {
      llvmDir = R__LLVMDIR;
   }
#endif

   fInterpreter = new cling::Interpreter(1, interpArgs, llvmDir); 
   fInterpreter->installLazyFunctionCreator(autoloadCallback);

   // Add the root include directory and etc/ to list searched by default.
   // Use explicit TCint::AddIncludePath() to avoid vtable: we're in the c'tor!
#ifndef ROOTINCDIR
   TCintWithCling::AddIncludePath(rootsys + "/include");
   TString dictDir = rootsys + "/lib";
#else
   TCintWithCling::AddIncludePath(ROOTINCDIR);
   TString dictDir = ROOTLIBDIR;
#endif

   clang::CompilerInstance * CI = fInterpreter->getCI ();
   CI->getPreprocessor().getHeaderSearchInfo().configureModules(dictDir.Data(), "NONE");
   const char* dictEntry = 0;
   void* dictDirPtr = gSystem->OpenDirectory(dictDir);
   while ((dictEntry = gSystem->GetDirEntry(dictDirPtr))) {
      static const char dictExt[] = "_dict.pcm";
      size_t lenDictEntry = strlen(dictEntry);
      if (lenDictEntry <= 9 || strcmp(dictEntry + lenDictEntry - 9, dictExt)) {
         continue;
      }
      //TString dictFile = dictDir + "/" + dictEntry;
      Info("LoadDictionaries", "Loading PCH %s", dictEntry);
      TString module(dictEntry);
      module.Remove(module.Length() - 4, 4);
      fInterpreter->processLine(std::string("__import_module__ ") + module.Data() + ";",
                                true /*raw*/);
   }
   gSystem->FreeDirectory(dictDirPtr);

   fMetaProcessor = new cling::MetaProcessor(*fInterpreter);

   // to pull in gPluginManager
   fMetaProcessor->process("#include \"TPluginManager.h\"");
}

//______________________________________________________________________________
TCintWithCling::~TCintWithCling()
{
   // Destroy the CINT+cling interpreter interface.
}

//______________________________________________________________________________
Long_t TCintWithCling::ProcessLine(const char *line, EErrorCode *error)
{
   // Let CINT process a command line.
   // If the command is executed and the result of G__process_cmd is 0,
   // the return value is the int value corresponding to the result of the command
   // (float and double return values will be truncated).

   Long_t ret = 0;
   // Store our line. line is a static buffer in TApplication
   // and we get called recursively through G__process_cmd.
   TString sLine(line);
   if ((sLine[0] == '#') || (sLine[0] == '.')) {
      // Preprocessor or special cmd, have cint process it too.
      // Do not actually run things, load theminstead:
      char haveX = sLine[1];
      if (haveX == 'x' || haveX == 'X')
         sLine[1] = 'L';
      TString sLineNoArgs(sLine);
      Ssiz_t posOpenParen = sLineNoArgs.Last('(');
      if (posOpenParen != kNPOS && sLineNoArgs.EndsWith(")")) {
         sLineNoArgs.Remove(posOpenParen, sLineNoArgs.Length() - posOpenParen);
      }
      ret = TCint::ProcessLine(sLineNoArgs, error);
      sLine[1] = haveX;
   }
   static const char *fantomline = "TRint::EndOfLineAction();";
   if (sLine == fantomline) {
      // end of line action, CINT-only.
      return TCint::ProcessLine(sLine, error);
   }
   TString aclicMode;
   TString arguments;
   TString io;
   TString fname;
   if (!strncmp(sLine.Data(), ".L", 2)) { // Load cmd, check for use of ACLiC.
      fname = gSystem->SplitAclicMode(sLine.Data()+3, aclicMode, arguments, io);
   }
   if (aclicMode.Length()) { // ACLiC, pass to cint and not to cling.
      ret = TCint::ProcessLine(sLine, error);
   }
   else {
      if (fMetaProcessor->process(sLine) > 0) {
         printf("...\n");
      }
   }

   return ret;
}

//______________________________________________________________________________
void TCintWithCling::PrintIntro()
{
   // Print CINT introduction and help message.

   Printf("\nCINT/ROOT C/C++ Interpreter version %s", G__cint_version());
   Printf(">>>>>>>>> cling-ified version <<<<<<<<<<");
   Printf("Type ? for help. Commands must be C++ statements.");
   Printf("Enclose multiple statements between { }.");
}

//______________________________________________________________________________
void TCintWithCling::AddIncludePath(const char *path)
{
   // Add the given path to the list of directories in which the interpreter
   // looks for include files. Only one path item can be specified at a
   // time, i.e. "path1:path2" is not supported.

   fInterpreter->AddIncludePath(path);
   TCint::AddIncludePath(path);
}

//______________________________________________________________________________
void TCintWithCling::InspectMembers(TMemberInspector&, void* obj, const char* clname)
{
   Printf("Inspecting class %s\n", clname);
}

