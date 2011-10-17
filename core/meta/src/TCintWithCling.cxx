// @(#)root/meta:$Id$
// Author: Fons Rademakers   01/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// This class defines an interface to the CINT C/C++ interpreter made   //
// by Masaharu Goto from HP Japan.                                      //
//                                                                      //
// CINT is an almost full ANSI compliant C/C++ interpreter.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TCling.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassEdit.h"
#include "TCollection.h"
#include "TBaseClass.h"
#include "TDataMember.h"
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
#include "THashList.h"
#include "THashTable.h"
#include "RConfigure.h"
#include "compiledata.h"
#include "cling/Interpreter/Interpreter.h"
#include "cling/MetaProcessor/MetaProcessor.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/HeaderSearchOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/DynamicLibrary.h"

#include <map>
#include <vector>
#include <set>
#include <string>
#include <cxxabi.h>

#undef USE_CLR

#ifdef USE_CLR
   #include "clr-scan.h"
#endif

#if G__CINTVERSION == 70030000
// Ignore SetGetLineFunc in Cint7
void G__SetGetlineFunc(char*(*)(const char* prompt),
                       void (*)(char* line)) {}
#endif

using namespace std;

R__EXTERN int optind;

static std::map<std::string, std::string>& class_to_library_map() {
   static std::map<std::string, std::string> s;
   return s;
}

static void loadClassToLibraryMap();

extern "C" int ScriptCompiler(const char *filename, const char *opt) {
   return gSystem->CompileMacro(filename, opt);
}

extern "C" int IgnoreInclude(const char *fname, const char *expandedfname) {
   return gROOT->IgnoreInclude(fname,expandedfname);
}

extern "C" void TCint_UpdateClassInfo(char *c, Long_t l) {
   TCint::UpdateClassInfo(c, l);
}

extern "C" int TCint_AutoLoadCallback(char *c, char *l) {
   ULong_t varp = G__getgvp();
   G__setgvp((Long_t)G__PVOID);
   string cls(c);
   int result =  TCint::AutoLoadCallback(cls.c_str(), l);
   G__setgvp(varp);
   return result;
}

static void* autoloadCallback(const std::string& mangled_name)
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
   std::map<std::string, std::string>::iterator iter = class_to_library_map().find(name);
   if (iter == class_to_library_map().end()) {
      // Not found in the map, all done.
      return 0;
   }
   //fprintf(stderr, "library: %s\n", iter->second.c_str());
   // Now we have the name of the library to load, so load it.
   std::string errmsg;
   bool load_failed = llvm::sys::DynamicLibrary::LoadLibraryPermanently(iter->second.c_str(), &errmsg);
   if (load_failed) {
      // The library load failed, all done.
      //fprintf(stderr, "load failed: %s\n", errmsg.c_str());
      return 0;
   }
   //fprintf(stderr, "load succeeded.\n");
   // Get the address of the function being called.
   void* addr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(mangled_name.c_str());
   //fprintf(stderr, "addr: %016lx\n", reinterpret_cast<unsigned long>(addr));
   return addr;
}

extern "C" void *TCint_FindSpecialObject(char *c, G__ClassInfo *ci, void **p1, void **p2) {
   return TCint::FindSpecialObject(c, ci, p1, p2);
}

TInterpreter* TCint_Factory(const char* name, const char* title) {
   return new TCint(name, title);
}



int TCint_GenerateDictionary(const std::vector<std::string> &classes,
                             const std::vector<std::string> &headers,
                             const std::vector<std::string> &fwdDecls,
                             const std::vector<std::string> &unknown)
{
   //This function automatically creates the "LinkDef.h" file for templated
   //classes then executes CompileMacro on it.
   //The name of the file depends on the class name, and it's not generated again
   //if the file exist.

   return 0;
}
 
int TCint_GenerateDictionary(const std::string &className,
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

// It is a "fantom" method to synchronize user keyboard input
// and ROOT prompt line (for WIN32)
const char *fantomline = "TRint::EndOfLineAction();";

void* TCint::fgSetOfSpecials = 0;

ClassImp(TCint)

//______________________________________________________________________________
TCint::TCint(const char *name, const char *title) :
   TInterpreter(name, title),
   fInterpreter(0),
   fMetaProcessor(0)
{
   // Initialize the CINT interpreter interface.

   fMore      = 0;
   fPrompt[0] = 0;
   fMapfile   = 0;
   fRootmapFiles = 0;
   fLockProcessLine = kTRUE;

   G__RegisterScriptCompiler(&ScriptCompiler);
   G__set_ignoreinclude(&IgnoreInclude);
   G__InitUpdateClassInfo(&TCint_UpdateClassInfo);
   G__InitGetSpecialObject(&TCint_FindSpecialObject);

   // check whether the compiler is available:
   char* path = gSystem->Which(gSystem->Getenv("PATH"), gSystem->BaseName(COMPILER));
   if (path && path[0]) {
      G__InitGenerateDictionary( &TCint_GenerateDictionary );
   }
   delete[] path;

   ResetAll();

#ifndef R__WIN32
   optind = 1;  // make sure getopt() works in the main program
#endif

#if defined(G__NOSTUBS)
# if defined(G__NOSTUBSTEST)
   EnableWrappers(gEnv->GetValue("Cint.EnableWrappers",1));
# else
   EnableWrappers(0);
# endif
#else
   EnableWrappers(1);
#endif

   // Make sure that ALL macros are seen as C++.
   G__LockCpp();

   fInterpreter = new cling::Interpreter(0, 0, 0, gSystem->ExpandPathName("$(LLVMDIR)"));
   fInterpreter->installLazyFunctionCreator(autoloadCallback);
   clang::CompilerInstance * CI = fInterpreter->getCI ();

   clang::LangOptions & langInfo = CI->getLangOpts ();
   //langInfo.C99         = 1;
   //langInfo.HexFloats   = 1;
   langInfo.BCPLComment = 1; // Only for C99/C++.
   langInfo.Digraphs    = 1; // C94, C99, C++.
   langInfo.CPlusPlus   = 1;
   //langInfo.CPlusPlus0x = 1;
   langInfo.CXXOperatorNames = 1;
   langInfo.Bool = 1;
   langInfo.NeXTRuntime = 1;
   langInfo.NoInline = 1;
   langInfo.Exceptions = 1;
   langInfo.GNUMode = 1;
   langInfo.NoInline = 1;
   langInfo.GNUInline = 1;
   langInfo.DollarIdents = 1;
   langInfo.POSIXThreads = 1;


   // Add the root include directory to list searched by default.
   // Use explicit TCint::AddIncludePath() to avoid vtable: we're in the c'tor!
#ifndef ROOTINCDIR
   TString include = gSystem->Getenv("ROOTSYS");
   include.Append("/include");
   TCint::AddIncludePath(include);
#else
   TCint::AddIncludePath(ROOTINCDIR);
#endif

   //TString pch_filename = gSystem->Getenv("ROOTSYS");
   //pch_filename.Append("/include/ROOT.pch");
   //CI->createPCHExternalASTSource(pch_filename.Data(), true, 0);

   fMetaProcessor = new cling::MetaProcessor(*fInterpreter);

   // to pull in gPluginManager
   fMetaProcessor->process("#include \"TPluginManager.h\"");
}

//______________________________________________________________________________
TCint::~TCint()
{
   // Destroy the CINT interpreter interface.

   if (fMore != -1) {
      // only close the opened files do not free memory:
      // G__scratch_all();
      G__close_inputfiles();
   }

   delete fMapfile;
   delete fRootmapFiles;
   gCint = 0;
}

//______________________________________________________________________________
void TCint::ClearFileBusy()
{
   // Reset CINT internal state in case a previous action was not correctly
   // terminated by G__init_cint() and G__dlmod().

   R__LOCKGUARD(gCINTMutex);

   G__clearfilebusy(0);
}

//______________________________________________________________________________
void TCint::ClearStack()
{
   // Delete existing temporary values

   R__LOCKGUARD(gCINTMutex);

   G__clearstack();
}

//______________________________________________________________________________
Int_t TCint::InitializeDictionaries()
{
   // Initialize all registered dictionaries. Normally this is already done
   // by G__init_cint() and G__dlmod().

   R__LOCKGUARD(gCINTMutex);

   return G__call_setup_funcs();
}

//______________________________________________________________________________
void TCint::EnableWrappers(bool value)
{
   // Enable call wrappers (also known as stubs) if value is true;
   // disable if value is false.

   G__enable_wrappers((int) value);
}

//______________________________________________________________________________
void TCint::EnableAutoLoading()
{
   // Enable the automatic loading of shared libraries when a class
   // is used that is stored in a not yet loaded library. Uses the
   // information stored in the class/library map (typically
   // $ROOTSYS/etc/system.rootmap).

   R__LOCKGUARD(gCINTMutex);

   G__set_class_autoloading_callback(&TCint_AutoLoadCallback);
   LoadLibraryMap();
}

//______________________________________________________________________________
void TCint::EndOfLineAction()
{
   // It calls a "fantom" method to synchronize user keyboard input
   // and ROOT prompt line.

   ProcessLineSynch(fantomline);
}

//______________________________________________________________________________
Bool_t TCint::IsLoaded(const char* filename) const
{
   // Return true if the file has already been loaded by cint.

   // We will try in this order:
   //   actual filename
   //   filename as a path relative to
   //            the include path
   //            the shared library path

   R__LOCKGUARD(gCINTMutex);

   G__SourceFileInfo file(filename);
   if (file.IsValid()) { return kTRUE; };

   char *next = gSystem->Which(TROOT::GetMacroPath(), filename, kReadPermission);
   if (next) {
      file.Init(next);
      delete [] next;
      if (file.IsValid()) { return kTRUE; };
   }

   TString incPath = gSystem->GetIncludePath(); // of the form -Idir1  -Idir2 -Idir3
   incPath.Append(":").Prepend(" ");
   incPath.ReplaceAll(" -I",":");       // of form :dir1 :dir2:dir3
   while ( incPath.Index(" :") != -1 ) {
      incPath.ReplaceAll(" :",":");
   }
   incPath.Prepend(".:");
# ifdef CINTINCDIR
   TString cintdir = CINTINCDIR;
# else
   TString cintdir = "$(ROOTSYS)/cint";
# endif
   incPath.Append(":");
   incPath.Append(cintdir);
   incPath.Append("/include:");
   incPath.Append(cintdir);
   incPath.Append("/stl");
   next = gSystem->Which(incPath, filename, kReadPermission);
   if (next) {
      file.Init(next);
      delete [] next;
      if (file.IsValid()) { return kTRUE; };
   }

   next = gSystem->DynamicPathName(filename,kTRUE);
   if (next) {
      file.Init(next);
      delete [] next;
      if (file.IsValid()) { return kTRUE; };
   }

   return kFALSE;
}

//______________________________________________________________________________
Int_t TCint::Load(const char *filename, Bool_t system)
{
   // Load a library file in CINT's memory.
   // if 'system' is true, the library is never unloaded.

   R__LOCKGUARD2(gCINTMutex);

   int i;
   if (!system)
      i = G__loadfile(filename);
   else
      i = G__loadsystemfile(filename);

   UpdateListOfTypes();

   return i;
}

//______________________________________________________________________________
void TCint::LoadMacro(const char *filename, EErrorCode *error)
{
   // Load a macro file in CINT's memory.

   ProcessLine(Form(".L %s", filename), error);
}

//______________________________________________________________________________
Long_t TCint::ProcessLine(const char *line, EErrorCode *error)
{
   // Let CINT process a command line.
   // If the command is executed and the result of G__process_cmd is 0,
   // the return value is the int value corresponding to the result of the command
   // (float and double return values will be truncated).

   Long_t ret = 0;
   // Store our line. line is a static buffer in TApplication
   // and we get called recursively through G__process_cmd.
   TString sLine(line);

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

         G__value local_res;
         G__setnull(&local_res);

         // It checks whether the input line contains the "fantom" method
         // to synchronize user keyboard input and ROOT prompt line
         if (sLine == fantomline) {
            G__free_tempobject();
            TCint::UpdateAllCanvases();
         } else {
            int local_error = 0;

            int prerun = G__getPrerun();
            G__setPrerun(0);
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
               ret = G__process_cmd((char *)sLineNoArgs.Data(), fPrompt, &fMore, &local_error, &local_res);
               sLine[1] = haveX;
            }
            TString aclicMode;
            TString arguments;
            TString io;
            TString fname;
            if (!strncmp(sLine.Data(), ".L", 2)) { // Load cmd, check for use of ACLiC.
               fname = gSystem->SplitAclicMode(sLine.Data()+3, aclicMode, arguments, io);
            }
            if (aclicMode.Length()) { // ACLiC, pass to cint and not to cling.
               ret = G__process_cmd((char *)sLine.Data(), fPrompt, &fMore, &local_error, &local_res);
            }
            else {
               if (true /*sLine != "#include <iostream>"*/) { 
                  // cling cannot parse <iostream> - REALLY?
                  if (fMetaProcessor->process(sLine) > 0) {
                     printf("...\n");
                  }
                  else
                  {
                     #ifdef USE_CLR
                        clang::CompilerInstance * CI = fInterpreter->getCI ();
                        clang::TranslationUnitDecl* tu = CI->getASTContext().getTranslationUnitDecl();
                        ClrScan (& CI->getASTContext(), tu);
                     #endif
                  }
               }
            }
            G__setPrerun(prerun);
            if (local_error == 0 && G__get_return(&fExitCode) == G__RETURN_EXIT2) {
               ResetGlobals();
               gApplication->Terminate(fExitCode);
            }
            if (error)
               *error = (EErrorCode)local_error;
         }

         if (ret == 0) ret = G__int_cast(local_res);

         gROOT->SetLineHasBeenProcessed();
      } else {
         ret = ProcessLineAsynch(sLine, error);
      }
   } else {
      if (gGlobalMutex && !gCINTMutex && fLockProcessLine) {
         gGlobalMutex->Lock();
         if (!gCINTMutex)
            gCINTMutex = gGlobalMutex->Factory(kTRUE);
         gGlobalMutex->UnLock();
      }
      R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);
      gROOT->SetLineIsProcessing();

      G__value local_res;
      G__setnull(&local_res);

      int local_error = 0;

      int prerun = G__getPrerun();
      G__setPrerun(0);
      if ((sLine[0] == '#') || (sLine[0] == '.')) {
         // Preprocessor or special cmd, have cint process it too.
         ret = G__process_cmd((char *)sLine.Data(), fPrompt, &fMore, &local_error, &local_res);
      }
      TString aclicMode;
      TString arguments;
      TString io;
      TString fname;
      if (!strncmp(sLine, ".L", 2)) { // Load cmd, check for use of ACLiC.
         fname = gSystem->SplitAclicMode(sLine.Data()+3, aclicMode, arguments, io);
      }
      if (aclicMode.Length()) { // ACLiC, pass to cint and not to cling.
         ret = G__process_cmd((char *)sLine.Data(), fPrompt, &fMore, &local_error, &local_res);
      }
      else {
         if (fMetaProcessor->process(sLine) > 0) {
            printf("...\n");
         }
      }
      G__setPrerun(prerun);
      if (local_error == 0 && G__get_return(&fExitCode) == G__RETURN_EXIT2) {
         ResetGlobals();
         exit(fExitCode);
      }
      if (error)
         *error = (EErrorCode)local_error;

      if (ret == 0) ret = G__int_cast(local_res);

      gROOT->SetLineHasBeenProcessed();
   }
   return ret;
}

//______________________________________________________________________________
Long_t TCint::ProcessLineAsynch(const char *line, EErrorCode *error)
{
   // Let CINT process a command line asynch.

   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCint::ProcessLineSynch(const char *line, EErrorCode *error)
{
   // Let CINT process a command line synchronously, i.e we are waiting
   // it will be finished.

   R__LOCKGUARD(fLockProcessLine ? gCINTMutex : 0);

   if (gApplication) {
      if (gApplication->IsCmdThread())
         return ProcessLine(line, error);
      return 0;
   }
   return ProcessLine(line, error);
}

//______________________________________________________________________________
Long_t TCint::Calc(const char *line, EErrorCode *error)
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
   result = (Long_t) G__int_cast(G__calc((char *)line));
   if (error) *error = (EErrorCode)G__lasterror();

#ifdef R__WIN32
   if (gApplication && gApplication->GetApplicationImp())
      gROOT->SetLineHasBeenProcessed();
#endif

   return result;
}

//______________________________________________________________________________
void TCint::PrintIntro()
{
   // Print CINT introduction and help message.

   Printf("\nCINT/ROOT C/C++ Interpreter version %s", G__cint_version());
   Printf(">>>>>>>>> cling-ified version <<<<<<<<<<");
   Printf("Type ? for help. Commands must be C++ statements.");
   Printf("Enclose multiple statements between { }.");
}

//______________________________________________________________________________
void TCint::SetGetline(const char*(*getlineFunc)(const char* prompt),
		       void (*histaddFunc)(const char* line))
{
   // Set a getline function to call when input is needed.
   G__SetGetlineFunc(getlineFunc, histaddFunc);
}


//______________________________________________________________________________
void TCint::RecursiveRemove(TObject *obj)
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
void TCint::Reset()
{
   // Reset the CINT state to the state saved by the last call to
   // TCint::SaveContext().

   R__LOCKGUARD(gCINTMutex);

   G__scratch_upto(&fDictPos);
}

//______________________________________________________________________________
void TCint::ResetAll()
{
   // Reset the CINT state to its initial state.

   R__LOCKGUARD(gCINTMutex);

   G__init_cint("cint +V");
   G__init_process_cmd();
}

//______________________________________________________________________________
void TCint::ResetGlobals()
{
   // Reset the CINT global object state to the state saved by the last
   // call to TCint::SaveGlobalsContext().

   R__LOCKGUARD(gCINTMutex);

   G__scratch_globals_upto(&fDictPosGlobals);
}

//______________________________________________________________________________
void TCint::ResetGlobalVar(void *obj)
{
   // Reset the CINT global object state to the state saved by the last
   // call to TCint::SaveGlobalsContext().

   R__LOCKGUARD(gCINTMutex);

   G__resetglobalvar(obj);
}

//______________________________________________________________________________
void TCint::RewindDictionary()
{
   // Rewind CINT dictionary to the point where it was before executing
   // the current macro. This function is typically called after SEGV or
   // ctlr-C after doing a longjmp back to the prompt.

   R__LOCKGUARD(gCINTMutex);

   G__rewinddictionary();
}

//______________________________________________________________________________
Int_t TCint::DeleteGlobal(void *obj)
{
   // Delete obj from CINT symbol table so it cannot be accessed anymore.
   // Returns 1 in case of success and 0 in case object was not in table.

   R__LOCKGUARD(gCINTMutex);

   return G__deleteglobal(obj);
}

//______________________________________________________________________________
void TCint::SaveContext()
{
   // Save the current CINT state.

   R__LOCKGUARD(gCINTMutex);

   G__store_dictposition(&fDictPos);
}

//______________________________________________________________________________
void TCint::SaveGlobalsContext()
{
   // Save the current CINT state of global objects.
   R__LOCKGUARD(gCINTMutex);

   G__store_dictposition(&fDictPosGlobals);
}

//______________________________________________________________________________
void TCint::UpdateListOfGlobals()
{
   // Update the list of pointers to global variables. This function
   // is called by TROOT::GetListOfGlobals().

   R__LOCKGUARD2(gCINTMutex);

   G__DataMemberInfo t, *a;
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         // first remove if already in list
         TGlobal *g = (TGlobal *)gROOT->fGlobals->FindObject(t.Name());
         if (g) {
            gROOT->fGlobals->Remove(g);
            delete g;
         }
         a = new G__DataMemberInfo(t);
         gROOT->fGlobals->Add(new TGlobal(a));
      }
   }
}

//______________________________________________________________________________
void TCint::UpdateListOfGlobalFunctions()
{
   // Update the list of pointers to global functions. This function
   // is called by TROOT::GetListOfGlobalFunctions().

   R__LOCKGUARD2(gCINTMutex);

   G__MethodInfo t, *a;
   void* vt =0;

   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
         Bool_t needToAdd = kTRUE;
         // first remove if already in list
         TList* listFuncs = ((THashTable*)(gROOT->fGlobalFunctions))->GetListForObject(t.Name());
         if (listFuncs && (vt = (void*)t.InterfaceMethod())) {
            Long_t prop = -1;
            TIter iFunc(listFuncs);
            TFunction* f = 0;
            Bool_t foundStart = kFALSE;
            while (needToAdd && (f = (TFunction*)iFunc())) {
               if (strcmp(f->GetName(),t.Name())) {
                  if (foundStart) break;
                  continue;
               }
               foundStart = kTRUE;
               if (vt == f->InterfaceMethod()) {
                  if (prop == -1)
                     prop = t.Property();
                  needToAdd = !((prop & G__BIT_ISCOMPILED)
                                || t.GetMangledName() == f->GetMangledName());
               }
            }
         }

         if (needToAdd) {
            a = new G__MethodInfo(t);
            gROOT->fGlobalFunctions->Add(new TFunction(a));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::UpdateListOfTypes()
{
   // Update the list of pointers to Datatype (typedef) definitions. This
   // function is called by TROOT::GetListOfTypes().

   R__LOCKGUARD2(gCINTMutex);

   // Remember the index of the last type that we looked at,
   // so that we don't keep reprocessing the same types.
   static int last_typenum = -1;

   // Also remember the count from the last time the dictionary
   // was rewound.  If it's been rewound since the last time we've
   // been called, then we recan everything.
   static int last_scratch_count = 0;
   int this_scratch_count = G__scratch_upto(0);
   if (this_scratch_count != last_scratch_count) {
      last_scratch_count = this_scratch_count;
      last_typenum = -1;
   }

   // Scan from where we left off last time.
   G__TypedefInfo t (last_typenum);
   while (t.Next()) {
      const char* name = t.Name();
      if (gROOT && gROOT->fTypes && t.IsValid() && name) {
         TDataType *d = (TDataType *)gROOT->fTypes->FindObject(name);
         // only add new types, don't delete old ones with the same name
         // (as is done in UpdateListOfGlobals()),
         // this 'feature' is being used in TROOT::GetType().
         if (!d) {
            gROOT->fTypes->Add(new TDataType(new G__TypedefInfo(t)));
         }
         last_typenum = t.Typenum();
      }
   }
}

//______________________________________________________________________________
void TCint::SetClassInfo(TClass *cl, Bool_t reload)
{
   // Set pointer to CINT's G__ClassInfo in TClass.

   R__LOCKGUARD2(gCINTMutex);

   if (!cl->fClassInfo || reload) {

      if (IsCINTClassInfo(cl->fClassInfo)) {
         delete (G__ClassInfo*)cl->fClassInfo;
      }
      cl->fClassInfo = 0;
      if (CheckClassInfoInternal(cl->GetName()) == kCINT) {

         G__ClassInfo *info = new G__ClassInfo(cl->GetName());
         cl->fClassInfo = info;


         Bool_t zombieCandidate = kFALSE;

         // In case a class contains an external enum, the enum will be seen as a
         // class. We must detect this special case and make the class a Zombie.
         // Here we assume that a class has at least one method.
         // We can NOT call TClass::Property from here, because this method
         // assumes that the TClass is well formed to do a lot of information
         // caching. The method SetClassInfo (i.e. here) is usually called during
         // the building phase of the TClass, hence it is NOT well formed yet.
         if (info->IsValid() &&
             !(info->Property() & (kIsClass|kIsStruct|kIsNamespace))) {
            zombieCandidate = kTRUE; // cl->MakeZombie();
         }

         if (!info->IsLoaded()) {
            if (info->Property() & (kIsNamespace)) {
               // Namespace can have a ClassInfo but no CINT dictionary per se
               // because they are auto-created if one of their contained
               // classes has a dictionary.
               zombieCandidate = kTRUE; // cl->MakeZombie();
            }

            // this happens when no CINT dictionary is available
            delete info;
            cl->fClassInfo = 0;
         }

         if (zombieCandidate && !TClassEdit::IsSTLCont(cl->GetName())) {
            cl->MakeZombie();
         }

      }
   }
}

//______________________________________________________________________________
Bool_t TCint::CheckClassInfo(const char *name, Bool_t autoload /*= kTRUE*/)
{
   // See CheckClassInfoInternal().
   return CheckClassInfoInternal(name, autoload) != kInvalidInterpreter;
}

//______________________________________________________________________________
TCint::EInterpreterKind TCint::CheckClassInfoInternal(const char *name, Bool_t autoload /*= kTRUE*/)
{
   // Checks if a class with the specified name is defined in CINT.
   // Returns kInvalidInterpreter if class is not defined.
   // Otherwise it returns the interpreter kind that the type is defined in.

   // In the case where the class is not loaded and belongs to a namespace
   // or is nested, looking for the full class name is outputing a lots of
   // (expected) error messages.  Currently the only way to avoid this is to
   // specifically check that each level of nesting is already loaded.
   // In case of templates the idea is that everything between the outer
   // '<' and '>' has to be skipped, e.g.: aap<pipo<noot>::klaas>::a_class

#if defined(R__BUILDING_CINT7) || defined(R__BUILDING_ONLYCINT7)
   if (Reflex::Instance::HasShutdown()) {
      return kFALSE;
   }
#endif

   R__LOCKGUARD(gCINTMutex);

   char *classname = new char[strlen(name)*2];
   strcpy(classname,name);

   char *current = classname;
   while (*current) {

      while (*current && *current != ':' && *current != '<')
         current++;

      if (!*current) break;

      if (*current == '<') {
         int level = 1;
         current++;
         while (*current && level > 0) {
            if (*current == '<') level++;
            if (*current == '>') level--;
            current++;
         }
         continue;
      }

      // *current == ':', must be a "::"
      if (*(current+1) != ':') {
         Error("CheckClassInfo", "unexpected token : in %s", classname);
         delete [] classname;
         return kInvalidInterpreter;
      }

      *current = '\0';
      G__ClassInfo info(classname);
      if (!info.IsValid()) {
         delete [] classname;
         return kInvalidInterpreter;
      }
      *current = ':';
      current += 2;
   }
   strcpy(classname,name);

   int flag = 2;
   if (!autoload) {
      flag = 3;
   }
   Int_t tagnum = G__defined_tagname(classname, flag); // This function might modify the name (to add space between >>).
   if (tagnum >= 0) {
      G__ClassInfo info(tagnum);
      // If autoloading is off then Property() == 0 for autoload entries.
      if (!autoload && !info.Property())
         return kCINT;
      if (info.Property() & (G__BIT_ISENUM | G__BIT_ISCLASS | G__BIT_ISSTRUCT | G__BIT_ISUNION | G__BIT_ISNAMESPACE)) {
         // We are now sure that the entry is not in fact an autoload entry.
         delete [] classname;
         return kCINT;
      }
   }
   G__TypedefInfo t(name);
   if (t.IsValid() && !(t.Property()&G__BIT_ISFUNDAMENTAL)) {
      delete [] classname;
      return kCINT;
   }

   delete [] classname;
   return kInvalidInterpreter;
}

//______________________________________________________________________________
void TCint::CreateListOfBaseClasses(TClass *cl)
{
   // Create list of pointers to base class(es) for TClass cl.

   R__LOCKGUARD2(gCINTMutex);

   if (!cl->fBase) {

      cl->fBase = new TList;

      G__BaseClassInfo t(*(G__ClassInfo *)cl->GetClassInfo()), *a;
      while (t.Next()) {
         // if name cannot be obtained no use to put in list
         if (t.IsValid() && t.Name()) {
            a = new G__BaseClassInfo(t);
            cl->fBase->Add(new TBaseClass(a, cl));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::CreateListOfDataMembers(TClass *cl)
{
   // Create list of pointers to data members for TClass cl.

   R__LOCKGUARD2(gCINTMutex);

   if (!cl->fData) {

      cl->fData = new TList;

      G__DataMemberInfo t(*(G__ClassInfo*)cl->GetClassInfo()), *a;
      while (t.Next()) {
         // if name cannot be obtained no use to put in list
         if (t.IsValid() && t.Name() && strcmp(t.Name(), "G__virtualinfo")) {
            a = new G__DataMemberInfo(t);
            cl->fData->Add(new TDataMember(a, cl));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::CreateListOfMethods(TClass *cl)
{
   // Create list of pointers to methods for TClass cl.

   R__LOCKGUARD2(gCINTMutex);

   if (!cl->fMethod) {

      cl->fMethod = new THashList;

      G__MethodInfo *a;
      G__MethodInfo t(*(G__ClassInfo*)cl->GetClassInfo());
      while (t.Next()) {
         // if name cannot be obtained no use to put in list
         if (t.IsValid() && t.Name()) {
            a = new G__MethodInfo(t);
            cl->fMethod->Add(new TMethod(a, cl));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::UpdateListOfMethods(TClass *cl)
{
   // Update the list of pointers to method for TClass cl, if necessary

   if (cl->fMethod) {
      R__LOCKGUARD2(gCINTMutex);

      G__ClassInfo *info = (G__ClassInfo*)cl->GetClassInfo();
      if (!info || cl->fMethod->GetEntries() == info->NMethods()) {
         return;
      }
      delete cl->fMethod;
      cl->fMethod = 0;
   }
   CreateListOfMethods(cl);
}

//______________________________________________________________________________
void TCint::CreateListOfMethodArgs(TFunction *m)
{
   // Create list of pointers to method arguments for TMethod m.

   R__LOCKGUARD2(gCINTMutex);

   if (!m->fMethodArgs) {

      m->fMethodArgs = new TList;

      G__MethodArgInfo t(*(G__MethodInfo *)m->fInfo), *a;
      while (t.Next()) {
         // if type cannot be obtained no use to put in list
         if (t.IsValid() && t.Type()) {
            a = new G__MethodArgInfo(t);
            m->fMethodArgs->Add(new TMethodArg(a, m));
         }
      }
   }
}


//______________________________________________________________________________
Int_t TCint::GenerateDictionary(const char *classes, const char *includes /* = 0 */, const char * /* options  = 0 */)
{
   // Generate the dictionary for the C++ classes listed in the first
   // argmument (in a semi-colon separated list).
   // 'includes' contains a semi-colon separated list of file to 
   // #include in the dictionary.  
   // For example:
   //    gInterpreter->GenerateDictionary("vector<vector<float> >;list<vector<float> >","list;vector");
   // or
   //    gInterpreter->GenerateDictionary("myclass","myclass.h;myhelper.h");
   
   if (classes == 0 || classes[0] == 0) return 0;
   
   // Split the input list
   std::vector<std::string> listClasses;
   for(const char *current = classes, *prev = classes; *current != 0; ++current) {
      if (*current == ';') {
         listClasses.push_back( std::string(prev,current-prev) );
         prev = current+1;
      } else if (*(current+1) == 0) {
         listClasses.push_back( std::string(prev,current+1-prev) );
         prev = current+1;
      }
   }
   std::vector<std::string> listIncludes;
   for(const char *current = includes, *prev = includes; *current != 0; ++current) {
      if (*current == ';') {
         listIncludes.push_back( std::string(prev,current-prev) );
         prev = current+1;
      } else if (*(current+1) == 0) {
         listIncludes.push_back( std::string(prev,current+1-prev) );
         prev = current+1;
      }
   }
   
   // Generate the temporary dictionary file
   return TCint_GenerateDictionary(listClasses,listIncludes, std::vector<std::string>(), std::vector<std::string>());
}


//______________________________________________________________________________
TString TCint::GetMangledName(TClass *cl, const char *method,
                             const char *params)
{
   // Return the CINT mangled name for a method of a class with parameters
   // params (params is a string of actual arguments, not formal ones). If the
   // class is 0 the global function list will be searched.

   R__LOCKGUARD2(gCINTMutex);

   G__CallFunc  func;
   Long_t       offset;

   if (cl)
      func.SetFunc((G__ClassInfo*)cl->GetClassInfo(), method, params, &offset);
   else {
      G__ClassInfo gcl;   // default G__ClassInfo is global environment
      func.SetFunc(&gcl, method, params, &offset);
   }
   return func.GetMethodInfo().GetMangledName();
}

//______________________________________________________________________________
TString TCint::GetMangledNameWithPrototype(TClass *cl, const char *method,
                                           const char *proto)
{
   // Return the CINT mangled name for a method of a class with a certain
   // prototype, i.e. "char*,int,float". If the class is 0 the global function
   // list will be searched.

   R__LOCKGUARD2(gCINTMutex);

   Long_t             offset;

   if (cl)
      return ((G__ClassInfo*)cl->GetClassInfo())->GetMethod(method, proto, &offset).GetMangledName();
   G__ClassInfo gcl;   // default G__ClassInfo is global environment
   return gcl.GetMethod(method, proto, &offset).GetMangledName();
}

//______________________________________________________________________________
void *TCint::GetInterfaceMethod(TClass *cl, const char *method,
                                const char *params)
{
   // Return pointer to CINT interface function for a method of a class with
   // parameters params (params is a string of actual arguments, not formal
   // ones). If the class is 0 the global function list will be searched.

   R__LOCKGUARD2(gCINTMutex);

   G__CallFunc  func;
   Long_t       offset;

   if (cl)
      func.SetFunc((G__ClassInfo*)cl->GetClassInfo(), method, params, &offset);
   else {
      G__ClassInfo gcl;   // default G__ClassInfo is global environment
      func.SetFunc(&gcl, method, params, &offset);
   }
   return (void *)func.InterfaceMethod();
}

//______________________________________________________________________________
void *TCint::GetInterfaceMethodWithPrototype(TClass *cl, const char *method,
                                             const char *proto)
{
   // Return pointer to CINT interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.

   R__LOCKGUARD2(gCINTMutex);

   G__InterfaceMethod f;
   Long_t             offset;

   if (cl)
      f = ((G__ClassInfo*)cl->GetClassInfo())->GetMethod(method, proto, &offset).InterfaceMethod();
   else {
      G__ClassInfo gcl;   // default G__ClassInfo is global environment
      f = gcl.GetMethod(method, proto, &offset).InterfaceMethod();
   }
   return (void *)f;
}

//______________________________________________________________________________
const char *TCint::GetInterpreterTypeName(const char *name, Bool_t full)
{
   // The 'name' is known to the interpreter, this function returns
   // the internal version of this name (usually just resolving typedefs)
   // This is used in particular to synchronize between the name used
   // by rootcint and by the run-time enviroment (TClass)
   // Return 0 if the name is not known.

   R__LOCKGUARD(gCINTMutex);

   if (!gInterpreter->CheckClassInfo(name)) return 0;
   G__ClassInfo cl(name);
   if (cl.IsValid()) {
      if (full) return cl.Fullname();
      else return cl.Name();
   }
   else return 0;
}

//______________________________________________________________________________
void TCint::Execute(const char *function, const char *params, int *error)
{
   // Execute a global function with arguments params.

   R__LOCKGUARD2(gCINTMutex);

   G__CallFunc  func;
   G__ClassInfo cl;
   Long_t       offset;

   // set pointer to interface method and arguments
   func.SetFunc(&cl, function, params, &offset);

   // call function
   func.Exec(0);
   if (error) *error = G__lasterror();
}

//______________________________________________________________________________
void TCint::Execute(TObject *obj, TClass *cl, const char *method,
                    const char *params, int *error)
{
   // Execute a method from class cl with arguments params.

   R__LOCKGUARD2(gCINTMutex);

   void       *address;
   Long_t      offset;
   G__CallFunc func;

   // If the actuall class of this object inherit 2nd (or more) from TObject,
   // 'obj' is unlikely to be the start of the object (as described by IsA()),
   // hence gInterpreter->Execute will improperly correct the offset.

   void *addr = cl->DynamicCast( TObject::Class(), obj, kFALSE);

   // set pointer to interface method and arguments
   func.SetFunc((G__ClassInfo*)cl->GetClassInfo(), method, params, &offset);

   // call function
   address = (void*)((Long_t)addr + offset);
   func.Exec(address);
   if (error) *error = G__lasterror();
}

//______________________________________________________________________________
void TCint::Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params,
                    int *error)
{
   // Execute a method from class cl with the arguments in array params
   // (params[0] ... params[n] = array of TObjString parameters).

   // Convert the TObjArray array of TObjString parameters to a character
   // string of comma separated parameters.
   // The parameters of type 'char' are enclosed in double quotes and all
   // internal quotes are escaped.

   if (!method) {
      Error("Execute","No method was defined");
      return;
   }

   TList *argList = method->GetListOfMethodArgs();

   // Check number of actual parameters against of expected formal ones

   Int_t nparms = argList->LastIndex()+1;
   Int_t argc   = params ? params->LastIndex()+1:0;

   if (nparms != argc) {
      Error("Execute","Wrong number of the parameters");
      return;
   }

   const char *listpar = "";
   TString complete(10);

   if (params)
   {
      // Create a character string of parameters from TObjArray
      TIter next(params);
      for (Int_t i = 0; i < argc; i ++)
      {
         TMethodArg *arg = (TMethodArg *) argList->At( i );
         G__TypeInfo type( arg->GetFullTypeName() );
         TObjString *nxtpar = (TObjString *)next();
         if (i) complete += ',';
         if (strstr( type.TrueName(), "char" )) {
            TString chpar('\"');
            chpar += (nxtpar->String()).ReplaceAll("\"","\\\"");
            // At this point we have to check if string contains \\"
            // and apply some more sophisticated parser. Not implemented yet!
            complete += chpar;
            complete += '\"';
         }
         else
            complete += nxtpar->String();
      }
      listpar = complete.Data();
   }

   Execute(obj, cl, (char *)method->GetName(), (char *)listpar, error);
}

//______________________________________________________________________________
Long_t TCint::ExecuteMacro(const char *filename, EErrorCode *error)
{
   // Execute a CINT macro.

   R__LOCKGUARD(gCINTMutex);

   return TApplication::ExecuteFile(filename, (int*)error);
}

//______________________________________________________________________________
const char *TCint::GetTopLevelMacroName() const
{
   // Return the file name of the current un-included interpreted file.
   // See the documentation for GetCurrentMacroName().

   G__SourceFileInfo srcfile(G__get_ifile()->filenum);
   while (srcfile.IncludedFrom().IsValid())
      srcfile = srcfile.IncludedFrom();

   return srcfile.Name();
}

//______________________________________________________________________________
const char *TCint::GetCurrentMacroName() const
{
   // Return the file name of the currently interpreted file,
   // included or not. Example to illustrate the difference between
   // GetCurrentMacroName() and GetTopLevelMacroName():
   // BEGIN_HTML <!--
   /* -->
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">inclfile.h</span>
      <!--div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;"-->
      <div class="code">
      <pre style="margin:0pt">#include &lt;iostream&gt;
void inclfunc() {
   std::cout &lt;&lt; "In inclfile.h" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCint::GetCurrentMacroName() returns  " &lt;&lt;
      TCint::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCint::GetTopLevelMacroName() returns " &lt;&lt;
      TCint::GetTopLevelMacroName() &lt;&lt; std::endl;
}</pre></div>
      <div style="clear:both"></div>
      <span style="color:#ffffff;background-color:#7777ff;padding-left:0.3em;padding-right:0.3em">mymacro.C</span>
      <div style="border:solid 1px #ffff77;background-color: #ffffdd;float:left;padding:0.5em;margin-bottom:0.7em;">
      <pre style="margin:0pt">#include &lt;iostream&gt;
#include "inclfile.h"
void mymacro() {
   std::cout &lt;&lt; "In mymacro.C" &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCint::GetCurrentMacroName() returns  " &lt;&lt;
      TCint::GetCurrentMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  TCint::GetTopLevelMacroName() returns " &lt;&lt;
      TCint::GetTopLevelMacroName() &lt;&lt; std::endl;
   std::cout &lt;&lt; "  Now calling inclfunc..." &lt;&lt; std::endl;
   inclfunc();
}</pre></div>
<div style="clear:both"></div>
<!-- */
// --> END_HTML
   // Running mymacro.C will print:
   //
   // root [0] .x mymacro.C
   // In mymacro.C
   //   TCint::GetCurrentMacroName() returns  ./mymacro.C
   //   TCint::GetTopLevelMacroName() returns ./mymacro.C
   //   Now calling inclfunc...
   // In inclfile.h
   //   TCint::GetCurrentMacroName() returns  inclfile.h
   //   TCint::GetTopLevelMacroName() returns ./mymacro.C

   return G__get_ifile()->name;
}


//______________________________________________________________________________
const char *TCint::TypeName(const char *typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // You need to use the result immediately before it is being overwritten.

   static char t[1024];
   char *s, *template_start;
   if (!strstr(typeDesc, "(*)(")) {
      s = (char*)strchr(typeDesc, ' ');
      template_start = (char*)strchr(typeDesc, '<');
      if (!strcmp(typeDesc, "long long"))
         strcpy(t, typeDesc);
      else if (!strncmp(typeDesc,"unsigned ",s+1-typeDesc))
         strcpy(t, typeDesc);
      // s is the position of the second 'word' (if any)
      // except in the case of templates where there will be a space
      // just before any closing '>': eg.
      //    TObj<std::vector<UShort_t,__malloc_alloc_template<0> > >*
      else if (s && (template_start==0 || (s < template_start)) )
         strcpy(t, s+1);
      else
         strcpy(t, typeDesc);
   }

   int l = strlen(t);
   while (l > 0 && (t[l-1] == '*' || t[l-1] == '&') ) t[--l] = 0;

   return t;
}

//______________________________________________________________________________
Int_t TCint::LoadLibraryMap(const char *rootmapfile)
{
   // Load map between class and library. If rootmapfile is specified a
   // specific rootmap file can be added (typically used by ACLiC).
   // In case of error -1 is returned, 0 otherwise.
   // Cint uses this information to automatically load the shared library
   // for a class (autoload mechanism).
   // See also the AutoLoadCallback() method below.

   R__LOCKGUARD(gCINTMutex);

   loadClassToLibraryMap();

   // open the [system].rootmap files
   if (!fMapfile) {
      fMapfile = new TEnv(".rootmap");
      fMapfile->IgnoreDuplicates(kTRUE);

      fRootmapFiles = new TObjArray;
      fRootmapFiles->SetOwner();

      // Make sure that this information will be useable by inserting our
      // autoload call back!
      G__set_class_autoloading_callback(&TCint_AutoLoadCallback);
   }

   // Load all rootmap files in the dynamic load path ((DY)LD_LIBRARY_PATH, etc.).
   // A rootmap file must end with the string ".rootmap".
   TString ldpath = gSystem->GetDynamicPath();
   if (ldpath != fRootmapLoadPath) {
      fRootmapLoadPath = ldpath;
#ifdef WIN32
      TObjArray *paths = ldpath.Tokenize(";");
#else
      TObjArray *paths = ldpath.Tokenize(":");
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
            void *dirp = gSystem->OpenDirectory(d);
            if (dirp) {
               if (gDebug > 3)
                  Info("LoadLibraryMap", "%s", d.Data());
               const char *f1;
               while ((f1 = gSystem->GetDirEntry(dirp))) {
                  TString f = f1;
                  if (f.EndsWith(".rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     if (!gSystem->AccessPathName(p, kReadPermission)) {
                        if (gDebug > 4)
                           Info("LoadLibraryMap", "   rootmap file: %s", p.Data());
                        fMapfile->ReadFile(p, kEnvGlobal);
                        fRootmapFiles->Add(new TObjString(p));
                     }
                  }
                  if (f.BeginsWith("rootmap")) {
                     TString p;
                     p = d + "/" + f;
                     FileStat_t stat;
                     if (gSystem->GetPathInfo(p, stat) == 0 && R_ISREG(stat.fMode))
                        Warning("LoadLibraryMap", "please rename %s to end with \".rootmap\"", p.Data());
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
      fRootmapFiles->Add(new TObjString(rootmapfile));
      fMapfile->IgnoreDuplicates(ignore);
   }

   TEnvRec *rec;
   TIter next(fMapfile->GetTable());

   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {

         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") continue;
         TString delim(" ");
         TObjArray *tokens = libs.Tokenize(delim);
         const char *lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0,8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (cls.Contains(":")) {
            // We have a namespace and we have to check it first
            int slen = cls.Length();
            for (int k = 0; k < slen; k++) {
               if (cls[k] == ':') {
                  if (k+1 >= slen || cls[k+1] != ':') {
                     // we expected another ':'
                     break;
                  }
                  if (k) {
                     TString base = cls(0, k);
                     if (base == "std") {
                        // std is not declared but is also ignored by CINT!
                        break;
                     } else {
                        // Only declared the namespace do not specify any library because
                        // the namespace might be spread over several libraries and we do not
                        // know (yet?) which one the user will need!

                        // But what if it's not a namespace but a class?
                        // Does CINT already know it?
                        const char* baselib = G__get_class_autoloading_table((char*)base.Data());
                        if ((!baselib || !baselib[0]) && !rec->FindObject(base))
                           G__set_class_autoloading_table((char*)base.Data(), (char*)"");
                     }
                     ++k;
                  }
               } else if (cls[k] == '<') {
                  // We do not want to look at the namespace inside the template parameters!
                  break;
               }
            }
         }
         G__set_class_autoloading_table((char*)cls.Data(), (char*)lib);
         G__security_recover(stderr); // Ignore any error during this setting.
         if (gDebug > 6) {
            const char *wlib = gSystem->DynamicPathName(lib, kTRUE);
            if (wlib)
               Info("LoadLibraryMap", "class %s in %s", cls.Data(), wlib);
            else
               Info("LoadLibraryMap", "class %s in %s (library does not exist)", cls.Data(), lib);
            delete [] wlib;
         }
         delete tokens;
      }
   }
   return 0;
}

//______________________________________________________________________________
static void loadClassToLibraryMap()
{
   static TEnv* map_file = 0;
   if (!map_file) {
      map_file = new TEnv(".rootmap");
      map_file->IgnoreDuplicates(kTRUE);
   }
   static TString rootmap_load_path;
   TString ldpath = gSystem->GetDynamicPath();
   if (ldpath != rootmap_load_path) {
      rootmap_load_path = ldpath;
#ifdef WIN32
      TObjArray* paths = ldpath.Tokenize(";");
#else // WIN32
      TObjArray* paths = ldpath.Tokenize(":");
#endif // WIN32
      for (int i = 0; i < paths->GetEntriesFast(); ++i) {
         TString d = ((TObjString*)paths->At(i))->GetString();
         bool skip = false;
         for (int j = 0; j < i; ++j) {
            TString pd = ((TObjString*)paths->At(j))->GetString();
            if (pd == d) {
               skip = true;
               break;
            }
         }
         if (skip) {
            continue;
         }
         void* dirp = gSystem->OpenDirectory(d);
         if (dirp) {
            for (const char* f1 = 0; (f1 = gSystem->GetDirEntry(dirp));) {
               TString f = f1;
               if (!f.EndsWith(".rootmap")) {
                  continue;
               }
               TString p;
               p = d + "/" + f;
               if (gSystem->AccessPathName(p, kReadPermission)) {
                  continue;
               }
               map_file->ReadFile(p, kEnvGlobal);
            }
         }
         gSystem->FreeDirectory(dirp);
      }
      delete paths;
   }
   TIter next(map_file->GetTable());
   TEnvRec* rec = 0;
   for (; (rec = (TEnvRec*) next());) {
      TString entry_name = rec->GetName();
      if (
         (entry_name.Length() > 8) &&
         !strncmp(entry_name.Data(), "Library.", 8)
      ) {
         TString entry_value = rec->GetValue();
         if (entry_value == "") {
            continue;
         }
         entry_name.Remove(0, 8);
         entry_name.ReplaceAll("@@", "::");
         entry_name.ReplaceAll("-", " ");
         if (entry_name.Contains(":")) {
            int slen = entry_name.Length();
            for (int i = 0; i < (slen - 1); ++i) {
               if (entry_name[i] == '<') {
                  break;
               }
               if (entry_name[i] != ':') {
                  continue;
               }
               if (entry_name[i+1] != ':') {
                  break;
               }
               if (!i) {
                  continue;
               }
               TString qual_id = entry_name(0, i);
               ++i;
               if (qual_id == "std") {
                  break;
               }
               std::map<std::string, std::string>::iterator iter =
                  class_to_library_map().find(qual_id.Data());
               if (
                  (
                     (iter == class_to_library_map().end()) ||
                     !iter->second.size()
                  ) &&
                  !rec->FindObject(qual_id)
               ) {
                  class_to_library_map()[qual_id.Data()] = "";
               }
            }
         }
         Ssiz_t from = 0;
         TString libname;
         entry_value.Tokenize(libname, from);
         class_to_library_map()[entry_name.Data()] = libname.Data();
      }
   }
}

//______________________________________________________________________________
Int_t TCint::RescanLibraryMap()
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
Int_t TCint::ReloadAllSharedLibraryMaps()
{
   // Reload the library map entries coming from all the loaded shared libraries,
   // after first unloading the current ones.
   // In case of error -1 is returned, 0 otherwise.

   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray *sharedLibL = sharedLibLStr.Tokenize(" ");
   const Int_t nrSharedLibs = sharedLibL->GetEntriesFast();

   for (Int_t ilib = 0; ilib < nrSharedLibs; ilib++) {
      const TString sharedLibStr = ((TObjString *)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      const Int_t ret = UnloadLibraryMap(sharedLibBaseStr);

      if (ret < 0) continue;

      TString rootMapBaseStr = sharedLibBaseStr;
      if (sharedLibBaseStr.EndsWith(".dll"))
         rootMapBaseStr.ReplaceAll(".dll","");
      else if (sharedLibBaseStr.EndsWith(".DLL"))
         rootMapBaseStr.ReplaceAll(".DLL","");
      else if (sharedLibBaseStr.EndsWith(".so"))
         rootMapBaseStr.ReplaceAll(".so","");
      else if (sharedLibBaseStr.EndsWith(".sl"))
         rootMapBaseStr.ReplaceAll(".sl","");
      else if (sharedLibBaseStr.EndsWith(".dl"))
         rootMapBaseStr.ReplaceAll(".dl","");
      else if (sharedLibBaseStr.EndsWith(".a"))
         rootMapBaseStr.ReplaceAll(".a","");
      else {
         Error("ReloadAllSharedLibraryMaps","Unknown library type %s",sharedLibBaseStr.Data());
         delete sharedLibL;
         return -1;
      }
      rootMapBaseStr += ".rootmap";

      const char *rootMap = gSystem->Which(gSystem->GetDynamicPath(),rootMapBaseStr);
      if (!rootMap) {
         Error("ReloadAllSharedLibraryMaps","Could not find rootmap %s in path",rootMap);
         delete [] rootMap;
         delete sharedLibL;
         return -1;
      }

      const Int_t status = LoadLibraryMap(rootMap);
      if (status < 0) {
         Error("ReloadAllSharedLibraryMaps","Error loading map %s",rootMap);
         delete [] rootMap;
         delete sharedLibL;
         return -1;
      }
      delete [] rootMap;
   }
   delete sharedLibL;

   return 0;
}

//______________________________________________________________________________
Int_t TCint::UnloadAllSharedLibraryMaps()
{
   // Unload the library map entries coming from all the loaded shared libraries.
   // Returns 0 if succesful

   const TString sharedLibLStr = GetSharedLibs();
   const TObjArray *sharedLibL = sharedLibLStr.Tokenize(" ");
   for (Int_t ilib = 0; ilib < sharedLibL->GetEntriesFast(); ilib++) {
      const TString sharedLibStr = ((TObjString *)sharedLibL->At(ilib))->GetString();
      const  TString sharedLibBaseStr = gSystem->BaseName(sharedLibStr);
      UnloadLibraryMap(sharedLibBaseStr);
   }
   delete sharedLibL;

   return 0;
}

//______________________________________________________________________________
Int_t TCint::UnloadLibraryMap(const char *library)
{
   // Unload library map entries coming from the specified library.
   // Returns -1 in case no entries for the specified library were found,
   // 0 otherwise.

   if (!fMapfile || !library || !*library)
      return 0;

   TEnvRec *rec;
   TIter next(fMapfile->GetTable());

   R__LOCKGUARD(gCINTMutex);

   Int_t ret = 0;

   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {

         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         if (libs == "") continue;
         TString delim(" ");
         TObjArray *tokens = libs.Tokenize(delim);
         const char *lib = ((TObjString*)tokens->At(0))->GetName();
         // convert "@@" to "::", we used "@@" because TEnv
         // considers "::" a terminator
         cls.Remove(0,8);
         cls.ReplaceAll("@@", "::");
         // convert "-" to " ", since class names may have
         // blanks and TEnv considers a blank a terminator
         cls.ReplaceAll("-", " ");
         if (cls.Contains(":")) {
            // We have a namespace and we have to check it first
            int slen = cls.Length();
            for (int k = 0; k < slen; k++) {
               if (cls[k] == ':') {
                  if (k+1 >= slen || cls[k+1] != ':') {
                     // we expected another ':'
                     break;
                  }
                  if (k) {
                     TString base = cls(0, k);
                     if (base == "std") {
                        // std is not declared but is also ignored by CINT!
                        break;
                     } else {
                        // Only declared the namespace do not specify any library because
                        // the namespace might be spread over several libraries and we do not
                        // know (yet?) which one the user will need!
                        //G__remove_from_class_autoloading_table((char*)base.Data());
                     }
                     ++k;
                  }
               } else if (cls[k] == '<') {
                  // We do not want to look at the namespace inside the template parameters!
                  break;
               }
            }
         }

         if (!strcmp(library, lib)) {
            if (fMapfile->GetTable()->Remove(rec) == 0) {
               Error("UnloadLibraryMap", "entry for <%s,%s> not found in library map table", cls.Data(), lib);
               ret = -1;
            }
            G__set_class_autoloading_table((char*)cls.Data(), (char*)-1);
            G__security_recover(stderr); // Ignore any error during this setting.
         }

         delete tokens;
      }
   }

   return ret;
}

//______________________________________________________________________________
Int_t TCint::AutoLoad(const char *cls)
{
   // Load library containing the specified class. Returns 0 in case of error
   // and 1 in case if success.

   R__LOCKGUARD(gCINTMutex);

   Int_t status = 0;

   if (!gROOT || !gInterpreter) return status;

   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldvalue = G__set_class_autoloading(0);

   // lookup class to find list of dependent libraries
   TString deplibs = gInterpreter->GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray *tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntriesFast()-1; i > 0; i--) {
         const char *deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 0)
               ::Info("TCint::AutoLoad", "loaded dependent library %s for class %s",
                      deplib, cls);
         } else
            ::Error("TCint::AutoLoad", "failure loading dependent library %s for class %s",
                    deplib, cls);
      }
      const char *lib = ((TObjString*)tokens->At(0))->GetName();

      if (lib[0]) {
         if (gROOT->LoadClass(cls, lib) == 0) {
            if (gDebug > 0)
               ::Info("TCint::AutoLoad", "loaded library %s for class %s",
                      lib, cls);
            status = 1;
         } else
            ::Error("TCint::AutoLoad", "failure loading library %s for class %s",
                    lib, cls);
      }
      delete tokens;
   }

   G__set_class_autoloading(oldvalue);
   return status;
}

//______________________________________________________________________________
Int_t TCint::AutoLoadCallback(const char *cls, const char *lib)
{
   // Load library containing specified class. Returns 0 in case of error
   // and 1 in case if success.

   R__LOCKGUARD(gCINTMutex);

   if (!gROOT || !gInterpreter || !cls || !lib) return 0;

   // calls to load libCore might come in the very beginning when libCore
   // dictionary is not fully loaded yet, ignore it since libCore is always
   // loaded
   if (strstr(lib, "libCore")) return 1;

   // lookup class to find list of dependent libraries
   TString deplibs = gInterpreter->GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
     if (gDebug > 0 && gDebug <= 4)
       ::Info("TCint::AutoLoadCallback", "loaded dependent library %s for class %s",
	      deplibs.Data(), cls);
      TString delim(" ");
      TObjArray *tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntriesFast()-1; i > 0; i--) {
         const char *deplib = ((TObjString*)tokens->At(i))->GetName();
         if (gROOT->LoadClass(cls, deplib) == 0) {
            if (gDebug > 4)
               ::Info("TCint::AutoLoadCallback", "loaded dependent library %s for class %s",
                      deplib, cls);
         } else {
            ::Error("TCint::AutoLoadCallback", "failure loading dependent library %s for class %s",
                      deplib, cls);
         }
      }
      delete tokens;
   }
   if (lib[0]) {
      if (gROOT->LoadClass(cls, lib) == 0) {
         if (gDebug > 0)
            ::Info("TCint::AutoLoadCallback", "loaded library %s for class %s",
            lib, cls);
         return 1;
      } else {
         ::Error("TCint::AutoLoadCallback", "failure loading library %s for class %s",
         lib, cls);
      }
   }
   return 0;
}

//______________________________________________________________________________
void *TCint::FindSpecialObject(const char *item, G__ClassInfo *type,
                               void **prevObj, void **assocPtr)
{
   // Static function called by CINT when it finds an un-indentified object.
   // This function tries to find the UO in the ROOT files, directories, etc.
   // This functions has been registered by the TCint ctor.

   if (!*prevObj || *assocPtr != gDirectory) {
      *prevObj = gROOT->FindSpecialObject(item, *assocPtr);
      if (!fgSetOfSpecials) fgSetOfSpecials = new std::set<TObject*>;
      if (*prevObj) ((std::set<TObject*>*)fgSetOfSpecials)->insert((TObject*)*prevObj);
   }

   if (*prevObj) type->Init(((TObject *)*prevObj)->ClassName());
   return *prevObj;
}

//______________________________________________________________________________
// Helper class for UpdateClassInfo
namespace {
   class TInfoNode {
   private:
      string fName;
      Long_t fTagnum;
   public:
      TInfoNode(const char *item, Long_t tagnum)
         : fName(item),fTagnum(tagnum)
      {}
      void Update() {
         TCint::UpdateClassInfoWork(fName.c_str(),fTagnum);
      }
   };
}

//______________________________________________________________________________
void TCint::UpdateClassInfo(char *item, Long_t tagnum)
{
   // Static function called by CINT when it changes the tagnum for
   // a class (e.g. after re-executing the setup function). In such
   // cases we have to update the tagnum in the G__ClassInfo used by
   // the TClass for class "item".

   R__LOCKGUARD(gCINTMutex);

   if (gROOT && gROOT->GetListOfClasses()) {

      static Bool_t entered = kFALSE;
      static vector<TInfoNode> updateList;
      Bool_t topLevel;

      if (entered) topLevel = kFALSE;
      else {
         entered = kTRUE;
         topLevel = kTRUE;
      }
      if (topLevel) {
         UpdateClassInfoWork(item,tagnum);
      } else {
         // If we are called indirectly from within another call to
         // TCint::UpdateClassInfo, we delay the update until the dictionary loading
         // is finished (i.e. when we return to the top level TCint::UpdateClassInfo).
         // This allows for the dictionary to be fully populated when we actually
         // update the TClass object.   The updating of the TClass sometimes
         // (STL containers and when there is an emulated class) forces the building
         // of the TClass object's real data (which needs the dictionary info).
         updateList.push_back(TInfoNode(item,tagnum));
      }
      if (topLevel) {
         while (!updateList.empty()) {
            TInfoNode current( updateList.back() );
            updateList.pop_back();
            current.Update();
         }
         entered = kFALSE;
      }
   }
}

//______________________________________________________________________________
void TCint::UpdateClassInfoWork(const char *item, Long_t tagnum)
{
   // This does the actual work of UpdateClassInfo.

   Bool_t load = kFALSE;
   if (strchr(item,'<') && TClass::GetClassShortTypedefHash()) {
      // We have a template which may have duplicates.

      TString resolvedItem(
       TClassEdit::ResolveTypedef(TClassEdit::ShortType(item,
          TClassEdit::kDropStlDefault).c_str(), kTRUE) );

      if (resolvedItem != item) {
         TClass* cl= (TClass*)gROOT->GetListOfClasses()->FindObject(resolvedItem);
         if (cl)
            load = kTRUE;
      }

      if (!load) {
         TIter next(TClass::GetClassShortTypedefHash()->GetListForObject(resolvedItem));

         while ( TClass::TNameMapNode* htmp =
              static_cast<TClass::TNameMapNode*> (next()) ) {
            if (resolvedItem == htmp->String()) {
               TClass* cl = gROOT->GetClass (htmp->fOrigName, kFALSE);
               if (cl) {
                  // we found at least one equivalent.
                  // let's force a reload
                  load = kTRUE;
                  break;
               }
            }
         }
      }
   }

   TClass *cl = gROOT->GetClass(item, load);
   if (cl) cl->ResetClassInfo(tagnum);
}

//______________________________________________________________________________
void TCint::UpdateAllCanvases()
{
   // Update all canvases at end the terminal input command.

   TIter next(gROOT->GetListOfCanvases());
   TVirtualPad *canvas;
   while ((canvas = (TVirtualPad *)next()))
      canvas->Update();
}

//______________________________________________________________________________
const char* TCint::GetSharedLibs()
{
   // Refresh the list of shared libraries and return it.

   fSharedLibs = "";

   G__SourceFileInfo cursor(0);
   while (cursor.IsValid()) {
      const char *filename = cursor.Name();
      if (filename==0) continue;
      Int_t len = strlen(filename);
      const char *end = filename+len;
      Bool_t needToSkip = kFALSE;
      if ( len>5 && ( (strcmp(end-4,".dll") == 0 ) || (strstr(filename,"Dict.")!=0)  || (strstr(filename,"MetaTCint")!=0)  ) ) {
         // Filter out the cintdlls
         static const char *excludelist [] = {
            "stdfunc.dll","stdcxxfunc.dll","posix.dll","ipc.dll","posix.dll"
            "string.dll","vector.dll","vectorbool.dll","list.dll","deque.dll",
            "map.dll", "map2.dll","set.dll","multimap.dll","multimap2.dll",
            "multiset.dll","stack.dll","queue.dll","valarray.dll",
            "exception.dll","stdexcept.dll","complex.dll","climits.dll",
            "libvectorDict.","libvectorboolDict.","liblistDict.","libdequeDict.",
            "libmapDict.", "libmap2Dict.","libsetDict.","libmultimapDict.","libmultimap2Dict.",
            "libmultisetDict.","libstackDict.","libqueueDict.","libvalarrayDict.",
            "libMetaTCint.","libMetaTCint7."
         };
         static const unsigned int excludelistsize = sizeof(excludelist)/sizeof(excludelist[0]);
         static int excludelen[excludelistsize] = {-1};
         if (excludelen[0] == -1) {
            for (unsigned int i = 0; i < excludelistsize; ++i)
               excludelen[i] = strlen(excludelist[i]);
         }
         const char* basename = gSystem->BaseName(filename);
         for (unsigned int i = 0; !needToSkip && i < excludelistsize; ++i)
            needToSkip = (!strncmp(basename, excludelist[i], excludelen[i]));
      }
      if (!needToSkip &&
           ((len>3 && strcmp(end-2,".a") == 0)    ||
            (len>4 && (strcmp(end-3,".sl") == 0   ||
                       strcmp(end-3,".dl") == 0   ||
                       strcmp(end-3,".so") == 0)) ||
            (len>5 && (strcasecmp(end-4,".dll") == 0)))) {
         if (!fSharedLibs.IsNull())
            fSharedLibs.Append(" ");
         fSharedLibs.Append(filename);
      }

      cursor.Next();
   }

   return fSharedLibs;
}

//______________________________________________________________________________
const char *TCint::GetClassSharedLibs(const char *cls)
{
   // Get the list of shared libraries containing the code for class cls.
   // The first library in the list is the one containing the class, the
   // others are the libraries the first one depends on. Returns 0
   // in case the library is not found.

   if (!cls || !*cls)
      return 0;

   // lookup class to find list of libraries
   if (fMapfile) {
      TString c = TString("Library.") + cls;
      // convert "::" to "@@", we used "@@" because TEnv
      // considers "::" a terminator
      c.ReplaceAll("::", "@@");
      // convert "-" to " ", since class names may have
      // blanks and TEnv considers a blank a terminator
      c.ReplaceAll(" ", "-");
      const char *libs = fMapfile->GetValue(c, "");
      return (*libs) ? libs : 0;
   }
   return 0;
}

//______________________________________________________________________________
const char *TCint::GetSharedLibDeps(const char *lib)
{
   // Get the list a libraries on which the specified lib depends. The
   // returned string contains as first element the lib itself.
   // Returns 0 in case the lib does not exist or does not have
   // any dependencies.

   if (!fMapfile || !lib || !lib[0])
      return 0;

   TString libname(lib);
   Ssiz_t idx = libname.Last('.');
   if (idx != kNPOS) {
      libname.Remove(idx);
   }
   TEnvRec *rec;
   TIter next(fMapfile->GetTable());

   size_t len = libname.Length();
   while ((rec = (TEnvRec*) next())) {
      const char *libs = rec->GetValue();
      if (!strncmp(libs, libname.Data(), len) && strlen(libs) >= len
          && (!libs[len] || libs[len] == ' ' || libs[len] == '.')) {
         return libs;
      }
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TCint::IsErrorMessagesEnabled() const
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout.

   return !G__const_whatnoerror();
}

//______________________________________________________________________________
Bool_t TCint::SetErrorMessages(Bool_t enable)
{
   // If error messages are disabled, the interpreter should suppress its
   // failures and warning messages from stdout. Return the previous state.

   if (enable)
      G__const_resetnoerror();
   else
      G__const_setnoerror();
   return !G__const_whatnoerror();
}

//______________________________________________________________________________
void TCint::AddIncludePath(const char *path)
{
   // Add the given path to the list of directories in which the interpreter
   // looks for include files. Only one path item can be specified at a
   // time, i.e. "path1:path2" is not supported.

   R__LOCKGUARD(gCINTMutex);

   char *incpath = gSystem->ExpandPathName(path);

   G__add_ipath(incpath);
   fInterpreter->AddIncludePath(path);
   
   delete [] incpath;
}

//______________________________________________________________________________
const char *TCint::GetIncludePath()
{
   // Refresh the list of include paths known to the interpreter and return it
   // with -I prepended.

   R__LOCKGUARD(gCINTMutex);

   fIncludePath = "";

   G__IncludePathInfo path;

   while (path.Next()) {
      const char *pathname = path.Name();
      fIncludePath.Append(" -I\"").Append(pathname).Append("\" ");
   }
#ifdef R__BUILDING_CINT7
# ifdef ROOTINCDIR
   fIncludePath.Append(" -I\"").Append(ROOTINCDIR);
# else
   fIncludePath.Append(" -I\"").Append(gRootDir).Append("/include");
# endif
   fIncludePath.Append("/cint7\" ");
#endif

   return fIncludePath;
}

//______________________________________________________________________________
const char *TCint::GetSTLIncludePath() const
{
   // Return the directory containing CINT's stl cintdlls.
   static TString stldir;
   if (!stldir.Length()) {
#ifdef CINTINCDIR
      stldir = CINTINCDIR;
#else
      stldir = gRootDir; stldir += "/cint";
#endif
      if (!stldir.EndsWith("/"))
         stldir += '/';
#if defined(R__BUILDING_CINT7) || (R__BUILDING_ONLYCINT7)
      stldir += "cint7/stl";
#else
      // Default to Cint5's directory
      stldir += "cint/stl";
#endif
   }
   return stldir;
}

//______________________________________________________________________________
//                      M I S C
//______________________________________________________________________________

int TCint::DisplayClass(FILE *fout,char *name,int base,int start) const
{
   // Interface to CINT function

   return G__display_class(fout,name,base,start);
}
//______________________________________________________________________________
int TCint::DisplayIncludePath(FILE *fout) const
{
   // Interface to CINT function

   return G__display_includepath(fout);
}
//______________________________________________________________________________
void  *TCint::FindSym(const char *entry) const
{
   // Interface to CINT function

   return G__findsym(entry);
}
//______________________________________________________________________________
void   TCint::GenericError(const char *error) const
{
   // Interface to CINT function
   G__genericerror(error);
}
//______________________________________________________________________________
Long_t TCint::GetExecByteCode() const
{
   // Interface to CINT function

   return (Long_t)G__exec_bytecode;
}

//______________________________________________________________________________
Long_t TCint::Getgvp() const
{
   // Interface to CINT function

   return (Long_t)G__getgvp();
}
//______________________________________________________________________________
const char *TCint::Getp2f2funcname(void *receiver) const
{
   // Interface to CINT function

   return G__p2f2funcname(receiver);
}
//______________________________________________________________________________
int    TCint::GetSecurityError() const
{
   // Interface to CINT function

   return G__get_security_error();
}
//______________________________________________________________________________
int    TCint::LoadFile(const char *path) const
{
   // Interface to CINT function

   return G__loadfile(path);
}
//______________________________________________________________________________
void   TCint::LoadText(const char *text) const
{
   // Interface to CINT function

   G__load_text(text);
}
//______________________________________________________________________________
const char *TCint::MapCppName(const char *name) const
{
   // Interface to CINT function

   return G__map_cpp_name(name);
}
//______________________________________________________________________________
void   TCint::SetAlloclockfunc(void (*p)()) const
{
   // Interface to CINT function

   G__set_alloclockfunc(p);
}
//______________________________________________________________________________
void   TCint::SetAllocunlockfunc(void (*p)()) const
{
   // Interface to CINT function

   G__set_allocunlockfunc(p);
}
//______________________________________________________________________________
int    TCint::SetClassAutoloading(int autoload) const
{
   // Interface to CINT function

   return G__set_class_autoloading(autoload);
}
//______________________________________________________________________________
void   TCint::SetErrmsgcallback(void *p) const
{
   // Interface to CINT function

   G__set_errmsgcallback(p);
}
//______________________________________________________________________________
void   TCint::Setgvp(Long_t gvp) const
{
   // Interface to CINT function

   G__setgvp(gvp);
}
//______________________________________________________________________________
void   TCint::SetRTLD_NOW() const
{
   // Interface to CINT function

   G__Set_RTLD_NOW();
}
//______________________________________________________________________________
void   TCint::SetRTLD_LAZY() const
{
   // Interface to CINT function

   G__Set_RTLD_LAZY();
}
//______________________________________________________________________________
void   TCint::SetTempLevel(int val) const
{
   // Interface to CINT function

   G__settemplevel(val);
}
//______________________________________________________________________________
int    TCint::UnloadFile(const char *path) const
{
   // Interface to CINT function

   return G__unloadfile(path);
}



//______________________________________________________________________________
// G__CallFunc interface
//______________________________________________________________________________
void  TCint::CallFunc_Delete(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   delete f;
}
//______________________________________________________________________________
void  TCint::CallFunc_Exec(CallFunc_t *func, void *address) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->Exec(address);
}
//______________________________________________________________________________
Long_t  TCint::CallFunc_ExecInt(CallFunc_t *func, void *address) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   return f->ExecInt(address);
}
//______________________________________________________________________________
Long_t  TCint::CallFunc_ExecInt64(CallFunc_t *func, void *address) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   return f->ExecInt64(address);
}
//______________________________________________________________________________
Double_t  TCint::CallFunc_ExecDouble(CallFunc_t *func, void *address) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   return f->ExecDouble(address);
}
//______________________________________________________________________________
CallFunc_t *TCint::CallFunc_Factory() const
{
   // Interface to CINT function

   G__CallFunc *f = new G__CallFunc();
   return f;
}
//______________________________________________________________________________
CallFunc_t *TCint::CallFunc_FactoryCopy(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f1 = (G__CallFunc*)func;
   G__CallFunc *f  = new G__CallFunc(*f1);
   return f;
}
//______________________________________________________________________________
MethodInfo_t *TCint::CallFunc_FactoryMethod(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   G__MethodInfo *info = new G__MethodInfo((*f).GetMethodInfo());
   return info;
}
//______________________________________________________________________________
void  TCint::CallFunc_Init(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->Init();
}
//______________________________________________________________________________
bool  TCint::CallFunc_IsValid(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   return f->IsValid();
}
//______________________________________________________________________________
void  TCint::CallFunc_ResetArg(CallFunc_t *func) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->ResetArg();
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArg(CallFunc_t *func, Long_t param) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArg(param);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArg(CallFunc_t *func, Double_t param) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArg(param);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArg(CallFunc_t *func, Long64_t param) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArg(param);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArg(CallFunc_t *func, ULong64_t param) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArg(param);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArgArray(CallFunc_t *func, Long_t *paramArr, Int_t nparam) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArgArray(paramArr,nparam);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetArgs(CallFunc_t *func, const char *param) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetArgs(param);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetFunc(CallFunc_t *func, ClassInfo_t *info, const char *method, const char *params, Long_t *offset) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   f->SetFunc((G__ClassInfo*)info,method,params,offset);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetFunc(CallFunc_t *func, MethodInfo_t *info) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   G__MethodInfo *minfo = (G__MethodInfo*)info;
   f->SetFunc(*minfo);
}
//______________________________________________________________________________
void  TCint::CallFunc_SetFuncProto(CallFunc_t *func, ClassInfo_t *info, const char *method, const char *proto, Long_t *offset) const
{
   // Interface to CINT function

   G__CallFunc *f = (G__CallFunc*)func;
   G__ClassInfo *cinfo = (G__ClassInfo*)info;
   f->SetFuncProto(cinfo,method,proto,offset);
}



//______________________________________________________________________________
// G__ClassInfo interface
//______________________________________________________________________________
Long_t   TCint::ClassInfo_ClassProperty(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->ClassProperty();
}
//______________________________________________________________________________
void  TCint::ClassInfo_Delete(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   delete info;
}
//______________________________________________________________________________
void  TCint::ClassInfo_Delete(ClassInfo_t *cinfo, void *arena) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   info->Delete(arena);
}
//______________________________________________________________________________
void  TCint::ClassInfo_DeleteArray(ClassInfo_t *cinfo, void *arena, bool dtorOnly) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   info->DeleteArray(arena,dtorOnly);
}
//______________________________________________________________________________
void  TCint::ClassInfo_Destruct(ClassInfo_t *cinfo, void *arena) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   info->Destruct(arena);
}
//______________________________________________________________________________
ClassInfo_t *TCint::ClassInfo_Factory() const
{
   // Interface to CINT function

   return new G__ClassInfo();
}
//______________________________________________________________________________
ClassInfo_t *TCint::ClassInfo_Factory(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return new G__ClassInfo(*info);
}
//______________________________________________________________________________
ClassInfo_t *TCint::ClassInfo_Factory(const char *name) const
{
   // Interface to CINT function

   return new G__ClassInfo(name);
}
//______________________________________________________________________________
ClassInfo_t *TCint::ClassInfo_Factory(G__value *pvalue) const
{
   // Interface to CINT function

   return new G__ClassInfo(*pvalue);
}
//______________________________________________________________________________
int TCint::ClassInfo_GetMethodNArg(ClassInfo_t *cinfo, const char *method,const char *proto) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   G__MethodInfo meth;
   if (info) {
      Long_t offset;
      meth = info->GetMethod(method,proto,&offset);
   }
   if (meth.IsValid()) return meth.NArg();
   return -1;
}
//______________________________________________________________________________
bool  TCint::ClassInfo_HasDefaultConstructor(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->HasDefaultConstructor();
}
//______________________________________________________________________________
bool  TCint::ClassInfo_HasMethod(ClassInfo_t *cinfo, const char *name) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->HasMethod(name);
}
//______________________________________________________________________________
void  TCint::ClassInfo_Init(ClassInfo_t *cinfo, const char *funcname) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   info->Init(funcname);
}
//______________________________________________________________________________
void  TCint::ClassInfo_Init(ClassInfo_t *cinfo, int tagnum) const
{
   // Interface to CINT function
   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   info->Init(tagnum);

}
//______________________________________________________________________________
bool  TCint::ClassInfo_IsBase(ClassInfo_t *cinfo, const char*name) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->IsBase(name);
}
//______________________________________________________________________________
bool  TCint::ClassInfo_IsEnum(const char*name) const
{
   // Interface to CINT function

   G__ClassInfo info(name);
   if (info.IsValid() && info.Property()&G__BIT_ISENUM) return kTRUE;
   return kFALSE;
}
//______________________________________________________________________________
bool  TCint::ClassInfo_IsLoaded(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->IsLoaded();
}
//______________________________________________________________________________
bool  TCint::ClassInfo_IsValid(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->IsValid();
}
//______________________________________________________________________________
bool  TCint::ClassInfo_IsValidMethod(ClassInfo_t *cinfo,  const char *method,const char *proto, Long_t *offset) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->GetMethod(method,proto,offset).IsValid();
}
//______________________________________________________________________________
int  TCint::ClassInfo_Next(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Next();
}
//______________________________________________________________________________
void  *TCint::ClassInfo_New(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->New();
}
//______________________________________________________________________________
void  *TCint::ClassInfo_New(ClassInfo_t *cinfo, int n) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->New(n);
}
//______________________________________________________________________________
void  *TCint::ClassInfo_New(ClassInfo_t *cinfo, int n, void *arena) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->New(n,arena);
}
//______________________________________________________________________________
void  *TCint::ClassInfo_New(ClassInfo_t *cinfo, void *arena) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->New(arena);
}
//______________________________________________________________________________
Long_t  TCint::ClassInfo_Property(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Property();
}
//______________________________________________________________________________
int   TCint::ClassInfo_RootFlag(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->RootFlag();
}
//______________________________________________________________________________
int   TCint::ClassInfo_Size(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Size();
}
//______________________________________________________________________________
Long_t  TCint::ClassInfo_Tagnum(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Tagnum();
}
//______________________________________________________________________________
const char *TCint::ClassInfo_FileName(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->FileName();
}
//______________________________________________________________________________
const char *TCint::ClassInfo_FullName(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Fullname();
}
//______________________________________________________________________________
const char *TCint::ClassInfo_Name(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::ClassInfo_Title(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->Title();
}
//______________________________________________________________________________
const char *TCint::ClassInfo_TmpltName(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *info = (G__ClassInfo*)cinfo;
   return info->TmpltName();
}



//______________________________________________________________________________
// G__BaseClassInfo interface
//______________________________________________________________________________
void  TCint::BaseClassInfo_Delete(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   delete info;
}
//______________________________________________________________________________
BaseClassInfo_t *TCint::BaseClassInfo_Factory(ClassInfo_t *cinfo) const
{
   // Interface to CINT function

   G__ClassInfo *cinfo1 = (G__ClassInfo*)cinfo;
   G__BaseClassInfo *info = new G__BaseClassInfo(*cinfo1);
   return info;
}
//______________________________________________________________________________
int  TCint::BaseClassInfo_Next(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function
   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Next();
}
//______________________________________________________________________________
int  TCint::BaseClassInfo_Next(BaseClassInfo_t *bcinfo, int onlyDirect) const
{
   // Interface to CINT function
   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Next(onlyDirect);
}
//______________________________________________________________________________
Long_t  TCint::BaseClassInfo_Offset(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Offset();
}
//______________________________________________________________________________
Long_t  TCint::BaseClassInfo_Property(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Property();
}
//______________________________________________________________________________
Long_t  TCint::BaseClassInfo_Tagnum(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Tagnum();
}
//______________________________________________________________________________
const char *TCint::BaseClassInfo_FullName(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Fullname();
}
//______________________________________________________________________________
const char *TCint::BaseClassInfo_Name(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::BaseClassInfo_TmpltName(BaseClassInfo_t *bcinfo) const
{
   // Interface to CINT function

   G__BaseClassInfo *info = (G__BaseClassInfo*)bcinfo;
   return info->TmpltName();
}

//______________________________________________________________________________
// G__DataMemberInfo interface
//______________________________________________________________________________
int   TCint::DataMemberInfo_ArrayDim(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->ArrayDim();
}
//______________________________________________________________________________
void  TCint::DataMemberInfo_Delete(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   delete info;
}
//______________________________________________________________________________
DataMemberInfo_t *TCint::DataMemberInfo_Factory(ClassInfo_t* clinfo /* = 0 */) const
{
   // Interface to CINT function
   G__ClassInfo* clinfo1 = (G__ClassInfo*) clinfo;
   if (clinfo1)
      return new G__DataMemberInfo(*clinfo1);
   return new G__DataMemberInfo();
}
//______________________________________________________________________________
DataMemberInfo_t *TCint::DataMemberInfo_FactoryCopy(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info1 = (G__DataMemberInfo*)dminfo;
   G__DataMemberInfo *info = new G__DataMemberInfo(*info1);
   return info;
}
//______________________________________________________________________________
bool   TCint::DataMemberInfo_IsValid(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->IsValid();
}
//______________________________________________________________________________
int   TCint::DataMemberInfo_MaxIndex(DataMemberInfo_t *dminfo, Int_t dim) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->MaxIndex(dim);
}
//______________________________________________________________________________
int  TCint::DataMemberInfo_Next(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Next();
}
//______________________________________________________________________________
Long_t TCint::DataMemberInfo_Offset(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Offset();
}
//______________________________________________________________________________
Long_t  TCint::DataMemberInfo_Property(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Property();
}
//______________________________________________________________________________
Long_t  TCint::DataMemberInfo_TypeProperty(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Type()->Property();
}
//______________________________________________________________________________
int   TCint::DataMemberInfo_TypeSize(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Type()->Size();
}
//______________________________________________________________________________
const char *TCint::DataMemberInfo_TypeName(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Type()->Name();
}
//______________________________________________________________________________
const char *TCint::DataMemberInfo_TypeTrueName(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Type()->TrueName();
}
//______________________________________________________________________________
const char *TCint::DataMemberInfo_Name(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::DataMemberInfo_Title(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->Title();
}
//______________________________________________________________________________
const char *TCint::DataMemberInfo_ValidArrayIndex(DataMemberInfo_t *dminfo) const
{
   // Interface to CINT function

   G__DataMemberInfo *info = (G__DataMemberInfo*)dminfo;
   return info->ValidArrayIndex();
}



//______________________________________________________________________________
// G__MethodInfo interface
//______________________________________________________________________________
void  TCint::MethodInfo_Delete(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   delete info;
}
//______________________________________________________________________________
void  TCint::MethodInfo_CreateSignature(MethodInfo_t *minfo, TString &signature) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   G__MethodArgInfo arg(*info);

   int ifirst = 0;
   signature = "(";
   while (arg.Next()) {
      if (ifirst) signature += ", ";
      if (arg.Type() == 0) break;
      signature += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         signature += " ";
         signature += arg.Name();
      }
      if (arg.DefaultValue()) {
         signature += " = ";
         signature += arg.DefaultValue();
      }
      ifirst++;
   }
   signature += ")";
}
//______________________________________________________________________________
MethodInfo_t *TCint::MethodInfo_Factory() const
{
   // Interface to CINT function

   G__MethodInfo *info = new G__MethodInfo();
   return info;
}
//______________________________________________________________________________
MethodInfo_t *TCint::MethodInfo_FactoryCopy(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info1 = (G__MethodInfo*)minfo;
   G__MethodInfo *info  = new G__MethodInfo(*info1);
   return info;
}
//______________________________________________________________________________
void *TCint::MethodInfo_InterfaceMethod(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   G__InterfaceMethod pfunc = info->InterfaceMethod();
   if (!pfunc) {
      struct G__bytecodefunc *bytecode = info->GetBytecode();

      if(bytecode) pfunc = (G__InterfaceMethod)G__exec_bytecode;
      else {
         pfunc = (G__InterfaceMethod)NULL;
      }
   }
   return (void*)pfunc;
}
//______________________________________________________________________________
bool  TCint::MethodInfo_IsValid(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->IsValid();
}
//______________________________________________________________________________
int   TCint::MethodInfo_NArg(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->NArg();
}
//______________________________________________________________________________
int   TCint::MethodInfo_NDefaultArg(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->NDefaultArg();
}
//______________________________________________________________________________
int   TCint::MethodInfo_Next(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Next();
}
//______________________________________________________________________________
Long_t  TCint::MethodInfo_Property(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Property();
}
//______________________________________________________________________________
void *TCint::MethodInfo_Type(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Type();
}
//______________________________________________________________________________
const char *TCint::MethodInfo_GetMangledName(MethodInfo_t *minfo) const
{
   // Interface to CINT function
   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->GetMangledName();
}
//______________________________________________________________________________
const char *TCint::MethodInfo_GetPrototype(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->GetPrototype();
}
//______________________________________________________________________________
const char *TCint::MethodInfo_Name(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::MethodInfo_TypeName(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Type()->Name();
}
//______________________________________________________________________________
const char *TCint::MethodInfo_Title(MethodInfo_t *minfo) const
{
   // Interface to CINT function

   G__MethodInfo *info = (G__MethodInfo*)minfo;
   return info->Title();
}

//______________________________________________________________________________
// G__MethodArgInfo interface
//______________________________________________________________________________
void  TCint::MethodArgInfo_Delete(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   delete info;
}
//______________________________________________________________________________
MethodArgInfo_t *TCint::MethodArgInfo_Factory() const
{
   // Interface to CINT function

   G__MethodArgInfo *info = new G__MethodArgInfo();
   return info;
}
//______________________________________________________________________________
MethodArgInfo_t *TCint::MethodArgInfo_FactoryCopy(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info1 = (G__MethodArgInfo*)marginfo;
   G__MethodArgInfo *info  = new G__MethodArgInfo(*info1);
   return info;
}
//______________________________________________________________________________
bool  TCint::MethodArgInfo_IsValid(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->IsValid();
}
//______________________________________________________________________________
int  TCint::MethodArgInfo_Next(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->Next();
}
//______________________________________________________________________________
Long_t TCint::MethodArgInfo_Property(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->Property();
}
//______________________________________________________________________________
const char *TCint::MethodArgInfo_DefaultValue(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->DefaultValue();;
}
//______________________________________________________________________________
const char *TCint::MethodArgInfo_Name(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::MethodArgInfo_TypeName(MethodArgInfo_t *marginfo) const
{
   // Interface to CINT function

   G__MethodArgInfo *info = (G__MethodArgInfo*)marginfo;
   return info->Type()->Name();
}


//______________________________________________________________________________
// G__TypeInfo interface
//______________________________________________________________________________
void  TCint::TypeInfo_Delete(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   delete info;
}
//______________________________________________________________________________
TypeInfo_t *TCint::TypeInfo_Factory() const
{
   // Interface to CINT function

   G__TypeInfo *info = new G__TypeInfo();
   return info;
}
//______________________________________________________________________________
TypeInfo_t *TCint::TypeInfo_Factory(G__value *pvalue) const
{
   // Interface to CINT function

   G__TypeInfo *info = new G__TypeInfo(*pvalue);
   return info;
}
//______________________________________________________________________________
TypeInfo_t *TCint::TypeInfo_FactoryCopy(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = new G__TypeInfo(*(G__TypeInfo*)tinfo);
   return info;
}
//______________________________________________________________________________
void  TCint::TypeInfo_Init(TypeInfo_t *tinfo, const char *funcname) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   info->Init(funcname);
}
//______________________________________________________________________________
bool  TCint::TypeInfo_IsValid(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->IsValid();
}
//______________________________________________________________________________
const char *TCint::TypeInfo_Name(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->Name();
}
//______________________________________________________________________________
Long_t  TCint::TypeInfo_Property(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->Property();
}
//______________________________________________________________________________
int   TCint::TypeInfo_RefType(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->Reftype();
}
//______________________________________________________________________________
int   TCint::TypeInfo_Size(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->Size();
}
//______________________________________________________________________________
const char *TCint::TypeInfo_TrueName(TypeInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypeInfo *info = (G__TypeInfo*)tinfo;
   return info->TrueName();
}


//______________________________________________________________________________
// G__TypedefInfo interface
//______________________________________________________________________________
void  TCint::TypedefInfo_Delete(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   delete info;
}
//______________________________________________________________________________
TypedefInfo_t *TCint::TypedefInfo_Factory() const
{
   // Interface to CINT function

   G__TypedefInfo *info = new G__TypedefInfo();
   return info;
}
//______________________________________________________________________________
TypedefInfo_t *TCint::TypedefInfo_FactoryCopy(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = new G__TypedefInfo(*(G__TypedefInfo*)tinfo);
   return info;
}
//______________________________________________________________________________
TypedefInfo_t  TCint::TypedefInfo_Init(TypedefInfo_t *tinfo, const char *funcname) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   info->Init(funcname);
}
//______________________________________________________________________________
bool  TCint::TypedefInfo_IsValid(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->IsValid();
}
//______________________________________________________________________________
Long_t  TCint::TypedefInfo_Property(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->Property();
}
//______________________________________________________________________________
int   TCint::TypedefInfo_Size(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->Size();
}
//______________________________________________________________________________
const char *TCint::TypedefInfo_TrueName(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->TrueName();
}
//______________________________________________________________________________
const char *TCint::TypedefInfo_Name(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->Name();
}
//______________________________________________________________________________
const char *TCint::TypedefInfo_Title(TypedefInfo_t *tinfo) const
{
   // Interface to CINT function

   G__TypedefInfo *info = (G__TypedefInfo*)tinfo;
   return info->Title();
}
