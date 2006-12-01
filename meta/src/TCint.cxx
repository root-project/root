// @(#)root/meta:$Name:  $:$Id: TCint.cxx,v 1.129 2006/11/30 23:19:47 pcanal Exp $
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

#include "TCint.h"
#include "G__ci.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassEdit.h"
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
#include "THashTable.h"

#include <vector>
#include <string>
using namespace std;

R__EXTERN int optind;

// reference cint includes
// make sure fproto.h is loaded (it was excluded in TCint.h)
//#undef G__FPROTO_H
//#include "fproto.h"
#include "Api.h"

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
   G__setgvp(G__PVOID);
   string cls(c);
   int result =  TCint::AutoLoadCallback(cls.c_str(), l);
   G__setgvp(varp);
   return result;
}

extern "C" void *TCint_FindSpecialObject(char *c, G__ClassInfo *ci, void **p1, void **p2) {
   return TCint::FindSpecialObject(c, ci, p1, p2);
}

// It is a "fantom" method to synchronize user keyboard input
// and ROOT prompt line (for WIN32)
const char *fantomline = "TRint::EndOfLineAction();";

ClassImp(TCint)

//______________________________________________________________________________
TCint::TCint(const char *name, const char *title) : TInterpreter(name, title)
{
   // Initialize the CINT interpreter interface.

   fMore      = 0;
   fPrompt[0] = 0;
   fMapfile   = 0;
   fLockProcessLine = kTRUE;

   G__RegisterScriptCompiler(&ScriptCompiler);
   G__set_ignoreinclude(&IgnoreInclude);
   G__InitUpdateClassInfo(&TCint_UpdateClassInfo);
   G__InitGetSpecialObject(&TCint_FindSpecialObject);

   fDictPos.ptype = 0;
   fDictPosGlobals.ptype = 0;

   ResetAll();

#ifndef R__WIN32
   optind = 1;  // make sure getopt() works in the main program
#endif

   // Make sure that ALL macros are seen as C++.
   G__LockCpp();
}

//______________________________________________________________________________
TCint::TCint(const TCint& ci) :
  TInterpreter(ci),
  fMore(ci.fMore),
  fExitCode(ci.fExitCode),
  fDictPos(ci.fDictPos),
  fDictPosGlobals(ci.fDictPosGlobals),
  fSharedLibs(ci.fSharedLibs),
  fIncludePath(ci.fIncludePath),
  fMapfile(ci.fMapfile),
  fLockProcessLine(ci.fLockProcessLine)
{
   //copy constructor
   strncpy(fPrompt,ci.fPrompt,64);
}


//______________________________________________________________________________
TCint& TCint::operator=(const TCint& ci)
{
   //assignement operator
   if(this!=&ci) {
      TInterpreter::operator=(ci);
      fMore=ci.fMore;
      fExitCode=ci.fExitCode;
      strncpy(fPrompt,ci.fPrompt,64);
      fDictPos=ci.fDictPos;
      fDictPosGlobals=ci.fDictPosGlobals;
      fSharedLibs=ci.fSharedLibs;
      fIncludePath=ci.fIncludePath;
      fMapfile=ci.fMapfile;
      fLockProcessLine=ci.fLockProcessLine;
   }
   return *this;
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

   free(fDictPos.ptype);
   free(fDictPosGlobals.ptype);
   delete fMapfile;
}

//______________________________________________________________________________
void TCint::ClearFileBusy()
{
   // Reset CINT internal state in case a previous action was not correctly
   // terminated by G__init_cint() and G__dlmod().

   G__clearfilebusy(0);
}

//______________________________________________________________________________
void TCint::ClearStack()
{
   // Delete existing temporary values

   G__clearstack();
}

//______________________________________________________________________________
Int_t TCint::InitializeDictionaries()
{
   // Initialize all registered dictionaries. Normally this is already done
   // by G__init_cint() and G__dlmod().

   return G__call_setup_funcs();
}

//______________________________________________________________________________
void TCint::EnableAutoLoading()
{
   // Enable the automatic loading of shared libraries when a class
   // is used that is stored in a not yet loaded library. Uses the
   // information stored in the class/library map (typically
   // $ROOTSYS/etc/system.rootmap).

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
   incPath.Append(":$ROOTSYS/cint/include:$ROOTSYS/cint/stl");
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

   Int_t ret = 0;
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
         if (strstr(line,fantomline)) {
            G__free_tempobject();
            TCint::UpdateAllCanvases();
         } else {
            int local_error = 0;

            ret = G__process_cmd((char *)line, fPrompt, &fMore, &local_error, &local_res);
            if (local_error == 0 && G__get_return(&fExitCode) == G__RETURN_EXIT2) {
               ResetGlobals();
               gApplication->Terminate(fExitCode);
            }
            if (error)
               *error = (EErrorCode)local_error;
         }

         if (ret==0) ret = G__int_cast(local_res);

         gROOT->SetLineHasBeenProcessed();
      } else
         ret = ProcessLineAsynch(line, error);
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

   if (gApplication && gApplication->IsCmdThread())
      return ProcessLine(line, error);
   return 0;
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
   Printf("Type ? for help. Commands must be C++ statements.");
   Printf("Enclose multiple statements between { }.");
}

//______________________________________________________________________________
void TCint::RecursiveRemove(TObject *obj)
{
   // Delete object from CINT symbol table so it can not be used anymore.
   // CINT object are always on the heap.

   if (obj->IsOnHeap()) {
      DeleteGlobal(obj);
   }
}

//______________________________________________________________________________
void TCint::Reset()
{
   // Reset the CINT state to the state saved by the last call to
   // TCint::SaveContext().

   G__scratch_upto(&fDictPos);
}

//______________________________________________________________________________
void TCint::ResetAll()
{
   // Reset the CINT state to its initial state.

   G__init_cint("cint +V");
   G__init_process_cmd();
}

//______________________________________________________________________________
void TCint::ResetGlobals()
{
   // Reset the CINT global object state to the state saved by the last
   // call to TCint::SaveGlobalsContext().

   G__scratch_globals_upto(&fDictPosGlobals);
}

//______________________________________________________________________________
void TCint::RewindDictionary()
{
   // Rewind CINT dictionary to the point where it was before executing
   // the current macro. This function is typically called after SEGV or
   // ctlr-C after doing a longjmp back to the prompt.

   G__rewinddictionary();
}

//______________________________________________________________________________
Int_t TCint::DeleteGlobal(void *obj)
{
   // Delete obj from CINT symbol table so it cannot be accessed anymore.
   // Returns 1 in case of success and 0 in case object was not in table.

   return G__deleteglobal(obj);
}

//______________________________________________________________________________
void TCint::SaveContext()
{
   // Save the current CINT state.

   G__store_dictposition(&fDictPos);
}

//______________________________________________________________________________
void TCint::SaveGlobalsContext()
{
   // Save the current CINT state of global objects.

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
            Int_t prop = -1;
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
   G__TypedefInfo t;
   while (t.Next()) {
      if (gROOT && gROOT->fTypes && t.IsValid() && t.Name()) {
         TDataType *d = (TDataType *)gROOT->fTypes->FindObject(t.Name());
         // only add new types, don't delete old ones with the same name
         // (as is done in UpdateListOfGlobals()),
         // this 'feature' is being used in TROOT::GetType().
         if (!d) {
            gROOT->fTypes->Add(new TDataType(new G__TypedefInfo(t)));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::SetClassInfo(TClass *cl, Bool_t reload)
{
   // Set pointer to CINT's G__ClassInfo in TClass.

   R__LOCKGUARD2(gCINTMutex);
   if (!cl->fClassInfo || reload) {

      delete cl->fClassInfo; cl->fClassInfo = 0;
      if (CheckClassInfo(cl->GetName())) {

         cl->fClassInfo = new G__ClassInfo(cl->GetName());

         // In case a class contains an external enum, the enum will be seen as a
         // class. We must detect this special case and make the class a Zombie.
         // Here we assume that a class has at least one method.
         // We can NOT call TClass::Property from here, because this method
         // assumes that the TClass is well formed to do a lot of information
         // caching. The method SetClassInfo (i.e. here) is usually called during
         // the building phase of the TClass, hence it is NOT well formed yet.
         if (cl->fClassInfo->IsValid() &&
             !(cl->fClassInfo->Property() & (kIsClass|kIsStruct))) {
            cl->MakeZombie();
         }

         if (!cl->fClassInfo->IsLoaded()) {
            // this happens when no CINT dictionary is available
            delete cl->fClassInfo;
            cl->fClassInfo = 0;
         }

      }
   }
}

//______________________________________________________________________________
Bool_t TCint::CheckClassInfo(const char *name)
{
   // Checks if a class with the specified name is defined in CINT.
   // Returns kFALSE is class is not defined.

   // In the case where the class is not loaded and belongs to a namespace
   // or is nested, looking for the full class name is outputing a lots of
   // (expected) error messages.  Currently the only way to avoid this is to
   // specifically check that each level of nesting is already loaded.
   // In case of templates the idea is that everything between the outer
   // '<' and '>' has to be skipped, e.g.: aap<pipo<noot>::klaas>::a_class

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
         return kFALSE;
      }

      *current = '\0';
      G__ClassInfo info(classname);
      if (!info.IsValid()) {
         delete [] classname;
         return kFALSE;
      }
      *current = ':';
      current += 2;
   }
   strcpy(classname,name);

   Int_t tagnum = G__defined_tagname(classname, 2); // This function might modify the name (to add space between >>).
   if (tagnum >= 0) return kTRUE;
   G__TypedefInfo t(name);
   if (t.IsValid() && !(t.Property()&G__BIT_ISFUNDAMENTAL)) return kTRUE;

   delete [] classname;
   return kFALSE;
}

//______________________________________________________________________________
void TCint::CreateListOfBaseClasses(TClass *cl)
{
   // Create list of pointers to base class(es) for TClass cl.

   R__LOCKGUARD2(gCINTMutex);
   if (!cl->fBase) {

      cl->fBase = new TList;

      G__BaseClassInfo t(*cl->GetClassInfo()), *a;
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

      G__DataMemberInfo t(*cl->GetClassInfo()), *a;
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

      cl->fMethod = new TList;

      G__MethodInfo t(*cl->GetClassInfo()), *a;
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
void TCint::CreateListOfMethodArgs(TFunction *m)
{
   // Create list of pointers to method arguments for TMethod m.

   R__LOCKGUARD2(gCINTMutex);
   if (!m->fMethodArgs) {

      m->fMethodArgs = new TList;

      G__MethodArgInfo t(*m->fInfo), *a;
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
      func.SetFunc(cl->GetClassInfo(), method, params, &offset);
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
      return cl->GetClassInfo()->GetMethod(method, proto, &offset).GetMangledName();
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
      func.SetFunc(cl->GetClassInfo(), method, params, &offset);
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
      f = cl->GetClassInfo()->GetMethod(method, proto, &offset).InterfaceMethod();
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
   func.SetFunc(cl->GetClassInfo(), method, params, &offset);

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

   if (gApplication)
      return gApplication->ProcessFile(filename, (int*)error);
   else
      /*G__value result =*/ G__exec_tempfile((char*)filename);
   return 0;  // could get return value from result, but what about return type?
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
Int_t TCint::LoadLibraryMap()
{
   // Load map between class and library. Cint uses this information to
   // automatically load the shared library for a class (autoload mechanism).
   // See also the AutoLoadCallback() method below.

   // open the [system].rootmap files
   if (!fMapfile) {
      fMapfile = new TEnv(".rootmap");

      if (!fMapfile->GetTable()->GetEntries())
         Error("LoadLibraryMap", "library map empty, no system.rootmap file\n"
               "found. ROOT not properly installed (run \"make install\").");

      // Load all rootmap files in the dynamic load path (LD_LIBRARY_PATH, etc.).
      // A rootmap file must start with the string "rootmap" and may be followed
      // by any extension, like rootmap_ModuleX, rootmap-Module-Y.
      TString ldpath = gSystem->GetDynamicPath();
#ifdef WIN32
      TObjArray *paths = ldpath.Tokenize(";");
#else
      TObjArray *paths = ldpath.Tokenize(":");
#endif
      paths->Sort();

      TString d;
      for (Int_t i = 0; i < paths->GetEntries(); i++) {
         if (d == ((TObjString*)paths->At(i))->GetString())
            continue;  // skip already seen directories
         d = ((TObjString*)paths->At(i))->GetString();

         void *dirp = gSystem->OpenDirectory(d);
         if (dirp) {
            const char *f;
            Bool_t gotrm = kFALSE;
            while ((f = gSystem->GetDirEntry(dirp))) {
               if (!strncasecmp(f, "rootmap", 7)) {
                  TString p;
                  p = d + "/" + f;
                  if (!gSystem->AccessPathName(p, kReadPermission)) {
                     if (gDebug > 1)
                        Info("LoadLibraryMap", "additional rootmap file: %s", p.Data());
                     fMapfile->ReadFile(p, kEnvGlobal);
                  }
                  gotrm = kTRUE;
               } else if (gotrm) {
                  break;  // no need to continue after last rootmap file
               }
            }
         }
         gSystem->FreeDirectory(dirp);
      }

      delete paths;
      if (!fMapfile->GetTable()->GetEntries()) {
         return -1;
      }
   }

   TEnvRec *rec;
   TIter next(fMapfile->GetTable());

   while ((rec = (TEnvRec*) next())) {
      TString cls = rec->GetName();
      if (!strncmp(cls.Data(), "Library.", 8) && cls.Length() > 8) {

         // get the first lib from the list of lib and dependent libs
         TString libs = rec->GetValue();
         TString delim(" ");
         TObjArray *tokens = libs.Tokenize(delim);
         char *lib = (char *)((TObjString*)tokens->At(0))->GetName();
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
                        G__set_class_autoloading_table((char*)base.Data(), "");
                     }
                     ++k;
                  }
               } else if (cls[k] == '<') {
                  // We do not want to look at the namespace inside the template parameters!
                  break;
               }
            }
         }
         G__set_class_autoloading_table((char*)cls.Data(), lib);
         G__security_recover(stderr); // Ignore any error during this setting.
         if (gDebug > 0)
            Info("LoadLibraryMap", "adding class %s in lib %s", cls.Data(), lib);
         delete tokens;
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t TCint::AutoLoad(const char *cls)
{
   // Load library containing specified class. Returns 0 in case of error
   // and 1 in case if success.

   Int_t status = 0;

   if (!gROOT || !gInterpreter) return status;

   // Prevent the recursion when the library dictionary are loaded.
   Int_t oldvalue = G__set_class_autoloading(0);

   // lookup class to find list of dependent libraries
   TString deplibs = gInterpreter->GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray *tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntries()-1; i > 0; i--) {
         const char *deplib = ((TObjString*)tokens->At(i))->GetName();
         gROOT->LoadClass(cls, deplib);
         if (gDebug > 0)
            ::Info("TCint::AutoLoad", "loaded dependent library %s for class %s",
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

   if (!gROOT || !gInterpreter || !cls || !lib) return 0;

   // calls to load libCore might come in the very beginning when libCore
   // dictionary is not fully loaded yet, ignore it since libCore is always
   // loaded
   if (strstr(lib, "libCore")) return 1;

   // lookup class to find list of dependent libraries
   TString deplibs = gInterpreter->GetClassSharedLibs(cls);
   if (!deplibs.IsNull()) {
      TString delim(" ");
      TObjArray *tokens = deplibs.Tokenize(delim);
      for (Int_t i = tokens->GetEntries()-1; i > 0; i--) {
         const char *deplib = ((TObjString*)tokens->At(i))->GetName();
         gROOT->LoadClass(cls, deplib);
         if (gDebug > 0)
            ::Info("TCint::AutoLoadCallback", "loaded dependent library %s for class %s",
                   deplib, cls);
      }
      delete tokens;
   }

   if (lib[0]) {
      if (gROOT->LoadClass(cls, lib) == 0) {
         if (gDebug > 0)
            ::Info("TCint::AutoLoadCallback", "loaded library %s for class %s",
            lib, cls);
         return 1;
      } else
         ::Error("TCint::AutoLoadCallback", "failure loading library %s for class %s",
         lib, cls);
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
         Update(fName.c_str(),fTagnum);
      }
      static void Update(const char *item, Long_t tagnum)
      {
         Bool_t load = kFALSE;
         if (strchr(item,'<')) {
            // We have a template which may have duplicates.

            TIter next( gROOT->GetListOfClasses() );
            TClass *cl;

            TString resolvedItem(
               TClassEdit::ResolveTypedef(TClassEdit::ShortType(item,
                  TClassEdit::kDropStlDefault).c_str(), kTRUE) );
            TString resolved;
            while ( (cl = (TClass*)next()) ) {
               resolved = TClassEdit::ResolveTypedef(TClassEdit::ShortType(cl->GetName(),
                  TClassEdit::kDropStlDefault).c_str(), kTRUE);
               if (resolved==resolvedItem) {
                  // we found at least one equivalent.
                  // let's force a reload
                  load = kTRUE;
               }
            }
         }

         TClass *cl = gROOT->GetClass(item, load);
         if (cl) cl->ResetClassInfo(tagnum);
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
         TInfoNode::Update(item,tagnum);
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
      if ( len>5 && (strcmp(end-4,".dll") == 0 ) ) {
         // Filter out the cintdlls
         const char *excludelist [] = {
            "stdfunc.dll","stdcxxfunc.dll","posix.dll","sys/ipc.dll",
            "string.dll","vector.dll","list.dll","deque.dll","map.dll",
            "map2.dll","set.dll","multimap.dll","multimap2.dll","multiset.dll",
            "stack.dll","queue.dll","valarray.dll","exception.dll","complex.dll"};
         for (unsigned int i=0; i < sizeof(excludelist)/sizeof(excludelist[0]); ++i) {
            if (strcmp(filename,excludelist[i])==0) { needToSkip = kTRUE; break; }
         }
      }
      if ( !needToSkip &&
           ( (len>3 && strcmp(end-2,".a") == 0) ||
             (len>4 && (strcmp(end-3,".sl") == 0 ||
                        strcmp(end-3,".dl") == 0 ||
                      strcmp(end-3,".so") == 0)) ||
             (len>5 && (strcmp(end-4,".dll") == 0 ||
                        strcmp(end-4,".DLL") == 0)))) {
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

   TEnvRec *rec;
   TIter next(fMapfile->GetTable());

   while ((rec = (TEnvRec*) next())) {
      size_t l = strlen(lib);
      const char *libs = rec->GetValue();
      if (!strncmp(libs, lib, l) && strlen(libs) > l) {
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

   char *incpath = gSystem->ExpandPathName(path);

   G__add_ipath(incpath);

   delete [] incpath;
}

//______________________________________________________________________________
const char *TCint::GetIncludePath()
{
   // Refresh the list of include paths known to the interpreter and return it
   // with -I prepended.

   fIncludePath = "";

   G__IncludePathInfo path;

   while (path.Next()) {
     const char *pathname = path.Name();
     fIncludePath.Append(" -I\"").Append(pathname).Append("\" ");
   }

   return fIncludePath;
}

