// @(#)root/meta:$Name:  $:$Id: TCint.cxx,v 1.8 2000/09/01 20:37:05 brun Exp $
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
#include "TROOT.h"
#include "TApplication.h"
#include "TGlobal.h"
#include "TDataType.h"
#include "TClass.h"
#include "TBaseClass.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TSystem.h"
#include "G__ci.h"


#ifdef WIN32
#  ifndef ROOT_TGWin32Command
#    include "TGWin32Command.h"
#    undef GetClassInfo
#  endif
#endif

R__EXTERN int optind;

// reference cint includes
// make sure fproto.h is loaded (it was excluded in TCint.h)
#undef G__FPROTO_H
#include "fproto.h"
#include "Api.h"

extern "C" void G__clearfilebusy(int);

extern "C" int ScriptCompiler(const char* filename, const char* opt) {
   return gSystem->CompileMacro(filename, opt);
}

extern "C" int IgnoreInclude(const char* fname,const char* expandedfname) {
   return gROOT->IgnoreInclude(fname,expandedfname);
}

extern "C" void TCint_UpdateClassInfo(char* c ,Long_t l) {
  TCint::UpdateClassInfo(c,l);
}

extern "C" void* TCint_FindSpecialObject(char * c, G__ClassInfo *ci, void ** p1, void ** p2) {
  return TCint::FindSpecialObject(c,ci,p1,p2);
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

   G__RegisterScriptCompiler(&ScriptCompiler);
   G__set_ignoreinclude(&IgnoreInclude);
   G__InitUpdateClassInfo(&TCint_UpdateClassInfo);
   G__InitGetSpecialObject(&TCint_FindSpecialObject);

   ResetAll();
#ifndef WIN32
   optind = 1;  // make sure getopt() works in the main program
#endif
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
}

//______________________________________________________________________________
void TCint::ClearFileBusy()
{
   // Reset CINT internal state in case a previous action was not correctly terminated
   // by G__init_cint() and G__dlmod().

   G__clearfilebusy(0);
}

//______________________________________________________________________________
Int_t TCint::InitializeDictionaries()
{
   // Initialize all registered dictionaries. Normally this is already done
   // by G__init_cint() and G__dlmod().

   return G__call_setup_funcs();
}

//______________________________________________________________________________
void TCint::EndOfLineAction()
{
   // It calls a "fantom" method to synchronize user keyboard input
   // and ROOT prompt line.

    ProcessLineSynch(fantomline);
}

//______________________________________________________________________________
void TCint::ExecThreadCB(TWin32SendClass *command)
{
   // This function must be called from the "Command thread only".

#ifdef WIN32
   char *line = (char *)(command->GetData(0));
   Int_t iret = ProcessLine((const char *)line);
   delete [] line;
   if (LOWORD(command->GetCOP()) == kSendWaitClass)
      ((TWin32SendWaitClass *)command)->Release();
   else
      delete command;
#else
   if (command) { }
#endif
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
     if (file.IsValid()) { return kTRUE; };
   }

   next = gSystem->DynamicPathName(filename,kTRUE);
   if (next) {
     file.Init(next);
     if (file.IsValid()) { return kTRUE; };
   }

   return kFALSE;
}

//______________________________________________________________________________
void TCint::LoadMacro(const char *filename)
{
   // Load a macro file in CINT's memory.

   ProcessLine(Form(".L %s", filename));
}

//______________________________________________________________________________
Int_t TCint::ProcessLine(const char *line)
{
   // Let CINT process a command line.

   Int_t ret = 0;

   if (gApplication && gApplication->IsCmdThread()) {
      gROOT->SetLineIsProcessing();

      // It checks whether the input line contains the "fantom" method
      // to synchronize user keyboard input and ROOT prompt line

      if (strstr(line,fantomline))
          TCint::UpdateAllCanvases();
      else
          ret = G__process_cmd((char *)line, fPrompt, &fMore, 0, 0);
      gROOT->SetLineHasBeenProcessed();
   } else
      ret = ProcessLineAsynch(line);

   return ret;
}

//______________________________________________________________________________
Int_t TCint::ProcessLineAsynch(const char *line)
{
   // Let CINT process a command line asynch.

#ifndef WIN32
   return ProcessLine(line);
#else
   char *cmd = new char[strlen(line)+1];
   strcpy(cmd,line);
   TWin32SendClass *code = new TWin32SendClass(this,(UInt_t)cmd,0,0,0);
   ExecCommandThread(code,kFALSE);
   return 0;
#endif
}

//______________________________________________________________________________
Int_t TCint::ProcessLineSynch(const char *line)
{
   // Let CINT process a command line synchronously, i.e we are waiting
   // it will be finished.

  if (gApplication && gApplication->IsCmdThread())
     return ProcessLine(line);
#ifdef WIN32
   char *cmd = new char[strlen(line)+1];
   strcpy(cmd,line);
   TWin32SendWaitClass code(this,(UInt_t)cmd,0,0,0);
   ExecCommandThread(&code,kFALSE);
   code.Wait();
#endif
   return 0;
}

//______________________________________________________________________________
Long_t TCint::Calc(const char *line)
{
   // Directly execute an executable statement (e.g. "func()", "3+5", etc.
   // however not declarations, like "Int_t x;").

   Long_t result;

#ifdef WIN32
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

   result = (Long_t) G__int(G__calc((char *)line));

#ifdef WIN32
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

   G__init_cint("cint");
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

   G__MethodInfo t, *a;
   while (t.Next()) {
      // if name cannot be obtained no use to put in list
      if (t.IsValid() && t.Name()) {
        // first remove if already in list
         TFunction *f = (TFunction *)gROOT->fGlobalFunctions->FindObject(t.Name());
         if (f) {
            gROOT->fGlobalFunctions->Remove(f);
            delete f;
         }
         a = new G__MethodInfo(t);
         gROOT->fGlobalFunctions->Add(new TFunction(a));
      }
   }
}

//______________________________________________________________________________
void TCint::UpdateListOfTypes()
{
   // Update the list of pointers to Datatype (typedef) definitions. This
   // function is called by TROOT::GetListOfTypes().

   G__TypedefInfo t, *a;
   while (t.Next()) {
      if (t.IsValid() && t.Name()) {
         TDataType *d = (TDataType *)gROOT->fTypes->FindObject(t.Name());
         // only add new types, don't delete old ones with the same name
         // (as is done in UpdateListOfGlobals())
         if (!d) {
            a = new G__TypedefInfo(t);
            gROOT->fTypes->Add(new TDataType(a));
         }
      }
   }
}

//______________________________________________________________________________
void TCint::SetClassInfo(TClass *cl)
{
   // Set pointer to CINT's G__ClassInfo in TClass.

   if (!cl->fClassInfo) {
      cl->fClassInfo = new G__ClassInfo(cl->GetName());

      if (!cl->fClassInfo->IsValid()) {
         // this happens when no CINT dictionary is available
         delete cl->fClassInfo;
         cl->fClassInfo = 0;
      }
   }
}

//______________________________________________________________________________
void TCint::CreateListOfBaseClasses(TClass *cl)
{
   // Create list of pointers to base class(es) for TClass cl.

   if (!cl->fBase) {

      cl->fBase = new TList(cl);

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

   if (!cl->fData) {

      cl->fData = new TList(cl);

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

   if (!cl->fMethod) {

      cl->fMethod = new TList(cl);

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

   if (!m->fMethodArgs) {

      m->fMethodArgs = new TList(m);

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
void *TCint::GetInterfaceMethod(TClass *cl, char *method, char *params)
{
   // Return pointer to CINT interface function for a method of a class with
   // parameters params (params is a string of actual arguments, not formal
   // ones). If the class is 0 the global function list will be searched.

   G__CallFunc  func;
   long         offset;

   if (cl)
      func.SetFunc(cl->GetClassInfo(), method, params, &offset);
   else {
      G__ClassInfo gcl;   // default G__ClassInfo is global environment
      func.SetFunc(&gcl, method, params, &offset);
   }
   return (void *)func.InterfaceMethod();
}

//______________________________________________________________________________
void *TCint::GetInterfaceMethodWithPrototype(TClass *cl, char *method, char *proto)
{
   // Return pointer to CINT interface function for a method of a class with
   // a certain prototype, i.e. "char*,int,float". If the class is 0 the global
   // function list will be searched.

   G__InterfaceMethod f;
   long               offset;

   if (cl)
      f = cl->GetClassInfo()->GetMethod(method, proto, &offset).InterfaceMethod();
   else {
      G__ClassInfo gcl;   // default G__ClassInfo is global environment
      f = gcl.GetMethod(method, proto, &offset).InterfaceMethod();
   }
   return (void *)f;
}

//______________________________________________________________________________
void TCint::Execute(const char *function, const char *params)
{
   // Execute a global function with arguments params.

   long         offset;
   G__CallFunc  func;
   G__ClassInfo cl;

   // set pointer to interface method and arguments
   func.SetFunc(&cl, (char*)function, (char*)params, &offset);

   // call function
   func.Exec(0);
}

//______________________________________________________________________________
void TCint::Execute(TObject *obj, TClass *cl, const char *method, const char *params)
{
   // Execute a method from class cl with arguments params.

   void       *address;
   long        offset;
   G__CallFunc func;

   // set pointer to interface method and arguments
   func.SetFunc(cl->GetClassInfo(), (char*)method, (char*)params, &offset);

   // call function
   address = (void*)((Long_t)obj + offset);
   func.Exec(address);
}

//______________________________________________________________________________
void TCint::Execute(TObject *obj, TClass *cl, TMethod *method, TObjArray *params)
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
           if (strstr( type.TrueName(), "char" ))
           {
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

   Execute(obj, cl, (char *)method->GetName(), (char *)listpar);
}

//______________________________________________________________________________
Int_t TCint::ExecuteMacro(const char *filename)
{
   // Execute a CINT macro.

   G__value result = G__exec_tempfile((char*)filename);
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
      // s is the position of the second 'word' (if any)
      // except in the case of templates where there will be a space
      // just before any closing '>': eg.
      //    TObj<std::vector<UShort_t,__malloc_alloc_template<0> > >*
      if (s && (template_start==0 || (s < template_start)) )
         strcpy(t, s+1);
      else
         strcpy(t, typeDesc);
   }

   int l = strlen(t);
   while (l > 0 && t[l-1] == '*') t[--l] = 0;

   return t;
}

//______________________________________________________________________________
void *TCint::FindSpecialObject(char *item, G__ClassInfo *type, void **prevObj,
                        void **assocPtr)
{
   // Static function called by CINT when it finds an un-indentified object.
   // This function tries to find the UO in the ROOT files, directories, etc.
   // This functions has been registered by the TCint ctor.

   if (!*prevObj || *assocPtr != gDirectory) {
      *prevObj  = gROOT->FindSpecialObject(item,*assocPtr);
   }

   if (*prevObj) type->Init(((TObject *)*prevObj)->ClassName());
   return *prevObj;
}

//______________________________________________________________________________
void TCint::UpdateClassInfo(char *item, Long_t tagnum)
{
   // Static function called by CINT when it changes the tagnum for
   // a class (e.g. after re-executing the setup function). In such
   // cases we have to update the tagnum in the G__ClassInfo used by
   // the TClass for class "item".

   if (gROOT && gROOT->GetListOfClasses()) {
      TClass *cl = gROOT->GetClass(item, kFALSE);
      if (cl) {
         G__ClassInfo *info = cl->GetClassInfo();
         if (info && info->Tagnum() != tagnum) {
            info->Init((int)tagnum);
            if (cl->fBase)
               cl->fBase->Delete();
            delete cl->fBase;
            cl->fBase = 0;
         }
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
      if (filename &&
          (strstr(filename,".sl")  != 0 ||
           strstr(filename,".dl")  != 0 ||
           strstr(filename,".so")  != 0 ||
           strstr(filename,".dll") != 0 ||
           strstr(filename,".DLL") != 0 ||
           strstr(filename,".a")   != 0)) {
         if (!fSharedLibs.IsNull())
            fSharedLibs.Append(" ");
         fSharedLibs.Append(filename);
      }

      cursor.Next();
   }

   return fSharedLibs;
}

//______________________________________________________________________________
void TCint::AddIncludePath(const char *path)
{
  // Add the given path to the list of directory in which the interpreter
  // look for include files.

  G__add_ipath((char*)path);

}

//______________________________________________________________________________
const char* TCint::GetIncludePath()
{
  // Refresh the list of include path known to the interpreter and return it,
  // with -I prepended.

  fIncludePath = "";

  G__IncludePathInfo path;

  while ( path.Next() ) {
    const char *pathname = path.Name();
    fIncludePath.Append(" -I").Append(pathname);
  } ;

  return fIncludePath;

}

