// @(#)root/base:$Id$
// Author: Fons Rademakers   26/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPluginManager                                                       //
//                                                                      //
// This class implements a plugin library manager. It keeps track of    //
// a list of plugin handlers. A plugin handler knows which plugin       //
// library to load to get a specific class that is used to extend the   //
// functionality of a specific base class and how to create an object   //
// of this class. For example, to extend the base class TFile to be     //
// able to read RFIO files one needs to load the plugin library         //
// libRFIO.so which defines the TRFIOFile class. This loading should    //
// be triggered when a given URI contains a regular expression defined  //
// by the handler.                                                      //
// Plugin handlers can be defined via macros in a list of plugin        //
// directories. With $ROOTSYS/etc/plugins the default top plugin        //
// directory specified in $ROOTSYS/etc/system.rootrc. Additional        //
// directories can be specified by adding them to the end of the list.  //
// Macros for identical plugin handlers in later directories will       //
// override previous ones (the inverse of normal search path behavior). //
// The macros must have names like <BaseClass>/PX0_<PluginClass>.C,     //
// e.g.:                                                                //
//    TFile/P10_TRFIOFile.C, TSQLServer/P20_TMySQLServer.C, etc.        //
// to allow easy sorting and grouping. If the BaseClass is in a         //
// namespace the directory must have the name NameSpace@@BaseClass as   //
// : is a reserved pathname character on some operating systems.        //
// Macros not beginning with 'P' and ending with ".C" are ignored.      //
// These macros typically look like:                                    //
//                                                                      //
//   void P10_TDCacheFile()                                             //
//   {                                                                  //
//       gPluginMgr->AddHandler("TFile", "^dcache", "TDCacheFile",      //
//          "DCache", "TDCacheFile(const char*,Option_t*)");            //
//   }                                                                  //
//                                                                      //
// Plugin handlers can also be defined via resources in the .rootrc     //
// file. Although now deprecated this method still works for backward   //
// compatibility, e.g.:                                                 //
//                                                                      //
//   Plugin.TFile:       ^rfio:   TRFIOFile    RFIO   "<constructor>"   //
//   Plugin.TSQLServer:  ^mysql:  TMySQLServer MySQL  "<constructor>"   //
//   +Plugin.TSQLServer: ^pgsql:  TPgSQLServer PgSQL  "<constructor>"   //
//   Plugin.TVirtualFitter: *     TFitter      Minuit "TFitter(Int_t)"  //
//                                                                      //
// Where the + in front of Plugin.TSQLServer says that it extends the   //
// existing definition of TSQLServer, useful when there is more than    //
// one plugin that can extend the same base class. The "<constructor>"  //
// should be the constructor or a static method that generates an       //
// instance of the specified class. Global methods should start with    //
// "::" in their name, like "::CreateFitter()".                         //
// Instead of being a shared library a plugin can also be a CINT        //
// script, so instead of libDialog.so one can have Dialog.C.            //
// The * is a placeholder in case there is no need for a URI to         //
// differentiate between different plugins for the same base class.     //
// For the default plugins see $ROOTSYS/etc/system.rootrc.              //
//                                                                      //
// Plugin handlers can also be registered at run time, e.g.:            //
//                                                                      //
//   gPluginMgr->AddHandler("TSQLServer", "^sapdb:",                    //
//                          "TSapDBServer", "SapDB",                    //
//             "TSapDBServer(const char*,const char*, const char*)");   //
//                                                                      //
// A list of currently defined handlers can be printed using:           //
//                                                                      //
//   gPluginMgr->Print(); // use option="a" to see ctors                //
//                                                                      //
// The use of the plugin library manager removes all textual references //
// to hard-coded class and library names and the resulting dependencies //
// in the base classes. The plugin manager is used to extend a.o.       //
// TFile, TSQLServer, TGrid, etc. functionality.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPluginManager.h"
#include "Varargs.h"
#include "TEnv.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TSortedList.h"
#include "THashList.h"
#include "THashTable.h"
#include "Varargs.h"
#include "TClass.h"
#include "TInterpreter.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TVirtualMutex.h"
#include "TSystem.h"
#include "TObjString.h"


TPluginManager *gPluginMgr;   // main plugin manager created in TROOT


ClassImp(TPluginHandler)

//______________________________________________________________________________
TPluginHandler::TPluginHandler(const char *base, const char *regexp,
                               const char *className, const char *pluginName,
                               const char *ctor, const char *origin):
   fBase(base),
   fRegexp(regexp),
   fClass(className),
   fPlugin(pluginName),
   fCtor(ctor),
   fOrigin(origin),
   fCallEnv(0),
   fMethod(0),
   fCanCall(0),
   fIsMacro(kFALSE),
   fIsGlobal(kFALSE)
{
   // Create a plugin handler. Called by TPluginManager.

   TString aclicMode, arguments, io;
   TString fname = gSystem->SplitAclicMode(fPlugin, aclicMode, arguments, io);
   Bool_t validMacro = kFALSE;
   if (fname.EndsWith(".C") || fname.EndsWith(".cxx") || fname.EndsWith(".cpp") ||
       fname.EndsWith(".cc"))
      validMacro = kTRUE;

   if (validMacro && gROOT->LoadMacro(fPlugin, 0, kTRUE) == 0)
      fIsMacro = kTRUE;

   if (fCtor.Contains("::")) {
      fIsGlobal = kTRUE;
      fCtor = fCtor.Strip(TString::kLeading, ':');
   }
}

//______________________________________________________________________________
TPluginHandler::~TPluginHandler()
{
   // Cleanup plugin handler object.

   delete fCallEnv;
}

//______________________________________________________________________________
Bool_t TPluginHandler::CanHandle(const char *base, const char *uri)
{
   // Check if regular expression appears in the URI, if so return kTRUE.
   // If URI = 0 always return kTRUE.

   if (fBase != base)
      return kFALSE;

   if (!uri || fRegexp == "*")
      return kTRUE;

   Bool_t wildcard = kFALSE;
   if (!fRegexp.MaybeRegexp())
      wildcard = kTRUE;

   TRegexp re(fRegexp, wildcard);
   TString ruri = uri;

   if (ruri.Index(re) != kNPOS)
      return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TPluginHandler::SetupCallEnv()
{
   // Setup ctor or static method call environment.

   fCanCall = -1;

   // check if class exists
   TClass *cl = TClass::GetClass(fClass);
   if (!cl && !fIsGlobal) {
      Error("SetupCallEnv", "class %s not found in plugin %s", fClass.Data(),
            fPlugin.Data());
      return;
   }

   // split method and prototype strings
   TString method = fCtor(0, fCtor.Index("("));
   TString proto  = fCtor(fCtor.Index("(")+1, fCtor.Index(")")-fCtor.Index("(")-1);

   if (fIsGlobal) {
      cl = 0;
      fMethod = gROOT->GetGlobalFunctionWithPrototype(method, proto, kTRUE);
   } else {
      fMethod = cl->GetMethodWithPrototype(method, proto);
   }

   if (!fMethod) {
      if (fIsGlobal)
         Error("SetupCallEnv", "global function %s not found", method.Data());
      else
         Error("SetupCallEnv", "method %s not found in class %s", method.Data(),
               fClass.Data());
      return;
   }

   if (!fIsGlobal && !(fMethod->Property() & kIsPublic)) {
      Error("SetupCallEnv", "method %s is not public", method.Data());
      return;
   }

   fCallEnv = new TMethodCall;
   fCallEnv->InitWithPrototype(cl, method, proto);

   fCanCall = 1;

   return;
}

//______________________________________________________________________________
Int_t TPluginHandler::CheckPlugin() const
{
   // Check if the plugin library for this handler exits. Returns 0
   // when it exists and -1 in case the plugin does not exist.

   if (fIsMacro) {
      if (TClass::GetClass(fClass)) return 0;
      return gROOT->LoadMacro(fPlugin, 0, kTRUE);
   } else
      return gROOT->LoadClass(fClass, fPlugin, kTRUE);
}

//______________________________________________________________________________
Int_t TPluginHandler::LoadPlugin()
{
   // Load the plugin library for this handler. Returns 0 on successful loading
   // and -1 in case the library does not exist or in case of error.

   if (fIsMacro) {
      if (TClass::GetClass(fClass)) return 0;
      return gROOT->LoadMacro(fPlugin);
   } else {
      // first call also loads dependent libraries declared via the rootmap file
      if (gROOT->LoadClass(fClass)) return 0;
      return gROOT->LoadClass(fClass, fPlugin);
   }
}

//______________________________________________________________________________
Long_t TPluginHandler::ExecPlugin(Int_t va_(nargs), ...)
{
   // Execute ctor for this plugin and return pointer to object of specific
   // class. User must cast the returned long to the correct class.
   // This method accepts a variable number of arguments to be passed
   // to the ctor, where nargs is the number of arguments, followed
   // by nargs arguments. Returns 0 in case of error.

   if (fCtor.IsNull()) {
      Error("ExecPlugin", "no ctor specified for this handler %s", fClass.Data());
      return 0;
   }

   if (!fCallEnv && !fCanCall)
      SetupCallEnv();

   if (fCanCall == -1)
      return 0;

   if (nargs < fMethod->GetNargs() - fMethod->GetNargsOpt() ||
       nargs > fMethod->GetNargs()) {
      Error("ExecPlugin", "nargs (%d) not consistent with expected number of arguments ([%d-%d])",
            nargs, fMethod->GetNargs() - fMethod->GetNargsOpt(),
            fMethod->GetNargs());
      return 0;
   }

   R__LOCKGUARD2(gCINTMutex);

   fCallEnv->ResetParam();

   if (nargs > 0) {
      TIter next(fMethod->GetListOfMethodArgs());
      TMethodArg *arg;

      va_list ap;
      va_start(ap, va_(nargs));

      for (int i = 0; i < nargs; i++) {
         arg = (TMethodArg*) next();
         TString type = arg->GetFullTypeName();
         TDataType *dt = gROOT->GetType(type);
         if (dt)
            type = dt->GetFullTypeName();
         if (arg->Property() & (kIsPointer | kIsArray | kIsReference))
            fCallEnv->SetParam((Long_t) va_arg(ap, void*));
         else if (type == "bool")
            fCallEnv->SetParam((Long_t) va_arg(ap, int));  // bool is promoted to int
         else if (type == "char" || type == "unsigned char")
            fCallEnv->SetParam((Long_t) va_arg(ap, int));  // char is promoted to int
         else if (type == "short" || type == "unsigned short")
            fCallEnv->SetParam((Long_t) va_arg(ap, int));  // short is promoted to int
         else if (type == "int" || type == "unsigned int")
            fCallEnv->SetParam((Long_t) va_arg(ap, int));
         else if (type == "long" || type == "unsigned long")
            fCallEnv->SetParam((Long_t) va_arg(ap, long));
         else if (type == "long long")
            fCallEnv->SetParam((Long64_t) va_arg(ap, Long64_t));
         else if (type == "unsigned long long")
            fCallEnv->SetParam((ULong64_t) va_arg(ap, ULong64_t));
         else if (type == "float")
            fCallEnv->SetParam((Double_t) va_arg(ap, double));  // float is promoted to double
         else if (type == "double")
            fCallEnv->SetParam((Double_t) va_arg(ap, double));
      }

      va_end(ap);
   }

   Long_t ret;
   fCallEnv->Execute(ret);

   return ret;
}

//______________________________________________________________________________
void TPluginHandler::Print(Option_t *opt) const
{
   // Print info about the plugin handler. If option is "a" print
   // also the ctor's that will be used.

   const char *exist = "";
   if (CheckPlugin() == -1)
      exist = " [*]";

   Printf("%-20s %-13s %-18s %s%s", fBase.Data(), fRegexp.Data(),
          fClass.Data(), fPlugin.Data(), exist);
   if (strchr(opt, 'a')) {
      if (strlen(exist) == 0) {
         TString lib = fPlugin;
         if (!lib.BeginsWith("lib"))
            lib = "lib" + lib;
         char *path = gSystem->DynamicPathName(lib, kTRUE);
         if (path) Printf("  [Lib:  %s]", path);
         delete [] path;
      }
      Printf("  [Ctor: %s]", fCtor.Data());
      Printf("  [origin: %s]", fOrigin.Data());
   }
}


ClassImp(TPluginManager)

//______________________________________________________________________________
TPluginManager::~TPluginManager()
{
   // Clean up the plugin manager.

   delete fHandlers;
   delete fBasesLoaded;
}

//______________________________________________________________________________
void TPluginManager::LoadHandlersFromEnv(TEnv *env)
{
   // Load plugin handlers specified in config file, like:
   //    Plugin.TFile:       ^rfio:    TRFIOFile      RFIO  "TRFIOFile(...)"
   //    Plugin.TSQLServer:  ^mysql:   TMySQLServer   MySQL "TMySQLServer(...)"
   //    +Plugin.TSQLServer: ^pgsql:   TPgSQLServer   PgSQL "TPgSQLServer(...)"
   // The + allows the extension of an already defined resource (see TEnv).

   if (!env) return;

   TIter next(env->GetTable());
   TEnvRec *er;

   while ((er = (TEnvRec*) next())) {
      const char *s;
      if ((s = strstr(er->GetName(), "Plugin."))) {
         // use s, i.e. skip possible OS and application prefix to Plugin.
         // so that GetValue() takes properly care of returning the value
         // for the specified OS and/or application
         const char *val = env->GetValue(s, (const char*)0);
         if (val) {
            Int_t cnt = 0;
            char *v = StrDup(val);
            s += 7;
            while (1) {
               TString regexp = strtok(!cnt ? v : 0, "; ");
               if (regexp.IsNull()) break;
               TString clss   = strtok(0, "; ");
               if (clss.IsNull()) break;
               TString plugin = strtok(0, "; ");
               if (plugin.IsNull()) break;
               TString ctor = strtok(0, ";\"");
               if (!ctor.Contains("("))
                  ctor = strtok(0, ";\"");
               AddHandler(s, regexp, clss, plugin, ctor, "TEnv");
               cnt++;
            }
            delete [] v;
         }
      }
   }
}

//______________________________________________________________________________
void TPluginManager::LoadHandlerMacros(const char *path)
{
   // Load all plugin macros from the specified path/base directory.

   void *dirp = gSystem->OpenDirectory(path);
   if (dirp) {
      if (gDebug > 0)
         Info("LoadHandlerMacros", "%s", path);
      TSortedList macros;
      macros.SetOwner();
      const char *f1;
      while ((f1 = gSystem->GetDirEntry(dirp))) {
         TString f = f1;
         if (f[0] == 'P' && f.EndsWith(".C")) {
            const char *p = gSystem->ConcatFileName(path, f);
            if (!gSystem->AccessPathName(p, kReadPermission)) {
               macros.Add(new TObjString(p));
            }
            delete [] p;
         }
      }
      // load macros in alphabetical order
      TIter next(&macros);
      TObjString *s;
      while ((s = (TObjString*)next())) {
         if (gDebug > 1)
            Info("LoadHandlerMacros", "   plugin macro: %s", s->String().Data());
         Long_t res;
         if ((res = gROOT->Macro(s->String(), 0, kFALSE)) < 0) {
            Error("LoadHandlerMacros", "pluging macro %s returned %ld",
                  s->String().Data(), res);
         }
      }
   }
   gSystem->FreeDirectory(dirp);
}

//______________________________________________________________________________
void TPluginManager::LoadHandlersFromPluginDirs(const char *base)
{
   // Load plugin handlers specified via macros in a list of plugin
   // directories. The $ROOTSYS/etc/plugins is the default top plugin directory
   // specified in $ROOTSYS/etc/system.rootrc. The macros must have names
   // like <BaseClass>/PX0_<PluginClass>.C, e.g.:
   //    TFile/P10_TRFIOFile.C, TSQLServer/P20_TMySQLServer.C, etc.
   // to allow easy sorting and grouping. If the BaseClass is in a namespace
   // the directory must have the name NameSpace@@BaseClass as : is a reserved
   // pathname character on some operating systems. Macros not beginning with
   // 'P' and ending with ".C" are ignored. If base is specified only plugin
   // macros for that base class are loaded. The macros typically
   // should look like:
   //   void P10_TDCacheFile()
   //   {
   //       gPluginMgr->AddHandler("TFile", "^dcache", "TDCacheFile",
   //          "DCache", "TDCacheFile(const char*,Option_t*,const char*,Int_t)");
   //   }
   // In general these macros should not cause side effects, by changing global
   // ROOT state via, e.g. gSystem calls, etc. However, in specific cases
   // this might be useful, e.g. adding a library search path, adding a specific
   // dependency, check on some OS or ROOT capability or downloading
   // of the plugin.

   if (!fBasesLoaded) {
      fBasesLoaded = new THashTable();
      fBasesLoaded->SetOwner();
      
      // make sure we have gPluginMgr availble in the plugin macros
      gInterpreter->InitializeDictionaries();
   }
   TString sbase = base;
   if (sbase != "") {
      sbase.ReplaceAll("::", "@@");
      if (fBasesLoaded->FindObject(sbase))
         return;
      fBasesLoaded->Add(new TObjString(sbase));
   }

   fReadingDirs = kTRUE;

   TString plugindirs = gEnv->GetValue("Root.PluginPath", (char*)0);
#ifdef WIN32
   TObjArray *dirs = plugindirs.Tokenize(";");
#else
   TObjArray *dirs = plugindirs.Tokenize(":");
#endif
   TString d;
   for (Int_t i = 0; i < dirs->GetEntriesFast(); i++) {
      d = ((TObjString*)dirs->At(i))->GetString();
      // check if directory already scanned
      Int_t skip = 0;
      for (Int_t j = 0; j < i; j++) {
         TString pd = ((TObjString*)dirs->At(j))->GetString();
         if (pd == d) {
            skip++;
            break;
         }
      }
      if (!skip) {
         if (sbase != "") {
            const char *p = gSystem->ConcatFileName(d, sbase);
            LoadHandlerMacros(p);
            delete [] p;
         } else {
            void *dirp = gSystem->OpenDirectory(d);
            if (dirp) {
               if (gDebug > 0)
                  Info("LoadHandlersFromPluginDirs", "%s", d.Data());
               const char *f1;
               while ((f1 = gSystem->GetDirEntry(dirp))) {
                  TString f = f1;
                  const char *p = gSystem->ConcatFileName(d, f);
                  LoadHandlerMacros(p);
                  fBasesLoaded->Add(new TObjString(f));
                  delete [] p;
               }
            }
            gSystem->FreeDirectory(dirp);
         }
      }
   }

   delete dirs;
   fReadingDirs = kFALSE;
}

//______________________________________________________________________________
void TPluginManager::AddHandler(const char *base, const char *regexp,
                                const char *className, const char *pluginName,
                                const char *ctor, const char *origin)
{
   // Add plugin handler to the list of handlers. If there is already a
   // handler defined for the same base and regexp it will be replaced.

   if (!fHandlers) {
      fHandlers = new TList;
      fHandlers->SetOwner();
   }

   // make sure there is no previous handler for the same case
   RemoveHandler(base, regexp);

   if (fReadingDirs)
      origin = gInterpreter->GetCurrentMacroName();

   TPluginHandler *h = new TPluginHandler(base, regexp, className,
                                          pluginName, ctor, origin);
   fHandlers->Add(h);
}

//______________________________________________________________________________
void TPluginManager::RemoveHandler(const char *base, const char *regexp)
{
   // Remove handler for the specified base class and the specified
   // regexp. If regexp=0 remove all handlers for the specified base.

   if (!fHandlers) return;

   TIter next(fHandlers);
   TPluginHandler *h;

   while ((h = (TPluginHandler*) next())) {
      if (h->fBase == base) {
         if (!regexp || h->fRegexp == regexp) {
            fHandlers->Remove(h);
            delete h;
         }
      }
   }
}

//______________________________________________________________________________
TPluginHandler *TPluginManager::FindHandler(const char *base, const char *uri)
{
   // Returns the handler if there exists a handler for the specified URI.
   // The uri can be 0 in which case the first matching plugin handler
   // will be returned. Returns 0 in case handler is not found.

   LoadHandlersFromPluginDirs(base);

   TIter next(fHandlers);
   TPluginHandler *h;

   while ((h = (TPluginHandler*) next())) {
      if (h->CanHandle(base, uri)) {
         if (gDebug > 0)
            Info("FindHandler", "found plugin for %s", h->GetClass());
         return h;
      }
   }

   if (gDebug > 2) {
      if (uri)
         Info("FindHandler", "did not find plugin for class %s and uri %s", base, uri);
      else
         Info("FindHandler", "did not find plugin for class %s", base);
   }

   return 0;
}

//______________________________________________________________________________
void TPluginManager::Print(Option_t *opt) const
{
   // Print list of registered plugin handlers. If option is "a" print
   // also the ctor's that will be used.

   if (!fHandlers) return;

   TIter next(fHandlers);
   TPluginHandler *h;
   Int_t cnt = 0, cntmiss = 0;

   Printf("=====================================================================");
   Printf("Base                 Regexp        Class              Plugin");
   Printf("=====================================================================");

   while ((h = (TPluginHandler*) next())) {
      cnt++;
      h->Print(opt);
      if (h->CheckPlugin() == -1)
         cntmiss++;
   }
   Printf("=====================================================================");
   Printf("%d plugin handlers registered", cnt);
   Printf("[*] %d %s not available", cntmiss, cntmiss==1 ? "plugin" : "plugins");
   Printf("=====================================================================\n");
}

//______________________________________________________________________________
Int_t TPluginManager::WritePluginMacros(const char *dir, const char *plugin) const
{
   // Write in the specified directory the plugin macros. If plugin is specified
   // and if it is a base class all macros for that base will be written. If it
   // is a plugin class name, only that one macro will be written. If plugin
   // is 0 all macros are written. Returns -1 if dir does not exist, 0 otherwise.

   const_cast<TPluginManager*>(this)->LoadHandlersFromPluginDirs();

   if (!fHandlers) return 0;

   TString d;
   if (!dir || !dir[0])
      d = ".";
   else
      d = dir;

   if (gSystem->AccessPathName(d, kWritePermission)) {
      Error("WritePluginMacros", "cannot write in directory %s", d.Data());
      return -1;
   }

   TString base;
   Int_t   idx = 0;

   TObjLink *lnk = fHandlers->FirstLink();
   while (lnk) {
      TPluginHandler *h = (TPluginHandler *) lnk->GetObject();
      if (plugin && strcmp(plugin, h->fBase) && strcmp(plugin, h->fClass)) {
         lnk = lnk->Next();
         continue;
      }
      if (base != h->fBase) {
         idx = 10;
         base = h->fBase;
      } else
         idx += 10;
      const char *dd = gSystem->ConcatFileName(d, h->fBase);
      TString sdd = dd;
      sdd.ReplaceAll("::", "@@");
      delete [] dd;
      if (gSystem->AccessPathName(sdd, kWritePermission)) {
         if (gSystem->MakeDirectory(sdd) < 0) {
            Error("WritePluginMacros", "cannot create directory %s", sdd.Data());
            return -1;
         }
      }
      TString fn;
      fn.Form("P%03d_%s.C", idx, h->fClass.Data());
      const char *fd = gSystem->ConcatFileName(sdd, fn);
      FILE *f = fopen(fd, "w");
      if (f) {
         fprintf(f, "void P%03d_%s()\n{\n", idx, h->fClass.Data());
         fprintf(f, "   gPluginMgr->AddHandler(\"%s\", \"%s\", \"%s\",\n",
                 h->fBase.Data(), h->fRegexp.Data(), h->fClass.Data());
         fprintf(f, "      \"%s\", \"%s\");\n", h->fPlugin.Data(), h->fCtor.Data());
         
         // check for different regexps cases for the same base + class and
         // put them all in the same macro
         TObjLink *lnk2 = lnk->Next();
         while (lnk2) {
            TPluginHandler *h2 = (TPluginHandler *) lnk2->GetObject();
            if (h->fBase != h2->fBase || h->fClass != h2->fClass)
               break;
            
            fprintf(f, "   gPluginMgr->AddHandler(\"%s\", \"%s\", \"%s\",\n",
                    h2->fBase.Data(), h2->fRegexp.Data(), h2->fClass.Data());
            fprintf(f, "      \"%s\", \"%s\");\n", h2->fPlugin.Data(), h2->fCtor.Data());
            
            lnk  = lnk2;
            lnk2 = lnk2->Next();
         }
         fprintf(f, "}\n");
         fclose(f);
      }
      delete [] fd;
      lnk = lnk->Next();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TPluginManager::WritePluginRecords(const char *envFile, const char *plugin) const
{
   // Write in the specified environment config file the plugin records. If
   // plugin is specified and if it is a base class all records for that
   // base will be written. If it is a plugin class name, only that one
   // record will be written. If plugin is 0 all macros are written.
   // If envFile is 0 or "" the records are written to stdout.
   // Returns -1 if envFile cannot be created or opened, 0 otherwise.

   const_cast<TPluginManager*>(this)->LoadHandlersFromPluginDirs();

   if (!fHandlers) return 0;

   FILE *fd;
   if (!envFile || !envFile[0])
      fd = stdout;
   else
      fd = fopen(envFile, "w+");

   if (!fd) {
      Error("WritePluginRecords", "error opening file %s", envFile);
      return -1;
   }

   TString base, base2;
   Int_t   idx = 0;

   TObjLink *lnk = fHandlers->FirstLink();
   while (lnk) {
      TPluginHandler *h = (TPluginHandler *) lnk->GetObject();
      if (plugin && strcmp(plugin, h->fBase) && strcmp(plugin, h->fClass)) {
         lnk = lnk->Next();
         continue;
      }
      if (base != h->fBase) {
         idx = 1;
         base = h->fBase;
         base2 = base;
         base2.ReplaceAll("::", "@@");
      } else
         idx += 1;

      if (idx == 1)
         fprintf(fd, "Plugin.%s: %s %s %s \"%s\"\n", base2.Data(), h->fRegexp.Data(),
                 h->fClass.Data(), h->fPlugin.Data(), h->fCtor.Data());
      else
         fprintf(fd, "+Plugin.%s: %s %s %s \"%s\"\n", base2.Data(), h->fRegexp.Data(),
                 h->fClass.Data(), h->fPlugin.Data(), h->fCtor.Data());

      // check for different regexps cases for the same base + class and
      // put them all in the same macro
      TObjLink *lnk2 = lnk->Next();
      while (lnk2) {
         TPluginHandler *h2 = (TPluginHandler *) lnk2->GetObject();
         if (h->fBase != h2->fBase || h->fClass != h2->fClass)
            break;

         fprintf(fd, "+Plugin.%s: %s %s %s \"%s\"\n", base2.Data(), h2->fRegexp.Data(),
                 h2->fClass.Data(), h2->fPlugin.Data(), h2->fCtor.Data());

         lnk  = lnk2;
         lnk2 = lnk2->Next();
      }
      lnk = lnk->Next();
   }

   if (envFile && envFile[0])
      fclose(fd);

   return 0;
}
