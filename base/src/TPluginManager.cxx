// @(#)root/base:$Name:  $:$Id: TPluginManager.cxx,v 1.8 2002/07/17 14:53:46 rdm Exp $
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
// by the handler. Handlers can be defined for example as resources     //
// in the .rootrc file, e.g.:                                           //
//                                                                      //
//   Plugin.TFile:       ^rfio:   TRFIOFile    RFIO   "<constructor>"   //
//   Plugin.TSQLServer:  ^mysql:  TMySQLServer MySQL  "<constructor>"   //
//   +Plugin.TSQLServer: ^pgsql:  TPgSQLServer PgSQL  "<constructor>"   //
//   Plugin.TVirtualFitter: *     TFitter      Minuit "TFitter(Int_t)"  //
//                                                                      //
// Where the + in front of Plugin.TSQLServer says that it extends the   //
// existing definition of TSQLServer, usefull when there is more than   //
// one plugin that can extend the same base class. The "<constructor>"  //
// should be the constructor or static method that generates an         //
// instance of the specified class. The * is a placeholder in case      //
// there is no need for a URI to differentiate between different        //
// plugins for the same base class. For the default plugins see         //
// $ROOTSYS/etc/system.rootrc.                                          //
//                                                                      //
// Plugin handlers can also be registered at run time, e.g.:            //
//                                                                      //
//   gROOT->GetPluginManager()->AddHandler("TSQLServer", "^sapdb:",     //
//                                         "TSapDBServer", "SapDB",     //
//             "TSapDBServer(const char*,const char*, const char*)");   //
//                                                                      //
// A list of currently defined handlers can be printed using:           //
//                                                                      //
//   gROOT->GetPluginManager()->Print(); // use option="a" to see ctors //
//                                                                      //
// The use of the plugin library manager removes all textual references //
// to hard-coded class and library names and the resulting dependencies //
// in the base classes. The plugin manager is used to extend a.o.       //
// TFile, TSQLServer, TGrid, etc. functionality.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPluginManager.h"
#include "TEnv.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TList.h"
#include "TOrdCollection.h"
#include "Varargs.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDataType.h"
#include "TMethodCall.h"

ClassImp(TPluginHandler)
ClassImp(TPluginManager)


//______________________________________________________________________________
TPluginHandler::TPluginHandler(const char *base, const char *regexp,
                               const char *className, const char *pluginName,
                               const char *ctor)
{
   // Create a plugin handler. Called by TPluginManager.

   fBase    = base;
   fRegexp  = regexp;
   fClass   = className;
   fPlugin  = pluginName;
   fCtor    = ctor;
   fCallEnv = 0;
   fCanCall = 0;
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

   if (!uri)
      return kTRUE;

   TRegexp re(fRegexp, kFALSE);
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
   TClass *cl = gROOT->GetClass(fClass);
   if (!cl) {
      Error("SetupCallEnv", "class %s not found in plugin %s", fClass.Data(),
            fPlugin.Data());
      return;
   }

   // split method and prototype strings
   TString method = fCtor(0, fCtor.Index("("));
   TString proto  = fCtor(fCtor.Index("(")+1, fCtor.Index(")")-fCtor.Index("(")-1);

   fMethod = cl->GetMethodWithPrototype(method, proto);
   if (!fMethod) {
      Error("SetupCallEnv", "method %s not found in class %s", method.Data(),
            fClass.Data());
      return;
   }

   if (!(fMethod->Property() & kIsPublic)) {
      Error("SetupCallEnv", "method %s in not public", method.Data());
      return;
   }

   fCallEnv = new TMethodCall;
   fCallEnv->InitWithPrototype(cl, method, proto);

   fCanCall = 1;

   return;
}

//______________________________________________________________________________
Int_t TPluginHandler::CheckPlugin()
{
   // Check if the plugin library for this handler exits. Returns 0 on
   // when it exists and -1 in case the library does not exist.

   return gROOT->LoadClass(fClass, fPlugin, kTRUE);
}

//______________________________________________________________________________
Int_t TPluginHandler::LoadPlugin()
{
   // Load the plugin library for this handler. Returns 0 on successful loading
   // and -1 in case the library does not exist or in case of error.

   return gROOT->LoadClass(fClass, fPlugin);
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

   if (!fCallEnv && fCanCall == 0)
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

   Long_t *args = 0;

   if (nargs > 0) {
      TIter next(fMethod->GetListOfMethodArgs());
      TMethodArg *arg;

      va_list ap;
      va_start(ap, va_(nargs));

      args = new Long_t[nargs];

      for (int i = 0; i < nargs; i++) {
         union {
           float f;
           long  l;
         } u;
         arg = (TMethodArg*) next();
         TString type = arg->GetFullTypeName();
         TDataType *dt = gROOT->GetType(type);
         if (dt)
            type = dt->GetFullTypeName();
         if (arg->Property() & (kIsPointer | kIsArray | kIsReference))
            args[i] = (Long_t) va_arg(ap, void*);
         else if (type == "bool")
            args[i] = (Long_t) va_arg(ap, int);  // bool is promoted to int
         else if (type == "char" || type == "unsigned char")
            args[i] = (Long_t) va_arg(ap, int);  // char is promoted to int
         else if (type == "short" || type == "unsigned short")
            args[i] = (Long_t) va_arg(ap, int);  // short is promoted to int
         else if (type == "int" || type == "unsigned int")
            args[i] = (Long_t) va_arg(ap, int);
         else if (type == "long" || type == "unsigned long")
            args[i] = (Long_t) va_arg(ap, long);
         else if (type == "float") {
            u.f = (Float_t) va_arg(ap, double);  // float is promoted to double
            args[i] = (Long_t) u.l;
         } else if (type == "double") {
            u.f = (Float_t) va_arg(ap, double);
            args[i] = (Long_t) u.l;
         }
      }

      va_end(ap);

      fCallEnv->SetParamPtrs(args);
   }

   Long_t ret;
   fCallEnv->Execute(ret);

   delete [] args;

   return ret;
}

//______________________________________________________________________________
TPluginManager::~TPluginManager()
{
   // Clean up the plugin manager.

   delete fHandlers;
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
         const char *val = env->GetValue(er->GetName(), (const char*)0);
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
               strtok(0, ";\"");
               TString ctor = strtok(0, ";\"");
               AddHandler(s, regexp, clss, plugin, ctor);
               cnt++;
            }
            delete [] v;
         }
      }
   }
}

//______________________________________________________________________________
void TPluginManager::AddHandler(const char *base, const char *regexp,
                                const char *className, const char *pluginName,
                                const char *ctor)
{
   // Add plugin handler to the list of handlers. If there is already a
   // handler defined for the same base and regexp it will be replaced.

   if (!fHandlers) {
      fHandlers = new TList;
      fHandlers->IsOwner();
   }

   // make sure there is no previous handler for the same case
   RemoveHandler(base, regexp);

   TPluginHandler *h = new TPluginHandler(base, regexp, className,
                                          pluginName, ctor);
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

   if (!fHandlers) return 0;

   TIter next(fHandlers);
   TPluginHandler *h;

   while ((h = (TPluginHandler*) next())) {
      if (h->CanHandle(base, uri)) {
         if (gDebug > 0)
            Info("FindHandler", "found plugin for %s", h->GetClass());
         return h;
      }
   }

   if (gDebug > 0)
      Info("FindHandler", "did not find plugin for %s", h->GetClass());

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

   printf("=====================================================================\n");
   printf("Base                 Regexp        Class              Plugin\n");
   printf("=====================================================================\n");

   while ((h = (TPluginHandler*) next())) {
      const char *exist = "";
      if (h->CheckPlugin() == -1)
         exist = " [*]";
      printf("%-20s %-13s %-18s %s%s\n", h->fBase.Data(), h->fRegexp.Data(),
             h->fClass.Data(), h->fPlugin.Data(), exist);
      if (strchr(opt, 'a'))
         printf("  [Ctor: %s]\n", h->fCtor.Data());
   }
   printf("=====================================================================\n");
   printf("[*] plugin not available\n");
   printf("=====================================================================\n\n");
}
