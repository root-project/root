// @(#)root/base:$Name:  $:$Id: TPluginManager.cxx,v 1.4 2002/01/27 17:22:47 rdm Exp $
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
// functionality of a specific base class. For example, to extend the   //
// base class TFile to be able to read RFIO files one needs to load     //
// the plugin library libRFIO.so which defines the TRFIOFile class.     //
// This loading should be triggered when a given URI contains a         //
// regular expression defined by the handler. Handlers can be defined   //
// for example as resources in the .rootrc file, e.g.:                  //
//                                                                      //
//   Plugin.TFile:       ^rfio:    TRFIOFile     RFIO                   //
//   Plugin.TSQLServer:  ^mysql:   TMySQLServer  MySQL                  //
//   +Plugin.TSQLServer: ^pgsql:   TPgSQLServer  PgSQL                  //
//                                                                      //
// Plugin handlers can also be registered at run time, e.g.:            //
//                                                                      //
//   gROOT->GetPluginManager()->AddHandler("TSQLServer", "^sapdb:",     //
//                                         "TSapDBServer", "SapDB");    //
//                                                                      //
// A list of currently defined handlers can be printed using:           //
//                                                                      //
//    gROOT->GetPluginManager()->Print();                               //
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

ClassImp(TPluginHandler)
ClassImp(TPluginManager)


//______________________________________________________________________________
TPluginHandler::TPluginHandler(const char *base, const char *regexp,
                               const char *className, const char *pluginName)
{
   // Create a plugin handler. Called by TPluginManager.

   fBase    = base;
   fRegexp  = regexp;
   fClass   = className;
   fPlugin  = pluginName;
}

//______________________________________________________________________________
Bool_t TPluginHandler::CanHandle(const char *base, const char *uri)
{
   // Check if regular expression appears in the URI, if so return kTRUE.

   if (fBase != base)
      return kFALSE;

   TRegexp re(fRegexp, kFALSE);
   TString ruri = uri;

   if (ruri.Index(re) != kNPOS)
      return kTRUE;
   return kFALSE;
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
TPluginManager::~TPluginManager()
{
   // Clean up the plugin manager.

   delete fHandlers;
}

//______________________________________________________________________________
void TPluginManager::LoadHandlersFromEnv(TEnv *env)
{
   // Load plugin handlers specified in config file, like:
   //    Plugin.TFile:       ^rfio:    TRFIOFile      RFIO
   //    Plugin.TSQLServer:  ^mysql:   TMySQLServer   MySQL
   //    +Plugin.TSQLServer: ^pgsql:   TPgSQLServer   PgSQL
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
               TString regexp = strtok(!cnt ? v : 0, ",; ");
               if (regexp.IsNull()) break;
               TString clss   = strtok(0, ",; ");
               if (clss.IsNull()) break;
               TString plugin = strtok(0, ",; ");
               if (plugin.IsNull()) break;
               AddHandler(s, regexp, clss, plugin);
               cnt++;
            }
            delete [] v;
         }
      }
   }
}

//______________________________________________________________________________
void TPluginManager::AddHandler(const char *base, const char *regexp,
                               const char *className, const char *pluginName)
{
   // Add plugin handler to the list of handlers. If there is already a
   // handler defined for the same base and regexp it will be replaced.

   if (!fHandlers) {
      fHandlers = new TList;
      fHandlers->IsOwner();
   }

   // make sure there is no previous handler for the same case
   RemoveHandler(base, regexp);

   TPluginHandler *h = new TPluginHandler(base, regexp, className, pluginName);
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
   // Returns 0 in case handler is not found.

   if (!fHandlers) return 0;

   TIter next(fHandlers);
   TPluginHandler *h;

   while ((h = (TPluginHandler*) next())) {
      if (h->CanHandle(base, uri)) {
         if (gDebug > 0)
            Printf("<TPluginManager::FindHandler>: found plugin for %s",
                   h->GetClass());
         return h;
      }
   }

   if (gDebug > 0)
      Printf("<TPluginManager::FindHandler>: did not find plugin for handling %s",
             uri);

   return 0;
}

//______________________________________________________________________________
void TPluginManager::Print(Option_t *) const
{
   // Print list of registered plugin handlers.

   if (!fHandlers) return;

   TIter next(fHandlers);
   TPluginHandler *h;

   printf("=====================================================================\n");
   printf("Base               Regexp          Class              Plugin\n");
   printf("=====================================================================\n");

   while ((h = (TPluginHandler*) next())) {
      const char *exist = "";
      if (h->CheckPlugin() == -1)
         exist = " [*]";
      printf("%-18s %-15s %-18s %s%s\n", h->fBase.Data(), h->fRegexp.Data(),
             h->fClass.Data(), h->fPlugin.Data(), exist);
   }
   printf("=====================================================================\n");
   printf("[*] plugin not available\n");
   printf("=====================================================================\n\n");
}
