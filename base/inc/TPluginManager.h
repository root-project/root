// @(#)root/base:$Name:  $:$Id: TPluginManager.h,v 1.3 2002/07/16 13:57:14 rdm Exp $
// Author: Fons Rademakers   26/1/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPluginManager
#define ROOT_TPluginManager


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

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TEnv;
class TList;
class TMethodCall;
class TPluginManager;


class TPluginHandler : public TObject {

friend class TPluginManager;

private:
   TString      fBase;      // base class which will be extended by plugin
   TString      fRegexp;    // regular expression which must be matched in URI
   TString      fClass;     // class to be loaded from plugin library
   TString      fPlugin;    // plugin library which should contain fClass
   TString      fCtor;      // ctor used to instantiate object of fClass
   TMethodCall *fCallEnv;   //!ctor method call environment
   TMethod     *fMethod;    //!ctor method
   Int_t        fCanCall;   //!if 1 fCallEnv is ok, -1 fCallEnv is not ok

   TPluginHandler() { fCallEnv = 0; fCanCall = 0; }
   TPluginHandler(const char *base, const char *regexp,
                  const char *className, const char *pluginName,
                  const char *ctor);

   ~TPluginHandler();

   const char *GetBase() const { return fBase; }
   const char *GetRegexp() const { return fRegexp; }
   const char *GetPlugin() const { return fPlugin; }
   const char *GetCtor() const { return fCtor; }

   Bool_t CanHandle(const char *base, const char *uri);
   void   SetupCallEnv();

public:
   const char *GetClass() const { return fClass; }
   Int_t       CheckPlugin();
   Int_t       LoadPlugin();
   Long_t      ExecPlugin(Int_t nargs, ...);

   ClassDef(TPluginHandler,1)  // Handler for plugin libraries
};


class TPluginManager : public TObject {

private:
   TList  *fHandlers;    // list of plugin handlers

public:
   TPluginManager() { fHandlers = 0; }
   ~TPluginManager();

   void   LoadHandlersFromEnv(TEnv *env);
   void   AddHandler(const char *base, const char *regexp,
                     const char *className, const char *pluginName,
                     const char *ctor = 0);
   void   RemoveHandler(const char *base, const char *regexp = 0);

   TPluginHandler *FindHandler(const char *base, const char *uri = 0);

   void   Print(Option_t *opt = "") const;

   ClassDef(TPluginManager,1)  // Manager for plugin handlers
};

#endif
