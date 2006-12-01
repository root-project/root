// @(#)root/reflex:$Name:  $:$Id: PluginService.cxx,v 1.1 2006/11/30 08:27:08 roiser Exp $
// Author: Pere Mato 2006

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/PluginService.h"
#include "Reflex/SharedLibrary.h"
#include "Reflex/Reflex.h"
#include "PluginFactoryMap.h"

#include <vector>

using namespace ROOT::Reflex;
using namespace std;

//-------------------------------------------------------------------------------
void* ROOT::Reflex::PluginService::Create( const string & name, 
                                       const Type & ret, 
                                       const vector<ValueObject> & arg) {
//-------------------------------------------------------------------------------
// Create a Plugin.
   static Object dummy; 
   vector<void*> argv;
   vector<Type>  argt;
   for ( vector<ValueObject>::const_iterator i = arg.begin(); i != arg.end(); i++ ){
      argv.push_back(i->Address());
      argt.push_back(Type(i->TypeOf(),0));  // Ignore argument CV qualifiers
   }
   Type signature = FunctionTypeBuilder(ret, argt);
   string factoryname = FactoryName(name);
   //---Look first is the member exists ----
   if( ! Instance().fFactories.FunctionMemberByName(factoryname) ) {
      string mapname = string(PLUGINSVC_FACTORY_NS) + "@@" + factoryname;
      
      int rett = Instance().LoadFactoryLib(mapname);
      if ( ! rett ) {
         if ( Debug() ) cout << "PluginService: Could not load library associated to plugin " << name << endl;
         return 0;
      }
   }
   Member m = Instance().fFactories.FunctionMemberByName(FactoryName(name), signature);
   if ( !m ) {
      if ( Debug() > 1 ) cout << "PluginService: Could not find factory for " << name << " with signature " << signature.Name() << endl;
      return 0; 
   }
   else {
      try {
         Object rett = m.Invoke(dummy, argv);
         return rett.Address();
      }
      catch (RuntimeError& e) {
         if ( Debug() ) cout << "PluginService: Got exception -> " << e.what() << endl;
         return 0;
      }
      catch (exception& e) {
         if ( Debug() ) cout << "PluginService: Got exception -> " << e.what() << endl;
         return 0;
      }      
   }
}


//-------------------------------------------------------------------------------
void* ROOT::Reflex::PluginService::CreateWithId(const Any& id,  
                                                std::string (*str)(const Any&),  
                                                bool(*cmp)(const Any&, const Any&), 
                                                const Type& ret, 
                                                const vector<ValueObject>& arg) {
//-------------------------------------------------------------------------------
// Create plugin with Id.
   static Object dummy; 
   vector<void*> argv;
   vector<Type>  argt;
   for ( vector<ValueObject>::const_iterator i = arg.begin(); i != arg.end(); i++ ){
      argv.push_back(i->Address());
      argt.push_back(Type(i->TypeOf(),0));  // Ignore argument CV qualifiers
   }
   Type signature = FunctionTypeBuilder(ret, argt);

   string factoryname = FactoryName(str(id));
   
   if( ! Instance().fFactories.FunctionMemberByName(factoryname) ) {
      string mapname = string(PLUGINSVC_FACTORY_NS) + "@@" + factoryname;
      int rett = Instance().LoadFactoryLib(mapname);
      if ( ! rett ) {
         if ( Debug() ) cout << "PluginSvc: Could not load library associated to plugin with ID" << str(id) << endl;
         return 0;
      }
   }
   
   //--- loop over members
   Member m;
   for (Member_Iterator it = Instance().fFactories.FunctionMember_Begin(); 
        it != Instance().fFactories.FunctionMember_End(); ++it ) {
      if (it->Properties().HasProperty("id")) {
         if ( cmp(it->Properties().PropertyValue("id"),id) ) {
            if (signature.IsEquivalentTo(it->TypeOf())) {
               m = *it;
               break;
            }
         }
      }
   }
   
   if ( !m ) {
      if ( Debug() > 1 ) cout << "PluginService: Could not find factory for " << str(id) << " with signature " << signature.Name() << endl;
      return 0; 
   }
   else {
      try {
         Object rett = m.Invoke(dummy, argv);
         return rett.Address();
      }
      catch (RuntimeError& e) {
         if ( Debug() ) cout << "PluginService: Got exception -> " << e.what() << endl;
         return 0;
      }
      catch (exception& e) {
         if ( Debug() ) cout << "PluginService: Got exception -> " << e.what() << endl;
         return 0;
      }      
   }
}


//-------------------------------------------------------------------------------
string ROOT::Reflex::PluginService::FactoryName(const string& name) {
//-------------------------------------------------------------------------------
// Create a factory name out of the parameter given.
   static string chars(":<> *&, ");
   string::size_type pos1 = name.find_first_not_of(' ');
   string::size_type pos2 = name.find_last_not_of(' ');
   string res = name.substr(pos1 == string::npos ? 0 : pos1, 
                            pos2 == string::npos ? name.length() - 1 : pos2 - pos1 + 1);
   for ( string::iterator i = res.begin(); i != res.end(); i++ ) 
      if ( chars.find(*i) != string::npos ) *i = '_';
   return res;
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PluginService::PluginService() : fDebugLevel(0) {
//-------------------------------------------------------------------------------
// Constructor.
   NamespaceBuilder(PLUGINSVC_FACTORY_NS);
   fFactories = Scope::ByName(PLUGINSVC_FACTORY_NS);
   fFactoryMap = new PluginFactoryMap();
}


//-------------------------------------------------------------------------------
ROOT::Reflex::PluginService::~PluginService() {
//-------------------------------------------------------------------------------
// Destructor.
   delete fFactoryMap;
}


//-------------------------------------------------------------------------------
PluginService& ROOT::Reflex::PluginService::Instance() {
//-------------------------------------------------------------------------------
// Get singleton.
   static PluginService s_instance;
   return s_instance;
}


//-------------------------------------------------------------------------------
int ROOT::Reflex::PluginService::Debug() {
//-------------------------------------------------------------------------------
// Get debug level.
   return Instance().fDebugLevel;
}
 

//-------------------------------------------------------------------------------
void ROOT::Reflex::PluginService::SetDebug(int l) {
//-------------------------------------------------------------------------------
// Set debug level.
   PluginFactoryMap::SetDebug(l);
   Instance().fDebugLevel = l;
}
    
 
//-------------------------------------------------------------------------------
int ROOT::Reflex::PluginService::LoadFactoryLib(const string& name) {
//-------------------------------------------------------------------------------
// Load libraries needed for a plugin to instantiate. 
   list<string> libs = Instance().fFactoryMap->GetLibraries(name);
   for ( list<string>::reverse_iterator i = libs.rbegin(); i != libs.rend(); i++ ) {
      SharedLibrary sl(*i);
      if ( sl.Load() ) {
         if ( Debug() ) cout << "PluginService: Loaded library  " << *i << endl;
      }
      else {
         if ( Debug() ) cout << "PluginService: Error loading library " << *i <<
                           endl << sl.Error() <<  endl;
         return 0;
      }
   }
   return 1;
}

