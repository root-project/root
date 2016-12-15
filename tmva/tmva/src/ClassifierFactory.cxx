// @(#)Root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Joerg Stelzer   <stelzer@cern.ch>        - DESY, Germany                  *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      DESY, Germany                                                             *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::ClassifierFactory
\ingroup TMVA
This is the MVA factory.
*/

#include "TMVA/ClassifierFactory.h"

#include "RtypesCore.h"
#include "TString.h"

#include <assert.h>
#include <iostream>

class TString;

///
/// Initialize static singleton pointer
///
TMVA::ClassifierFactory* TMVA::ClassifierFactory::fgInstance = 0;

////////////////////////////////////////////////////////////////////////////////
/// access to the ClassifierFactory singleton
/// creates the instance if needed

TMVA::ClassifierFactory& TMVA::ClassifierFactory::Instance()
{
   if (!fgInstance) fgInstance = new TMVA::ClassifierFactory();

   return *fgInstance;
}

////////////////////////////////////////////////////////////////////////////////
/// destroy the singleton instance

void TMVA::ClassifierFactory::DestroyInstance()
{
   if (fgInstance!=0) delete fgInstance;
}

////////////////////////////////////////////////////////////////////////////////
/// registers a classifier creator function under the method type name

Bool_t TMVA::ClassifierFactory::Register( const std::string &name, Creator creator )
{
   if(fCalls.find(name) != fCalls.end())
      {
         std::cerr << "ClassifierFactory<>::Register - " << name << " already exists" << std::endl;
         return false;
      }

   return fCalls.insert(CallMap::value_type(name, creator)).second;
}

////////////////////////////////////////////////////////////////////////////////
/// unregisters a classifier type name

Bool_t TMVA::ClassifierFactory::Unregister( const std::string &name )
{
   return fCalls.erase(name) == 1;
}

////////////////////////////////////////////////////////////////////////////////
/// creates the method if needed based on the method name using the
/// creator function the factory has stored

TMVA::IMethod* TMVA::ClassifierFactory::Create( const std::string &name,
                                                const TString& job,
                                                const TString& title,
                                                DataSetInfo& dsi,
                                                const TString& option )
{
   // additional options are passed to the creator function (the
   // method constructor)

   CallMap::const_iterator it = fCalls.find(name);

   // handle unknown algorithm request
   if (it == fCalls.end()) {
      std::cerr << "ClassifierFactory<>::Create - don't know anything about " << name << std::endl;
      assert(0);
   }

   return (it->second)(job, title, dsi, option);
}

////////////////////////////////////////////////////////////////////////////////
/// creates the method if needed based on the method name using the
/// creator function the factory has stored

TMVA::IMethod* TMVA::ClassifierFactory::Create( const std::string &name,
                                                DataSetInfo& dsi,
                                                const TString& weightfile )
{
   // additional options are passed to the creator function (the
   // second method constructor)

   CallMap::const_iterator it = fCalls.find(name);

   // handle unknown algorithm request
   if (it == fCalls.end()) {
      std::cerr << "ClassifierFactory<>::Create - don't know anything about " << name << std::endl;
      assert(0);
   }

   return (it->second)("", "", dsi, weightfile);
}

////////////////////////////////////////////////////////////////////////////////
/// returns a vector of the method type names of registered methods

const std::vector<std::string> TMVA::ClassifierFactory::List() const
{
   std::vector<std::string> svec;

   CallMap::const_iterator it = fCalls.begin();
   for (; it != fCalls.end(); ++it) svec.push_back(it -> first);

   return svec;
}

////////////////////////////////////////////////////////////////////////////////
/// prints the registered method type names

void TMVA::ClassifierFactory::Print() const
{
   std::cout << "Print: ClassifierFactory<> knows about " << fCalls.size() << " objects" << std::endl;

   CallMap::const_iterator it = fCalls.begin();
   for (; it != fCalls.end(); ++it) std::cout << "Registered object name " << it -> first << std::endl;
}
