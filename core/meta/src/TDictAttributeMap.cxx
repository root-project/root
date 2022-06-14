// @(#)root/meta:$Id:$
// Author: Bianca-Cristina Cristescu   03/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TDictAttributeMap
The ROOT object has a list of properties which are stored and
retrieved using TDictAttributeMap.
TDictAttributeMap maps the property keys of the object to their
values.
*/

#include "TDictAttributeMap.h"
#include "THashTable.h"
#include "TNamed.h"
#include "TParameter.h"


ClassImp(TDictAttributeMap);

////////////////////////////////////////////////////////////////////////////////
///Default constructor.

TDictAttributeMap::TDictAttributeMap()
{
   fStringProperty.SetOwner(kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
///Default destructor.

TDictAttributeMap::~TDictAttributeMap()
{
}

////////////////////////////////////////////////////////////////////////////////
///Add a property with a String value to the TDictAttributeMap.
///Parameters: key and char* value of the property.

void TDictAttributeMap::AddProperty(const char* key, const char* value)
{
   //Add the property pair name - Int value to the hash table.
   fStringProperty.Add(new TNamed(key, value));
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TDictAttributeMap::HasKey(const char* key) const
{
   //Check whether the class has a property using the key.

   if (fStringProperty.FindObject(key))
      return true;
   return false;
}

////////////////////////////////////////////////////////////////////////////////
///Access the value of a String property using the key.

const char* TDictAttributeMap::GetPropertyAsString(const char* key) const
{
   //Copy object into found to avoid calling the function two times.
   TObject* found = fStringProperty.FindObject(key);
   if(found)
      return found->GetTitle();
   else
      //Show an error message if the key is not found.
      Error("GetPropertyAsString"
      , "Could not find property with String value for this key: %s", key);
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
///Remove a String property from the attribute map specified by the key.
///Returns the TString property removed or NULL if the property does not exist.

TString TDictAttributeMap::RemovePropertyString(const char* key)
{
   TObject *property = fStringProperty.FindObject(key);
   if (property) {
      fStringProperty.Remove(property);
      return property->GetTitle();
   }
   return TString(0);
}

Bool_t TDictAttributeMap::RemoveProperty(const char* key)
{
   //Remove a property from the attribute map specified by the key.
   //Returns true if property exists and was removed, false if property
   //does not exist.

   if (TObject *property = fStringProperty.FindObject(key)) {
      fStringProperty.Remove(property);
      return true;
   }
   return false;
}

////////////////////////////////////////////////////////////////////////////////
///Deletes all the properties of the class.

void TDictAttributeMap::Clear(Option_t* /*option = ""*/)
{
   fStringProperty.Delete();
}
