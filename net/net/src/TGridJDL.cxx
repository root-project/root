// @(#)root/net:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   28/9/2004
//         Jancurova.lucia@cern.ch Slovakia  29/9/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridJDL                                                             //
//                                                                      //
// Abstract base class to generate JDL files for job submission to the  //
// Grid.                                                                //
//                                                                      //
// Related classes are TGLiteJDL.                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGridJDL.h"
#include "TObjString.h"
#include "Riostream.h"


ClassImp(TGridJDL);

////////////////////////////////////////////////////////////////////////////////
/// Cleanup.

TGridJDL::~TGridJDL()
{
   Clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Clears the JDL information.

void TGridJDL::Clear(const Option_t*)
{
   fMap.DeleteAll();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets a value. If the entry already exists the old one is replaced.

void TGridJDL::SetValue(const char *key, const char *value)
{
   TObject *object = fMap.FindObject(key);
   TPair *pair = dynamic_cast<TPair*>(object);
   if (pair) {
      TObject *oldObject = pair->Key();
      if (oldObject) {
         TObject *oldValue = pair->Value();

         fMap.Remove(oldObject);
         delete oldObject;
         oldObject = 0;

         if (oldValue) {
            delete oldValue;
            oldValue = 0;
         }
      }
   }

   fMap.Add(new TObjString(key), new TObjString(value));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value corresponding to the provided key. Return 0 in case
/// key is not found.

const char *TGridJDL::GetValue(const char *key)
{
   if (!key)
      return 0;

   TObject *object = fMap.FindObject(key);
   if (!object)
      return 0;

   TPair *pair = dynamic_cast<TPair*>(object);
   if (!pair)
      return 0;

   TObject *value = pair->Value();
   if (!value)
      return 0;

   TObjString *string = dynamic_cast<TObjString*>(value);
   if (!string)
      return 0;

   return string->GetName();
}

////////////////////////////////////////////////////////////////////////////////
/// Sets a value. If the entry already exists the old one is replaced.

void TGridJDL::SetDescription(const char *key, const char* description)
{
   TObject *object = fDescriptionMap.FindObject(key);
   TPair *pair = dynamic_cast<TPair*>(object);
   if (pair) {
      TObject *oldObject = pair->Key();
      if (oldObject) {
         TObject *oldValue = pair->Value();

         fDescriptionMap.Remove(oldObject);
         delete oldObject;
         oldObject = 0;

         if (oldValue) {
            delete oldValue;
            oldValue = 0;
         }
      }
   }

   fDescriptionMap.Add(new TObjString(key), new TObjString(description));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the value corresponding to the provided key. Return 0 in case
/// key is not found.

const char *TGridJDL::GetDescription(const char *key)
{
   if (!key)
      return 0;

   TObject *object = fDescriptionMap.FindObject(key);
   if (!object)
      return 0;

   TPair *pair = dynamic_cast<TPair*>(object);
   if (!pair)
      return 0;

   TObject *value = pair->Value();
   if (!value)
      return 0;

   TObjString *string = dynamic_cast<TObjString*>(value);
   if (!string)
      return 0;

   return string->GetName();
}

////////////////////////////////////////////////////////////////////////////////
/// Adds quotes to the provided string.
///  E.g. Value --> "Value"

TString TGridJDL::AddQuotes(const char *value)
{
   TString temp = TString("\"");
   temp += value;
   temp += "\"";

   return temp;
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a value to a key value which hosts a set of values.
/// E.g. InputSandbox: {"file1","file2"}

void TGridJDL::AddToSet(const char *key, const char *value)
{
   const char *oldValue = GetValue(key);
   TString newString;
   if (oldValue)
      newString = oldValue;
   if (newString.IsNull()) {
      newString = "{";
   } else {
      newString.Remove(newString.Length()-1);
      newString += ",";
   }

   newString += AddQuotes(value);
   newString += "}";

   SetValue(key, newString);
}

////////////////////////////////////////////////////////////////////////////////
/// Adds a value to a key value which hosts a set of values.
/// E.g. InputSandbox: {"file1","file2"}

void TGridJDL::AddToSetDescription(const char *key, const char *description)
{
   const char *oldValue = GetDescription(key);
   TString newString;
   if (oldValue)
      newString = oldValue;
   newString += description;

   SetDescription(key, newString);
}
////////////////////////////////////////////////////////////////////////////////
/// Generates the JDL snippet.

TString TGridJDL::Generate()
{
   TString output("");

   TIter next(&fMap);
   TIter nextDescription(&fDescriptionMap);
   TObject *object = 0;
   TObject *objectD = 0;
   while ((object = next())) {
      TObjString *key = dynamic_cast<TObjString*>(object);
      if (key) {
         TObject *value = fMap.GetValue(object);
         TObjString *valueobj = dynamic_cast<TObjString*>(value);

         if (valueobj) {
            nextDescription.Reset();
            while ((objectD = nextDescription())) {
               TObjString *keyD = dynamic_cast<TObjString*>(objectD);
               if (keyD) {
                  TObject *valueD = fDescriptionMap.GetValue(objectD);
                  TObjString *valueobjD = dynamic_cast<TObjString*>(valueD);
                  if (valueobjD && !strcmp(key->GetName(), keyD->GetName())){
                     //Info("",Form("%s %s",key->GetString().Data(),keyD->GetString().Data()));
                     output += "# ";
                     output += valueobjD->GetName();
                     output += "\n";
                  }
               }
            }
            output += key->GetName();
            output += " = ";
            output += valueobj->GetName();
            output += ";\n\n";
         }
      }
   }

   return output;
}
