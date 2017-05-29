// @(#)root/meta:$Id$
// Author: Bianca-Cristina Cristescu   03/07/13

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDictAttributeMap
#define ROOT_TDictAttributeMap


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDictAttributeMap                                                    //
//                                                                      //
// Dictionary of attributes of a TClass.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TObject.h"
#include "THashTable.h"


class TDictAttributeMap : public TObject
{
public:

   TDictAttributeMap();
   virtual ~TDictAttributeMap();

   void        AddProperty(const char* key, const char* value);
   Bool_t      HasKey(const char* key) const;
   const char  *GetPropertyAsString(const char* key) const;
   Int_t       GetPropertySize() const { return fStringProperty.GetSize(); }
   TString     RemovePropertyString(const char* key);
   Bool_t      RemoveProperty(const char* key);
   void        Clear(Option_t* option = "");

private:

   THashTable     fStringProperty;         //all properties of String type

   ClassDef(TDictAttributeMap,2)  // Container for name/value pairs of TDictionary attributes
};

#endif // ROOT_TDictAttributeMap

