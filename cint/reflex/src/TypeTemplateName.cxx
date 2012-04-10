// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/internal/TypeTemplateName.h"

#include "Reflex/TypeTemplate.h"
#include "Reflex/internal/TypeTemplateImpl.h"
#include "Reflex/Type.h"
#include "Reflex/Tools.h"

#include "stl_hash.h"
#include <vector>

//-------------------------------------------------------------------------------
typedef __gnu_cxx::hash_multimap<const std::string*, Reflex::TypeTemplate> Name2TypeTemplate_t;
typedef std::vector<Reflex::TypeTemplate> TypeTemplateVec_t;


//-------------------------------------------------------------------------------
static Name2TypeTemplate_t&
sTypeTemplates() {
//-------------------------------------------------------------------------------
// Static wrapper around the type template map.
   static Name2TypeTemplate_t* t = 0;

   if (!t) {
      t = new Name2TypeTemplate_t;
   }
   return *t;
}


//-------------------------------------------------------------------------------
static TypeTemplateVec_t&
sTypeTemplateVec() {
//-------------------------------------------------------------------------------
// Static wrapper around the type template vector.
   static TypeTemplateVec_t* t = 0;

   if (!t) {
      t = new TypeTemplateVec_t;
   }
   return *t;
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplateName::TypeTemplateName(const char* name,
                                           TypeTemplateImpl* typeTemplateImpl)
//-------------------------------------------------------------------------------
   : fName(name),
   fTypeTemplateImpl(typeTemplateImpl) {
   // Constructor.
   fThisTypeTemplate = new TypeTemplate(this);
   sTypeTemplates().insert(std::pair<const std::string* const, TypeTemplate>(&fName, *fThisTypeTemplate));
   sTypeTemplateVec().push_back(*fThisTypeTemplate);
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplateName::~TypeTemplateName() {
//-------------------------------------------------------------------------------
// Destructor.
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::TypeTemplateName::ByName(const std::string& name,
                                 size_t nTemplateParams) {
//-------------------------------------------------------------------------------
// Lookup a type template by its name.
   typedef Name2TypeTemplate_t::iterator IT;
   IT lower = sTypeTemplates().find(&name);

   if (lower != sTypeTemplates().end()) {
      if (!nTemplateParams) {
         return lower->second;
      } else {
         std::pair<IT, IT> bounds = sTypeTemplates().equal_range(&name);

         for (IT it = bounds.first; it != bounds.second; ++it) {
            if (it->second.TemplateParameterSize() == nTemplateParams) {
               return it->second;
            }
         }
      }
   }
   return Dummy::TypeTemplate();
} // ByName


//-------------------------------------------------------------------------------
void
Reflex::TypeTemplateName::CleanUp() {
//-------------------------------------------------------------------------------
// Do the final cleanup for the type templates.
   for (TypeTemplateVec_t::iterator it = sTypeTemplateVec().begin(); it != sTypeTemplateVec().end(); ++it) {
      TypeTemplateName* tn = (TypeTemplateName*) it->Id();
      TypeTemplate* t = tn->fThisTypeTemplate;

      if (t) {
         t->Unload();
      }
      delete t;
      delete tn;
   }
}


//-------------------------------------------------------------------------------
void
Reflex::TypeTemplateName::DeleteTypeTemplate() const {
//-------------------------------------------------------------------------------
// Remove a type template dictionary information.
   delete fTypeTemplateImpl;
   fTypeTemplateImpl = 0;
}


//-------------------------------------------------------------------------------
std::string
Reflex::TypeTemplateName::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
// Return the name of this type template.
   if (0 != (mod & (SCOPED | S))) {
      return fName;
   } else { return Tools::GetBaseName(fName); }
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::TypeTemplateName::ThisTypeTemplate() const {
//-------------------------------------------------------------------------------
// Return the type template corresponding to this type template name.
   return *fThisTypeTemplate;
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate
Reflex::TypeTemplateName::TypeTemplateAt(size_t nth) {
//-------------------------------------------------------------------------------
// Return teh nth type template.
   if (nth < sTypeTemplateVec().size()) {
      return sTypeTemplateVec()[nth];
   }
   return Dummy::TypeTemplate();
}


//-------------------------------------------------------------------------------
size_t
Reflex::TypeTemplateName::TypeTemplateSize() {
//-------------------------------------------------------------------------------
// Return the number of type templates declared.
   return sTypeTemplateVec().size();
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate_Iterator
Reflex::TypeTemplateName::TypeTemplate_Begin() {
//-------------------------------------------------------------------------------
// Return the begin iterator of the type template collection
   return sTypeTemplateVec().begin();
}


//-------------------------------------------------------------------------------
Reflex::TypeTemplate_Iterator
Reflex::TypeTemplateName::TypeTemplate_End() {
//-------------------------------------------------------------------------------
// Return the end iterator of the type template collection
   return sTypeTemplateVec().end();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_TypeTemplate_Iterator
Reflex::TypeTemplateName::TypeTemplate_RBegin() {
//-------------------------------------------------------------------------------
// Return the RBegin iterator of the type template collection
   return ((const std::vector<TypeTemplate> &)sTypeTemplateVec()).rbegin();
}


//-------------------------------------------------------------------------------
Reflex::Reverse_TypeTemplate_Iterator
Reflex::TypeTemplateName::TypeTemplate_REnd() {
//-------------------------------------------------------------------------------
// Return the rend iterator of the type template collection
   return ((const std::vector<TypeTemplate> &)sTypeTemplateVec()).rend();
}
