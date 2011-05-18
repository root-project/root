// @(#)root/reflex:$Id$
// Author: Axel Naumann, 2009

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_ScopedType
#define Reflex_ScopedType

// Include files
#include "Reflex/Type.h"
#include "Reflex/Scope.h"
#include "Reflex/internal/ScopeBase.h"
#include <map>
#include <vector>

namespace Reflex {
/**
 * @class Class Class.h Reflex/Class.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class ScopedType: public TypeBase,
   public ScopeBase {
public:
   /** constructor */
   ScopedType(const char* name, size_t size, TYPE typeType,
              const std::type_info& ti, const Type& finalType = Dummy::Type(),
              unsigned int modifiers = 0, REPRESTYPE represType = REPRES_NOTYPE);

   /** destructor */
   virtual ~ScopedType() {}

   /**
    * operator Scope will return the corresponding scope of this type if
    * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
    */
   operator Scope() const;

   /**
    * the operator Type will return a corresponding Type object to the At if
    * applicable (i.e. if the Scope is also a Type e.g. Class, Union, Enum)
    */
   operator Type() const;

   /**
    * Hide this class from any lookup by appending the string " @HIDDEN@" to its name.
    */
   virtual void HideName() const;

   /**
    * Un-Hide class type from any lookup by removing the string " @HIDDEN@" to its name.
    */
   virtual void UnhideName() const;


   /**
    * DeclaringScope will return a pointer to the At of this one
    * @return pointer to declaring At
    */
   virtual Scope DeclaringScope() const;


   /**
    * IsPrivate will check if the scope access is private
    * @return true if scope access is private
    */
   virtual bool IsPrivate() const;

   /**
    * IsProtected will check if the scope access is protected
    * @return true if scope access is protected
    */
   virtual bool IsProtected() const;

   /**
    * IsPublic will check if the scope access is public
    * @return true if scope access is public
    */
   virtual bool IsPublic() const;

   /**
    * Name will return the Name of the class
    * @return Name of class
    */
   virtual std::string Name(unsigned int mod = 0) const;

   /**
    * SimpleName returns the name of the type as a reference. It provides a
    * simplified but faster generation of a type name. Attention currently it
    * is not guaranteed that Name() and SimpleName() return the same character
    * layout of a name (ie. spacing, commas, etc. )
    * @param pos will indicate where in the returned reference the requested name starts
    * @param mod The only 'mod' support is SCOPED
    * @return name of type
    */
   virtual const char* SimpleName(size_t& pos,
                                  unsigned int mod = 0) const;

   /**
    * Properties will return a pointer to the PropertyNth list attached
    * to this item
    * @return pointer to PropertyNth list
    */
   virtual PropertyList Properties() const;


   /**
    * return the type name
    */
   TypeName* TypeNameGet() const;

   /**
    * @return all the modifiers
    */
   virtual unsigned int Modifiers() const;

   /**
    * @return all the modifiers
    */
   virtual void SetModifiers(unsigned int modifiers);

protected:
   /**
    * The modifiers of this scoped type
    */
   unsigned int fModifiers;

}; // class Class
} //namespace Reflex


//-------------------------------------------------------------------------------
inline Reflex::ScopedType::operator
Reflex::Scope() const {
//-------------------------------------------------------------------------------
   return ScopeBase::operator Scope();
}


//-------------------------------------------------------------------------------
inline Reflex::ScopedType::operator
Reflex::Type() const {
//-------------------------------------------------------------------------------
   return TypeBase::operator Type();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope
Reflex::ScopedType::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline void
Reflex::ScopedType::HideName() const {
//-------------------------------------------------------------------------------
   TypeBase::HideName();
   ScopeBase::HideName();
}


//-------------------------------------------------------------------------------
inline void
Reflex::ScopedType::UnhideName() const {
   //-------------------------------------------------------------------------------
   TypeBase::UnhideName();
   ScopeBase::UnhideName();
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopedType::IsPrivate() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PRIVATE);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopedType::IsProtected() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PROTECTED);
}


//-------------------------------------------------------------------------------
inline bool
Reflex::ScopedType::IsPublic() const {
//-------------------------------------------------------------------------------
   return 0 != (fModifiers & PUBLIC);
}


//-------------------------------------------------------------------------------
inline std::string
Reflex::ScopedType::Name(unsigned int mod) const {
//-------------------------------------------------------------------------------
   return ScopeBase::Name(mod);
}


//-------------------------------------------------------------------------------
inline const char*
Reflex::ScopedType::SimpleName(size_t& pos,
                               unsigned int mod) const {
//-------------------------------------------------------------------------------
   return ScopeBase::SimpleName(pos, mod);
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyList
Reflex::ScopedType::Properties() const {
//-------------------------------------------------------------------------------
   return TypeBase::Properties();
}


//-------------------------------------------------------------------------------
inline Reflex::TypeName*
Reflex::ScopedType::TypeNameGet() const {
//-------------------------------------------------------------------------------
   return fTypeName;
}


//-------------------------------------------------------------------------------
inline unsigned int
Reflex::ScopedType::Modifiers() const {
   //-------------------------------------------------------------------------------
   return fModifiers;
}


//-------------------------------------------------------------------------------
inline void
Reflex::ScopedType::SetModifiers(unsigned int modifiers) {
   //-------------------------------------------------------------------------------
   fModifiers = modifiers;
}


#endif // Reflex_ScopedType
