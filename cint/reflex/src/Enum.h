// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Enum
#define Reflex_Enum

// Include files
#include "Reflex/internal/TypeBase.h"
#include "Reflex/internal/ScopeBase.h"

namespace Reflex {

   // forward declarations
   class DictionaryGenerator;


   /**
   * @class Enum Enum.h Reflex/Enum.h
   * @author Stefan Roiser
   * @date 24/11/2003
   * @ingroup Ref
   */
   class Enum : public TypeBase, public ScopeBase {

   public:

      /** default constructor */
      Enum( const char * enumType,
         const std::type_info & ti,
         unsigned int modifiers );


      /** destructor */
      virtual ~Enum();


      /**
      * operator Scope will return the corresponding scope of this type if
      * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
      */                                       
      operator Scope () const;


      /**
      * operator Type will return the corresponding Type object
      * @return Type corresponding to this TypeBase
      */
      operator Type () const;


      /**
      * AddDataMember will add the information about a data MemberAt
      * @param dm pointer to data MemberAt
      */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * nam,
         const Type & typ,
         size_t offs,
         unsigned int modifiers = 0 ) const;
      virtual void AddDataMember( Member &output,
                                 const char * nam,
                                 const Type & typ,
                                 size_t offs,
                                 unsigned int modifiers = 0 ) const;
      

      /**
      * DataMemberAt will return the nth data MemberAt of the At
      * @param  nth data MemberAt
      * @return pointer to data MemberAt
      */
      virtual Member DataMemberAt( size_t nth ) const;


      /**
      * DataMemberByName will return the MemberAt with Name
      * @param  Name of data MemberAt
      * @return data MemberAt
      */
      virtual Member DataMemberByName( const std::string & nam ) const;


      /**
      * DataMemberSize will return the number of data members of this At
      * @return number of data members
      */
      virtual size_t DataMemberSize() const;


      virtual Member_Iterator DataMember_Begin() const;
      virtual Member_Iterator DataMember_End() const;
      virtual Reverse_Member_Iterator DataMember_RBegin() const;
      virtual Reverse_Member_Iterator DataMember_REnd() const;


      /**
      * DeclaringScope will return a pointer to the At of this one
      * @return pointer to declaring At
      */
      virtual Scope DeclaringScope() const;


      /**
      * GenerateDict will produce the dictionary information of this type
      * @param generator a reference to the dictionary generator instance
      */
      virtual void GenerateDict(DictionaryGenerator &generator) const;

      /**
       * Hide this type from any lookup by appending the string " @HIDDEN@" to its name.
       */
      virtual void HideName() const;
      
      /**
       * Un-Hide this type from any lookup by removing the string " @HIDDEN@" to its name.
       */
      virtual void UnhideName() const;
      
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
      * MemberByName will return the first MemberAt with a given Name
      * @param  MemberAt Name
      * @return pointer to MemberAt
      */
      virtual Member MemberByName( const std::string & nam,
         const Type & signature ) const;


      /**
      * MemberAt will return the nth MemberAt of the At
      * @param  nth MemberAt
      * @return pointer to nth MemberAt
      */
      virtual Member MemberAt( size_t nth ) const;


      /**
      * MemberSize will return the number of members
      * @return number of members
      */
      virtual size_t MemberSize() const;


      virtual Member_Iterator Member_Begin() const;
      virtual Member_Iterator Member_End() const;
      virtual Reverse_Member_Iterator Member_RBegin() const;
      virtual Reverse_Member_Iterator Member_REnd() const;


      virtual std::string Name( unsigned int mod = 0 ) const;


      /**
      * Properties will return a pointer to the PropertyNth list attached
      * to this item
      * @return pointer to PropertyNth list
      */
      virtual PropertyList Properties() const;

   private:

      /**
      * The modifiers of this enum
      */
      unsigned int fModifiers;

   }; // class Enum
} //namespace Reflex

#include "Reflex/internal/OwnedMember.h"

//-------------------------------------------------------------------------------
inline Reflex::Enum::operator Reflex::Scope () const {
//-------------------------------------------------------------------------------
   return ScopeBase::operator Scope ();
}


//-------------------------------------------------------------------------------
inline Reflex::Enum::operator Reflex::Type () const {
//-------------------------------------------------------------------------------
   return TypeBase::operator Type ();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator Reflex::Enum::DataMember_Begin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator Reflex::Enum::DataMember_End() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator Reflex::Enum::DataMember_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator Reflex::Enum::DataMember_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_REnd();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator Reflex::Enum::Member_Begin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_Begin();
}


//-------------------------------------------------------------------------------
inline Reflex::Member_Iterator Reflex::Enum::Member_End() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_End();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator Reflex::Enum::Member_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_RBegin();
}


//-------------------------------------------------------------------------------
inline Reflex::Reverse_Member_Iterator Reflex::Enum::Member_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_REnd();  
}


//-------------------------------------------------------------------------------
inline std::string Reflex::Enum::Name( unsigned int mod ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::Name( mod );
}


//-------------------------------------------------------------------------------
inline void Reflex::Enum::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   ScopeBase::AddDataMember( dm );
}


//-------------------------------------------------------------------------------
inline void Reflex::Enum::AddDataMember( const char * nam,
                                               const Type & typ,
                                               size_t offs,
                                               unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   ScopeBase::AddDataMember(nam, typ, offs, modifiers);
}

//-------------------------------------------------------------------------------
inline void Reflex::Enum::AddDataMember( Member &output,
                                        const char * nam,
                                        const Type & typ,
                                        size_t offs,
                                        unsigned int modifiers ) const {
   //-------------------------------------------------------------------------------
   ScopeBase::AddDataMember(output, nam, typ, offs, modifiers);
}


//-------------------------------------------------------------------------------
inline Reflex::Member Reflex::Enum::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberAt( nth );
}


//-------------------------------------------------------------------------------
inline Reflex::Member Reflex::Enum::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberByName( nam );
}


//-------------------------------------------------------------------------------
inline size_t Reflex::Enum::DataMemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberSize();
}


//-------------------------------------------------------------------------------
inline Reflex::Scope Reflex::Enum::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline void Reflex::Enum::HideName() const {
//-------------------------------------------------------------------------------
   TypeBase::HideName();
   ScopeBase::HideName();
}

//-------------------------------------------------------------------------------
inline void Reflex::Enum::UnhideName() const {
   //-------------------------------------------------------------------------------
   TypeBase::UnhideName();
   ScopeBase::UnhideName();
}

//-------------------------------------------------------------------------------
inline bool Reflex::Enum::IsPrivate() const {
//-------------------------------------------------------------------------------
   return 0 != ( fModifiers & PRIVATE );
}


//-------------------------------------------------------------------------------
inline bool Reflex::Enum::IsProtected() const {
//-------------------------------------------------------------------------------
   return 0 != ( fModifiers & PROTECTED );
}


//-------------------------------------------------------------------------------
inline bool Reflex::Enum::IsPublic() const {
//-------------------------------------------------------------------------------
   return 0 != ( fModifiers & PUBLIC );
}


//-------------------------------------------------------------------------------
inline Reflex::Member Reflex::Enum::MemberByName( const std::string & nam,
                                                              const Type & signature ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberByName( nam, signature );
}


//-------------------------------------------------------------------------------
inline Reflex::Member Reflex::Enum::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t Reflex::Enum::MemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberSize();
}


//-------------------------------------------------------------------------------
inline Reflex::PropertyList Reflex::Enum::Properties() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Properties();
}

#endif // Reflex_Enum

