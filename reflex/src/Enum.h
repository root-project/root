// @(#)root/reflex:$Name: HEAD $:$Id: Enum.h,v 1.6 2006/03/06 12:51:46 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Enum
#define ROOT_Reflex_Enum

// Include files
#include "Reflex/TypeBase.h"
#include "Reflex/ScopeBase.h"

namespace ROOT {
   namespace Reflex {

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
               const std::type_info & ti );


         /** destructor */
         virtual ~Enum();


         /**
          * operator Scope will return the corresponding scope of this type if
          * applicable (i.e. if the Type is also a Scope e.g. Class, Union, Enum)
          */                                       
         operator Scope() const;


         /**
          * operator Type will return the corresponding Type object
          * @return Type corresponding to this TypeBase
          */
         operator Type() const;


         /**
          * AddDataMember will add the information about a data MemberAt
          * @param dm pointer to data MemberAt
          */
         virtual void AddDataMember( const Member & dm ) const;
         virtual void AddDataMember( const char * nam,
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


         /**
          * Properties will return a pointer to the PropertyNth list attached
          * to this item
          * @return pointer to PropertyNth list
          */
         virtual PropertyList Properties() const;

      }; // class Enum
   } //namespace Reflex
} //namespace ROOT

#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
inline ROOT::Reflex::Enum::operator ROOT::Reflex::Scope () const {
//-------------------------------------------------------------------------------
   return ScopeBase::operator Scope();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Enum::operator ROOT::Reflex::Type () const {
//-------------------------------------------------------------------------------
   return TypeBase::operator Type();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Enum::DataMember_Begin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Enum::DataMember_End() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::DataMember_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::DataMember_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMember_REnd();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Enum::Member_Begin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_Begin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member_Iterator ROOT::Reflex::Enum::Member_End() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_End();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::Member_RBegin() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_RBegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::Member_REnd() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Member_REnd();  
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Enum::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
   ScopeBase::AddDataMember( dm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Enum::AddDataMember( const char * nam,
                                               const Type & typ,
                                               size_t offs,
                                               unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
   ScopeBase::AddDataMember(nam, typ, offs, modifiers);
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::DataMemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberAt( nth );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::DataMemberByName( const std::string & nam ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberByName( nam );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Enum::DataMemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DataMemberSize();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Enum::DeclaringScope() const {
//-------------------------------------------------------------------------------
   return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::MemberByName( const std::string & nam,
                                                              const Type & signature ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberByName( nam, signature );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::MemberAt( size_t nth ) const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberAt( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Enum::MemberSize() const {
//-------------------------------------------------------------------------------
   return ScopeBase::MemberSize();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Enum::Properties() const {
//-------------------------------------------------------------------------------
   return ScopeBase::Properties();
}

#endif // ROOT_Reflex_Enum

