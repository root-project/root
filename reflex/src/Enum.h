// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
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
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * nam,
                                  const Type & typ,
                                  size_t offs,
                                  unsigned int modifiers = 0 ) const;


      /**
       * DataMemberNth will return the nth data MemberNth of the ScopeNth
       * @param  nth data MemberNth
       * @return pointer to data MemberNth
       */
      virtual Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberNth will return the MemberNth with Name
       * @param  Name of data MemberNth
       * @return data MemberNth
       */
      virtual Member DataMemberNth( const std::string & nam ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      virtual size_t DataMemberCount() const;


      virtual Member_Iterator DataMember_Begin() const;
      virtual Member_Iterator DataMember_End() const;
      virtual Reverse_Member_Iterator DataMember_Rbegin() const;
      virtual Reverse_Member_Iterator DataMember_Rend() const;


      /**
       * DeclaringScope will return a pointer to the ScopeNth of this one
       * @return pointer to declaring ScopeNth
       */
      virtual Scope DeclaringScope() const;


      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param  MemberNth Name
       * @return pointer to MemberNth
       */
      virtual Member MemberNth( const std::string & nam,
                             const Type & signature ) const;


      /**
       * MemberNth will return the nth MemberNth of the ScopeNth
       * @param  nth MemberNth
       * @return pointer to nth MemberNth
       */
      virtual Member MemberNth( size_t nth ) const;


      /**
       * MemberCount will return the number of members
       * @return number of members
       */
      virtual size_t MemberCount() const;


      virtual Member_Iterator Member_Begin() const;
      virtual Member_Iterator Member_End() const;
      virtual Reverse_Member_Iterator Member_Rbegin() const;
      virtual Reverse_Member_Iterator Member_Rend() const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      virtual PropertyList PropertyListGet() const;

    }; // class Enum
  } //namespace Reflex
} //namespace ROOT

#include "Reflex/Member.h"

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
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::DataMember_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::DataMember_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMember_Rend();
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
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::Member_Rbegin() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_Rbegin();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Reverse_Member_Iterator ROOT::Reflex::Enum::Member_Rend() const {
//-------------------------------------------------------------------------------
  return ScopeBase::Member_Rend();  
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
inline ROOT::Reflex::Member ROOT::Reflex::Enum::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::DataMemberNth( const std::string & nam ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nam );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Enum::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Scope ROOT::Reflex::Enum::DeclaringScope() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DeclaringScope();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::MemberNth( const std::string & nam,
                                                           const Type & signature ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( nam, signature );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Enum::MemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Enum::MemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Enum::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return ScopeBase::PropertyListGet();
}

#endif // ROOT_Reflex_Enum

