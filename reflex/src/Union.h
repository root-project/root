// @(#)root/reflex:$Name:$:$Id:$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef ROOT_Reflex_Union
#define ROOT_Reflex_Union

// Include Files
#include "Reflex/TypeBase.h"
#include "Reflex/ScopeBase.h"

namespace ROOT {
  namespace Reflex {

    
    /**
     * @class Union Union.h Reflex/Union.h
     * @author Stefan Roiser
     * @date 24/11/2003
     * @ingroup Ref
     */
    class Union : public TypeBase, public ScopeBase {

    public:

      /** constructor */
      Union( const char * unionType,
             size_t size,
             const std::type_info & TypeInfo ) ;


      /** destructor */
      virtual ~Union();


      /**
       * AddDataMember will add the information about a data MemberNth
       * @param dm pointer to data MemberNth
       */
      virtual void AddDataMember( const Member & dm ) const;
      virtual void AddDataMember( const char * Name,
                                  const Type & TypeNth,
                                  size_t Offset,
                                  unsigned int modifiers = 0 ) const;


      /**
       * DataMemberNth will return the nth data MemberNth of the ScopeNth
       * @param  nth data MemberNth
       * @return pointer to data MemberNth
       */
      Member DataMemberNth( size_t nth ) const;


      /**
       * DataMemberCount will return the number of data members of this ScopeNth
       * @return number of data members
       */
      size_t DataMemberCount() const;

      /**
       * MemberNth will return the nth MemberNth of the ScopeNth
       * @param  nth MemberNth
       * @return pointer to nth MemberNth
       */
      Member MemberNth( size_t nth ) const;

      /**
       * MemberNth will return the first MemberNth with a given Name
       * @param  MemberNth Name
       * @return pointer to MemberNth
       */
      Member MemberNth( const std::string & Name ) const;

      /**
       * MemberCount will return the number of members
       * @return number of members
       */
      size_t MemberCount() const;


      /**
       * PropertyListGet will return a pointer to the PropertyNth list attached
       * to this item
       * @return pointer to PropertyNth list
       */
      PropertyList PropertyListGet() const;

    }; // class Union

  } // namespace Reflex
} // namespace ROOT

#include "Reflex/Member.h"

//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Union::AddDataMember( const Member & dm ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddDataMember( dm );
}


//-------------------------------------------------------------------------------
inline void ROOT::Reflex::Union::AddDataMember( const char * Name,
                                                const Type & TypeNth,
                                                size_t Offset,
                                                unsigned int modifiers ) const {
//-------------------------------------------------------------------------------
  ScopeBase::AddDataMember(Name, TypeNth, Offset, modifiers );
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Union::DataMemberNth( size_t nth ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberNth( nth );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Union::DataMemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::DataMemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::Member ROOT::Reflex::Union::MemberNth( const std::string & Name ) const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberNth( Name );
}


//-------------------------------------------------------------------------------
inline size_t ROOT::Reflex::Union::MemberCount() const {
//-------------------------------------------------------------------------------
  return ScopeBase::MemberCount();
}


//-------------------------------------------------------------------------------
inline ROOT::Reflex::PropertyList ROOT::Reflex::Union::PropertyListGet() const {
//-------------------------------------------------------------------------------
  return ScopeBase::PropertyListGet();
}

#endif // ROOT_Reflex_Union

