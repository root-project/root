// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_Union
#define Reflex_Union

// Include Files
#include "Reflex/Member.h"
#include "ScopedType.h"

namespace Reflex {
/**
 * @class Union Union.h Reflex/Union.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class Union: public ScopedType {
public:
   /** constructor */
   Union(const char* typ, size_t size, const std::type_info& ti, unsigned int modifiers, TYPE unionType = UNION);

   /** destructor */
   virtual ~Union();

   /**
    * IsComplete will return true if all classes and BaseAt classes of this
    * class are resolved and fully known in the system
    */
   virtual bool IsComplete() const;

public:
   /**
    * AddFunctionMember will add the information about a function MemberAt
    * @param fm pointer to function MemberAt
    */
   virtual void AddFunctionMember(const Member& fm) const;
   virtual Member AddFunctionMember(const char* nam,
                                    const Type& typ,
                                    StubFunction stubFP,
                                    void* stubCtx = 0,
                                    const char* params = 0,
                                    unsigned int modifiers = 0) const;

private:
   /** boolean is true if the whole object is resolved */
   mutable bool fCompleteType;

   /**
    * short cut to constructors
    * @label constructors
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1..*
    */
   mutable std::vector<Member> fConstructors;

   /**
    * short cut to destructor
    * @label destructor
    * @link aggregation
    * @clientCardinality 1
    * @supplierCardinality 1
    */
   mutable Member fDestructor;

}; // class Union

} // namespace Reflex

#endif // Reflex_Union
