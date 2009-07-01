// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_DataMember
#define Reflex_DataMember

// Include files
#include "Reflex/internal/MemberBase.h"

namespace Reflex {
// forward declarations
class TypeBase;
class Type;
class DictionaryGenerator;


/**
 * @class DataMember DataMember.h Reflex/DataMember.h
 * @author Stefan Roiser
 * @date 24/11/2003
 * @ingroup Ref
 */
class DataMember: public MemberBase {
public:
   /** default constructor */
   DataMember(const char* nam, const Type& typ, size_t offs, unsigned int modifiers = 0, char* interpreterOffset = 0);


   /** destructor */
   virtual ~DataMember();


   /** return Name of data MemberAt */
   std::string Name(unsigned int mod = 0) const;


   /** Get the MemberAt value (as void*) */
   Object Get(const Object& obj) const;


   /**
    * GenerateDict will produce the dictionary information of this type
    * @param generator a reference to the dictionary generator instance
    */
   virtual void GenerateDict(DictionaryGenerator& generator) const;


   /** return the Offset of the MemberAt */
   size_t Offset() const;
   void InterpreterOffset(char*);
   char*& InterpreterOffset() const;


   /** Set the MemberAt value */

   /*void Set( const Object & instance,
      const Object & value ) const;*/
   void Set(const Object& instance,
            const void* value) const;

private:
   /** Offset of the MemberAt */
   size_t fOffset;
   char* fInterpreterOffset;

};    // class DataMember
} //namespace Reflex


inline size_t
Reflex::DataMember::Offset() const {
   return fOffset;
}


inline void
Reflex::DataMember::InterpreterOffset(char* offset) {
   //fOffset = reinterpret_cast<size_t>(offset);
   fInterpreterOffset = offset;
}


inline char*&
Reflex::DataMember::InterpreterOffset() const {
   //return *reinterpret_cast<char**>(const_cast<size_t*>(&fOffset));
   return *const_cast<char**>(&fInterpreterOffset);
}


#endif // Reflex_DataMember
