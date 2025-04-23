// $Id: RelationalGlobalHeadTagTable.h,v 1.3 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALGLOBALHEADTAGTABLE_H
#define RELATIONALCOOL_RELATIONALGLOBALHEADTAGTABLE_H

// Include files
#include "CoolKernel/StorageType.h"

// Local include files
#include "uppercaseString.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalGlobalHeadTagTable RelationalGlobalHeadTagTable.h
   *
   *  Relational schema of the table storing COOL global HEAD tags.
   *
   *  @author Andrea Valassi
   *  @date   2007-02-15
   */

  namespace RelationalGlobalHeadTagTable {

    inline const std::string defaultTableName( const std::string& prefix )
    {
      return uppercaseString(prefix) + "HEADTAGS";
    }

    namespace columnNames
    {
      static const std::string nodeId  = "NODE_ID";
      static const std::string tagId   = "TAG_ID";
      // Tag type (NOT NULL): 0(unknown), 1(head), 2(user)
      // Tag type must be equal to 1 (CHECK constraint) for the head tag table
      static const std::string tagType = "TAG_TYPE";
    }

    namespace columnTypeIds
    {
      static const StorageType::TypeId nodeId  = StorageType::UInt32;
      static const StorageType::TypeId tagId   = StorageType::UInt32;
      static const StorageType::TypeId tagType = StorageType::UInt16;
    }

    namespace columnTypes
    {
      typedef UInt32 nodeId;
      typedef UInt32 tagId;
      typedef UInt16 tagType;
    }

    /// Get the record specification of the global user tag table
    const IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALGLOBALHEADTAGTABLE_H
