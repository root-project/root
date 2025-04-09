// $Id: RelationalGlobalTagTable.h,v 1.20 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALGLOBALTAGTABLE_H
#define RELATIONALCOOL_RELATIONALGLOBALTAGTABLE_H

// Include files
#include "CoolKernel/StorageType.h"

// Local include files
#include "uppercaseString.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalGlobalTagTable RelationalGlobalTagTable.h
   *
   *  Relational schema of the table storing COOL global tags.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-03-01
   */

  namespace RelationalGlobalTagTable {

    inline const std::string defaultTableName( const std::string& prefix )
    {
      return uppercaseString(prefix) + "TAGS";
    }

    namespace columnNames
    {
      static const std::string nodeId = "NODE_ID";
      static const std::string tagId = "TAG_ID";
      static const std::string tagName = "TAG_NAME";
      static const std::string tagLockStatus = "TAG_LOCK_STATUS";
      static const std::string tagDescription = "TAG_DESCRIPTION";
      // *** START *** 3.0.0 schema extensions (task #4396)
      // Tag type (NOT NULL): 0(unknown), 1(head), 2(user)
      static const std::string tagType = "TAG_TYPE";
      // **** END **** 3.0.0 schema extensions (task #4396)
      static const std::string sysInsTime = "SYS_INSTIME";
    }

    namespace columnTypeIds
    {
      static const StorageType::TypeId nodeId         = StorageType::UInt32;
      static const StorageType::TypeId tagId          = StorageType::UInt32;
      static const StorageType::TypeId tagName        = StorageType::String255;
      static const StorageType::TypeId tagLockStatus  = StorageType::UInt16;
      static const StorageType::TypeId tagDescription = StorageType::String255;
      // *** START *** 3.0.0 schema extensions (task #4396)
      static const StorageType::TypeId tagType        = StorageType::UInt16;
      // **** END **** 3.0.0 schema extensions (task #4396)
      static const StorageType::TypeId sysInsTime     = StorageType::String255;
    }

    namespace columnTypes
    {
      typedef UInt32 nodeId;
      typedef UInt32 tagId;
      typedef String255 tagName;
      typedef UInt16 tagLockStatus;
      typedef String255 tagDescription;
      // *** START *** 3.0.0 schema extensions (task #4396)
      typedef UInt16 tagType;
      // **** END **** 3.0.0 schema extensions (task #4396)
      typedef String255 sysInsTime;
    }

    /// Get the record specification of the global tag table
    const IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALGLOBALTAGTABLE_H
