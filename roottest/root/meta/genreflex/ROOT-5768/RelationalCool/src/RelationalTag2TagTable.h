// $Id: RelationalTag2TagTable.h,v 1.10 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTAG2TAGTABLE_H
#define RELATIONALCOOL_RELATIONALTAG2TAGTABLE_H

// Local include files
#include "uppercaseString.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalTag2TagTable RelationalTag2TagTable.h
   *
   *  Relational schema of the table storing
   *  parent-child relations between HVS global tags.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-23
   */

  namespace RelationalTag2TagTable {

    inline const std::string defaultTableName( const std::string& prefix ) {
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input prefix is uppercase anyway...
      return uppercaseString(prefix) + "TAG2TAG";
    }

    inline const std::string sequenceName( const std::string& tableName ) {
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input table name is uppercase anyway...
      return uppercaseString(tableName) + "_SEQ";
    }

    namespace columnNames {
      static const std::string parentNodeId = "PARENT_NODEID";
      static const std::string parentTagId = "PARENT_TAGID";
      static const std::string childNodeId = "CHILD_NODEID";
      static const std::string childTagId = "CHILD_TAGID";
      static const std::string sysInsTime = "SYS_INSTIME";
    }

    namespace columnTypeIds {
      static const StorageType::TypeId parentNodeId = StorageType::UInt32;
      static const StorageType::TypeId parentTagId  = StorageType::UInt32;
      static const StorageType::TypeId childNodeId  = StorageType::UInt32;
      static const StorageType::TypeId childTagId   = StorageType::UInt32;
      // TEMPORARY! Should be Time?
      static const StorageType::TypeId sysInsTime   = StorageType::String255;
    }

    namespace columnTypes {
      typedef UInt32 parentNodeId;
      typedef UInt32 parentTagId;
      typedef UInt32 childNodeId;
      typedef UInt32 childTagId;
      // TEMPORARY! Should be Time?
      typedef String255 sysInsTime;
    }

    /// Get the record specification of the tag table
    const IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALTAG2TAGTABLE_H
