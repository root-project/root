// $Id: RelationalIovTablesTable.h,v 1.3 2009-12-17 17:05:54 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALIOVTABLESTABLE_H
#define RELATIONALCOOL_RELATIONALIOVTABLESTABLE_H 1

// Include files
#include "CoolKernel/StorageType.h"

// Local include files
#include "VersionNumber.h"
#include "uppercaseString.h"

namespace cool
{

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalIovTablesTable RelationalIovTablesTable.h
   *
   *  Relational schema of the table storing metadata of COOL IOV tables.
   *
   *  @author Andrea Valassi
   *  @date   2007-02-15
   */

  namespace RelationalIovTablesTable {

    inline const std::string defaultTableName( const std::string& prefix )
    {
      return uppercaseString(prefix) + "IOVTABLES";
    }

    namespace columnNames
    {
      static const
      std::string iovTableName = "IOVTABLE_NAME";
      static const
      std::string iovTableSchemaVersion = "IOVTABLE_SCHEMA_VERSION";
      static const
      std::string iovTableVersioningMode = "IOVTABLE_VERSIONING";
      static const
      std::string iovTableInsertionTime = "IOVTABLE_INSTIME";
      static const
      std::string payloadSpecDesc = "PAYLOAD_SPEC";
      static const
      std::string payloadInline = "PAYLOAD_INLINE";
      static const
      std::string payloadExtRef = "PAYLOAD_EXTREF";
    }

    namespace columnTypeIds
    {
      static const
      StorageType::TypeId iovTableName = StorageType::String255;
      static const
      StorageType::TypeId iovTableSchemaVersion = StorageType::String255;
      static const
      StorageType::TypeId iovTableVersioningMode = StorageType::Int32;
      static const
      StorageType::TypeId iovTableInsertionTime = StorageType::String255;
      static const
      StorageType::TypeId payloadSpecDesc = StorageType::String64k;
      static const
      StorageType::TypeId payloadInline = StorageType::UInt16;
      static const
      StorageType::TypeId payloadExtRef = StorageType::String64k;
    }

    namespace columnTypes
    {
      typedef String255 iovTableName;
      typedef String255 iovTableSchemaVersion;
      typedef Int32 iovTableVersioningMode;
      typedef String255 iovTableInsertionTime;
      typedef String64k payloadSpecDesc;
      typedef UInt16 payloadInline;
      typedef String64k payloadExtRef;
    }

    /// Get the record specification of the iovTables table.
    const IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALIOVTABLESTABLE_H
