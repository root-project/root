// $Id: RelationalDatabaseTable.h,v 1.30 2009-12-17 17:05:54 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALDATABASETABLE_H
#define RELATIONALCOOL_RELATIONALDATABASETABLE_H 1

// Include files
#include "CoolKernel/StorageType.h"

// Local include files
#include "uppercaseString.h"

namespace cool
{

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalDatabaseTable RelationalDatabaseTable.h
   *
   *  Relational schema of the main table for a COOL conditions "database".
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-12-16
   */

  namespace RelationalDatabaseTable
  {

    /// The name of the main table for the given "conditions database name".
    /// This is the only hardcoded mapping that cannot be changed by the user.
    inline const std::string tableName( const std::string& dbName )
    {
      // TEMPORARY? AV 04.04.2005
      // The RelationalDatabaseId constructor throws an exception if dbname
      // is not uppercase: this is meant to prevent users from believing that
      // dbnames are case-sensitive (e.g., from believing that "myDb", "mydb"
      // and "MYDB" are three different databases). Eventually, case-sensitive
      // dbnames may be supported by mapping them to different case-sensitive
      // table names (such as "myDb_DB_ATTRIBUTES" or "MYDB_DB_ATTRIBUTES").
      // Note that ALL tables are presently uppercase: if we decide to go for
      // case-sensitive table names, all other Table classes must be changed!
      // FIXME: presently the dbname is uppercase anyway...
      return uppercaseString(dbName) + "_DB_ATTRIBUTES";
    }

    namespace columnNames
    {
      static const std::string attributeName = "DB_ATTRIBUTE_NAME";
      static const std::string attributeValue = "DB_ATTRIBUTE_VALUE";
    }

    namespace columnTypeIds
    {
      static const StorageType::TypeId attributeName = StorageType::String255;
      static const StorageType::TypeId attributeValue = StorageType::String4k;
    }

    namespace columnTypes
    {
      typedef String255 attributeName;
      typedef String4k attributeValue;
    }

    namespace attributeNames
    {
      static const
      std::string defaultTablePrefix = "DEFAULT_TABLE_PREFIX";
      // *** START *** 3.0.0 schema extensions (task #4307)
      static const
      std::string iovTablesTableName = "IOVTABLES_TABLE_NAME";
      static const
      std::string channelTablesTableName = "CHANNELTABLES_TABLE_NAME";
      // **** END **** 3.0.0 schema extensions (task #4307)
      static const
      std::string nodeTableName = "NODE_TABLE_NAME";
      static const
      std::string tagTableName = "TAG_TABLE_NAME";
      // *** START *** 3.0.0 schema extensions (task #4396)
      static const
      std::string headTagTableName = "HEADTAG_TABLE_NAME";
      static const
      std::string userTagTableName = "USERTAG_TABLE_NAME";
      // **** END **** 3.0.0 schema extensions (task #4396)
      static const
      std::string tag2TagTableName = "TAG2TAG_TABLE_NAME";
      static const
      std::string tagSharedSequenceName = "TAG_SHAREDSEQ_NAME";
      static const
      std::string iovSharedSequenceName = "IOV_SHAREDSEQ_NAME";
      static const
      std::string release = "RELEASE";
      static const
      std::string cvsCheckout = "CVS_CHECKOUT";
      static const
      std::string cvsCheckin = "CVS_CHECKIN";
      static const
      std::string schemaVersion = "SCHEMA_VERSION";
      static const
      std::string lastReplication = "LAST_REPLICATION";
      static const
      std::string lastReplicationSource = "LAST_REPLICATION_SOURCE";
    }

    /// Get the record specification of the main database table.
    const cool::IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALDATABASETABLE_H
