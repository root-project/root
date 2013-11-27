// $Id: RelationalTagTable.h,v 1.19 2012-06-29 13:53:56 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTAGTABLE_H
#define RELATIONALCOOL_RELATIONALTAGTABLE_H

// Include files
#include <cstdio> // For sprintf on gcc45
#include <cstring>

// Local include files
#include "uppercaseString.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalTagTable RelationalTagTable.h
   *
   *  Relational schema of the table storing COOL local tags.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-02-05
   */

  namespace RelationalTagTable {

    inline const std::string defaultTableName
    ( const std::string& prefix, unsigned nodeId ) {
      char tableName[] = "Fxxxx_TAGS";
      snprintf( tableName, strlen(tableName)+1, // Fix Coverity SECURE_CODING
                "F%4.4u_TAGS", nodeId );
      return uppercaseString( prefix ) + std::string( tableName );
    }

    inline const std::string sequenceName( const std::string& tableName ) {
      return uppercaseString(tableName) + "_SEQ";
    }

    namespace columnNames {
      static const std::string tagId = "TAG_ID";
      static const std::string tagName = "TAG_NAME";
      static const std::string tagDescription = "TAG_DESCRIPTION";
      static const std::string sysInsTime = "SYS_INSTIME";
    }

    namespace columnTypeIds {
      static const StorageType::TypeId tagId          = StorageType::UInt32;
      static const StorageType::TypeId tagName        = StorageType::String255;
      static const StorageType::TypeId tagDescription = StorageType::String255;
      // TEMPORARY! Should be Time?
      static const StorageType::TypeId sysInsTime     = StorageType::String255;
    }

    namespace columnTypes {
      typedef UInt32 tagId;
      typedef String255 tagName;
      typedef String255 tagDescription;
      // TEMPORARY! Should be Time?
      typedef String255 sysInsTime;
    }

    /// Get the RecordSpecification of the tag table
    const IRecordSpecification& tableSpecification();

  }

}

#endif // RELATIONALCOOL_RELATIONALTAGTABLE_H
