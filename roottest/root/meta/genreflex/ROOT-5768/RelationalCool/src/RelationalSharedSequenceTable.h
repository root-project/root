#ifndef RELATIONALCOOL_RELATIONALSHAREDSEQUENCETABLE_H
#define RELATIONALCOOL_RELATIONALSHAREDSEQUENCETABLE_H

// Include files
#include "CoolKernel/StorageType.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalSharedSequenceTable RelationalSharedSequenceTable.h
   *
   *  Relational schema of the new "sequence" tables for tag
   *  (tag and IOV) numbering in individual folder sets (folders).
   *
   *  These new sequence tables are designed to be shared by
   *  several nodes, hence they have a nodeId column as primary key.
   *  Two such tables are foreseen, one for tags and one for IOVS.
   *  Note that IOV numbering in each individual folders uses PKs
   *  that are globally unique across all channels in the folder.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-31
   */

  namespace RelationalSharedSequenceTable
  {

    namespace columnNames
    {
      static const std::string nodeId = "NODE_ID";
      static const std::string currentValue = "CURRENT_VALUE";
      static const std::string lastModDate = "LASTMOD_DATE";
    }

    namespace columnTypeIds
    {
      static const StorageType::TypeId nodeId = StorageType::UInt32;
      static const StorageType::TypeId currentValue = StorageType::UInt32;
      static const StorageType::TypeId lastModDate = StorageType::String255;
    }

    namespace columnTypes
    {
      typedef UInt32 nodeId;
      typedef UInt32 currentValue;
      typedef String255 lastModDate;
    }

    /// Get the RecordSpecification of the sequence table.
    /// If the flag is true, include only the first column (the nodeId).
    const IRecordSpecification&
    tableSpecification( bool nodeIdOnly = false );

  }

}

#endif // RELATIONALCOOL_RELATIONALSHAREDSEQUENCETABLE_H
