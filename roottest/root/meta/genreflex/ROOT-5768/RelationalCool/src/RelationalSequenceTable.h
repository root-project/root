#ifndef RELATIONALCOOL_RELATIONALSEQUENCETABLE_H
#define RELATIONALCOOL_RELATIONALSEQUENCETABLE_H

// Include files
#include "CoolKernel/StorageType.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;

  /** @namespace cool::RelationalSequenceTable RelationalSequenceTable.h
   *
   *  Relational schema of the tables implementing "sequence" functionalities.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-04-15
   */

  namespace RelationalSequenceTable {

    namespace columnNames {
      static const std::string sequenceName = "SEQUENCE_NAME";
      static const std::string currentValue = "CURRENT_VALUE";
      static const std::string lastModDate = "LASTMOD_DATE";
    }

    namespace columnTypeIds {
      static const StorageType::TypeId sequenceName = StorageType::String255;
      static const StorageType::TypeId currentValue = StorageType::UInt32;
      static const StorageType::TypeId lastModDate  = StorageType::String255;
    }

    namespace columnTypes {
      typedef String255 sequenceName;
      typedef UInt32 currentValue;
      typedef String255 lastModDate;
    }

    /// Get the RecordSpecification of the sequence table.
    /// If the flag is true, include only the first column (the sequence name).
    const IRecordSpecification&
    tableSpecification( bool seqNameOnly = false );

  }

}

#endif // RELATIONALCOOL_RELATIONALSEQUENCETABLE_H
