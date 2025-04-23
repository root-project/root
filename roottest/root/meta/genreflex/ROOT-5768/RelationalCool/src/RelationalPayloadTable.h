// $Id: RelationalPayloadTable.h,v 1.8 2012-07-02 17:03:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALPAYLOADTABLE_H
#define RELATIONALCOOL_RELATIONALPAYLOADTABLE_H 1

// Include files
#include <cstdio> // For sprintf on gcc45
#include <cstring>
#include <memory>

// Local include files
//#include "CoolKernel/PayloadMode.h"
#include "PayloadMode.h" // TEMPORARY
#include "CoolKernel/Record.h"
#include "CoolKernel/RecordSpecification.h"
#include "uppercaseString.h"

namespace cool
{

  /** @class RelationalPayloadTable RelationalPayloadTable.h
   *
   *  Relational schema of the table storing COOL payload
   *
   *
   *  @author Martin Wache
   *  @date   2008-10-16
   */

  class RelationalPayloadTable
  {


  public:

    static const std::string defaultTableName
    ( const std::string& prefix, const unsigned nodeId ) {
      char tableName[] = "Fxxxx_PAYLOAD";
      snprintf( tableName, strlen(tableName)+1, // Fix Coverity SECURE_CODING
                "F%4.4u_PAYLOAD", nodeId );
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the prefix is uppercase anyway...
      return uppercaseString( prefix ) + std::string( tableName );
    }

    static const std::string sequenceName( const std::string& tableName ) {
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input table name is uppercase anyway...
      return uppercaseString(tableName) + "_SEQ";
    }

    struct columnNames {
      // separate payload mode
      static const std::string payloadId() { return "PAYLOAD_ID"; }
      // vector payload mode
      static const std::string payloadSetId() { return "PAYLOAD_SET_ID"; }
      // vector payload mode
      static const std::string payloadItemId() { return "PAYLOAD_ITEM_ID"; }
      static const std::string p_sysInsTime() { return "P_SYS_INSTIME"; }
    };

    struct columnTypeIds {
      // separate payload mode
      static const StorageType::TypeId payloadId    = StorageType::UInt32;
      // vector payload mode
      static const StorageType::TypeId payloadSetId = StorageType::UInt32;
      // vector payload mode
      static const StorageType::TypeId payloadItemId= StorageType::UInt32;
      static const StorageType::TypeId p_sysInsTime = StorageType::String255;
    };

    struct columnTypes {
      // separate payload mode
      typedef UInt32 payloadId;
      // vector payload mode
      typedef UInt32 payloadSetId;
      // vector payload mode
      typedef UInt32 payloadItemId;
      typedef String255 p_sysInsTime;
    };

  public:

    // Destructor
    virtual ~RelationalPayloadTable();

    // returns the minimum specification (without any payload) for a payload table
    static const IRecordSpecification& defaultSpecification( PayloadMode::Mode pMode);

    /// Returns the table specification for the given payload columns
    static const RecordSpecification
    tableSpecification( const IRecordSpecification& payloadSpec,
                        PayloadMode::Mode pMode );

    static const coral::AttributeList
    rowAttributeList( const coral::AttributeList& payload,
                      PayloadMode::Mode pMode );

  };

}
#endif // RELATIONALCOOL_RELATIONALPAYLOADTABLE_H
