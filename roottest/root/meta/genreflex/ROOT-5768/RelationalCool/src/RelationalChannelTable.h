// $Id: RelationalChannelTable.h,v 1.19 2012-06-29 13:53:56 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALCHANNELTABLE_H
#define RELATIONALCOOL_RELATIONALCHANNELTABLE_H

// Include files
#include <cstdio> // For sprintf on gcc45
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "CoralBase/MessageStream.h"
#include "CoolKernel/ChannelId.h"
#include "CoolKernel/StorageType.h"

// Local iclude files
#include "uppercaseString.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;
  class RelationalDatabase;
  class RelationalFolder;
  class RelationalQueryMgr;
  class RelationalTableRow;

  /** @namespace RelationalChannelTable RelationalChannelTable.h
   *
   *  Relational schema of the table storing COOL channel metadata.
   *
   *  @author Sven A. Schmidt and Marco Clemencic
   *  @date   2006-04-27
   */

  class RelationalChannelTable {

  public:

    RelationalChannelTable( const RelationalDatabase& db,
                            const RelationalFolder& folder );

    static const std::string defaultTableName
    ( const std::string& prefix, unsigned nodeId ) {
      char tableName[] = "Fxxxx_CHANNELS";
      snprintf( tableName, strlen(tableName)+1, // Fix Coverity SECURE_CODING
                "F%4.4u_CHANNELS", nodeId );
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input prefix is uppercase anyway...
      return uppercaseString( prefix ) + std::string( tableName );
    }

    struct columnNames {
      static const std::string channelId() { return "CHANNEL_ID"; }
      static const std::string lastObjectId() { return "LAST_OBJECT_ID"; }
      static const std::string hasNewData() { return "HAS_NEW_DATA"; }
      static const std::string channelName() { return "CHANNEL_NAME"; }
      static const std::string description() { return "DESCRIPTION"; }
    };

    struct columnTypeIds {
      static const StorageType::TypeId channelId    = StorageType::UInt32;
      static const StorageType::TypeId lastObjectId = StorageType::UInt32;
      static const StorageType::TypeId hasNewData   = StorageType::Bool;
      static const StorageType::TypeId channelName  = StorageType::String255;
      static const StorageType::TypeId description  = StorageType::String255;
    };

    struct columnTypes {
      typedef UInt32 channelId;
      typedef UInt32 lastObjectId;
      typedef Bool hasNewData;
      typedef String255 channelName;
      typedef String255 description;
    };

    /// Get the record specification of the channel table
    static const IRecordSpecification& tableSpecification();

    /// Fetches the channel table row for the given channel id.
    const RelationalTableRow
    fetchRowForId( const ChannelId& channelId ) const;

    /// Fetches the channel table row for the given channel name.
    const RelationalTableRow
    fetchRowForChannelName( const std::string& channelName ) const;

    /// Lists channels (fetches all rows)
    const std::vector<ChannelId> listChannels() const;

    /// Lists channels with names (fetches all rows)
    const std::map<ChannelId,std::string> listChannelsWithNames() const;

    /// Returns the table name.
    const std::string& tableName() const { return m_tableName; }

  private:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const { return *m_log; }

    /// Get the RelationalQueryMgr associated to this table
    RelationalQueryMgr& queryMgr() const { return m_queryMgr; }

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Relational query manager
    RelationalQueryMgr& m_queryMgr;

    /// Channel table name
    std::string m_tableName;

  };

}
#endif // RELATIONALCOOL_RELATIONALCHANNELTABLE_H
