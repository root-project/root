// $Id: RelationalObjectTable.h,v 1.127 2012-06-29 13:53:56 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALOBJECTTABLE_H
#define RELATIONALCOOL_RELATIONALOBJECTTABLE_H

// Include files
#include <cstdio> // For sprintf on gcc45
#include <cstring>
#include <memory>
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/MessageStream.h"
#include "CoolKernel/IObject.h"
//#include "CoolKernel/PayloadMode.h"
#include "PayloadMode.h" // TEMPORARY
#include "CoolKernel/Record.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoolKernel/ValidityKey.h"
#include "CoolKernel/IRecordSelection.h"

// Local include files
#include "RelationalQueryMgr.h"
#include "uppercaseString.h"

namespace cool
{

  // Forward declarations
  class ChannelSelection;
  class IRelationalQueryDefinition;
  class RelationalFolder;
  class RelationalObjectTableRow;
  class RelationalTagMgr;

  /** @class RelationalObjectTable RelationalObjectTable.h
   *
   *  Relational schema of the table storing COOL conditions "objects"
   *
   *  Relational implementation of the object table queries
   *  in termns of the RelationalQueryMgr.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-12-16
   */

  class RelationalObjectTable
  {

    friend class RelationalObjectMgrTest;
    friend class RelationalObjectTableTest;

  public:

    static const std::string defaultTableName
    ( const std::string& prefix, unsigned nodeId )
    {
      char tableName[] = "Fxxxx_IOVS";
      snprintf( tableName, strlen(tableName)+1, // Fix Coverity SECURE_CODING
                "F%4.4u_IOVS", nodeId );
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the prefix is uppercase anyway...
      return uppercaseString( prefix ) + std::string( tableName );
    }

    static const std::string sequenceName( const std::string& tableName )
    {
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input table name is uppercase anyway...
      return uppercaseString(tableName) + "_SEQ";
    }

    struct columnNames
    {
      static const std::string objectId() { return "OBJECT_ID"; }
      static const std::string channelId() { return "CHANNEL_ID"; }
      static const std::string iovSince() { return "IOV_SINCE"; }
      static const std::string iovUntil() { return "IOV_UNTIL"; }
      static const std::string userTagId() { return "USER_TAG_ID"; }
      static const std::string sysInsTime() { return "SYS_INSTIME"; }
      static const std::string lastModDate() { return "LASTMOD_DATE"; }
      static const std::string originalId() { return "ORIGINAL_ID"; }
      static const std::string newHeadId() { return "NEW_HEAD_ID"; }
      // SEPARATEPAYLOAD mode
      static const std::string payloadId() { return "PAYLOAD_ID"; }
      // VECTORPAYLOAD mode
      static const std::string payloadSetId() { return "PAYLOAD_SET_ID"; }
      // VECTORPAYLOAD mode
      static const std::string payloadSize() { return "PAYLOAD_N_ITEMS"; }
    };

    struct columnTypeIds
    {
      static const StorageType::TypeId objectId    = StorageType::UInt32;
      static const StorageType::TypeId channelId   = StorageType::UInt32;
      static const StorageType::TypeId iovSince    = StorageType::UInt63;
      static const StorageType::TypeId iovUntil    = StorageType::UInt63;
      static const StorageType::TypeId userTagId   = StorageType::UInt32;
      // TEMPORARY! Should be Time?
      static const StorageType::TypeId sysInsTime  = StorageType::String255;
      // TEMPORARY! Should be Time?
      static const StorageType::TypeId lastModDate = StorageType::String255;
      static const StorageType::TypeId originalId  = StorageType::UInt32;
      static const StorageType::TypeId newHeadId   = StorageType::UInt32;
      // SEPARATEPAYLOAD mode
      static const StorageType::TypeId payloadId   = StorageType::UInt32;
      // VECTORPAYLOAD mode
      static const StorageType::TypeId payloadSetId  = StorageType::UInt32;
      // VECTORPAYLOAD mode
      static const StorageType::TypeId payloadSize = StorageType::UInt32;
    };

    struct columnTypes
    {
      typedef UInt32 objectId;
      typedef UInt32 channelId;
      typedef UInt63 iovSince;
      typedef UInt63 iovUntil;
      typedef UInt32 userTagId;
      // TEMPORARY! Should be Time?
      typedef String255 sysInsTime;
      // TEMPORARY! Should be Time?
      typedef String255 lastModDate;
      typedef UInt32 originalId;
      typedef UInt32 newHeadId;
      // SEPARATEPAYLOAD mode
      typedef UInt32 payloadId;
      // VECTORPAYLOAD mode
      typedef UInt32 payloadSetId;
      // VECTORPAYLOAD mode
      typedef UInt32 payloadSize;
    };

  public:

    // Destructor
    virtual ~RelationalObjectTable();

    /// Constructor
    RelationalObjectTable( const RelationalQueryMgr& queryMgr,
                           const RelationalFolder& folder );

    /// Returns the payload specification
    const IRecordSpecification& payloadSpecification() const
    {
      return m_payloadSpecification;
    }

    /// Returns the table specification
    const IRecordSpecification& tableSpecification() const
    {
      return m_tableSpecification;
    }

    /// Compute the table specification for the given payload columns
    static const
    RecordSpecification tableSpecification
    ( const IRecordSpecification& payloadSpec,
      PayloadMode::Mode type );

    /// Compute an AttributeList for the given payload columns
    static const
    coral::AttributeList rowAttributeList
    ( const coral::AttributeList& payload,
      PayloadMode::Mode mode = PayloadMode::INLINEPAYLOAD );

    /// Define the ORDER clause for the SV, MV HEAD and MV Tag selections.
    static const
    std::vector<std::string> orderByClause
    ( const ChannelSelection& channels,
      const std::string& objectTableName = "" );

    /// Define the query for the SV selection.
    std::auto_ptr<IRelationalQueryDefinition>
    queryDefinitionSV( const ValidityKey& since,
                       const ValidityKey& until,
                       const ChannelSelection& channels,
                       const IRecordSelection* payloadQuery = 0,
                       bool optimizeClobs = false,
                       bool fetchPayload = true );

    /// Define the query for the MV user tag and HEAD selection;
    /// user tags are assumed unless isHeadTag(userTagName) is true.
    std::auto_ptr<IRelationalQueryDefinition>
    queryDefinitionHeadAndUserTag( const ValidityKey& since,
                                   const ValidityKey& until,
                                   const ChannelSelection& channels,
                                   const std::string& userTagName,
                                   const IRecordSelection* payloadQuery = 0,
                                   bool optimizeClobs = false,
                                   bool fetchPayload = true );

    /// Define the query for the MV tag selection.
    std::auto_ptr<IRelationalQueryDefinition>
    queryDefinitionTag( const ValidityKey& since,
                        const ValidityKey& until,
                        const ChannelSelection& channels,
                        //unsigned int tagId );
                        const std::string& tagName,
                        const IRecordSelection* payloadQuery = 0,
                        bool optimizeClobs = false,
                        bool fetchPayload = true );

    /// Define the query for all four: SV, MV user tag, MV HEAD and MV tag.
    /// By default this defines the SV query (pTagId==0, isUserTag is ignored).
    /// The HEAD case is triggered if (*pTagId)==0 (isUserTag is ignored).
    std::auto_ptr<IRelationalQueryDefinition>
    queryDefinitionGeneric( const ValidityKey& since,
                            const ValidityKey& until,
                            const ChannelSelection& channels,
                            const unsigned int* pTagId = 0,
                            bool isUserTag = true,
                            const IRecordSelection* payloadQuery = 0,
                            bool optimizeClobs = false,
                            bool fetchPayload = true );

  public:

    /// Returns the table name.
    const std::string& objectTableName() const { return m_objectTableName; }

    /// Returns the associated tag table name.
    const std::string& tagTableName() const { return m_tagTableName; }

    /// Returns the associated IOV2tag table name.
    const std::string& object2TagTableName() const
    {
      return m_object2TagTableName;
    }

    /// Returns the associated channel table name.
    const std::string& channelTableName() const { return m_channelTableName; }

    /*
    /// TEMPORARY! This is ONLY needed by David's VerificationClient
    /// Fetch one IOV row (lookup for last inserted row in 1 channel - SV)
    /// [Rows are inserted with strictly ascending objectIds: this method
    /// fetches the row with the highest objectId in the given channel]
    RelationalObjectTableRow fetchLastRowSV( const ChannelId& channelId,
                                             bool fetchPayload = false );
    */

    /// Fetch the last row in a given tag (the highest object_id)
    const RelationalObjectTableRow
    fetchLastRowForTagId( unsigned int tagId,
                          bool fetchPayload = false ) const;

    /// Fetch one IOV row (lookup at 1 time in 1 channel in HEAD tag - MV)
    /// \todo TODO sas 2006-04-11 - this method was mainly used in findObject
    /// which is now using browseObjects instead. There's one other use
    /// (apart from tests) in RalDatabase that could be eliminated.
    const RelationalObjectTableRow fetchRowAtTimeInHead
    ( const ValidityKey& pointInTime,
      const ChannelId& channelId,
      unsigned int userTagId = 0 );

    /*
    /// Fetch one IOV row (lookup at 1 time in 1 channel in 1 tag - MV)
    /// \todo TODO sas 2006-04-11 - this method was mainly used in findObject
    /// which is now using browseObjects instead. There's no other use
    /// (apart from tests).
    RelationalObjectTableRow fetchRowAtTimeInTag
    ( const ValidityKey& pointInTime,
      const ChannelId& channelId,
      const std::string& tagName );
    */

    std::auto_ptr< std::vector<RelationalObjectTableRow> >
    fetchRowsBtTimesInHead
    ( const ValidityKey& since,
      const ValidityKey& until,
      const ChannelId& channelId,
      unsigned int userTagId,
      unsigned int maxRows );

    /// Utility method to fetch the HEAD object table rows for tagging
    /// The query only fetches the meta data (object_id, since, until,...)
    /// in order to avoid reading the payload and be more lightweight.
    /// Therefore the normal fetchObjectTableRows method is not used.
    std::vector<RelationalObjectTableRow> fetchRowsForTaggingCurrentHead
    ( PayloadMode::Mode pMode );

    /// Utility method to fetch the object table rows 'asOfDate' for tagging.
    /// The query only fetches the meta data (object_id, since, until,...)
    /// in order to avoid reading the payload and be more lightweight.
    /// Therefore the normal fetchObjectTableRows method is not used.
    std::vector<RelationalObjectTableRow> fetchRowsForTaggingHeadAsOfDate
    ( const ITime& asOfDate, PayloadMode::Mode pMode );

    /// Utility method to fetch the object table rows 'asOfObjectId'
    /// for tagging. The object with id 'objectId' is included.
    /// The query only fetches the meta data (object_id, since, until,...)
    /// in order to avoid reading the payload and be more lightweight.
    /// Therefore the normal fetchObjectTableRows method is not used.
    std::vector<RelationalObjectTableRow> fetchRowsForTaggingHeadAsOfObjectId
    ( unsigned int asOfObjectId, PayloadMode::Mode pMode );

    /// Returns the name of the object_id sequence associated with this table.
    const std::string sequenceName()
    {
      // TEMPORARY? AV 04.04.2005
      // FIXME: presently the input table name is uppercase anyway...
      return uppercaseString( objectTableName() ) + "_SEQ";
    }

    /// Returns the table specification for the given payload columns
    static const IRecordSpecification& defaultSpecification();

    /// Does a channel with this id exist?
    /// This is needed by dropChannel (throw if the channel has any IOVs).
    bool existsChannel( const ChannelId& channelId ) const;

  protected:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Get the RelationalQueryMgr associated to this table
    const RelationalQueryMgr& queryMgr() const { return m_queryMgr; }

    /// Fetch one IOV row (lookup by 1 objectId - SV and MV)
    const RelationalObjectTableRow
    fetchRowForId( unsigned int objectId,
                   bool fetchPayload = false ) const;

  private:

    /// Copy constructor is private
    RelationalObjectTable( const RelationalObjectTable& rhs );

    /// Assignment operator is private
    RelationalObjectTable& operator=( const RelationalObjectTable& rhs );

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Relational query manager
    const RelationalQueryMgr& m_queryMgr;

  protected: // TEMPORARY? For the RelationalObjectTableTest friend only

    /// Relational tag manager
    RelationalTagMgr& m_tagMgr;

  private:

    /// Payload specification
    RecordSpecification m_payloadSpecification;

    /// Full table specification
    RecordSpecification m_tableSpecification;

    /// IOV table name
    std::string m_objectTableName;

    /// Tag table name
    std::string m_tagTableName;

    /// IOV2tag table name
    std::string m_object2TagTableName;

    /// Channel table name
    std::string m_channelTableName;

    /// Payload table name
    std::string m_payloadTableName;

    /// Payload mode
    PayloadMode::Mode m_payloadMode;

  protected: // TEMPORARY? For the RelationalObjectTableTest friend only

    /// Node id for this node
    unsigned int m_nodeId;

  };

}

#endif // RELATIONALCOOL_RELATIONALOBJECTTABLE_H
