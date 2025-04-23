// $Id: RelationalObjectMgr.h,v 1.45 2012-01-30 16:13:01 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALOBJECTMGR_H
#define RELATIONALCOOL_RELATIONALOBJECTMGR_H

// Include files
#include <boost/shared_ptr.hpp>
#include <memory>
#include "CoralBase/MessageStream.h"

// Local include files
#include "ObjectId.h"
#include "RelationalFolder.h"
#include "RelationalObjectPtr.h"
#include "RelationalPayloadTableRow.h"
#include "SimpleObject.h"

namespace cool
{

  // Forward declarations
  class RelationalDatabase;
  class RelationalQueryMgr;
  class RelationalTagMgr;
  class SimpleObject;

  // Type definitions
  typedef std::map< ChannelId, ValidityKey > ChannelIdValidityKeyMap;
  typedef std::map< ChannelId, ObjectId > ChannelIdObjectIdMap;
  typedef boost::shared_ptr< RelationalObjectTableRow > RelationalObjectTableRowPtr;
  typedef boost::shared_ptr< RelationalPayloadTableRow > RelationalPayloadTableRowPtr;

  /// @class RelationalObjectMgr RelationalObjectMgr.h
  ///
  /// RelationalObjectMgr handles object queries of a COOL database instance.
  ///
  /// @author Sven A. Schmidt
  /// @date   2006-04-19
  ///
  class RelationalObjectMgr
  {

    friend class RelationalObjectMgrTest;

  public:

    // Virtual destructor
    virtual ~RelationalObjectMgr() {}

    /// Constructor from a RelationalDatabase reference.
    RelationalObjectMgr( const RelationalDatabase& db );

    /// Find THE conditions object valid at the given point in time
    IObjectPtr findObject( const RelationalFolder* folder,
                           const ValidityKey& pointInTime,
                           const ChannelId& channelId,
                           const std::string& tagName = "" ) const;

    /// Browse the objects in a given folder for a given
    /// channel selection within the given tag.
    /// The iterator will retrieve only ONE object at any given validity point
    /// for a given channel. The default order is channel, since if a channel
    /// range is specified.
    /// For SINGLE_VERSION folders, throw exception if tag != "" is specified.
    IObjectIteratorPtr
    browseObjects( const RelationalFolder* folder,
                   const ValidityKey& since,
                   const ValidityKey& until,
                   const ChannelSelection& channels,
                   const std::string& tagName = "",
                   const IRecordSelection* payloadQuery = 0,
                   const bool countOnly = false ) const;

    /// Store a list of conditions objects in the specified order
    void storeObjects
    ( RelationalFolder* folder,
      const std::vector<RelationalObjectPtr> & objects,
      bool userTagOnly = false ) const;

    /// Set a new finite end-of-validity value for all SV objects in a given
    /// channel selection whose end-of-validity is currently infinite.
    /// Throws an Exception if called on a MV folder, or if any of the
    /// selected channels contains a not open ended IOV at the point until.
    /// Returns the number of actually truncated IOVs.
    int truncateObjectValidity( const RelationalFolder* folder,
                                const ValidityKey& until,
                                const ChannelSelection& channels ) const;

    /// Create a new channel
    void createChannel( const RelationalFolder* folder,
                        const ChannelId& channelId,
                        const std::string& channelName,
                        const std::string& description = "" ) const;

    /// Drop a channel
    bool dropChannel( const RelationalFolder* folder,
                      const ChannelId& channelId ) const;

    /// Get the RelationalDatabase
    const RelationalDatabase& db() const
    {
      return m_db;
    }

    /// Get the RelationalQueryMgr
    const RelationalQueryMgr& queryMgr() const
    {
      return db().queryMgr();
    }

    /// Get the RelationalTagMgr
    RelationalTagMgr& tagMgr() const
    {
      return db().tagMgr();
    }

    /// Get the IRelationalTransactionMgr
    boost::shared_ptr<IRelationalTransactionMgr> transactionMgr() const
    {
      return db().transactionMgr();
    }

  protected:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Insert a new channel table row.
    void insertChannelTableRow( const std::string& channelTableName,
                                const ChannelId& channelId,
                                unsigned int lastObjectId,
                                bool hasNewData,
                                const std::string& channelName,
                                const std::string& description ) const;

    /// Updates lastObjectId and hasNewData for the given channel.
    /// Column lastObjectId is updated only if hasNewData is false; if
    /// hasNewData is true, the input lastObjectId must be 0 (else exception).
    /// Check that lastObjectId is expected to be 0 if hasNewData is false.
    /// If the channel does not exist, create it with the given metadata.
    void updateChannelTable( const std::string& channelTableName,
                             const ChannelId& channelId,
                             unsigned int lastObjectId,
                             bool hasNewData ) const;

    /// Bulk update lastObjectId and hasNewData for the given channels
    /// (but lastObjectId is updated only if hasNewData is false).
    /// If hasNewData is true, also check whether bulk update fails because
    /// some channels are missing, and in that case use single row update and
    /// create any missing channels (NB this relies on hasNewData being
    /// different from false in the database only while updating/inserting).
    void bulkUpdateChannelTable
    ( const std::string& channelTableName,
      const std::map< ChannelId, unsigned int >& updateDataMap,
      bool hasNewData ) const;

    /// Bulk update the IOV until values in the object table row with the
    /// given ids in the objectIdNewUntil map.
    void bulkUpdateObjectTableIov
    ( const std::string& objectTableName,
      const std::map<unsigned int,ValidityKey>& objectIdNewUntil ) const;

    /// Update the newHeadId value in the object table rows overlapping with
    /// the given since, until interval.
    /// Returns true if some rows have been updated.
    bool
    bulkUpdateObjectTableNewHeadId( const std::string& objectTableName,
                                    const std::string& channelTableName,
                                    const SOVector& updateNewHeads,
                                    unsigned int userTagId ) const;

    /// Stores the given list of rows in the given iov table
    /// NB: This method assumes that the objectIds of the rows are set
    /// externally and that the objectId sequence is advanced. Since SV and MV
    /// use different objectId assignment, this is necessary to be able to use
    /// the some storage method for both. Only the insertionTime is updated
    /// before flushing the rows to the database.
    void bulkInsertObjectTableRows
    ( const RelationalFolder* folder,
      const std::vector<RelationalObjectTableRowPtr>& rows,
      const std::vector<RelationalPayloadTableRowPtr>& payloadRows
      ) const;

    /// Updates open-ended IOVs for the given vector of intersectors. This list
    /// is expected to contain one ValidityKey 'since' per channel indicating
    /// the potential 'closing' IOV.
    /// Throws an exception if any object with an until > since is found \e
    /// unless this until is equal to ValidityKeyMax.
    void updateSingleVersionIovs
    ( RelationalFolder* folder,
      const std::pair<ChannelIdValidityKeyMap, ChannelIdObjectIdMap>&
      intersectors,
      const std::vector<RelationalObjectPtr>& objects ) const;

    /// Fetches all object table rows for channels that have new data.
    /// This method is used in multi channel bulk insertion and selects
    /// the rows in (potential) need of open IOV updating.
    void fetchLastRowsWithNewData
    ( RelationalFolder* folder,
      std::vector<RelationalObjectTableRow>& rows ) const;

    /// Stores objects in 'SingleVersion' mode
    void storeSingleVersionObjects
    ( RelationalFolder* folder,
      const std::vector<RelationalObjectPtr>& objects ) const;

    /// Stores objects in 'MultiVersion' mode
    void storeMultiVersionObjects
    ( RelationalFolder* folder,
      const std::vector<RelationalObjectPtr>& objects,
      bool userTagOnly = false ) const;

    /// Validates the given objects for SV storage (with respect to overlapping
    /// IOVs). Returns a map of channelIds with the lowest IOV per channel
    /// of the objects for updating open-ended IOVs in the persistent store
    /// and validating the insertion and a map of the last object ids per
    /// channel which is used to update the channel table.
    std::pair<ChannelIdValidityKeyMap, ChannelIdObjectIdMap>
    processSingleVersionObjects
    ( const std::vector<RelationalObjectPtr>& objects,
      std::vector<RelationalObjectTableRowPtr>& rows,
      std::vector<RelationalPayloadTableRowPtr>& payloadRows,
      unsigned int objectIdOffset,
      unsigned int payloadIdOffset,
      PayloadMode::Mode pMode ) const;

    /// Processes 'MultiVersion' objects for bulk insertion:
    ///    - analyse the objects and insert system objects if required
    ///    - find and return potential 'intersections' from the given
    ///      objects that might cause splitting in the HEAD
    ///    - create consistent references references (original_id,
    ///      new_head_id) among the objects. In order to do this without
    ///      having to access the objectId sequence for each new system
    ///      object being inserted, the objectIdOffset is used and the
    ///      objectIds inside processMultiVersionObjects are created from this
    ///      offset.
    ///    - idToIndex is used to bookkeep the object id to 'rows' index
    ///      mapping. This is subsequently used by mergeWithHead to know
    ///      where to insert further system objects.
    ///   Throws an TagIsLocked exception if partiallyLocked==true and the IOVs
    ///   of the objects overlap
    std::vector<SimpleObject> processMultiVersionObjects // OUT
    ( const std::vector<RelationalObjectPtr>& objects, // IN
      std::vector<RelationalObjectTableRowPtr>& rows, // OUT
      std::vector<RelationalPayloadTableRowPtr>& payloadRows, // OUT
      unsigned int objectIdOffset, // IN
      std::map<unsigned int, unsigned int>& idToIndex, // OUT
      unsigned int userTagId = 0, // IN
      bool partiallyLocked = false, // IN
      unsigned int pIdOffset = 0,
      PayloadMode::Mode pMode = PayloadMode::INLINEPAYLOAD ) const; // IN

    /// Merges the previously processed 'MultiVersion' objects (by
    /// processMultiVersionObjects()) with the persistent head
    /// Throws TagIsLocked exception if partiallyLocked==true and
    /// there are overlapping IOVs
    void mergeWithHead( RelationalFolder* folder,
                        const std::vector<SimpleObject> splitters,
                        std::vector<RelationalObjectTableRowPtr>& rows,
                        std::map<unsigned int, unsigned int>& idToIndex,
                        unsigned int userTagId = 0,
                        bool partiallyLocked = false ) const;

    /// Creates a system object row templated from the given row
    /// The iov and the object id are updated with the given values,
    /// original_id is set to the original row's object_id
    RelationalObjectTableRowPtr
    createSystemObjectRow( const RelationalObjectTableRow& origRow,
                           unsigned int objectId,
                           const ValidityKey& since,
                           const ValidityKey& until ) const;

  private:

    /// Standard constructor is private
    RelationalObjectMgr();

    /// Copy constructor is private
    RelationalObjectMgr( const RelationalObjectMgr& rhs );

    /// Assignment operator is private
    RelationalObjectMgr& operator=( const RelationalObjectMgr& rhs );

  private:

    /// The RelationalDatabase reference
    const RelationalDatabase& m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

} // namespace


#endif // RELATIONALCOOL_RELATIONALOBJECTMGR_H
