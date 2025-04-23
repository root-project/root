// $Id: RelationalFolder.h,v 1.178 2012-07-08 20:02:33 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALFOLDER_H
#define RELATIONALCOOL_RELATIONALFOLDER_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Disable warning C4250 on Windows (inheritance via dominance)
// Copied from SEAL (Dictionary/Reflection/src/Tools.h)
#ifdef WIN32
#pragma warning ( disable : 4250 )
#endif

// Include files
#include <memory>
#include <vector>
#include "CoralBase/MessageStream.h"
#include "CoolKernel/ChannelSelection.h"
#include "CoolKernel/FolderSpecification.h"
#include "CoolKernel/IFolder.h"
#include "CoolKernel/InternalErrorException.h"
#include "CoolKernel/RecordSpecification.h"

// Local include files
#include "RelationalDatabase.h"
#include "RelationalHvsNode.h"
#include "VersionNumber.h"

namespace cool {

  /** @class RelationalFolder RelationalFolder.h
   *
   *  Relational implementation of a COOL condition database "folder".
   *
   *  Also represents implementation within COOL of an HVS leaf node.
   *  Multiple virtual inheritance from IFolder and RelationalHvsNode
   *  (diamond virtual inheritance of IHvsNodeRecord abstract interface).
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2004-11-24
   */

  class RelationalFolder : virtual public IFolder,
                           virtual public RelationalHvsNode
  {

  public:

    /// Row cache size for all bulk operations
    static unsigned int bulkOpRowCacheSize();

    /// Folder schema version for this class
    static const VersionNumber 
    folderSchemaVersion( PayloadMode::Mode payloadMode )
    {
      // Summary of changes since software release COOL_2_0_0:
      // - COOL_2_0 creates '2.0.0' folders.
      // - COOL_2_1 still creates '2.0.0' folders (as agreed with users),
      //   even if these actually have an additional 5D index in the IOV table
      //   with respect to those created using COOL_2_0_0.
      // - COOL_2_2 creates '2.0.1' folders: these have a FK on channelId
      //   from the IOV to the channels table (only enforced on Oracle/MySQL);
      //   also, the MySQL default type for VARCHAR(255) is now BINARY.
      //   COOL_2_2 continues to read data using the 2.0.0 folder queries.
      // - COOL_2_3/2_4/2_5/2_6/2_7 create '2.0.1' folders just like COOL_2_2.
      // - COOL_2_8 may create '2.0.1' or '2.8.0' folders,
      //   the latter being folders with a separate payload table.
      // - COOL_2_9 may create '2.0.1', '2.8.0' and '2.9.0' folders,
      //   the latter being folders with vector payload in a separate table.
      if ( payloadMode == PayloadMode::INLINEPAYLOAD ) return "2.0.1";
      else if ( payloadMode == PayloadMode::SEPARATEPAYLOAD ) return "2.8.0";
      else if ( payloadMode == PayloadMode::VECTORPAYLOAD ) return "2.9.0";
      else throw cool::InternalErrorException
             ( "Unknown folder schema version",
               "RelationalFolder::folderSchemaVersion" );
    }

    /// Check if a folder schema version is supported by this class
    static bool isSupportedSchemaVersion( const VersionNumber& schemaVersion );

    /// Folder schema payload mode
    /// [WARNING: counterintuitive! inlineMode=0 is inline, 1 is not...]
    static UInt16 folderSchemaPayloadMode( PayloadMode::Mode payloadMode )
    {
      if ( payloadMode == PayloadMode::INLINEPAYLOAD ) return 0;  // inline
      else return 1;  // not inline (payload table)
    }

    /// Constructor to create a RelationalFolder from a node table row
    /// Inlined with the base class non-standard constructors as otherwise
    /// the compiler attempts to use the base class standard constructors
    /// (Windows compilation error C2248: standard constructors are private)
    RelationalFolder( const RelationalDatabasePtr& db,
                      const coral::AttributeList& row )
      : RelationalHvsNodeRecord( row )
      , RelationalHvsNode( db, row )
      , m_log( new coral::MessageStream( "RelationalFolder" ) )
      , m_folderSpec( versioningMode( row ), payloadSpecification( row ),
#ifdef COOL290VP
                      payloadMode( row ) 
#else
                      payloadMode( row ) == PayloadMode::SEPARATEPAYLOAD 
#endif
                      )
      , m_publicFolderAttributes() // fill it in initialize
    {
      initialize( row );
    }

    /// Destructor.
    virtual ~RelationalFolder();

  public:

    /// Return the folder specification.
    const IFolderSpecification& folderSpecification() const {
      return m_folderSpec;
    }

    /// Return the payload specification of the folder.
    const IRecordSpecification& payloadSpecification() const {
      return folderSpecification().payloadSpecification();
    }

    /// Return the folder versioning mode.
    FolderVersioning::Mode versioningMode() const {
      return folderSpecification().versioningMode();
    }

    /// Return the folder payload mode.
    PayloadMode::Mode payloadMode() const 
    {
#ifdef COOL290VP
      return folderSpecification().payloadMode();
#else
      return folderSpecification().hasPayloadTable() 
        ? PayloadMode::SEPARATEPAYLOAD 
        : PayloadMode::INLINEPAYLOAD;
#endif
    }

    /// Return true if the folder uses a separate payload table.
    bool hasPayloadTable() const 
    {
      return folderSpecification().hasPayloadTable();
    }

    /// Return the 'attributes' of the folder
    /// (implementation-specific properties not exposed in the API).
    const IRecord& folderAttributes() const;

    /*
    /// Declare that some payload columns reference external payload.
    /// The referencedEntity URL contains the path to the external container.
    /// Only FKs to relational tables within the same database are supported
    /// so far with the syntax "local://schema=USERNAME;table=TABLENAME".
    /// Non-relational implementation of CoolKernel should throw an exception.
    /// Returns true in case of success, false in case of any error.
    bool declareExternalReference
    ( const std::string& name,
      const std::vector< std::string >& attributes,
      const std::string& referencedEntity,
      const std::vector< std::string >& referencedAttributes );

    /// List the external reference constraints.
    const std::vector< std::string > externalReferences() const;

    /// Retrieve the properties of an external reference constraint
    const IFolder::ExternalReference&
    externalReference( const std::string& name ) const;
    */

    /// Activate/deactivate a storage buffer for bulk insertion of objects.
    /// If the buffer was used and is deactivated, flush the buffer.
    void setupStorageBuffer( bool useBuffer = true );

    /// Flush the storage buffer (execute the bulk insertion of objects).
    /// If the buffer is not used, ignore this command.
    void flushStorageBuffer();

    /// Store an object in a given channel with the given IOV and data payload.
    /// If the buffer is used, only register the object for later storage.
    void storeObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const IRecord& payload,
                      const ChannelId& channelId,
                      const std::string& userTagName = "",
                      bool userTagOnly = false );

    void storeObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const std::vector<IRecordPtr>& payload,
                      const ChannelId& channelId,
                      const std::string& userTagName = "",
                      bool userTagOnly = false );

    /// Store an object in a given channel with the given IOV and payloadId.
    /// The payload must already be stored in the payload table.
    void storeObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const unsigned int payloadId,
                      const ChannelId& channelId,
                      const std::string& userTagName = "",
                      bool userTagOnly = false );

    /// Store an object in a given channel with the given IOV and payloadSetId.
    /// The payload must already be stored in the payload table.
    void storeObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const unsigned int payloadSetId,
                      const unsigned int payloadSize,
                      const ChannelId& channelId,
                      const std::string& userTagName = "",
                      bool userTagOnly = false );

    // OBSOLETE - kept for backward compatibility
    void storeObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const coral::AttributeList& payload,
                      const ChannelId& channelId,
                      const std::string& userTagName = "",
                      bool userTagOnly = false );

    // common method called by the methods with the different parameters above
    void storeObject( RelationalObjectPtr& object,
                      bool userTagOnly = false );

    /// Set a new finite end-of-validity value for all SV objects in a given
    /// channel selection whose end-of-validity is currently infinite.
    /// The channel selection is specified through a ChannelSelection object.
    /// Throws an Exception if called on a MV folder, or if any of the
    /// selected channels contains a not open ended IOV at the point until.
    /// Returns the number of actually truncated IOVs.
    virtual int truncateObjectValidity( const ValidityKey& until,
                                        const ChannelSelection& channels );

    /// Find the ONE object in a given channel valid at the given point in time
    /// in the given tag ("" and "HEAD" both indicate the folder HEAD).
    /// Tag names, except for "HEAD", are case sensitive.
    /// If the buffer is used, it must be flushed first (else throw exception).
    /// Throws an ObjectNotFound exception if no such object exists.
    /// For single version folders, the tag must be either "" or "HEAD",
    /// otherwise a TagNotFound exception is thrown.
    IObjectPtr findObject( const ValidityKey& pointInTime,
                           const ChannelId& channelId,
                           const std::string& tagName="" ) const;

    /// Find the objects for a given channel selection at the given point
    /// in time in the given tag ("" and "HEAD" both indicate the folder HEAD).
    /// Tag names, except for "HEAD", are case sensitive.
    /// The channel selection is specified through a ChannelSelection object.
    /// The iterator will retrieve only ONE object for each selected channel
    /// (or none if there is no valid IOV at pointInTime in the channel).
    /// The iterator returns objects ordered by channel: the order clause
    /// optionally specified in the ChannelSelection object is ignored.
    /// For single version folders, the tag must be either "" or "HEAD",
    /// otherwise a TagNotFound exception is thrown.
    IObjectIteratorPtr
    findObjects( const ValidityKey& pointInTime,
                 const ChannelSelection& channels,
                 const std::string& tagName = "" ) const;

    /// Browse the objects for a given channel selection within the given time
    /// range in the given tag ("" and "HEAD" both indicate the folder HEAD).
    /// Tag names, except for "HEAD", are case sensitive.
    /// The validity range is inclusive at both ends as well, i.e.
    /// since = t1, until = t2 will select an object with since = t2.
    /// The channel selection is specified through a ChannelSelection object.
    /// The iterator will retrieve only ONE object at any given validity point
    /// for a given channel. The default order is channel, since.
    /// For single version folders, the tag must be either "" or "HEAD",
    /// otherwise a TagNotFound exception is thrown.
    IObjectIteratorPtr
    browseObjects( const ValidityKey& since,
                   const ValidityKey& until,
                   const ChannelSelection& channels,
                   const std::string& tagName = "",
                   const IRecordSelection* payloadQuery = 0 ) const;

    /// Internal method
    IObjectIteratorPtr
    browseObjects( const ValidityKey& since,
                   const ValidityKey& until,
                   const ChannelSelection& channels,
                   const std::string& tagName,
                   bool prefetchAll,
                   const IRecordSelection* payloadQuery ) const;

    /// Return the object count for the given selection. The selection
    /// parameters are the same as for browseObjects (as is the actual
    /// selection).
    unsigned int 
    countObjects( const ValidityKey& since,
                  const ValidityKey& until,
                  const ChannelSelection& channels,
                  const std::string& tagName = "",
                  const IRecordSelection* payloadQuery = 0 ) const;

    /// Change the object prefetch policy (default is prefetchAll=true).
    /// If prefetchAll is true, the browse() methods prefetch all IOVs and
    /// return IObjectIterator's that are wrappers to IObjectPtr vectors.
    /// If prefetchAll is false, the browse() methods prefetch only a few
    /// IOVs at a time and return IObjectIterator's that are wrappers to
    /// server-side cursors: a transaction and the cursor are opened when
    /// the iterator is created, and these are only closed when the iterator
    /// is explicitly deleted (or the shared pointer goes out of scope).
    /// No other query against the database can be performed during that time.
    void setPrefetchAll( bool prefetchAll ) 
    {
      m_prefetchAll = prefetchAll;
    }

    /// to be documented
    /// new browseObjects methods returning IObjectVector
    IObjectVectorPtr 
    fetchObjectsAtTime( const ValidityKey& pointInTime,
                        const ChannelSelection& channels,
                        const std::string& tagName = "" ) const;
    
    /// to be documented
    /// new browseObjects methods returning IObjectVector
    IObjectVectorPtr 
    fetchObjectsInRange( const ValidityKey& since,
                         const ValidityKey& until,
                         const ChannelSelection& channels,
                         const std::string& tagName = "" ) const;

    /// Associates the objects that are the current HEAD
    /// with the given tag name and tag description.
    void tagCurrentHead( const std::string& tagName,
                         const std::string& description = "" ) const;

    /// Clones the tag "tagName" as user tag "tagClone" by reinserting
    /// the IOVs with the user tag. Does not modify the current head.
    /// This method is non-const because it sets m_useBuffer to use bulk
    /// insertion and eventually restores the original setting of m_useBuffer.
    void cloneTagAsUserTag( const std::string& tagName,
                            const std::string& tagClone,
                            const std::string& description = "",
                            bool forceOverwriteTag = false );

    /// Associates the objects that were the HEAD at 'asOfDate'
    /// with the given tag name and tag description.
    void tagHeadAsOfDate( const ITime& asOfDate,
                          const std::string& tagName,
                          const std::string& description = "" ) const;

    /// TEMPORARY? Can be commented out until included in the public API?
    /// Tags the objects in the given folder
    /// that were at the HEAD at 'asOfObjectId', at the time the object with
    /// id 'asOfObjectId' was inserted (and including that object)
    /// with the given tagName and description. Throws a TagExists
    /// exception if a tag by that name already exists.
    void tagHeadAsOfObjectId( unsigned int asOfObjectId,
                              const std::string& tagName,
                              const std::string& description );

    /// Tags the given objects of the given folder. Throws a TagExists
    /// exception if a tag by that name already exists.
    void tagObjectList( const std::string& tagName,
                        const std::string& description,
                        const std::vector<RelationalObjectTableRow>& objects ) const;

    /// Inserts the given objects for the given tag id in the given
    /// object2Tag table
    void insertObject2TagTableRows
    ( const std::string& object2TagTableName,
      unsigned int tagId,
      const std::string& insertionTime,
      const std::vector<RelationalObjectTableRow>& objects,
      PayloadMode::Mode pMode ) const;

    /// Delete rows with tagId from the object2tag table.
    /// Returns the number of deleted rows.
    unsigned int
    deleteObject2TagTableRows( const std::string& object2TagTableName,
                               unsigned int tagId ) const;

    /// Delete rows with tagId from the object table.
    /// Returns the number of deleted rows.
    unsigned int
    deleteObjectTableRowsForUserTag( const std::string& objectTableName,
                                     unsigned int tagId ) const;

    /// Insertion time of the last inserted IOV in a tag defined for this node.
    const Time
    insertionTimeOfLastObjectInTag( const std::string& tagName ) const;

    /// Deletes a tag from this folder and from the global "tag namespace".
    /// The tag name is available for tagging again afterwards.
    void deleteTag( const std::string& tagName );

    /// Does this user tag exist? - This starts a transaction
    /// Tag names are case sensitive.
    bool existsUserTag( const std::string& userTagName ) const;

    /// Does this user tag exist? - This does not start a transaction
    /// Tag names are case sensitive.
    bool __existsUserTag( const std::string& userTagName ) const;

    /// Rename a payload item
    void renamePayload( const std::string& oldName,
                        const std::string& newName );

    /// Rename a payload item - This does not start a transaction
    void __renamePayload( const std::string& oldName,
                          const std::string& newName );

    /// Add new payload fields with the given names and storage types,
    /// setting their values for all existing IOVS to the given values.
    void extendPayloadSpecification( const IRecord& record );

    /// Add new payload fields with the given names and storage types,
    /// setting their values for all existing IOVS to the given values.
    /// - This does not start a transaction
    void __extendPayloadSpecification( const IRecord& record );

    /*
    /// Does this user tag exist in the object table?
    /// THIS DOES NOT START A TRANSACTION
    bool existsUserTagInObjectTable( UInt32 userTagId ) const;

    /// Does this user tag exist in the object2tag table?
    /// THIS DOES NOT START A TRANSACTION
    bool existsTagInObject2TagTable( UInt32 userTagId ) const;
    */

    /*
    /// Does this tag defined in this node have any relation (i.e. does
    /// it reference a parent tag or is it referenced by any children)?
    /// Returns false if this tag does not exist in this node.
    /// OVERLOADED RelationalHvsNode method.
    bool isTagUsed( UInt32 tagId ) const;
    */

    /// Does this user tag exist in the object table?
    /// THIS DOES NOT START A TRANSACTION
    static bool 
    existsUserTagInObjectTable( const RelationalQueryMgr& queryMgr,
                                UInt32 userTagId,
                                const std::string& objectTableName );

    /// Does this user tag exist in the object2tag table?
    /// THIS DOES NOT START A TRANSACTION
    static bool 
    existsTagInObject2TagTable( const RelationalQueryMgr& queryMgr,
                                UInt32 tagId,
                                const std::string& object2TagTableName );

  public:

    // ----- CHANNEL MANAGEMENT -----

    /// Return the list of existing channels (ordered by ascending channelId).
    const std::vector<ChannelId> listChannels() const;

    /// Return the map of id->name for all existing channels.
    const std::map<ChannelId,std::string> listChannelsWithNames() const;

    /// Create a new channel with the given id, name and description.
    /// Throw ChannelExists if the id/name is already used by another channel.
    /// Throw InvalidChannelName if the channel name is invalid: all valid
    /// channel names must be between 1 and 255 characters long; they must
    /// start with a letter and must contain only letters, numbers or the '_'
    /// character (these constraints may be relaxed in a future COOL release).
    /// Throw an Exception if the description is longer than 255 characters.
    void createChannel( const ChannelId& channelId,
                        const std::string& channelName,
                        const std::string& description = "" );

    /// Drop (delete) an existing channel, given its id.
    /// Return true if the channel is dropped as expected. Return false
    /// (without throwing any exception) if the channel does not exist any
    /// more on exit from this method, but it did not exist to start with.
    /// Throw an Exception if the channel cannot be dropped.
    /// NB: in the COOL_2_2_0 API, a channel can only be dropped if it exists
    /// in the channel table but contains no IOVs (an exception is thrown if
    /// it does contain IOVs); in later COOL releases, the semantics of this
    /// method may change to mean 'drop the channel and any IOVs it contains'.
    bool dropChannel( const ChannelId& channelId );

    /// Set the name of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw ChannelExists if the name is already used by another channel.
    void setChannelName( const ChannelId& channelId,
                         const std::string& channelName );

    /// Return the name of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    const std::string channelName( const ChannelId& channelId ) const;

    /// Return the id of a channel, given its name.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw ChannelNotFound if no channel exists with this name.
    ChannelId channelId( const std::string& channelName ) const;

    /// Does a channel with this id exist?
    bool existsChannel( const ChannelId& channelId ) const;

    /// Does a channel with this name exist?
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    bool existsChannel( const std::string& channelName ) const;

    /// Set the description of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw an Exception if the description is longer than 255 characters.
    void setChannelDescription( const ChannelId& channelId,
                                const std::string& description );

    /// Return the description of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    const std::string channelDescription( const ChannelId& channelId ) const;

  public:

    /// Return the IOV table name for this folder.
    const std::string& objectTableName() const;

    /// Return the payload table name for this folder.
    const std::string& payloadTableName() const;

    /// Return the tag table name for this folder.
    const std::string& tagTableName() const;

    /// Return the IOV2tag table name for this folder.
    const std::string& object2TagTableName() const;

    /// Return the channel table name for this folder.
    const std::string& channelTableName() const;

    /// Return wether bulk storage is active
    bool useBuffer() const
    {
      return m_useBuffer;
    };

  public:

    /// Get the extended payload specification from the given folder row
    static const RecordSpecification
    payloadSpecification( const coral::AttributeList& folderRow );

    /// Get the versioning mode from the given folder row
    static FolderVersioning::Mode
    versioningMode( const coral::AttributeList& folderRow );

    /// Get the payload mode from the given folder row
    static PayloadMode::Mode
    payloadMode( const coral::AttributeList& folderRow );

    /// Get the payload table name from the given folder row
    static const std::string payloadTableName
    ( const coral::AttributeList& folderRow );

    /// Get the object table name from the given folder row
    static const std::string objectTableName
    ( const coral::AttributeList& folderRow );

    /// Get the tag table name from the given folder row
    static const std::string tagTableName
    ( const coral::AttributeList& folderRow );

    /// Get the object2tag table name from the given folder row
    static const std::string object2TagTableName
    ( const coral::AttributeList& folderRow );

    /// Get the channel table name from the given folder row
    static const std::string channelTableName
    ( const coral::AttributeList& folderRow );

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize( const coral::AttributeList& row );

    /// Standard constructor is private
    RelationalFolder();

    /// Copy constructor is private
    RelationalFolder( const RelationalFolder& rhs );

    /// Assignment operator is private
    RelationalFolder& operator=( const RelationalFolder& rhs );

  private:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Folder attribute specification for the RelationalFolder class
    static const RecordSpecification& folderAttributesSpecification
    ( FolderVersioning::Mode versioningMode );

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Folder specification
    FolderSpecification m_folderSpec;

    /// External reference constraint map
    //std::map< std::string, IFolder::ExternalReference > m_externalReferences;

    /// Use a storage buffer?
    bool m_useBuffer;

    /// Object storage buffer
    std::vector<RelationalObjectPtr> m_objectBuffer;

    /// Insert into the user tag branch only?
    bool m_userTagOnly;

    /// IOV table name
    std::string m_objectTableName;

    /// payload table name
    std::string m_payloadTableName;

    /// Tag table name
    std::string m_tagTableName;

    /// IOV2tag table name
    std::string m_object2TagTableName;

    /// Channel table name
    std::string m_channelTableName;

    /// Public attributes of the folder
    Record m_publicFolderAttributes;

    /// Prefetch all objects into vectors?
    bool m_prefetchAll;

  };

}

#endif
