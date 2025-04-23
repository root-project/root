// $Id: TransRelFolder.h,v 1.3 2012-07-08 20:02:34 avalassi Exp $
#ifndef RELATIONALCOOL_TRANSRELFOLDER_H
#define RELATIONALCOOL_TRANSRELFOLDER_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/IHvsNode.h"
/*
#include <memory>
#include <vector>
#include "CoralBase/MessageStream.h"
#include "CoolKernel/ChannelSelection.h"
#include "CoolKernel/FolderSpecification.h"
#include "CoolKernel/IFolder.h"
#include "CoolKernel/InternalErrorException.h"
#include "CoolKernel/RecordSpecification.h"
*/
// Local include files
#include "RelationalException.h"
#include "RelationalFolder.h"
#include "RelationalTransaction.h"
#include "TransRelHvsNode.h"

namespace cool {

  /** @class TransRelFolder TransRelFolder.h
   *
   *  Transaction aware wrapper around RelationalFolder 
   *
   *  Automatically starts a transaction and commits it afterwards.
   *  The only exception is the bulk insertion: here a transaction
   *  is started when the first object is inserted, and the
   *  transaction is committed when flushStorageBuffer() is called.
   *  This behaviour is different to previous COOL versions.
   *
   *  @author Martin Wache
   *  @date   2010-10-21
   */

  class TransRelFolder : virtual public IFolder
                       , virtual public TransRelHvsNode
  {

  public:
    /// Return the folder specification.
    virtual const IFolderSpecification& folderSpecification() const
    {
      return m_folder->folderSpecification();
    };

    /// Return the payload specification of the folder.
    virtual const IRecordSpecification& payloadSpecification() const
    {
      return m_folder->payloadSpecification();
    };

    /// Return the folder versioning mode.
    virtual FolderVersioning::Mode versioningMode() const
    {
      return m_folder->versioningMode();
    };

    /// Return the 'attributes' of the folder
    /// (implementation-specific properties not exposed in the API).
    virtual const IRecord& folderAttributes() const
    {
      return m_folder->folderAttributes();
    };

    /// POSTPONED:
    /// 1. RAL does not support creating FKs on existing tables (ALTER table)
    /// 2. Need to encode the extRef as string and store it in folder table
    /*
    /// Declare that some payload columns reference external payload.
    /// The referencedEntity URL contains the path to the external container.
    /// Only FKs to relational tables within the same database are supported
    /// so far with the syntax "local://schema=USERNAME;table=TABLENAME".
    /// Non-relational implementation of CoolKernel should throw an exception.
    /// Returns true in case of success, false in case of any error.
    virtual bool declareExternalReference
    ( const std::string& name,
    const std::vector< std::string >& attributes,
    const std::string& referencedEntity,
    const std::vector< std::string >& referencedAttributes ) = 0;

    /// List the external reference constraints.
    virtual const std::vector< std::string > externalReferences() const = 0;

    /// Retrieve the properties of an external reference constraint
    virtual const IFolder::ExternalReference&
    externalReference( const std::string& name ) const = 0;
    */

    /// Activate/deactivate a storage buffer for bulk insertion of objects.
    /// Flush the buffer when deactivating an active one and commits
    /// active automatic transactions.
    virtual void setupStorageBuffer( bool useBuffer = true );

    /// Flush the storage buffer (execute the bulk insertion of objects).
    /// Rethrows exceptions encountered during bulk storage. See storeObject
    /// for possible errors.
    /// Commits active automatic transactions.
    virtual void flushStorageBuffer();

    /// Store an object in a given channel with the given IOV and data payload.
    /// If the buffer is used, only register the object for later storage.
    /// The following exceptions are thrown:
    /// - ValidityIntervalBackwards if the given since is larger than until
    /// - ValidityKeyOutOfBoundaries if since or until is out of bounds
    /// Throw an Exception in SingleVersion mode if the given IOV overlaps
    /// with an existing IOV (unless this existing IOV is open-ended).
    /// If a user tag is specified, insert the IOV into both the user tag HEAD
    /// and the global HEAD (but if userTagOnly is true, insert the IOV
    /// into the user tag HEAD only and not into the gloabl HEAD).
    virtual void storeObject( const ValidityKey& since,
                              const ValidityKey& until,
                              const IRecord& payload,
                              const ChannelId& channelId,
                              const std::string& userTagName = "",
                              bool userTagOnly = false );

#ifdef COOL290VP
    virtual void storeObject( const ValidityKey& since,
                              const ValidityKey& until,
                              const std::vector<IRecordPtr>& payload,
                              const ChannelId& channelId,
                              const std::string& userTagName = "",
                              bool userTagOnly = false );
#endif

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    virtual void storeObject( const ValidityKey& since,
                              const ValidityKey& until,
                              const coral::AttributeList& payload,
                              const ChannelId& channelId,
                              const std::string& userTagName = "",
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
    virtual IObjectPtr findObject( const ValidityKey& pointInTime,
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
    virtual IObjectIteratorPtr
    findObjects( const ValidityKey& pointInTime,
                 const ChannelSelection& channels,
                 const std::string& tagName = "" ) const;

    /// Browse the objects for a given channel selection within the given time
    /// range in the given tag ("" and "HEAD" both indicate the folder HEAD).
    /// Tag names, except for "HEAD", are case sensitive.
    /// The validity range is inclusive at both ends as well, i.e.
    /// since = t1, until = t2 will select an object with since = t2.
    /// The channel selection is specified through a ChannelSelection object.
    /// The iterator will return only ONE object at any given validity point
    /// in each channel, in the order specified in the ChannelSelection
    /// (default is by 'channel, since', alternatively by 'since, channel').
    /// For single version folders, the tag must be either "" or "HEAD",
    /// otherwise a TagNotFound exception is thrown.
    virtual IObjectIteratorPtr
    browseObjects( const ValidityKey& since,
                   const ValidityKey& until,
                   const ChannelSelection& channels,
                   const std::string& tagName = "",
                   const IRecordSelection* payloadQuery = 0 ) const;

    /// Count the number of objects that would be returned by the
    /// browseObjects method for the same selection parameters.
    /// Warning: calling browseObjects after calling this method
    /// may result in a different number of objects, since the two
    /// methods are executed in two separate database transactions.
    /// You may also use the browseObjects method directly after calling
    /// setPrefetchAll(false), and use the size() method of the iterator to
    /// retrieve the number of objects that this iterator will return: on
    /// transaction-safe backends, this should return the same number as the
    /// two database queries are executed in the same read only transaction.
    virtual unsigned int
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
    virtual void setPrefetchAll( bool prefetchAll )
    {
      return m_folder->setPrefetchAll( prefetchAll );
    };

    /// Rename a payload item.
    virtual void renamePayload( const std::string& oldName,
                                const std::string& newName );

    /// Add new payload fields with the given names and storage types,
    /// setting their values for all existing IOVS to the given values.
    virtual void extendPayloadSpecification( const IRecord& record );

    // ----- TAG MANAGEMENT -----

    /// Associates the objects that are the current HEAD
    /// with the given tag name and tag description.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws a ReservedHeadTag exception if tagging using "" or "HEAD".
    /// Else throws a FolderIsSingleVersion exception for SV folders.
    /// Throws a TagExists exception if a tag with the given name already
    /// exists (whether in the same or another folder: tag names are global,
    /// they cannot be used in more than one folder at the same time).
    /// Within a folder, changing the objects associated to an existing tag
    /// ("retagging") is achieved by calling deleteTag() and tagging again.
    virtual void
    tagCurrentHead( const std::string& tagName,
                    const std::string& description = "" ) const;

    /// Clones the tag "tagName" as user tag "tagClone" by reinserting
    /// the IOVs with the user tag. Does not modify the current head.
    virtual void
    cloneTagAsUserTag( const std::string& tagName,
                       const std::string& tagClone,
                       const std::string& description = "",
                       bool forceOverwriteTag = false );

    /// Associates the objects that were the HEAD at 'asOfDate'
    /// with the given tag name and tag description.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws a ReservedHeadTag exception if tagging using "" or "HEAD".
    /// Else throws a FolderIsSingleVersion exception for SV folders.
    /// Throws a TagExists exception if a tag with the given name already
    /// exists (whether in the same or another folder: tag names are global,
    /// they cannot be used in more than one folder at the same time).
    /// Within a folder, changing the objects associated to an existing tag
    /// ("retagging") is achieved by calling deleteTag() and tagging again.
    virtual void
    tagHeadAsOfDate( const ITime& asOfDate,
                     const std::string& tagName,
                     const std::string& description = "" ) const;

    /// Insertion time of the last inserted IOV in a tag defined for this node.
    /// This represents the first possible tagging date for the tag
    /// (for tagging via tagCurrentHead or tagHeadAsOfDate), i.e.
    /// the first date at which the IOVs in this tag were the HEAD.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Insertion time of last inserted IOV is returned for "" and "HEAD".
    /// For all other tag names, throws TagNotFound if tag does not exist
    /// (or node is a folder set or a single version folder).
    virtual const Time
    insertionTimeOfLastObjectInTag( const std::string& tagName ) const;

    /// Deletes a tag from this folder and from the global "tag namespace":
    /// the tag name is available for tagging again afterwards.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws a ReservedHeadTag exception if tagging using "" or "HEAD".
    /// For all other tag names, throws a TagNotFound exception if the tag
    /// does not exist or the folder is a single version folder.
    virtual void deleteTag( const std::string& tagName );

    /// Does this user tag exist?
    /// Tag names are case sensitive.
    virtual bool existsUserTag( const std::string& userTagName ) const;

    // ----- CHANNEL MANAGEMENT -----

    /// Return the list of existing channels (ordered by ascending channelId).
    virtual const std::vector<ChannelId> listChannels() const;

    /// Return the map of id->name for all existing channels.
    virtual const std::map<ChannelId,std::string> listChannelsWithNames() const;

    /// Create a new channel with the given id, name and description.
    /// Throw ChannelExists if the id/name is already used by another channel.
    /// Throw InvalidChannelName if the channel name is invalid: all valid
    /// channel names must be between 1 and 255 characters long; they must
    /// start with a letter and must contain only letters, numbers or the '_'
    /// character (these constraints may be relaxed in a future COOL release).
    /// Throw an Exception if the description is longer than 255 characters.
    virtual void createChannel( const ChannelId& channelId,
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
    virtual bool dropChannel( const ChannelId& channelId );

    /// Set the name of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw ChannelExists if the name is already used by another channel.
    virtual void setChannelName( const ChannelId& channelId,
                                 const std::string& channelName );

    /// Return the name of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    virtual const std::string channelName( const ChannelId& channelId ) const;

    /// Return the id of a channel, given its name.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw ChannelNotFound if no channel exists with this name.
    virtual ChannelId channelId( const std::string& channelName ) const;

    /// Does a channel with this id exist?
    virtual bool existsChannel( const ChannelId& channelId ) const;

    /// Does a channel with this name exist?
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    virtual bool existsChannel( const std::string& channelName ) const;

    /// Set the description of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    /// Throw InvalidChannelName if the channel name is invalid (see above).
    /// Throw an Exception if the description is longer than 255 characters.
    virtual void setChannelDescription( const ChannelId& channelId,
                                        const std::string& description );

    /// Return the description of a channel, given its id.
    /// Throw ChannelNotFound if no channel exists with this id.
    virtual const std::string
    channelDescription( const ChannelId& channelId ) const;

  protected:

    const RelationalDatabase& db() const { return m_folder->db(); }

  public:

    /// some tests need access to the RelationalFolder
    /// do *not* use this outside of test classes!
    RelationalFolder *getRelFolder()
    {
      return m_folder;
    };


    TransRelFolder( const IFolderPtr& folderPtr )
      : TransRelHvsNode( folderPtr.get() )
      , m_folder( dynamic_cast<RelationalFolder*>( folderPtr.get() ) )
      , m_folderPtr( folderPtr )
      , m_trans( 0 )
    {
      if ( !m_folder )
        throw PanicException("TransRelFolder constructor called without a"
                             " pointer to a RelationalFolder",
                             "TransRelFolder::TransRelFolder");

    };

    /// Destructor.
    virtual ~TransRelFolder()
    {
    };

  private:

    /// Standard constructor is private
    TransRelFolder();

    /// Copy constructor is private
    TransRelFolder( const TransRelFolder& rhs );

    /// Assignment operator is private
    TransRelFolder& operator=( const TransRelFolder& rhs );

  private:

    /// the wrapped RelationalFolder
    RelationalFolder *m_folder;

    /// shared pointer to the wrapped folder
    IFolderPtr m_folderPtr;

    /// the active transaction, if any
    std::auto_ptr<RelationalTransaction> m_trans;

    Record m_nodeAttributes;

  };

}

#endif
