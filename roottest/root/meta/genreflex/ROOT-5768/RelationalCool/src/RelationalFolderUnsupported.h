// $Id: RelationalFolderUnsupported.h,v 1.28 2010-08-05 11:53:25 mwache Exp $
#ifndef RELATIONALCOOL_RELATIONALFOLDERUNSUPPORTED_H
#define RELATIONALCOOL_RELATIONALFOLDERUNSUPPORTED_H

// Disable warning C4250 on Windows (inheritance via dominance)
// Copied from SEAL (Dictionary/Reflection/src/Tools.h)
#ifdef WIN32
#pragma warning ( disable : 4250 )
#endif

// Include files
#include <memory>
#include "CoralBase/MessageStream.h"
#include "CoolKernel/IFolder.h"
#include "CoolKernel/Time.h"

// Local include files
#include "RelationalDatabase.h"
#include "RelationalException.h"
#include "RelationalHvsNode.h"

namespace cool {

  //--------------------------------------------------------------------------

  /** @class RelationalFolderUnsupported RelationalFolderUnsupported.h
   *
   *  UNSUPPORTED relational implementation of a COOL condition db "folder".
   *
   *  Also represents implementation within COOL of an HVS leaf node.
   *  Multiple virtual inheritance from IFolder and RelationalHvsNode
   *  (diamond virtual inheritance of IHvsNodeRecord abstract interface).
   *
   *  Within the COOL 2.0 software, this represents a handle to a folder
   *  created using the COOL 2.1 software and implementing the new 2.1 schema.
   *  Such a folder cannot be opened for reading or writing using the 2.0
   *  software (its contents cannot be read or modified): only its generic
   *  properties (those retrieved from the node table) can be queried.
   *
   *  Within the COOL 2.2.2 software, this may also represent a handle
   *  to a folder with schema version 2.0.0, that is no longer supported.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-09
   */

  class RelationalFolderUnsupported : virtual public IFolder,
                                      virtual public RelationalHvsNode {

  public:

    /// Ctor to create a RelationalFolderUnsupported from a node table row
    /// Inlined with the base class non-standard constructors as otherwise
    /// the compiler attempts to use the base class standard constructors
    /// (Windows compilation error C2248: standard constructors are private)
    RelationalFolderUnsupported( const RelationalDatabasePtr& db,
                                 const coral::AttributeList& row )
      : RelationalHvsNodeRecord( row )
      , RelationalHvsNode( db, row )
      , m_log
      ( new coral::MessageStream( "RelationalFolderUnsupported" ) )
      , m_publicFolderAttributes() // fill it in initialize
    {
      initialize( row );
    }

    /// Destructor.
    virtual ~RelationalFolderUnsupported();

    /// Return the 'attributes' of the folder
    /// (implementation-specific properties not exposed in the API).
    const IRecord& folderAttributes() const;

  public:

    // -- THE FOLLOWING METHODS ALL THROW --

    /// Return the folder specification.
    const IFolderSpecification& folderSpecification() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the payload specification of the folder.
    const IRecordSpecification& payloadSpecification() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the folder versioning mode.
    FolderVersioning::Mode versioningMode() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return true if the folder uses a separate payload table.
    bool hasPayloadTable() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Activate/deactivate a storage buffer for bulk insertion of objects.
    /// If the buffer was used and is deactivated, flush the buffer.
    void setupStorageBuffer( bool /*useBuffer*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Flush the storage buffer (execute the bulk insertion of objects).
    /// If the buffer is not used, ignore this command.
    void flushStorageBuffer()
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Store an object in a given channel with the given IOV and data payload.
    void storeObject( const ValidityKey& /*since*/,
                      const ValidityKey& /*until*/,
                      const IRecord& /*payload*/,
                      const ChannelId& /*channelId*/,
                      const std::string& /*userTagName*/,
                      bool /*userTagOnly*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Store an object in a given channel with the given IOV and data payload.
    void storeObject( const ValidityKey& /*since*/,
                      const ValidityKey& /*until*/,
                      const std::vector<IRecordPtr>& /*payload*/,
                      const ChannelId& /*channelId*/,
                      const std::string& /*userTagName*/,
                      bool /*userTagOnly*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// DEPRECATED - added for easier compatibility with COOL 1.3
    /// (this is likely to be removed in a future COOL release).
    void storeObject( const ValidityKey& /*since*/,
                      const ValidityKey& /*until*/,
                      const coral::AttributeList& /*payload*/,
                      const ChannelId& /*channelId*/,
                      const std::string& /*userTagName = ""*/,
                      bool /*userTagOnly*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Set a new finite end-of-validity value for all SV objects in a given
    /// channel selection whose end-of-validity is currently infinite.
    int truncateObjectValidity( const ValidityKey& /*until*/,
                                const ChannelSelection& /*channels*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Find the ONE object in a given channel valid at the given point in time
    /// in the given tag ("" and "HEAD" both indicate the folder HEAD).
    IObjectPtr findObject( const ValidityKey& /*pointInTime*/,
                           const ChannelId& /*channelId*/,
                           const std::string& /*tagName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Find the objects for a given channel selection at the given point
    /// in time in the given tag ("" and "HEAD" both indicate the folder HEAD).
    IObjectIteratorPtr findObjects( const ValidityKey& /*pointInTime*/,
                                    const ChannelSelection& /*channels*/,
                                    const std::string& /*tagName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Browse the objects for a given channel selection within the given time
    /// range in the given tag ("" and "HEAD" both indicate the folder HEAD).
    IObjectIteratorPtr browseObjects( const ValidityKey& /*since*/,
                                      const ValidityKey& /*until*/,
                                      const ChannelSelection& /*channels*/,
                                      const std::string& /*tagName*/,
                                      const IRecordSelection* /*payloadQuery*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Rename a payload item
    void renamePayload( const std::string& /*oldName*/,
                        const std::string& /*newName*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Add new payload fields with the given names and storage types,
    /// setting their values for all existing IOVS to the given values.
    void extendPayloadSpecification( const IRecord& /*record*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Count the number of objects that would be returned by the
    /// browseObjects method for the same selection parameters.
    unsigned int countObjects( const ValidityKey& /*since*/,
                               const ValidityKey& /*until*/,
                               const ChannelSelection& /*channels*/,
                               const std::string& /*tagName = ""*/,
                               const IRecordSelection* /*payloadQuery = 0*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Change the object prefetch policy (default is prefetchAll=true).
    void setPrefetchAll( bool /*prefetchAll*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Associates the objects that are the current HEAD
    /// with the given tag name and tag description.
    void tagCurrentHead( const std::string& /*tagName*/,
                         const std::string& /*description = ""*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Clones the tag "tagName" as user tag "tagClone" by reinserting
    /// the IOVs with the user tag. Does not modify the current head.
    void cloneTagAsUserTag( const std::string& /*tagName*/,
                            const std::string& /*tagClone*/,
                            const std::string& /*description = ""*/,
                            bool /*forceOverwrite = false*/)
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Associates the objects that were the HEAD at 'asOfDate'
    /// with the given tag name and tag description.
    void tagHeadAsOfDate( const ITime& /*asOfDate*/,
                          const std::string& /*tagName*/,
                          const std::string& /*description = ""*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Insertion time of the last inserted IOV in a tag defined for this node.
    const Time
    insertionTimeOfLastObjectInTag( const std::string& /*tagName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Deletes a tag from this folder and from the global "tag namespace".
    void deleteTag( const std::string& /*tagName*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Does this user tag exist?
    bool existsUserTag( const std::string& /*userTagName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the list of existing channels (ordered by ascending channelId).
    const std::vector<ChannelId> listChannels() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the map of id->name for all existing channels.
    const std::map<ChannelId,std::string> listChannelsWithNames() const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Create a new channel with the given id, name and description.
    void createChannel( const ChannelId& /*channelId*/,
                        const std::string& /*channelName*/,
                        const std::string& /*description = ""*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };


    /// Drop a channel with the given id
    bool dropChannel( const ChannelId& /*channelId*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };


    /// Set the name of a channel, given its id.
    void setChannelName( const ChannelId& /*channelId*/,
                         const std::string& /*channelName*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the name of a channel, given its id.
    const std::string channelName( const ChannelId& /*channelId*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the id of a channel, given its name.
    ChannelId channelId( const std::string& /*channelName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Does a channel with this id exist?
    bool existsChannel( const ChannelId& /*channelId*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Does a channel with this name exist?
    bool existsChannel( const std::string& /*channelName*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Set the description of a channel, given its id.
    void setChannelDescription( const ChannelId& /*channelId*/,
                                const std::string& /*description*/ )
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

    /// Return the description of a channel, given its id.
    const std::string
    channelDescription( const ChannelId& /*channelId*/ ) const
    {
      throw UnsupportedFolderSchema
        ( fullPath(), schemaVersion(), "RelationalFolderUnsupported" );
    };

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize( const coral::AttributeList& row );

    /// Standard constructor is private
    RelationalFolderUnsupported();

    /// Copy constructor is private
    RelationalFolderUnsupported( const RelationalFolderUnsupported& rhs );

    /// Assignment operator is private
    RelationalFolderUnsupported&
    operator=( const RelationalFolderUnsupported& rhs );

  private:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Folder attribute spec for the RelationalFolderUnsupported class
    static const RecordSpecification& folderAttributesSpecification();

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Public attributes of the folder
    Record m_publicFolderAttributes;

  };

}

#endif
