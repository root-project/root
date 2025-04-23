// $Id: RelationalDatabase.h,v 1.221 2012-01-30 17:42:51 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALDATABASE_H
#define RELATIONALCOOL_RELATIONALDATABASE_H 1

// First of all, set/unset COOL290, COOL300 and COOL_HAS_CPP11 macros
#include "CoolKernel/VersionInfo.h"

// Include files
#include <boost/shared_ptr.hpp>
#include <memory>
#include <vector>
#include "CoolKernel/ChannelSelection.h"
//#include "CoolKernel/PayloadMode.h"
#include "PayloadMode.h" // TEMPORARY
#include "CoolKernel/IDatabase.h"
#ifdef COOL300
#include "CoolKernel/ITransaction.h"
#endif
#include "CoolKernel/Record.h"
#include "CoolKernel/ValidityKey.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/MessageStream.h"

// Local include files
#include "IHvsTagMgr.h"
#include "RelationalDatabaseId.h"
#include "RelationalDatabasePtr.h"
#include "RelationalObjectPtr.h"

namespace cool
{

  // Forward declarations
  class ChannelSelection;
  class IRelationalTransactionMgr;
  class ISessionMgr;
  class RelationalFolder;
  class RelationalFolderSet;
  class RelationalNodeMgr;
  class RelationalObjectMgr;
  class RelationalObjectTable;
  class RelationalObjectTableRow;
  class RelationalQueryMgr;
  class RelationalSchemaMgr;
  class RelationalTableRow;
  class RelationalTagMgr;

  /**  
   * FIXME move somewhere else?
   * 
   * Enumeration for AccessMode
   *
   * To be used in parameter list, to avoid
   * boolean parameters.
   */

  enum AccessMode
  {
    ReadWrite,
    ReadOnly
  };


  /** @class RelationalDatabase RelationalDatabase.h
   *
   *  Generic relational implementation of one COOL "condition database"
   *  instance (deployed on a specific physical infrastructure).
   *
   *  Abstract base class for specific relational implementations
   *  sharing the same relational database schema (RAL, MySQL, ...).
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */


  class RelationalDatabase : public IDatabase
  {

    friend class RelationalDatabaseTest;
    friend class RalDatabaseTest;
    friend class RalDatabaseTest_extendedSpec;

  public:

    // --- Implementation of the IDatabase interface. ---

    /// Return the global identifier of the database
    /// [WARNING: any visible passwords are masked out].
    const DatabaseId& databaseId() const;

    /// Return the 'attributes' of the database
    /// (implementation-specific properties not exposed in the API).
    /// Throws DatabaseNotOpen if the database is not open.
    const IRecord& databaseAttributes() const;

    /// Helper methods, does the ususal checks if
    /// database is open, rw/ro, a transaction active, etc
    /// Throws exceptions if the
    void checkDbOpenTransaction( const std::string& domain,
                                 cool::AccessMode mode ) const;

    /*
    /// Does the database support this payload specification?
    bool isValidPayloadSpecification( const IRecordSpecification& spec );
    */

    /*
    /// Does the database support this channel specification?
    bool isValidChannelSpecification( const IRecordSpecification& spec );
    */

    /// Create a new folder set and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if an invalid versioning mode has been specified.
    /// Throws an Exception if the user does not have writer privileges.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IFolderSetPtr createFolderSet
    ( const std::string& fullPath,
      const std::string& description = "",
      bool createParents = false ) = 0;

    /// Does this folder set exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    bool existsFolderSet( const std::string& folderSetName );

    /// Retrieve an existing folderset and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderSetNotFound if the folderset does not exist.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IFolderSetPtr getFolderSet( const std::string& fullPath ) = 0;

    /// Create a new folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if an invalid versioning mode has been specified.
    /// Throws an Exception if the user does not have writer privileges.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IFolderSpecification& folderSpec,
      const std::string& description = "",
      bool createParents = false ) = 0;

    /// DEPRECATED: use IFolderSpecification instead of IRecordSpecification!
    /// This is similar to the COOL1.3.3 API (with IRecordSpecification
    /// instead of ExtendedAttributeListSpecification), for easier porting of
    /// user code, but it is likely to be removed in a future COOL release.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IRecordSpecification& payloadSpec,
      const std::string& description = "",
      FolderVersioning::Mode mode = FolderVersioning::SINGLE_VERSION,
      bool createParents = false ) = 0;

    /// Does this folder exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    bool existsFolder( const std::string& fullPath );

    /// Retrieve an existing folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderNotFound if the folder does not exist.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IFolderPtr getFolder( const std::string& fullPath ) = 0;

    /// Return the list of existing nodes
    /// (in ascending/descending alphabetical order).
    const std::vector<std::string> listAllNodes( bool ascending = true );

    /// Drop an existing node (folder or folder set).
    /// Also delete any tags associated to the node.
    /// Return true if the node and all its structures are dropped as expected.
    /// Return false (without throwing any exception) if the node and
    /// all its structures do not exist any more on exit from this method,
    /// but the node or some of its structures did not exist to start with.
    /// Throw an Exception if the node schema version is more recent than
    /// the schema version supported by the current COOL software release.
    /// Throw an Exception if the node or one of its structures cannot
    /// be dropped (i.e. continue to exist on exit from this method).
    /// Throw an Exception if the node is a non-empty folder set.
    /// Throw an Exception if any associated tags cannot be deleted.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool dropNode( const std::string& fullPath ) = 0;

    /// HVS: does this tag exist?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    bool existsTag( const std::string& tagName ) const
    {
      return hvsTagMgr().existsTag( tagName );
    }

    /// HVS: return the node type (inner/leaf) where this tag name can be used.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    IHvsNode::Type tagNameScope( const std::string& tagName ) const
    {
      return hvsTagMgr().tagNameScope( tagName );
    }

    /// HVS: return the names of the nodes where this tag is defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    const std::vector<std::string>
    taggedNodes( const std::string& tagName ) const
    {
      return hvsTagMgr().taggedNodes( tagName );
    }

    /// Is the database 'open'?
    /// NB Note the difference between 'open' and 'connected': the database
    /// is 'connected' if the connection to the database backend has been
    /// established; it is 'open' if the management table has been read.
    bool isOpen() const;

    /// (Re)opens the database.
    void openDatabase();

    /// Closes the database.
    void closeDatabase();

    /// Return the "COOL database name".
    const std::string& databaseName() const;

#ifdef COOL300
    /// Start a new transaction and enter manual transaction mode
    virtual ITransactionPtr startTransaction() = 0;
#endif

    // --- Other public methods. ---

    /// Return the list of folders inside the given folderset
    /// (in ascending/descending alphabetical order).
    const std::vector<std::string>
    listFolders( const RelationalFolderSet* folderset,
                 bool ascending = true ) const;

    /// Return the list of foldersets inside the given folderset
    /// (in ascending/descending alphabetical order).
    const std::vector<std::string>
    listFolderSets( const RelationalFolderSet* folderset,
                    bool ascending = true ) const;

    /// Get a constant reference to the HVS tag manager
    const IHvsTagMgr& hvsTagMgr() const;

    /// Return the RelationalDatabasePtr
    /// PURE VIRTUAL method implemented in subclasses.
    virtual RelationalDatabasePtr relationalDbPtr() = 0;

    /// Return the default table prefix (from the database attributes)
    const std::string defaultTablePrefix() const;

    /// Return the name of the main table (from the db name)
    const std::string mainTableName() const;

    // *** START *** 3.0.0 schema extensions (task #4307)
    /// Return the name of the iovTables table (from the db attributes)
    const std::string iovTablesTableName() const;

    /// Return the name of the channelTables table (from the db attributes)
    const std::string channelTablesTableName() const;
    // **** END **** 3.0.0 schema extensions (task #4307)

    /// Return the name of the folder table (from the db attributes)
    const std::string nodeTableName() const;

    /// Return the name of the global tag table (from the db attributes)
    const std::string globalTagTableName() const;

    // *** START *** 3.0.0 schema extensions (task #4396)
    /// Return the name of the global head tag table (from the db attributes)
    const std::string globalHeadTagTableName() const;

    /// Return the name of the global user tag table (from the db attributes)
    const std::string globalUserTagTableName() const;
    // **** END **** 3.0.0 schema extensions (task #4396)

    /// Return the name of the tag2tag table (from the db attributes)
    const std::string tag2TagTableName() const;

    /// Return the name of the tag shared sequence (from the db attributes)
    const std::string tagSharedSequenceName() const;

    /// Return the name of the IOV shared sequence (from the db attributes)
    const std::string iovSharedSequenceName() const;

    /// Get a RelationalObjectTable for the given folder.
    /// The concrete class can only be created by the concrete database.
    /// The RelationalFolder parameter is only used to obtain
    /// the associated table names and is *not* retained.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<RelationalObjectTable>
    relationalObjectTable( const RelationalFolder& folder ) const = 0;

    /// Update the description for the given node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void updateNodeTableDescription
    ( const std::string& fullPath,
      const std::string& description ) const = 0;

    /// Get the IRelationalTransactionMgr
    boost::shared_ptr<IRelationalTransactionMgr> transactionMgr() const;

    /// Return the StorageType singleton for the given type name.
    /// Throw a RelationalException if no storage type exists with that name.
    static const StorageType& storageType( const std::string& name );

    /// Get a string representation of a RecordSpecification
    static const std::string
    encodeRecordSpecification( const IRecordSpecification& recordSpec );

    /// Decode a RecordSpecification from its string representation
    static const RecordSpecification
    decodeRecordSpecification( const std::string& encodedSpec );

    /// Get the ISessionMgr
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<ISessionMgr> sessionMgr() const = 0;

    /// Get the RelationalQueryMgr
    RelationalQueryMgr& queryMgr() const;

    /// Get the RelationalSchemaMgr
    RelationalSchemaMgr& schemaMgr() const;

    /// Get the RelationalNodeMgr
    RelationalNodeMgr& nodeMgr() const;

    /// Get the RelationalTagMgr
    RelationalTagMgr& tagMgr() const;

    /// Get the RelationalObjectMgr
    const RelationalObjectMgr& objectMgr() const;

    /// Is this a valid name for a payload field of a folder?
    /// Payload field names must have between 1 and 30 characters (including
    /// only letters, digits or '_'), must start with a letter and cannot start
    /// with the "COOL_" prefix (in any lowercase/uppercase combination).
    static bool isValidPayloadFieldName( const std::string& name );

    /// Return the list of all existing tables (within a transaction)
    const std::vector<std::string> listAllTables() const;

    /// Return the list of all existing tables (no transaction)
    const std::vector<std::string> __listAllTables() const;

    /// Required access mode to the database.
    /// Delegated to RalSessionMgr.
    virtual bool isReadOnly() const = 0;

  protected:

    /// The following methods are all protected: only subclasses can
    /// instantiate or delete this class and create, drop or open a database

    /// Destructor
    virtual ~RelationalDatabase();

    /// Constructor
    RelationalDatabase( const DatabaseId& dbId );

    /// Create a new database with default attributes.
    /// Default attributes are those specific for a RelationalDatabase.
    void createDatabase();

    /// Create a new database with non-default attributes.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createDatabase( const IRecord& dbAttr ) = 0;

    /// Drop the database.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool dropDatabase() = 0;

    /// Fetch the database attributes (fetch all rows from the main table).
    const Record fetchDatabaseAttributes() const;

    /// AV - TO BE REMOVED
    /// Fetch one node row (lookup by 1 node fullPath)
    const RelationalTableRow
    fetchNodeTableRow( const std::string& fullPath ) const;

    /// AV - TO BE REMOVED
    /// Fetch one node row (lookup by 1 nodeId)
    const RelationalTableRow
    fetchNodeTableRow( unsigned int nodeId ) const;

    /// AV - TO BE REMOVED
    /// Fetch one node row (lookup with given WHERE clause and bind variables)
    const RelationalTableRow
    fetchNodeTableRow( const std::string& whereClause,
                       const Record& whereData ) const;

    /// WARNING: UNUSED! (AV)
    /// Fetch the tag table row for the given tagname in the given tag table
    RelationalTableRow
    fetchTagTableRow( const std::string& tagTableName,
                      const std::string& tagName );

    /// WARNING: USED ONLY IN TESTS! (AV)
    /// Fetch the object2Tag table row for the given tagId, objectId
    RelationalTableRow
    fetchObject2TagTableRow( const std::string& tagTableName,
                             unsigned int tagId,
                             unsigned int objectId,
                             PayloadMode::Mode pMode );

    /// Does this node exist?
    bool existsNode( const std::string& fullPath );

    /// Return the list of nodes inside the given nodeId with the attribute
    /// isLeaf as specified (ordered by name asc/desc)
    const std::vector<std::string>
    listNodes( unsigned int nodeId,
               bool isLeaf,
               bool ascending = true ) const;

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Set the RelationalQueryMgr (transfer ownership)
    void setQueryMgr( std::auto_ptr<RelationalQueryMgr> queryMgr );

    /// Set the RelationalSchemaMgr (transfer ownership)
    void setSchemaMgr( std::auto_ptr<RelationalSchemaMgr> schemaMgr );

    /// Set the RelationalNodeMgr (transfer ownership)
    void setNodeMgr( std::auto_ptr<RelationalNodeMgr> nodeMgr );

    /// Set the RelationalTagMgr (transfer ownership)
    void setTagMgr( std::auto_ptr<RelationalTagMgr> tagMgr );

    /// Set the RelationalObjectMgr (transfer ownership)
    void setObjectMgr( std::auto_ptr<RelationalObjectMgr> objectMgr );

    /// Set the IRelationalTransactionMgr (shared ownership)
    void setTransactionMgr
    ( boost::shared_ptr<IRelationalTransactionMgr> mgr );

    /// Database attribute specification for the RelationalDatabase class
    static
    const IRecordSpecification& databaseAttributesSpecification();

    /// Check whether this software library can read the given schema.
    /// Returns true if the schema can be read without schema evolution.
    /// Returns false if the schema requires schema evolution.
    /// Throws an IncompatibleReleaseNumber if the schema is newer than
    /// this software library or no schema evolution is possible.
    bool areReleaseAndSchemaCompatible
    ( const std::string releaseNumber,
      const std::string schemaVersion ) const;

    /// Validate the payload specification.
    /// Throws InvalidPayloadSpecification if the payload specification is
    /// invalid: there can be at most 900 fields, including up to 10 BLOB
    /// fields and up to 200 String255 fields; field names must be between
    /// 1 and 30 characters (including only letters, digits or '_'), must
    /// start with a letter and cannot start with the "COOL_" prefix (in any
    /// combination of lowercase and uppercase letters).
    void validatePayloadSpecification( const IRecordSpecification& spec );

  private:

    /// Is the database 'connected'?
    /// Delegated to RalSessionMgr.
    /// [NB Note the difference between 'open' and 'connected': the database
    /// is 'connected' if the connection to the database backend has been
    /// established; it is 'open' if the management table has been read].
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool isConnected() const = 0;

    /// (Re)connect to the database.
    /// Delegated to RalSessionMgr.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void connect() = 0;

    /// Disconnect from the database.
    /// Delegated to RalSessionMgr.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void disconnect() = 0;

  private:

    /// Standard constructor is private
    RelationalDatabase();

    /// Copy constructor is private
    RelationalDatabase( const RelationalDatabase& rhs );

    /// Assignment operator is private
    RelationalDatabase& operator=( const RelationalDatabase& rhs );

  protected:

    /// Attributes of the database
    Record m_dbAttr;

    /// Is the database open?
    bool m_isOpen;

    /// Global identifier of the database
    RelationalDatabaseId m_relationalDbId;

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// RelationalQueryMgr (owned by this instance)
    std::auto_ptr<RelationalQueryMgr> m_queryMgr;

    /// RelationalSchemaMgr (owned by this instance)
    std::auto_ptr<RelationalSchemaMgr> m_schemaMgr;

    /// RelationalNodeMgr (owned by this instance)
    std::auto_ptr<RelationalNodeMgr> m_nodeMgr;

    /// RelationalTagMgr (owned by this instance)
    std::auto_ptr<RelationalTagMgr> m_tagMgr;

    /// RelationalObjectMgr (owned by this instance)
    std::auto_ptr<RelationalObjectMgr> m_objectMgr;

    /// IRelationalTransactionMgr (shared ownership)
    boost::shared_ptr<IRelationalTransactionMgr> m_transactionMgr;

  };

}

#endif // RELATIONALCOOL_RELATIONALDATABASE_H
