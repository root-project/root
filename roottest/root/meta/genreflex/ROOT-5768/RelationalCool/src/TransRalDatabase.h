// $Id: TransRalDatabase.h,v 1.2 2012-01-30 17:11:50 avalassi Exp $
#ifndef RELATIONALCOOL_TRANSRALDATABASE_H 
#define RELATIONALCOOL_TRANSRALDATABASE_H 1 

// First of all, set/unset COOL290, COOL300 and COOL_HAS_CPP11 macros
#include "CoolKernel/VersionInfo.h"

// Include files
#include "RalDatabase.h"
#include "RelationalException.h"

namespace cool
{
  // Forward declarations
  class TransRalDatabase;

  // Type definitions
  typedef boost::shared_ptr<TransRalDatabase> TransRalDatabasePtr;

 /** @class TransRalDatabase TransRalDatabase.h 
   *
   *  Transaction aware wrapper around a RalDatabase object
   *
   *  @author Martin Wache 
   *  @date   2010-10-29
   */

  class TransRalDatabase : public IDatabase
  {
  public:
    /// Constructor.
    TransRalDatabase( const IDatabasePtr& db )
      : m_db( dynamic_cast<RalDatabase*>( db.get() ) )
      , m_dbPtr( db )
    {
      if ( ! m_db )
        throw PanicException("TransRalDatabase constructor called without"
                             " a pointer to a RalDatabase",
                             "TransRalDatabase::TransRalDatabase");
    }

    /// Destructor.
    virtual ~TransRalDatabase() {}

    /// Return the global identifier of the database
    /// [WARNING: any visible passwords are masked out].
    virtual const DatabaseId& databaseId() const
    {
      return m_db->databaseId();
    }

    /// Return the 'attributes' of the database
    /// (implementation-specific properties not exposed in the API).
    /// Throws DatabaseNotOpen if the database is not open.
    virtual const IRecord& databaseAttributes() const
    {
      return m_db->databaseAttributes();
    }

    /// Create a new folder set and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if an invalid versioning mode has been specified.
    /// Throws an Exception if the user does not have writer privileges.
    virtual IFolderSetPtr createFolderSet
    ( const std::string& fullPath,
      const std::string& description = "",
      bool createParents = false );

    /// Does this folder set exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    virtual bool existsFolderSet( const std::string& folderSetName );

    /// Retrieve an existing folderset and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderSetNotFound if the folderset does not exist.
    virtual IFolderSetPtr getFolderSet( const std::string& fullPath );

    /// Create a new folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws InvalidPayloadSpecification if the payload specification is
    /// invalid: there can be at most 900 fields, including up to 10 BLOB
    /// fields and up to 200 String255 fields; field names must be between
    /// 1 and 30 characters (including only letters, digits or '_'), must
    /// start with a letter and cannot start with the "COOL_" prefix (in any
    /// combination of lowercase and uppercase letters).
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if the user does not have writer privileges.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IFolderSpecification& folderSpec,
      const std::string& description = "",
      bool createParents = false );

    /// DEPRECATED: use IFolderSpecification instead of IRecordSpecification!
    /// This is similar to the COOL1.3.3 API (with IRecordSpecification
    /// instead of ExtendedAttributeListSpecification), for easier porting of
    /// user code, but it is likely to be removed in a future COOL release.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IRecordSpecification& payloadSpec,
      const std::string& description = "",
      FolderVersioning::Mode mode = FolderVersioning::SINGLE_VERSION,
      bool createParents = false );

    /// Does this folder exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    virtual bool existsFolder( const std::string& fullPath );

    /// Retrieve an existing folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderNotFound if the folder does not exist.
    virtual IFolderPtr getFolder( const std::string& fullPath );

    /// Return the list of existing nodes
    /// (in ascending/descending alphabetical order).
    virtual const std::vector<std::string> listAllNodes( bool ascending = true );

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
    virtual bool dropNode( const std::string& fullPath );

    /// HVS: does this tag exist?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTag( const std::string& tagName ) const;

    /// HVS: return the node type (inner/leaf) where this tag name can be used.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    virtual IHvsNode::Type tagNameScope( const std::string& tagName ) const;

    /// HVS: return the names of the nodes where this tag is defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    virtual const std::vector<std::string>
    taggedNodes( const std::string& tagName ) const;

    /// Is the database 'open'?
    virtual bool isOpen() const
    {
      return db().isOpen();
    }

    /// (Re)opens the database.
    virtual void openDatabase();

    /// Closes the database.
    virtual void closeDatabase()
    {
      db().closeDatabase();
    }

    /// Return the "COOL database name".
    virtual const std::string& databaseName() const
    {
      return db().databaseName();
    }

#ifdef COOL300
    /// Start a new transaction and enter manual transaction mode
    virtual ITransactionPtr startTransaction();
#endif

    /// returns the wrapped RalDatabase, only to be used in tests
    RalDatabase *getRalDb() const
    {
      return &db();
    }

    // -- this is not part of the IFolder API, but part
    // of RalDatabase

    /// Refresh the database
    /// Keep and refresh nodes if the flag is true, else drop them
    void refreshDatabase( bool keepNodes = false );

    /// Create a new database with default attributes
    /// Default attributes are those specific for a RelationalDatabase
    /// Expose in the public API a protected RelationalDatabase method.
    void createDatabase();

    /// Create a new database with non-default attributes.
    /// Throw a RelationalException if the given attributes are invalid.
    void createDatabase( const IRecord& dbAttr );

    /// Drop the database
    bool dropDatabase();

  private:
    /// Standard constructor is private
    TransRalDatabase();

    /// Copy constructor is private
    TransRalDatabase( const TransRalDatabase& rhs );

    /// Assignment operator is private
    TransRalDatabase& operator=( const TransRalDatabase& rhs );

    /// returns a reference to the wrapped database
    RalDatabase& db() const
    {
      return *m_db;
    }

    /// the wrapped RalDatabase
    RalDatabase *m_db;

    /// shared pointer to the wrapped Database
    IDatabasePtr m_dbPtr;
  };

}

#endif // COOLKERNEL_IDATABASE_H
