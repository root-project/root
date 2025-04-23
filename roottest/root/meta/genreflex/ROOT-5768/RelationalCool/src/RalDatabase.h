// $Id: RalDatabase.h,v 1.240 2012-01-30 17:40:32 avalassi Exp $
#ifndef RELATIONALCOOL_RALDATABASE_H
#define RELATIONALCOOL_RALDATABASE_H 1

// First of all, set/unset COOL290, COOL300 and COOL_HAS_CPP11 macros
#include "CoolKernel/VersionInfo.h"

// Include files
#ifndef COOL_HAS_CPP11
#include <boost/enable_shared_from_this.hpp>
#endif
#include "RelationalAccess/ISessionProxy.h"

// Local include files
#include "CoralConnectionServiceProxy.h"
#include "ISessionMgr.h"
#include "RelationalDatabase.h"

// Disable icc warning 444: boost::enable_shared_from_this has non virtual dtor
#ifdef __ICC
#pragma warning (push)
#pragma warning (disable: 444)
#endif

namespace cool
{

  // Forward declarations
  class RalDatabase;
  class RalObjectMgr;
  class RelationalObjectTableRow;
  class RelationalSequence;
  class RelationalTableRow;
  class SimpleObject;
  class TransRalDatabase;

  // Type definitions
  typedef boost::shared_ptr<RalDatabase> RalDatabasePtr;

  /** @class RalDatabase RalDatabase.h
   *
   *  RAL implementation of one COOL "condition database" instance
   *  (deployed on a specific physical infrastructure).
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */

  class RalDatabase : public RelationalDatabase
#ifdef COOL_HAS_CPP11
                    , public std::enable_shared_from_this<RalDatabase>
#else
                    , public boost::enable_shared_from_this<RalDatabase>
#endif
  {

    friend class MemoryConsumptionTest;
    friend class RalDatabaseTest;
    friend class RalDatabaseTest_extendedSpec;
    friend class RalDatabaseTest_versioning;
    friend class RalSequenceTest;
    friend class RelationalObjectMgrTest;
    friend class RelationalObjectTableTest;

    // Only the RalDatabaseSvc can instantiate or delete a RalDatabase
    friend class RalDatabaseSvc;
    // also the wrapper needs to be able to create databases
    friend class TransRalDatabase;

    // Also the RalSchemaEvolution manager can access all internal methods
    friend class RalSchemaEvolution;

    // Only the boost shared pointer can delete a RalDatabase: see
    // http://www.boost.org/libs/smart_ptr/sp_techniques.html#preventing_delete
    class deleter;
    friend class deleter;

  public:

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const
    {
      return RelationalDatabase::log();
    }

    /// Get the ISessionMgr
    boost::shared_ptr<ISessionMgr> sessionMgr() const;

    /// Required access mode to the database.
    bool isReadOnly() const
    {
      return sessionMgr()->isReadOnly();
    }

    /// Returns the database session.
    /// Delegated to RalSessionMgr.
    coral::ISessionProxy& session() const;

    /// Create a new folder set
    /// Starts a transaction
    IFolderSetPtr createFolderSet
    ( const std::string& fullPath,
      const std::string& description = "",
      bool createParents = false );

    /// Retrieve an existing folderset and return the corresponding manager
    /// Throw an exception if the folderset does not exist
    IFolderSetPtr getFolderSet( const std::string& fullPath );

    /// Return the folderset manager for a given row
    IFolderSetPtr getFolderSet( const RelationalTableRow& row );

    /// Create a new folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if there is no connection to the database.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if an invalid versioning mode has been specified.
    IFolderPtr createFolder
    ( const std::string& fullPath,
      const IFolderSpecification& folderSpec,
      const std::string& description = "",
      bool createParents = false );

    /// DEPRECATED: this is likely to be removed in the next major release
    /// (similar to the COOL133 API, with IRecordSpecification instead of
    /// ExtendedAttributeListSpecification: use IFolderSpecification instead).
    IFolderPtr createFolder
    ( const std::string& fullPath,
      const IRecordSpecification& payloadSpec,
      const std::string& description = "",
      FolderVersioning::Mode mode = FolderVersioning::SINGLE_VERSION,
      bool createParents = false );

    /// Create a new folder and return the corresponding manager
    /// The ownership of the folder manager instance is shared
    IFolderPtr createFolder
    ( const std::string& fullPath,
      const IRecordSpecification& payloadSpec,
      const std::string& description,
      FolderVersioning::Mode mode,
      bool createParents,
      PayloadMode::Mode payloadMode );

    /// Retrieve an existing folder and return the corresponding manager
    /// Throw an exception if the folder does not exist
    IFolderPtr getFolder( const std::string& fullPath );

    /// Return the folder manager for a given row
    IFolderPtr getFolder( const RelationalTableRow& row );

    /// Retrieve all existing folders and folder sets
    /// (preload the cache - used for ReadOnly connections only)
    void preloadAllNodes();

    /// Drop an existing node.
    bool dropNode( const std::string& fullPath );

    /// Get a RelationalObjectTable for the given folder.
    /// The concrete class can only be created by the concrete database.
    /// The RelationalFolder parameter is only used to obtain
    /// the associated table names and is *not* retained.
    boost::shared_ptr<RelationalObjectTable>
    relationalObjectTable( const RelationalFolder& folder ) const;

    /// Return the RelationalDatabasePtr
    RelationalDatabasePtr relationalDbPtr()
    {
      return shared_from_this();
    }

    /// Return the connection state of the database.
    /// This is different from isOpen(): isConnected() refers only to the
    /// connection to the server, a prerequisite to the actual opening of
    /// a COOL "database" (which implies reading the management tables).
    /// Delegated to RalSessionMgr.
    bool isConnected() const;

    /// (Re)connect to the database.
    /// Delegated to RalSessionMgr.
    void connect();

    /// Close the database connection.
    /// Delegated to RalSessionMgr.
    void disconnect();

#ifdef COOL300
    /// Start a new transaction and enter manual transaction mode
    virtual ITransactionPtr startTransaction();
#endif

    /// Refresh the database
    /// Keep and refresh nodes if the flag is true, else drop them
    void refreshDatabase( bool keepNodes = false );

    /// Refresh all nodes.
    void refreshAllNodes();

    /// Refresh an existing node.
    void refreshNode( const std::string& fullPath );

    /// Drops all nodes (keep the root node if required)
    bool dropAllNodes( bool keepRoot=false );

  protected:

    /// The following methods are all protected: only a RalDatabaseSvc can
    /// instantiate or delete this class and create, drop or open a database

    /// Constructor
    /// Throws a RelationalException if the RelationalService handle is NULL.
    /// Throws a RelationalException if a connection cannot be established.
    RalDatabase( CoralConnectionServiceProxyPtr ppConnSvc,
                 const DatabaseId& dbId,
                 bool readOnly );

    /// Destructor
    virtual ~RalDatabase();

    /// Create a new database with default attributes
    /// Default attributes are those specific for a RelationalDatabase
    /// Expose in the public API a protected RelationalDatabase method.
    void createDatabase() {
      return RelationalDatabase::createDatabase();
    }

    /// Create a new database with non-default attributes.
    /// Throw a RelationalException if the given attributes are invalid.
    void createDatabase( const IRecord& dbAttr );

    /// Drop the database
    bool dropDatabase();

  private:

    /// Standard constructor is private
    RalDatabase();

    /// Copy constructor is private
    RalDatabase( const RalDatabase& rhs );

    /// Assignment operator is private
    RalDatabase& operator=( const RalDatabase& rhs );

    /// Update the description for the given node
    void updateNodeTableDescription( const std::string& fullPath,
                                     const std::string& description ) const;

    /// Creates a new entry in the folder table
    /// Returns the node id of the new entry
    UInt32 insertNodeTableRow
    ( const std::string& fullPath,
      const std::string& description,
      bool createParents,
      bool isLeaf,
      const std::string& payloadSpecDesc,
      FolderVersioning::Mode versioningMode,
      PayloadMode::Mode payloadMode = PayloadMode::INLINEPAYLOAD );

    /// Set the useTimeout flag
    void setUseTimeout( bool flag ) { m_useTimeout = flag; }

    /// Private deleter class
    class deleter
    {
    public:
      void operator()( RalDatabase* pDb ) {
        delete pDb;
      }
    };

  private:

    /// Map of all nodes (cache - only used for ReadOnly connections)
    /// NB Do not use IFolderPtr/IFolderSetPtr: RalDatabase is never deleted!
    std::map< std::string, RelationalTableRow* > m_nodes;

    /// Switch on the schema change timeout (ORA-01466 problem)
    bool m_useTimeout;

    /// Session manager connected to the database
    /// (created by this instance, but ownership shared, e.g. with query mgr)
    boost::shared_ptr<ISessionMgr> m_sessionMgr;

  };

}

// Reenable icc warning 444
#ifdef __ICC
#pragma warning (pop)
#endif

#endif // RELATIONALCOOL_RALDATABASE_H
