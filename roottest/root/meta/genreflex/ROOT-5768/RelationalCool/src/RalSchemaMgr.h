// $Id: RalSchemaMgr.h,v 1.37 2012-01-30 17:06:03 avalassi Exp $
#ifndef RELATIONALCOOL_RALSCHEMAMGR_H
#define RELATIONALCOOL_RALSCHEMAMGR_H

// Include files
#include <boost/shared_ptr.hpp>
#include "RelationalAccess/ISessionProxy.h"

// Local include files
#include "RelationalSchemaMgr.h"

namespace cool {

  // Forward declarations
  class ISessionMgr;

  /** @class RalSchemaMgr RalSchemaMgr.h
   *
   *  CORAL implementation of the manager
   *  of the COOL relational database schema.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2006-03-10
   */

  class RalSchemaMgr : public RelationalSchemaMgr
  {

  public:

    /// Constructor from a RelationalDatabase and a RalSessionMgr
    /// Inlined with the base class non-standard constructor as otherwise
    /// the compiler attempts to use the base class standard constructor
    /// (Windows compilation error C2248: standard constructor is private)
    RalSchemaMgr( const RelationalDatabase& aDb,
                  const boost::shared_ptr<ISessionMgr>& sessionMgr )
      : RelationalSchemaMgr( aDb )
      , m_sessionMgr( sessionMgr )
    {
      initialize();
    }

    /// Destructor
    virtual ~RalSchemaMgr();

    /// Drop a table (return false if the table does not exist)
    bool dropTable( const std::string& tableName ) const;

    /// Creates a CORAL table description from a RecordSpecification
    std::auto_ptr<coral::TableDescription> createTableDescription
    ( const std::string& tableName,
      const IRecordSpecification& payloadSpec,
      const std::string& primaryKey = "" ) const;

    /// Creates the main table
    void createMainTable( const std::string& tableName ) const;

    /// Fill the main table
    void fillMainTable( const std::string& tableName,
                        const coral::AttributeList& dbAttr ) const;

    /// Creates the iovTables table
    void createIovTablesTable
    ( const std::string& iovTablesTableName ) const;

    /// Creates the channelTables table
    void createChannelTablesTable
    ( const std::string& channelTablesTableName ) const;

    /// Creates the node table
    /// TEMPORARY - pass the table prefix to create the root fs tag sequence
    void createNodeTable( const std::string& nodeTableName,
                          const std::string& defaultTablePrefix ) const;

    /// Creates the global tag table
    void createGlobalTagTable( const std::string& globalTagTableName,
                               const std::string& nodeTableName ) const;

    /// Creates the global head tag table
    void createGlobalHeadTagTable
    ( const std::string& globalHeadTagTableName,
      const std::string& globalTagTableName ) const;

    /// Creates the global user tag table
    void createGlobalUserTagTable
    ( const std::string& globalUserTagTableName,
      const std::string& globalTagTableName ) const;

    /// Creates the tag2tag HVS table
    void createTag2TagTable( const std::string& tag2tagTableName,
                             const std::string& globalTagTableName,
                             const std::string& nodeTableName ) const;

    /// Creates the tag2tag HVS table FK references to the tag table
    /// *** This is needed by the coolReplicateDB utility ***
    void createTag2TagFKs( const std::string& tag2tagTableName,
                           const std::string& globalTagTableName ) const;

    /// Creates the tag2tag HVS table FK references to the tag table
    /// *** This is needed by the coolReplicateDB utility ***
    bool dropTag2TagFKs( const std::string& tag2tagTableName,
                         bool verbose = true ) const;

    /// Creates a shared sequence table
    void createSharedSequence( const std::string& sharedSequenceName,
                               const std::string& nodeTableName ) const;

    /// Creates the tag sequence for a node
    void createTagSequence( const std::string& seqName ) const;

    /// Creates the tag table for a leaf node
    void createTagTable( const std::string& tableName ) const;

    /// Creates the channel table for a leaf node
    void createChannelTable( const std::string& tableName ) const;

    /// Creates the object2tag (iov2tag) table for a leaf node
    void createObject2TagTable
    ( const std::string& object2tagTableName,
      const std::string& objectTableName,
      const std::string& tagTableName,
      const std::string& payloadTableName,
      PayloadMode::Mode pMode ) const;

    /// Creates a separate payload table
    void createPayloadTable( const std::string& payloadTableName,
                             const std::string& objectTableName,
                             const IRecordSpecification& payloadSpec,
                             PayloadMode::Mode pMode ) const;

    /// Creates the object (iov) table for a leaf node
    void createObjectTable
    ( const std::string& objectTableName,
      const std::string& channelTableName,
      const IRecordSpecification& payloadSpec,
      FolderVersioning::Mode versioningMode,
      const std::string& payloadTableName,
      PayloadMode::Mode payloadMode ) const;

    /// Creates the FK references from the object table to the channel table
    /// *** This is needed by the coolReplicateDB utility ***
    void createObjectChannelFK( const std::string& objectTableName,
                                const std::string& channelTableName ) const;

    /// Drops the FK reference from the object table to the channel table
    /// *** This is needed by the coolReplicateDB utility ***
    void dropObjectChannelFK( const std::string& objectTableName ) const;

    /// Creates the FK references from the object table to the payload table
    /// *** This is needed by the coolReplicateDB utility ***
    void createObjectPayloadFK( const std::string& objectTableName,
                                const std::string& payloadTableName ) const;

    /// Drops the FK reference from the object table to the channel table
    /// *** This is needed by the coolReplicateDB utility ***
    void dropObjectPayloadFK( const std::string& objectTableName ) const;

    /// creates the self reference from payload set id to object id
    void createObjectToObjectIdFK( const std::string& objectTableName ) const;

    /// creates the self reference from payload set id to object id
    void createObjectToOriginalIdFK( const std::string& objectTableName ) const;

    /// creates the FK from payload set id to the object table object id
    void createPayloadToObjectIdFK( const std::string& payloadTableName, 
                                    const std::string& objectTableName ) const;


    /// Rename a column of the given table
    void renameColumnInTable( const std::string& tableName,
                              const std::string& oldColumnName,
                              const std::string& newColumnName ) const;

    /// Add columns to the given table
    void addColumnsToTable( const std::string& tableName,
                            const IRecord& columnSpecAndValues ) const;

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize();

    /// Standard constructor is private
    RalSchemaMgr();

    /// Copy constructor is private
    RalSchemaMgr( const RalSchemaMgr& rhs );

    /// Assignment operator is private
    RalSchemaMgr& operator=( const RalSchemaMgr& rhs );

    /// Returns the database session
    /// Delegated to RalSessionMgr.
    coral::ISessionProxy& session() const;

    /// Return the server technology for the current connection.
    /// Delegated to RalSessionMgr.
    const std::string& databaseTechnology() const;

    /// Returns the server version for the current connection.
    /// Delegated to RalSessionMgr.
    const std::string& serverVersion() const;

  private:

    /// Handle to the RalSessionMgr (shared ownership)
    boost::shared_ptr<ISessionMgr> m_sessionMgr;

  };

}

#endif // RELATIONALCOOL_RALSCHEMAMGR_H
