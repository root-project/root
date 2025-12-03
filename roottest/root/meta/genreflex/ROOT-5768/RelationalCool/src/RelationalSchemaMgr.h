// $Id: RelationalSchemaMgr.h,v 1.42 2012-01-30 14:56:20 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALSCHEMAMGR_H
#define RELATIONALCOOL_RELATIONALSCHEMAMGR_H

// Include files
#include <memory>
#include "CoolKernel/FolderVersioning.h"
//#include "CoolKernel/PayloadMode.h"
#include "PayloadMode.h" // TEMPORARY
#include "CoralBase/AttributeList.h" // SAS 19.06.2006 needed only on Mac/gcc4?
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/TableDescription.h"

namespace cool {

  // Forward declarations
  class IRecordSpecification;
  class RelationalDatabase;
  class RelationalQueryMgr;

  /** @class RelationalSchemaMgr RelationalSchemaMgr.h
   *
   *  Abstract base class for the manager
   *  of the COOL relational database schema.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-15
   */

  class RelationalSchemaMgr {

  public:

    /// Destructor
    virtual ~RelationalSchemaMgr();

    /// Drop a table (return false if the table does not exist)
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool dropTable( const std::string& tableName ) const = 0;

    /// Creates the main table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createMainTable( const std::string& tableName ) const = 0;

    /// Fill the main table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void fillMainTable( const std::string& tableName,
                                const coral::AttributeList& dbAttr ) const = 0;

    // *** START *** 3.0.0 schema extensions (task #4307)
    /// Creates the iovTables table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createIovTablesTable
    ( const std::string& iovTablesTableName ) const = 0;

    /// Creates the channelTables table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createChannelTablesTable
    ( const std::string& channelTablesTableName ) const = 0;
    // **** END **** 3.0.0 schema extensions (task #4307)

    /// Creates the node table
    /// TEMPORARY - pass the table prefix to create the root fs tag sequence
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    void createNodeTable( const std::string& nodeTableName,
                          const std::string& defaultTablePrefix ) const = 0;

    /// Creates the global tag table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    void createGlobalTagTable( const std::string& globalTagTableName,
                               const std::string& nodeTableName ) const = 0;

    // *** START *** 3.0.0 schema extensions (task #4396)
    /// Creates the global head tag table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createGlobalHeadTagTable
    ( const std::string& globalHeadTagTableName,
      const std::string& globalTagTableName ) const = 0;

    /// Creates the global user tag table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createGlobalUserTagTable
    ( const std::string& globalUserTagTableName,
      const std::string& globalTagTableName ) const = 0;
    // **** END **** 3.0.0 schema extensions (task #4396)

    /// Creates the tag2tag HVS table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    void createTag2TagTable( const std::string& tag2tagTableName,
                             const std::string& globalTagTableName,
                             const std::string& nodeTableName ) const = 0;

    /// Creates the tag2tag HVS table FK references to the tag table
    /// *** This is needed by the coolReplicateDB utility ***
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    void createTag2TagFKs( const std::string& tag2tagTableName,
                           const std::string& globalTagTableName ) const = 0;

    /// Creates the tag2tag HVS table FK references to the tag table
    /// *** This is needed by the coolReplicateDB utility ***
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    bool dropTag2TagFKs( const std::string& tag2tagTableName,
                         bool verbose = true ) const = 0;

    /// Creates a shared sequence table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual
    void createSharedSequence( const std::string& sharedSequenceName,
                               const std::string& nodeTableName ) const = 0;

    /// Creates the tag sequence for a node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createTagSequence( const std::string& seqName ) const = 0;

    /// Creates the tag table for a leaf node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createTagTable( const std::string& tableName ) const = 0;

    /// Creates the channel table for a leaf node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createChannelTable( const std::string& tableName ) const = 0;

    /// Creates the object2tag (iov2tag) table for a leaf node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createObject2TagTable
    ( const std::string& object2tagTableName,
      const std::string& objectTableName,
      const std::string& tagTableName,
      const std::string& payloadTableName,
      PayloadMode::Mode pMode ) const = 0;

    /// Creates a separate payload table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createPayloadTable
    ( const std::string& payloadTableName,
      const std::string& objectTableName,
      const IRecordSpecification& payloadSpec,
      PayloadMode::Mode pMode ) const = 0;

    /// Creates the object (iov) table for a leaf node
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createObjectTable
    ( const std::string& objectTableName,
      const std::string& channelTableName,
      const IRecordSpecification& payloadSpec,
      FolderVersioning::Mode versioningMode,
      const std::string& payloadTableName,
      PayloadMode::Mode payloadMode = PayloadMode::INLINEPAYLOAD ) const = 0;

    /// Creates the FK references from the object table to the channel table
    /// *** This is needed by the coolReplicateDB utility ***
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void createObjectChannelFK
    ( const std::string& objectTableName,
      const std::string& channelTableName ) const = 0;

    /// Drops the FK reference from the object table to the channel table
    /// *** This is needed by the coolReplicateDB utility ***
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void dropObjectChannelFK
    ( const std::string& objectTableName ) const = 0;

    /// Rename a column of the given table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void renameColumnInTable
    ( const std::string& tableName,
      const std::string& oldColumnName,
      const std::string& newColumnName ) const = 0;

    /// Add columns to the given table
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void addColumnsToTable
    ( const std::string& tableName,
      const IRecord& columnSpecAndValues ) const = 0;

    /// Is this a valid name for a column of a relational table?
    virtual bool isValidColumnName( const std::string& name ) const;

  protected:

    /// Constructor from a RelationalDatabase reference
    RelationalSchemaMgr( const RelationalDatabase& db );

    /// Get the RelationalDatabase reference
    const RelationalDatabase& db() const { return m_db; }

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

    /// Get a relational query manager
    const RelationalQueryMgr& queryMgr() const;

  private:

    /// Standard constructor is private
    RelationalSchemaMgr();

    /// Copy constructor is private
    RelationalSchemaMgr( const RelationalSchemaMgr& rhs );

    /// Assignment operator is private
    RelationalSchemaMgr& operator=( const RelationalSchemaMgr& rhs );

  protected:

    /// Reference to the RelationalDatabase
    const RelationalDatabase& m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RELATIONALSCHEMAMGR_H
