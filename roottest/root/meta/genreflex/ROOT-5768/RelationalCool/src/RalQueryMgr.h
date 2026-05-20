// $Id: RalQueryMgr.h,v 1.81 2012-06-29 13:19:47 avalassi Exp $
#ifndef RELATIONALCOOL_RALQUERYMGR_H
#define RELATIONALCOOL_RALQUERYMGR_H

// Include files
#include "RelationalAccess/IQuery.h"

// Local include files
#include "RelationalQueryMgr.h"

namespace cool
{

  // Forward declarations
  class IRelationalQueryDefinition;
  class ISessionMgr;

  /** @class RalQueryMgr RalQueryMgr.h
   *
   *  Manager of relational queries executed using RAL.
   *  Manager of relational DML operations executed using RAL.
   *
   *  Transactions are NOT handled by this class.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2005-10-11
   */

  class RalQueryMgr : public RelationalQueryMgr
  {

  public:

    /// Constructor from an ISessionMgr const reference.
    RalQueryMgr( const ISessionMgr& sessionMgr );

    /// Constructor from a RalSessionMgr shared pointer.
    //RalQueryMgr( const boost::shared_ptr<ISessionMgr>& sessionMgr );

    /// Destructor.
    virtual ~RalQueryMgr();

    /// Clone this query manager.
    //RelationalQueryMgr* clone() const;

    /// Checks if a table exists in the schema.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    bool existsTable( const std::string& tableName ) const;

    /// Create a new empty CORAL query.
    /// This is needed if subqueries must be defined for this query.
    std::auto_ptr<coral::IQuery> newQuery() const;

    /// Prepare a CORAL query definition from one or more tables.
    void prepareQueryDefinition
    ( coral::IQueryDefinition& coralDef,
      const IRelationalQueryDefinition& coolDef,
      bool countStarSubquery = false ) const;

    /// Prepare a CORAL query from one or more tables.
    /// A pointer to an AttributeList data must be passed to be overwritten
    /// (a new AL with the right spec is created and used as data buffer).
    std::auto_ptr<coral::IQuery> prepareQuery
    ( const IRelationalQueryDefinition& coolDef,
      boost::shared_ptr<coral::AttributeList>& dataBuffer ) const;

    /// Prepare and execute a CORAL query
    /// (and start/stop the relevant timing report).
    /// The caller becomes the owenr of the returned cursor.
    IRelationalCursor* prepareAndExecuteQuery
    ( const IRelationalQueryDefinition& coolDef,
      boost::shared_ptr<coral::AttributeList>& dataBuffer ) const;

    /// Fetch an ordered set of rows from one or more tables as a vector.
    /// If nExp>0, throws TooManyRowsFound if more than nExp rows are found.
    /// If nExp>0, throws an exception if fewer than nExp rows are found.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    const std::vector<RelationalTableRow> fetchOrderedRows
    ( const IRelationalQueryDefinition& coolDef,
      const std::string& description = "",
      UInt32 nExp = 0,
      bool forUpdate = false ) const;

    /// Fetch a "select COUNT(*)" row count from one or more tables.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    UInt32 countRows
    ( const IRelationalQueryDefinition& coolDef,
      const std::string& description = "" ) const;

    /// Returns a new IRelationalBulkOperation object for performing a bulk
    /// insert operation specifying the input data buffer and the number of
    /// rows that should be cached on the client.
    boost::shared_ptr<IRelationalBulkOperation>
    bulkInsertTableRows( const std::string& tableName,
                         const Record& dataBuffer,
                         int rowCacheSize ) const;

    /// Insert one row into one table.
    void insertTableRow( const std::string& tableName,
                         const Record& data ) const;

    /// Returns a new IRelationalBulkOperation object for performing a bulk
    /// update operation specifying the input data buffer and the number of
    /// rows that should be cached on the client.
    boost::shared_ptr<IRelationalBulkOperation>
    bulkUpdateTableRows( const std::string& tableName,
                         const std::string& setClause,
                         const std::string& whereClause,
                         const Record& dataBuffer,
                         int rowCacheSize ) const;

    /// Update rows in one table.
    /// If nExp>0, throws an exception if more than nExp rows are updated.
    /// If nExp>0, throws an exception if fewer than nExp rows are updated.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    UInt32 updateTableRows
    ( const std::string& tableName,
      const std::string& setClause,
      const std::string& whereClause,
      const Record& updateData,
      UInt32 nExp = 0 ) const;

    /// Delete rows from one table.
    /// Throws an exception if #rows deleted is different from the expectation
    /// (taken from nExp if > 0; queried internally if nExp = 0).
    /// Implements RelationalQueryMgr pure abstract virtual method.
    UInt32 deleteTableRows
    ( const std::string& tableName,
      const std::string& whereClause,
      const Record& whereData,
      UInt32 nExp = 0 ) const;

    /// Delete all rows from one table (truncate the table).
    /// Implements RelationalQueryMgr pure abstract virtual method.
    void deleteAllTableRows
    ( const std::string& tableName ) const;

    /// Get a RelationalSequenceMgr.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    RelationalSequenceMgr& sequenceMgr() const
    {
      return *m_sequenceMgr;
    }

    /// Build the appropriate backend-specific SQL expression
    /// to compute the server-side time in the format used by COOL.
    /// Implements RelationalQueryMgr pure abstract virtual method.
    const std::string serverTimeClause() const;

    /// Execute a CORAL query (and start/stop the relevant timing report)
    static coral::ICursor& executeQuery( coral::IQuery& query );

    /// Increment a CORAL cursor (and start/stop the relevant timing report)
    static bool cursorNext( coral::ICursor& cursor );

    /// Return the server technology for the current connection.
    /// Supported technologies: "Oracle ", "MySQL", "SQLite", "frontier".
    const std::string databaseTechnology() const;

    /// Return the server technology version for the current connection.
    /// This ultimately corresponds to coral::IConnection::serverVersion().
    const std::string serverVersion() const;

    /// Return the schema name for the current connection.
    const std::string schemaName() const;

  private:

    /// Copy constructor is private (fix Coverity MISSING_COPY)
    RalQueryMgr( const RalQueryMgr& rhs );

    /// Assignment operator is private (fix Coverity MISSING_ASSIGN)
    RalQueryMgr& operator=( const RalQueryMgr& rhs );

  private:

    /// Reference to the ISessionMgr (not owned by this instance)
    const ISessionMgr* m_sessionMgr;

    /// Handle to the ISessionMgr (shared ownership)
    /// [NB this cannot be a reference because of RalQueryMgr::clone()!]
    //boost::shared_ptr<ISessionMgr> m_sessionMgr;

    /// RelationalSequenceMgr (owned by this instance)
    RelationalSequenceMgr* m_sequenceMgr;

  };

}

#endif // RELATIONALCOOL_RALQUERYMGR_H
