// $Id: RelationalQueryMgr.h,v 1.80 2012-01-30 19:37:36 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALQUERYMGR_H
#define RELATIONALCOOL_RELATIONALQUERYMGR_H

// Include files
#include <memory>
#include <string>
#include <boost/shared_ptr.hpp>
#include "CoolKernel/types.h"
#include "CoolKernel/Record.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/MessageStream.h"

namespace cool 
{

  // Forward declarations
  class IRelationalBulkOperation;
  class IRelationalCursor;
  class IRelationalQueryDefinition;
  class RelationalSequenceMgr;
  class RelationalTableRow;

  /** @class RelationalQueryMgr RelationalQueryMgr.h
   *  -----------------------------------------------------------------------
   *  RelationalQueryMgr: abstract base class for a manager
   *  of relational queries (SELECT statements).
   *
   *  Only three methods are pure virtual and implemented in a derived class:
   *  a check for the existence of a table in the schema,
   *  a generic query for many ordered rows from many aliased tables,
   *  and a generic COUNT(*) query from many aliased tables.
   *
   *  Some simpler queries are implemented within this class in terms
   *  of the two more generic pure abstract methods described above.
   *
   *  Transactions are NOT handled by this class.
   *  -----------------------------------------------------------------------
   *  RelationalUpdateMgr: abstract base class for a manager
   *  of relational DML operations (INSERT/UPDATE/DELETE statements).
   *
   *  Transactions are NOT handled by this class.
   *
   *  TEMPORARY? These functionalities are presently in RelationalQueryMgr.
   *  -----------------------------------------------------------------------
   *  RelationalSequenceMgr: return a reference to the abstract interface
   *  for a manager of COOL relational 'sequences'.
   *
   *  Transactions are NOT handled by this class.
   *
   *  TEMPORARY? These functionalities are presently in RelationalQueryMgr.
   *  -----------------------------------------------------------------------
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-10-11
   */

  class RelationalQueryMgr
  {

  public:

    /// Destructor
    virtual ~RelationalQueryMgr();

    /// Clone this query manager.
    /// PURE VIRTUAL method implemented in subclasses.
    //virtual RelationalQueryMgr* clone() const = 0;

    /// Checks if a table exists in the schema.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool existsTable( const std::string& tableName ) const = 0;

    /// Prepare and execute a CORAL query
    /// (and start/stop the relevant timing report).
    /// The caller becomes the owenr of the returned cursor.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual IRelationalCursor* prepareAndExecuteQuery
    ( const IRelationalQueryDefinition& coolDef,
      boost::shared_ptr<coral::AttributeList>& dataBuffer ) const = 0;

    /// Fetch an ordered set of rows from one or more tables as a vector.
    /// If nExp>0, throws TooManyRowsFound if more than nExp rows are found.
    /// If nExp>0, throws an exception if fewer than nExp rows are found.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual const std::vector<RelationalTableRow> fetchOrderedRows
    ( const IRelationalQueryDefinition& coolDef,
      const std::string& description = "",
      UInt32 nExp = 0,
      bool forUpdate = false ) const = 0;

    /// Fetch a "select COUNT(*)" row count from one or more tables.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual UInt32 countRows
    ( const IRelationalQueryDefinition& coolDef,
      const std::string& description = "" ) const = 0;

    /// Returns a new IRelationalBulkOperation object for performing a bulk
    /// insert operation specifying the input data buffer and the number of
    /// rows that should be cached on the client.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<IRelationalBulkOperation>
    bulkInsertTableRows( const std::string& tableName,
                         const Record& dataBuffer,
                         int rowCacheSize ) const = 0;

    /// Insert one row into one table.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void insertTableRow( const std::string& tableName,
                                 const Record& data ) const = 0;

    /// Returns a new IRelationalBulkOperation object for performing a bulk
    /// update operation specifying the input data buffer and the number of
    /// rows that should be cached on the client.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<IRelationalBulkOperation>
    bulkUpdateTableRows( const std::string& tableName,
                         const std::string& setClause,
                         const std::string& whereClause,
                         const Record& dataBuffer,
                         int rowCacheSize ) const = 0;

    /// Update rows in one table.
    /// If nExp>0, throws an exception if more than nExp rows are updated.
    /// If nExp>0, throws an exception if fewer than nExp rows are updated.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual UInt32 updateTableRows
    ( const std::string& tableName,
      const std::string& setClause,
      const std::string& whereClause,
      const Record& updateData,
      UInt32 nExp = 0 ) const = 0;

    /// Delete rows from one table.
    /// Throws an exception if #rows deleted is different from the expectation
    /// (taken from nExp if > 0; queried internally if nExp = 0).
    /// PURE VIRTUAL method implemented in subclasses.
    virtual UInt32 deleteTableRows
    ( const std::string& tableName,
      const std::string& whereClause,
      const Record& whereData,
      UInt32 nExp = 0 ) const = 0;

    /// Delete all rows from one table (truncate the table).
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void deleteAllTableRows
    ( const std::string& tableName ) const = 0;

    /// Get a RelationalSequenceMgr.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual RelationalSequenceMgr& sequenceMgr() const = 0;

    /// Build the appropriate backend-specific SQL expression
    /// to compute the server-side time in the format used by COOL.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual const std::string serverTimeClause() const = 0;

    /// Build the appropriate backend-specific SQL expression
    /// to compute the server-side time in the format used by COOL.
    static const std::string serverTimeClause( const std::string& technology );

    /// Return the server technology for the current connection.
    /// Supported technologies: "Oracle ", "MySQL", "SQLite", "frontier".
    /// PURE VIRTUAL method implemented in subclasses.
    virtual const std::string databaseTechnology() const = 0;

    /// Return the server technology version for the current connection.
    /// This ultimately corresponds to coral::IConnection::serverVersion().
    /// PURE VIRTUAL method implemented in subclasses.
    virtual const std::string serverVersion() const = 0;

    /// Return the schema name for the current connection.
    /// PURE VIRTUAL method implemented in subclasses.
    virtual const std::string schemaName() const = 0;

  protected:

    /// Constructor is protected
    RelationalQueryMgr();

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const
    {
      return *m_log;
    }

  private:

    /// Copy constructor is private
    RelationalQueryMgr( const RelationalQueryMgr& rhs );

    /// Assignment operator is private
    RelationalQueryMgr& operator=( const RelationalQueryMgr& rhs );

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RELATIONALQUERYMGR_H
