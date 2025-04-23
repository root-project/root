// $Id: RelationalException.h,v 1.41 2012-06-29 13:19:48 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALEXCEPTION_H
#define RELATIONALCOOL_RELATIONALEXCEPTION_H 1

// Include files
#include "CoolKernel/Exception.h"

// Local include files
#include "VersionInfo.h"

namespace cool
{

  //--------------------------------------------------------------------------

  /** @class RelationalException RelationalException.h
   *
   *  Base exception class for the relational implementations of COOL.
   *  Derived from SEAL Exception just like the POOL RelationalException.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-10
   */

  class RelationalException : public Exception {

  public:

    /// Constructor
    RelationalException( const std::string& message,
                         const std::string& methodName = "" );

    /// Copy constructor
    RelationalException( const RelationalException& rhs );

    /// Destructor
    virtual ~RelationalException() throw();

  private:

    /// Assignment operator is private (fix Coverity COPY_WITHOUT_ASSIGN)
    RelationalException& operator=( const RelationalException& rhs );

  };

  //--------------------------------------------------------------------------

  /** @class PanicException PanicException.h
   *
   *  Exception class for PANIC situations (asserts).
   *  These are situations which should not normally happen in user code:
   *  their meaning is that the code must be extended to cover new use cases.
   *
   *  @author Andrea Valassi
   *  @date   2007-10-26
   */

  class PanicException : public RelationalException {

  public:

    /// Constructor
    explicit PanicException( const std::string& message,
                             const std::string& domain )
      : RelationalException( message, domain )
    {
    }

    /// Destructor
    virtual ~PanicException() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class NodeTableRowNotFound
   *
   *  Exception thrown when the row for a node with a given name
   *  or nodeId cannot be found in the node table.
   */

  class NodeTableRowNotFound : public RelationalException {

  public:

    /// Constructor
    explicit NodeTableRowNotFound( const std::string& methodName )
      : RelationalException( "Table row for specified node not found",
                             methodName ) {}

    /// Constructor
    explicit NodeTableRowNotFound( const std::string& fullPath,
                                   const std::string& methodName )
      : RelationalException( "Table row for node with name=" + fullPath
                             + " not found", methodName ) {}

    /// Constructor
    explicit NodeTableRowNotFound( unsigned int nodeId,
                                   const std::string& methodName )
      : RelationalException( "", methodName )
    {
      std::stringstream msg;
      msg << "Table row for node with nodeId=" << nodeId << " not found";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~NodeTableRowNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FolderSpecificTableNotFound
   *
   *  Exception thrown when a folder-specific table (object table,
   *  tag table, object2tag table) cannot be found, even if its name
   *  was retrieved from the corresponding node table row: this may
   *  indicate either data corruption or DDL changes by another process
   *  during the job lifetime (folder has been dropped in the meantime).
   *
   */

  class FolderSpecificTableNotFound : public RelationalException {

  public:

    /// Constructor
    explicit FolderSpecificTableNotFound( const std::string& fullPath,
                                          const std::string& methodName )
      : RelationalException( "", methodName )
    {
      std::stringstream msg;
      msg << "PANIC! Folder-specific table "
          << "not found for folder " << fullPath << ": this indicates"
          << " either data corruption (if node table row exists)"
          << " or DDL changes by another process during your job lifetime";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~FolderSpecificTableNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TableNotFound
   *
   *  Exception thrown when a table cannot be found.
   *  This is generally a "PANIC" situation where the name of the table
   *  was retrieved from the database and should exist: this may indicate
   *  either data corruption or DDL changes by another process during the
   *  job lifetime (e.g., one folder has been dropped in the meantime).
   *
   */

  class TableNotFound : public RelationalException {

  public:

    /// Constructor
    explicit TableNotFound( const std::string& tableName,
                            const std::string& methodName )
      : RelationalException( "", methodName )
    {
      std::stringstream msg;
      msg << "Table " << tableName << " not found:"
          << " this may indicate a PANIC situation with data corruption"
          << " or DDL changes by another process during your job lifetime";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~TableNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TableNotDropped
   *
   *  Exception thrown when a table cannot be dropped.
   *
   */

  class TableNotDropped : public RelationalException {

  public:

    /// Constructor
    explicit TableNotDropped( const std::string& table,
                              const std::string& methodName )
      : RelationalException( "Could not drop table/sequence '" + table + "'",
                             methodName ) {}

    /// Destructor
    virtual ~TableNotDropped() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RowNotDeleted
   *
   *  Exception thrown when a table row cannot be deleted.
   *
   */

  class RowNotDeleted : public RelationalException {

  public:

    /// Constructor
    explicit RowNotDeleted( const std::string& message,
                            const std::string& methodName )
      : RelationalException( message, methodName ) {}

    /// Destructor
    virtual ~RowNotDeleted() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RowNotUpdated
   *
   *  Exception thrown when a table row cannot be updated.
   *
   */

  class RowNotUpdated : public RelationalException {

  public:

    /// Constructor
    explicit RowNotUpdated( const std::string& message,
                            const std::string& methodName )
      : RelationalException( message, methodName ) {}

    /// Destructor
    virtual ~RowNotUpdated() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class RowNotInserted
   *
   *  Exception thrown when a table row cannot be inserted.
   *
   */

  class RowNotInserted : public RelationalException {

  public:

    /// Constructor
    explicit RowNotInserted( const std::string& message,
                             const std::string& methodName )
      : RelationalException( message, methodName ) {}

    /// Destructor
    virtual ~RowNotInserted() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class NoRowsFound
   *
   *  Exception thrown when a query returns zero table rows
   *  and at least one was expected.
   *
   */

  class NoRowsFound : public RelationalException {

  public:

    /// Constructor
    explicit NoRowsFound( const std::string& message,
                          const std::string& methodName )
      : RelationalException( message, methodName ) {}

    /// Destructor
    virtual ~NoRowsFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TooManyRowsFound
   *
   *  Exception thrown when a query returns more table rows than expected.
   *
   */

  class TooManyRowsFound : public RelationalException {

  public:

    /// Constructor
    explicit TooManyRowsFound( const std::string& message,
                               const std::string& methodName )
      : RelationalException( message, methodName ) {}

    /// Destructor
    virtual ~TooManyRowsFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class IncompatibleReleaseNumber
   *
   *  Exception thrown when the software library release number is not
   *  compatible with the database schema it is attempting to handle.
   *
   */

  class IncompatibleReleaseNumber : public RelationalException
  {

  public:

    /// Constructor
    explicit IncompatibleReleaseNumber
    ( const std::string& message,
      const std::string& methodName )
      : RelationalException
    ( "IncompatibleReleaseNumber exception. " + message, methodName ) {}

    /// Destructor
    virtual ~IncompatibleReleaseNumber() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class InvalidColumnName
   *
   *  Exception thrown when attempting to create or rename a column
   *  of a relational table using an invalid name.
   *
   *  Names of column must have between 1 and 30 characters (including only
   *  letters, digits or the '_' character ) and must start with a letter.
   *
   */

  class InvalidColumnName : public RelationalException
  {

  public:

    /// Constructor
    explicit InvalidColumnName( const std::string& name,
                                const std::string& methodName )
      : RelationalException
    ( "Invalid column name '" + name + "'", methodName ) {}

    /// Destructor
    virtual ~InvalidColumnName() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class UnsupportedFolderSchema
   *
   *  Exception thrown when attempting to open a folder whose schema
   *  version is newer than that supported by the present software
   *  release (or has been obsoleted and is no longer supported).
   */

  class UnsupportedFolderSchema : public RelationalException
  {

  public:

    /// Constructor
    explicit UnsupportedFolderSchema( const std::string& fullPath,
                                      const VersionNumber& schemaVersion,
                                      const std::string& domain )
      : RelationalException( "", domain )
    {
      std::stringstream msg;
      msg << "Schema version " << schemaVersion << " of folder '"
          << fullPath << "' is not supported by this software release "
          << VersionInfo::release << " (using default database schema version "
          << VersionInfo::schemaVersion << ")";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit UnsupportedFolderSchema( const std::string& message,
                                      const std::string& domain )
      : RelationalException( message, domain )
    {
    }

    /// Destructor
    virtual ~UnsupportedFolderSchema() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class UnsupportedFolderSetSchema
   *
   *  Exception thrown when attempting to open a folder set whose schema
   *  version is newer than that supported by the present software
   *  release (or has been obsoleted and is no longer supported).
   */

  class UnsupportedFolderSetSchema : public RelationalException
  {

  public:

    /// Constructor
    explicit UnsupportedFolderSetSchema( const std::string& fullPath,
                                         const VersionNumber& schemaVersion,
                                         const std::string& domain )
      : RelationalException( "", domain )
    {
      std::stringstream msg;
      msg << "Schema version " << schemaVersion << " of folder set '"
          << fullPath << "' is not supported by this software release "
          << VersionInfo::release << " (using default database schema version "
          << VersionInfo::schemaVersion << ")";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit UnsupportedFolderSetSchema( const std::string& message,
                                         const std::string& domain )
      : RelationalException( message, domain )
    {
    }

    /// Destructor
    virtual ~UnsupportedFolderSetSchema() throw() {}

  };

  //--------------------------------------------------------------------------

}

#endif // RELATIONALCOOL_RELATIONALEXCEPTION_H
