// $Id: IDatabaseSvc.h,v 1.40 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IDATABASESVC_H
#define COOLKERNEL_IDATABASESVC_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/DatabaseId.h"
#include "CoolKernel/pointers.h"

namespace cool
{

  /** @class IDatabaseSvc IDatabaseSvc.h
   *
   *  Abstract interface to the top-level service to create, drop or open
   *  COOL "conditions databases".
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */

  class IDatabaseSvc
  {

  public:

    /// Destructor
    virtual ~IDatabaseSvc() {}

    /// Create a new database and return the corresponding manager.
    /// Use the default database attributes for the relevant implementation.
    /// The ownership of the database manager instance is shared.
    /// Please also see \ref sec_handles on connection handling.
    /// Throw an Exceptions if
    /// - any of the management tables cannot be created
    /// - any of the indexes cannot be created
    /// - any of the constraints (primary/foreign keys) cannot be created
    /// - the db attributes cannot be written to the main management table
    /// - the root folderset "/" cannot be inserted into the node table
    virtual IDatabasePtr createDatabase( const DatabaseId& dbId ) const = 0;

    /*
    /// Create a new database and return the corresponding manager.
    /// Specify non-default database attributes.
    /// The ownership of the database manager instance is shared.
    virtual IDatabasePtr createDatabase( const DatabaseId& dbId,
                                         const IRecord& dbAttr ) const = 0;
    */

    /// Open an existing database and return the corresponding manager.
    /// The ownership of the database manager instance is shared.
    /// Please also see \ref sec_handles on connection handling.
    /// Throw an Exception if the database schema version is not
    /// supported by this COOL software release.
    /// Throw DatabaseDoesNotExist if the database does not exist.
    virtual IDatabasePtr openDatabase( const DatabaseId& dbId,
                                       bool readOnly = true ) const = 0;

    /// Drop an existing database.
    /// Return true if all database structures are dropped as expected.
    /// Return false (without throwing any exception) if the database and
    /// all its structures do not exist any more on exit from this method,
    /// but the database or some of its structures did not exist to start with.
    /// Throw an Exception if the database schema version, or the schema
    /// of one of the nodes in this database, is more recent than the
    /// schema version supported by the current COOL software release.
    /// Throw an Exception if the database or one of its structures
    /// cannot be dropped (i.e. continues to exist on exit from this method).
    virtual bool dropDatabase( const DatabaseId& dbId ) const = 0;

    /// Retrieve the version number of the database service software.
    virtual const std::string serviceVersion() const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IDatabaseSvc& operator=( const IDatabaseSvc& rhs );
#endif

  };

}
#endif // COOLKERNEL_IDATABASESVC_H
