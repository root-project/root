// $Id: RalDatabaseSvc.h,v 1.35 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RALDATABASESVC_H
#define RELATIONALCOOL_RALDATABASESVC_H 1

// Include files
#include <memory>
#include "CoralBase/AttributeList.h"
#include "CoralBase/MessageStream.h"
#include "CoolKernel/IDatabaseSvc.h"

// Local include files
#include "CoralConnectionServiceProxy.h"

namespace cool
{

  /** @class RalDatabaseSvc RalDatabaseSvc.h
   *
   *  Top-level service to create, drop or open "conditions databases"
   *  using the RAL implementation of COOL.
   *
   *  Implementation as a SEAL service inspired from RelationalService.
   *  Many thanks to Rado for his help with the SEAL component model!
   *
   *  This class will typically be used as a singleton in a user application.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */

  class RalDatabaseSvc : public IDatabaseSvc {

  public:

    /// Constructor
    RalDatabaseSvc( coral::IConnectionService& connSvc );

    /// Destructor
    virtual ~RalDatabaseSvc();

    /// Create a new database and return the corresponding manager
    /// Use the default database attributes for the relevant implementation
    /// The ownership of the database manager instance is shared
    IDatabasePtr createDatabase( const DatabaseId& dbId ) const;


    /*
    /// Create a new database and return the corresponding manager
    /// Specify non-default database attributes in an attribute list
    /// The ownership of the database manager instance is shared
    IDatabasePtr createDatabase( const DatabaseId& dbId,
                                 const coral::AttributeList& dbAttr ) const;
    */

    /// Open an existing database and return the corresponding manager
    /// The ownership of the database manager instance is shared
    IDatabasePtr openDatabase( const DatabaseId& dbId,
                               bool readOnly ) const;

    /// Drop an existing database.
    bool dropDatabase( const DatabaseId& dbId ) const;

    /// "Refresh" an existing database: drop all folders and associated tables,
    /// but simply delete rows from global tables, without dropping them.
    /// Keep and refresh nodes if the flag is true, else drop them.
    /// *** WARNING: this is to be used for the unit tests only! ***
    void refreshDatabase( const DatabaseId& dbId,
                          bool keepNodes = false ) const;

    /// Retrieve the version number of the database service software.
    const std::string serviceVersion() const;

    /// Get the CORAL connection service used by the application.
    coral::IConnectionService& connectionSvc() const;

    /// Get the CORAL connection service used by the application.
    CoralConnectionServiceProxyPtr ppConnectionSvc() const
    {
      return m_ppConnSvc;
    };

  private:

    /// Initialize the service
    void initialize();

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Shared pointer to the CORAL connection service pointer.
    /// When the database service is deleted, this points to a null pointer.
    CoralConnectionServiceProxyPtr m_ppConnSvc;

  };

}

#endif // RELATIONALCOOL_RALDATABASESVC_H
