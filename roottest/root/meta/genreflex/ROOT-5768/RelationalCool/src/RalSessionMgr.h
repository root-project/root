// $Id: RalSessionMgr.h,v 1.35 2012-03-07 14:17:28 avalassi Exp $
#ifndef RELATIONALCOOL_RALSESSIONMGR_H
#define RELATIONALCOOL_RALSESSIONMGR_H

// Include files
#include <memory>
#include <boost/shared_ptr.hpp>
#include "CoolKernel/DatabaseId.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/AccessMode.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/ISchema.h"
#include "RelationalAccess/ISessionProxy.h"
#include "RelationalAccess/ISessionProperties.h"

// Local include files
#include "CoralConnectionServiceProxy.h"
#include "ISessionMgr.h"
#include "RelationalDatabaseId.h"

namespace cool
{

  std::string RalConnectionString( const RelationalDatabaseId& dbId );

  /** @class RalSessionMgr RalSessionMgr.h
   *
   *  Manager of relational database connections via a RAL session,
   *  implementing the abstract ISessionMgr interface.
   *
   *  This class knows nothing about COOL tables.
   *  It is only concerned with relational database connections.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-10-24
   */

  class RalSessionMgr : public ISessionMgr
  {

  public:

    /// The constructor automatically connects to the database.
    RalSessionMgr( CoralConnectionServiceProxyPtr ppConnSvc,
                   const DatabaseId& dbId,
                   bool readOnly );

    /// The destructor automatically disconnects from the database.
    virtual ~RalSessionMgr();

    /// Return the server technology for the current connection.
    /// Supported technologies: "Oracle", "MySQL", "SQLite", "frontier".
    /// This ultimately corresponds to coral::IDomain::flavorName()
    /// (grep m_flavorName in the four XxxAccess/src/Domain.cpp),
    /// because it is equal to ConnectionHandle::technologyName(),
    /// which is equal to ConnectionParams::technologyName(), which is
    /// set to IDomain::flavorName() in ReplicaCatalogue::getReplicas.
    /// *** WARNING!!! THIS MAY CHANGE IN LATER VERSIONS OF THE CODE!!! ***
    /// New (not for production!): for URLs using a middle tier, this method
    /// returns the properties of the remote database, not of the middle tier
    const std::string databaseTechnology() const;

    /// Return the server technology version for the current connection.
    /// This ultimately corresponds to coral::IConnection::serverVersion()
    /// (grep serverVersion in the four XxxAccess/src/Connection.cpp),
    /// because it is equal to ConnectionHandle::serverVersion(),
    /// which is equal to IConnection::serverVersion().
    /// *** WARNING!!! THIS MAY CHANGE IN LATER VERSIONS OF THE CODE!!! ***
    /// New (not for production!): for URLs using a middle tier, this method
    /// returns the properties of the remote database, not of the middle tier
    const std::string serverVersion() const;

    /// Return the schema name for the current connection.
    const std::string schemaName() const
    {
      return m_session->nominalSchema().schemaName();
    }

    /// Return the connection state of the database.
    /// [Note that this is subtly different from RelationalDatabase::isOpen!]
    bool isConnected() const;

    /// (Re)connect to the database.
    void connect();

    /// Close the database connection.
    void disconnect();

    /// Get a reference to the RAL database session.
    /// Throw an exception if there is no connection to the database.
    coral::ISessionProxy& session() const;

    /// Required access mode to the database.
    bool isReadOnly() const
    {
      return ( m_sessionMode==ReadOnly || m_sessionMode==ReadOnlyManyTx );
    }

    /// Required access mode to the database.
    bool isReadOnlyManyTx() const
    {
      return ( m_sessionMode==ReadOnlyManyTx );
    }

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const;

  private:

    /// Standard constructor is private
    RalSessionMgr();

    /// Copy constructor is private
    RalSessionMgr( const RalSessionMgr& rhs );

    /// Assignment operator is private
    RalSessionMgr& operator=( const RalSessionMgr& rhs );

  private:

    /// Get a reference to the CORAL connection service.
    coral::IConnectionService& connectionSvc() const;

    /// Shared pointer to the CORAL connection service pointer.
    /// When the database service is deleted, this points to a null pointer.
    CoralConnectionServiceProxyPtr m_ppConnSvc;

    /// Global identifier of the database
    RelationalDatabaseId m_relationalDbId;

    /// SEAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// RAL session (owned by this instance) connected to the database
    coral::ISessionProxy* m_session;

    /// Session access mode (see bug #90949)
    typedef enum { ReadOnly, Update, ReadOnlyManyTx } SessionMode;

    /// Required access mode to the database
    SessionMode m_sessionMode;

  };

}

#endif // RELATIONALCOOL_RALSESSIONMGR_H
