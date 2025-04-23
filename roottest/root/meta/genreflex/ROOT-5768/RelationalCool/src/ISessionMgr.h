// $Id: ISessionMgr.h,v 1.2 2012-03-07 14:17:28 avalassi Exp $
#ifndef RELATIONALCOOL_ISESSIONMGR_H
#define RELATIONALCOOL_ISESSIONMGR_H

// Include files
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/ISessionProxy.h"

// Local include files

namespace cool 
{

  /** @class ISessionMgr ISessionMgr.h
   *
   *  Pure virtual interface to a manager of a relational database session.
   *  
   *  This interface knows nothing about tables specific to COOL.
   *  It is only concerned with generic relational databases functionalities
   *  (like those available through CORAL, which has no predefined schema).
   *
   *  This interface was first added as part of the patch for task #6154.
   *
   *  @author Martin Wache
   *  @date   2010-11-26
   */

  class ISessionMgr 
  {

  public:

    /// The destructor automatically disconnects from the database.
    virtual ~ISessionMgr() {}

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
    virtual const std::string databaseTechnology() const = 0;

    /// Return the server technology version for the current connection.
    /// This ultimately corresponds to coral::IConnection::serverVersion()
    /// (grep serverVersion in the four XxxAccess/src/Connection.cpp),
    /// because it is equal to ConnectionHandle::serverVersion(),
    /// which is equal to IConnection::serverVersion().
    /// *** WARNING!!! THIS MAY CHANGE IN LATER VERSIONS OF THE CODE!!! ***
    /// New (not for production!): for URLs using a middle tier, this method
    /// returns the properties of the remote database, not of the middle tier
    virtual const std::string serverVersion() const = 0;

    /// Return the schema name for the current connection.
    virtual const std::string schemaName() const = 0;

    /// Return the connection state of the database.
    /// [Note that this is subtly different from RelationalDatabase::isOpen!]
    virtual bool isConnected() const = 0;

    /// (Re)connect to the database.
    virtual void connect() = 0;

    /// Close the database connection.
    virtual void disconnect() = 0;

    /// Get a reference to the RAL database session.
    /// Throw an exception if there is no connection to the database.
    virtual coral::ISessionProxy& session() const = 0;

    /// Required access mode to the database.
    virtual bool isReadOnly() const = 0;
    
    /// Required access mode to the database.
    virtual bool isReadOnlyManyTx() const = 0;
    
    /// Get a CORAL MessageStream
    virtual coral::MessageStream& log() const = 0;

  };

}

#endif 
