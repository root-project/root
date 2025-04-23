// $Id: CoralConnectionServiceProxy.h,v 1.6 2010-12-21 15:35:29 avalassi Exp $
#ifndef CORALCONNECTIONSERVICEPROXY_H
#define CORALCONNECTIONSERVICEPROXY_H 1

// Include files
#include <iostream>
#include <boost/shared_ptr.hpp>
#include "CoralBase/boost_thread_headers.h"
#include "RelationalAccess/IConnectionService.h"

namespace cool
{

  /** @class CoralConnectionServicePtr CoralConnectionServicePtr.h
   *
   *  Sharable thread-safe proxy to a coral::IConnectionService.
   *
   *  @author Andrea Valassi
   *  @date   2008-04-10
   */

  class CoralConnectionServiceProxy : virtual public coral::IConnectionService
  {

  public:

    /// Constructor
    CoralConnectionServiceProxy( coral::IConnectionService* pConnSvc = 0 );

    /// Destructor
    ~CoralConnectionServiceProxy() override;

    /// Get the pointer - throws if the pointer is 0.
    const coral::IConnectionService* getICS() const;

    /// Get the pointer - throws if the pointer is 0.
    coral::IConnectionService* getICS();

    /// Reset the pointer.
    void resetICS( coral::IConnectionService* pConnSvc = 0 )
    {
      //std::cout << "CoralConnectionServiceProxy::resetICS" << std::endl;
      boost::mutex::scoped_lock lock( m_mutex );
      m_pConnSvc = pConnSvc;
    }

    /**
     * Returns a session proxy object for the specified connection string
     * and access mode.
     */
    coral::ISessionProxy* connect( const std::string& connectionName,
                                   coral::AccessMode accessMode = coral::Update ) override
    {
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->connect( connectionName, accessMode );;
    }

    /**
     * Returns a session proxy object for the specified connection string, role
     * and access mode.
     */
    coral::ISessionProxy* connect( const std::string& connectionName,
                                   const std::string& asRole,
                                   coral::AccessMode accessMode = coral::Update ) override
    {
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->connect( connectionName, asRole, accessMode );
    }

    /**
     * Returns the configuration object for the service.
     */
    coral::IConnectionServiceConfiguration& configuration() override
    {
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->configuration();
    }

    /**
     * Cleans up the connection pool from the unused connection, according to
     * the policy defined in the configuration.
     */
    void purgeConnectionPool() override
    {
      //std::cout << "CoralConnectionServiceProxy::purgeConnectionPool" << std::endl;
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->purgeConnectionPool();
    }

    /**
     * Returns the monitoring reporter
     */
    const coral::IMonitoringReporter& monitoringReporter() const override
    {
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->monitoringReporter();
    }

    /**
     * Returns the object which controls the web cache
     */
    coral::IWebCacheControl& webCacheControl() override
    {
      boost::mutex::scoped_lock lock( m_mutex );
      return getICS()->webCacheControl();
    }

  private:

    /// The coral::IConnectionService pointer.
    coral::IConnectionService* m_pConnSvc;

    /// The mutex lock (mutable because it must be modified in const methods).
    mutable boost::mutex m_mutex;

  };

  // Type definition
  typedef boost::shared_ptr<CoralConnectionServiceProxy> CoralConnectionServiceProxyPtr;

}
#endif // CORALCONNECTIONSERVICEPTR_H
