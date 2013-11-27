// $Id: Application.h,v 1.31 2012-07-08 20:02:32 avalassi Exp $
#ifndef COOLAPPLICATION_APPLICATION_H
#define COOLAPPLICATION_APPLICATION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolApplication/IApplication.h"

namespace cool
{

  /**********************************************************************
   **  NB: this class should only be used in _STANDALONE_ applications **
   **  that cannot retrieve the IDatabaseSvc as a plugin because they  **
   **  do not directly manage the loading of plugins into contexts.    **
   ***********************************************************************
   *
   *  @class Application Application.h
   *
   *  COOL application class.
   *
   *  This class takes care of loading all plugins needed to run COOL, using
   *  the relevant plugin technology. The implementation details (presently
   *  based on SEAL, later on CORAL or ROOT) are hidden from the public API.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2007-01-17
   */

  /// AV 18.01.2007 - do NOT use virtual inheritance here!
  /// For reasons that are still unclear, this makes it impossible to fetch
  /// the COOL database service (the concrete RalDatabaseSvc is loaded and
  /// instantiated, but it cannot be dynamic cast to an IDatabaseSvc...?
  /// AV 05.07.2007 - removed all unnecessary virtual inheritance (task #4879).
  //class Application : virtual public IApplication

  class Application : public IApplication
  {

  public:

    /// Constructor from a CORAL ConnectionService (if one is provided,
    /// the user is responsible to keep it alive until the application is
    /// alive; if none is provided, a new own one is created if necessary).
    Application( coral::IConnectionService* connSvc = 0 );

    /// Destructor.
    virtual ~Application();

    /// Retrieve a reference to the COOL database service.
    IDatabaseSvc& databaseService()
    {
      return m_application->databaseService();
    }

    /// Get the output level threshold for COOL (and CORAL) messages.
    /// *** WARNING: this may actually return a shared (static) value. ***
    MSG::Level outputLevel()
    {
      return m_application->outputLevel();
    }

    /// Set the output level threshold for COOL (and CORAL) messages.
    /// *** WARNING: this may actually change a shared (static) value. ***
    void setOutputLevel( MSG::Level level )
    {
      m_application->setOutputLevel( level );
    }

    /// Get the SEAL context (if any) associated with this application.
    /// *** WARNING: throws an exception for applications not using SEAL. ***
    seal::Context* context() const
    {
      return m_application->context();
    }

    /// Get the CORAL connection service (if any) used in this application.
    /// *** WARNING: throws an exception for applications not using CORAL. ***
    coral::IConnectionService& connectionSvc() const
    {
      return m_application->connectionSvc();
    }

#ifdef COOL290CO
  private:

    /// Copy constructor is private (fix Coverity MISSING_COPY bug #95363)
    Application( const Application& rhs );

    /// Assignment operator is private (fix Coverity MISSING_ASSIGN bug #95363)
    Application& operator=( const Application& rhs );
#endif

  private:

    /// Privately owned implementation class to which all calls are delegated.
    IApplication* m_application;

  };

}
#endif // COOLAPPLICATION_APPLICATION_H
