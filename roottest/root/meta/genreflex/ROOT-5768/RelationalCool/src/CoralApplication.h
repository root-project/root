// $Id: CoralApplication.h,v 1.11 2012-06-29 13:19:47 avalassi Exp $
#ifndef RELATIONALCOOL_CORALAPPLICATION_H
#define RELATIONALCOOL_CORALAPPLICATION_H 1

// Include files
#include <memory>
#include "CoolKernel/IApplication.h"
#include "CoralBase/MessageStream.h"

namespace cool
{

  /** @class CoralApplication CoralApplication.h
   *
   *  CORAL-based implementation of a COOL application class.
   *
   *  This class provides a concrete implementation of the public Application
   *  class, while hiding from the public API all CORAL-based details.
   *
   *  @author Andrea Valassi
   *  @date   2007-04-08
   *
   */

  class CoralApplication : public IApplication {

  public:

    /// Constructor from a CORAL ConnectionService (if one is provided,
    /// the user is responsible to keep it alive until the application is
    /// alive; if none is provided, a new own one is created if necessary).
    CoralApplication( coral::IConnectionService* connSvc = 0 );

    /// Destructor
    virtual ~CoralApplication();

    /// Retrieve a reference to the COOL database service.
    IDatabaseSvc& databaseService();

    /// Get the output level threshold for the message service.
    /// *** WARNING: this may actually return a shared (static) value. ***
    MSG::Level outputLevel();

    /// Set the output level threshold for the message service.
    /// *** WARNING: this may actually change a shared (static) value. ***
    void setOutputLevel( MSG::Level level );

    /// Get the SEAL context (if any) associated with this application.
    /// *** WARNING: throws an exception for applications not using SEAL. ***
    seal::Context* context() const;

    /// Get the CORAL connection service (if any) used in this application.
    /// *** WARNING: throws an exception for applications not using CORAL. ***
    coral::IConnectionService& connectionSvc() const;

  private:

    /// Copy constructor is private (fix Coverity MISSING_COPY)
    CoralApplication( const CoralApplication& rhs );

    /// Assignment operator is private (fix Coverity MISSING_ASSIGN)
    CoralApplication& operator=( const CoralApplication& rhs );

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// Handler of SEAL PluginManager feedback
    //static void feedback ( seal::PluginManager::FeedbackData data );

    /// Plugin label for the technology-independent top-level DatabaseService
    //static const std::string& databaseServiceLabel();

  private:

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// CORAL connection service
    coral::IConnectionService* m_connSvc;

    /// Own CORAL connection service?
    bool m_ownConnSvc;

    /// COOL database service
    cool::IDatabaseSvc* m_dbSvc;

  };

}
#endif // RELATIONALCOOL_CORALAPPLICATION_H
