// $Id: IApplication.h,v 1.7 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IAPPLICATION_H
#define COOLKERNEL_IAPPLICATION_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/MessageLevels.h"

// Forward declarations
namespace seal
{
  class Context;
}
namespace coral
{
  class IConnectionService;
}

namespace cool
{

  // Forward declarations
  class IDatabaseSvc;

  /**********************************************************************
   **  NB: this class should only be used in _STANDALONE_ applications **
   **  that cannot retrieve the IDatabaseSvc as a plugin because they  **
   **  do not directly manage the loading of plugins into contexts.    **
   **********************************************************************
   *
   *  @class IApplication IApplication.h
   *
   *  Abstract interface to a COOL application class.
   *
   *  This class takes care of loading all plugins needed to run COOL, using
   *  the relevant plugin technology. The implementation details (presently
   *  based on SEAL, later on CORAL or ROOT) are hidden from the public API.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-17
   */

  class IApplication
  {

  public:

    /// Destructor.
    virtual ~IApplication() {}

    /// Retrieve a reference to the COOL database service.
    virtual IDatabaseSvc& databaseService() = 0;

    /// Get the output level threshold for COOL (and CORAL) messages.
    /// *** WARNING: this may actually return a shared (static) value. ***
    virtual MSG::Level outputLevel() = 0;

    /// Set the output level threshold for COOL (and CORAL) messages.
    /// *** WARNING: this may actually change a shared (static) value. ***
    virtual void setOutputLevel( MSG::Level level ) = 0;

    /// Get the SEAL context (if any) associated with this application.
    /// *** WARNING: throws an exception for applications not using SEAL. ***
    virtual seal::Context* context() const = 0;

    /// Get the CORAL connection service (if any) used in this application.
    /// *** WARNING: throws an exception for applications not using CORAL. ***
    virtual coral::IConnectionService& connectionSvc() const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IApplication& operator=( const IApplication& rhs );
#endif

  };

}
#endif // COOLKERNEL_IAPPLICATION_H
