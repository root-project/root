// $Id: DatabaseSvcFactory.h,v 1.8 2009-12-16 17:39:59 avalassi Exp $
#ifndef COOLAPPLICATION_DATABASESVCFACTORY_H
#define COOLAPPLICATION_DATABASESVCFACTORY_H 1

namespace cool
{

  // Forward declarations
  class IDatabaseSvc;

  /**********************************************************************
   **  NB: this class should only be used in _STANDALONE_ applications **
   **  that cannot retrieve the IDatabaseSvc as a plugin because they  **
   **  do not directly manage the loading of plugins into contexts.    **
   ***********************************************************************
   *
   *  @class DatabaseSvcFactory DatabaseSvcFactory.h
   *
   *  Factory of standalone DatabaseSvc instances.
   *
   *  This class takes care of loading all plugins needed to run COOL, using
   *  the relevant plugin technology. The implementation details (presently
   *  based on SEAL, later on CORAL or ROOT) are hidden from the public API.
   *
   *  The application used to retrieve the service is a singleton owned by the
   *  class: the same service is returned by successive calls to databaseSvc().
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-08
   */

  class DatabaseSvcFactory {

  public:

    /// Retrieve a reference to the COOL database service.
    static IDatabaseSvc& databaseService();

  };

}

#endif // COOLAPPLICATION_DATABASESVCFACTORY_H
