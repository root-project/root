#ifndef RELATIONALACCESS_IVIEWFACTORY_H
#define RELATIONALACCESS_IVIEWFACTORY_H

#include "IQueryDefinition.h"

namespace coral {

  // forward declarations
  class IView;

  /**
   * Class IViewFactory
   *
   * Interface for the definition and creation of new views in the current schema.
   */
  class IViewFactory : virtual public IQueryDefinition {
  public:
    /// Destructor
    ~IViewFactory() override {}

    /**
     * Creates a new view with the specified
     * name and the current query definition.
     * In case the view already exists a ViewAlreadyExistingException is thrown.
     */
    virtual IView& create( const std::string& viewName ) = 0;

    /**
     * Creates or replaces in case it exists a view with the specified
     * name and the current query definition.
     */
    virtual IView& createOrReplace( const std::string& viewName ) = 0;
  };

}

#endif
