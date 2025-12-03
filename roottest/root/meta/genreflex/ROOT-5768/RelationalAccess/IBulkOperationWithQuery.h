#ifndef IBULKOPERATIONWITHQUERY_H
#define IBULKOPERATIONWITHQUERY_H

#include "IBulkOperation.h"

namespace coral {

  // forward declarations
  class IQueryDefinition;

  /**
   * Class IBulkOperationWithQuery
   * Interface performing bulk DML operations where a query is involved.
   */
  class IBulkOperationWithQuery : virtual public IBulkOperation {
  public:
    /// Destructor
    ~IBulkOperationWithQuery() override {}

    /**
     * Returns a reference to the underlying query definition,
     * so that it can be filled-in by the client.
     */
    virtual IQueryDefinition& query() = 0;
  };

}

#endif
