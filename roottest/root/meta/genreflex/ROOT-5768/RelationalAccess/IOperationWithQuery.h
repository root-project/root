#ifndef RELATIONALACCESS_IOPERATIONWITHQUERY_H
#define RELATIONALACCESS_IOPERATIONWITHQUERY_H

namespace coral {

  // forward declarations
  class IQueryDefinition;

  /**
   * Class IOperationWithQuery
   * Interface for executing DML operations involving queries.
   */
  class IOperationWithQuery {
  public:
    /// Destructor
    virtual ~IOperationWithQuery() {}

    /**
     * Returns a reference to the underlying query definition,
     * so that it can be filled-in by the client.
     */
    virtual IQueryDefinition& query() = 0;

    /**
     * Executes the operation and returns the number of rows affected.
     */
    virtual long execute() = 0;
  };

}

#endif
