#ifndef RELATIONALACCESS_IBULKOPERATION_H
#define RELATIONALACCESS_IBULKOPERATION_H

namespace coral {

  /**
   * Class IBulkOperation
   *
   * Interface for the execution of bulk operations.
   */
  class IBulkOperation {
  public:
    /// Destructor
    virtual ~IBulkOperation() {}

    /**
     * Processes the next iteration
     */
    virtual void processNextIteration() = 0;

    /**
     * Flushes the data on the client side to the server.
     */
    virtual void flush() = 0;
  };

}
#endif
