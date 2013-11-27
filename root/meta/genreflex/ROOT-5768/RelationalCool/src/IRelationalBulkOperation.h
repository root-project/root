#ifndef RELATIONALCOOL_IRELATIONALBULKOPERATION_H
#define RELATIONALCOOL_IRELATIONALBULKOPERATION_H

namespace cool {

  /**
   * Class IRelationalBulkOperation
   *
   * Abstract interface for the execution of bulk operations.
   */

  class IRelationalBulkOperation {

  public:

    /// Destructor
    virtual ~IRelationalBulkOperation() {}

    /// Processes the next iteration
    virtual void processNextIteration() = 0;

    /// Flushes the data on the client side to the server.
    virtual void flush() = 0;

    /// Get the row cache size associated to this bulk operation
    /// (the bulk operation is auto-flushed if this is exceeded)
    virtual int rowCacheSize() const = 0;

  };

}
#endif
