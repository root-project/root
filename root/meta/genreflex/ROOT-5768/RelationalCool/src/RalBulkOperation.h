#ifndef RELATIONALCOOL_RALBULKOPERATION_H
#define RELATIONALCOOL_RALBULKOPERATION_H

// Include files
#include <boost/shared_ptr.hpp>
#include "RelationalAccess/IBulkOperation.h"

// Local include files
#include "IRelationalBulkOperation.h"

namespace cool {

  /**
   * Class RalBulkOperation
   *
   * Wrapper for a coral::IBulkOperation.
   *
   */

  class RalBulkOperation : public IRelationalBulkOperation {

  public:

    /// Destructor
    virtual ~RalBulkOperation() {}

    /// Constructor
    RalBulkOperation( const boost::shared_ptr<coral::IBulkOperation> bulkOp,
                      int rowCacheSize )
      : m_bulkOperation( bulkOp ), m_rowCacheSize( rowCacheSize ) {}

    /// Processes the next iteration
    void processNextIteration()
    {
      m_bulkOperation->processNextIteration();
    }

    /// Flushes the data on the client side to the server.
    void flush()
    {
      m_bulkOperation->flush();
    }

    /// Get the row cache size associated to this bulk operation
    /// (the bulk operation is auto-flushed if this is exceeded)
    int rowCacheSize() const
    {
      return m_rowCacheSize;
    }

  private:

    RalBulkOperation();
    RalBulkOperation( const RalBulkOperation& );
    RalBulkOperation& operator=( const RalBulkOperation& );

  private:

    boost::shared_ptr<coral::IBulkOperation> m_bulkOperation;
    const int m_rowCacheSize;

  };

}
#endif
