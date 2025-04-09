// $Id: RelationalTransaction.h,v 1.10 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTRANSACTION_H
#define RELATIONALCOOL_RELATIONALTRANSACTION_H

// Include files
#include <boost/shared_ptr.hpp>

namespace cool
{

  // Forward declarations
  class IRelationalTransactionMgr;

  /** @class RelationalTransaction RelationalTransaction.h
   *
   *  Generic implementation of a relational database transaction.
   *
   *  A transaction is started when this class is instantiated.
   *
   *  An explicit commit() call is necessary for committing the transaction.
   *  When the instance goes out of scope, the transaction is rolled back.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2006-03-10
   */

  class RelationalTransaction
  {

  public:

    /// Destructor
    virtual ~RelationalTransaction();

    /// Constructor from a IRelationalTransactionMgr
    RelationalTransaction
    ( const boost::shared_ptr<IRelationalTransactionMgr>& transactionMgr,
      bool readOnly = false );

    /// Commit the transaction
    void commit();

    /// Rollback the transaction
    void rollback();

  private:

    /// Standard constructor is private
    RelationalTransaction();

    /// Copy constructor is private
    RelationalTransaction( const RelationalTransaction& rhs );

    /// Assignment operator is private
    RelationalTransaction& operator=( const RelationalTransaction& rhs );

  private:

    /// Handle to the IRelationalTransactionMgr (shared ownership)
    boost::shared_ptr<IRelationalTransactionMgr> m_transactionMgr;

  };

} // namespace

#endif
