// $Id: IRelationalTransactionMgr.h,v 1.6 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_IRELATIONALTRANSACTIONMGR_H
#define RELATIONALCOOL_IRELATIONALTRANSACTIONMGR_H

namespace cool {

  /** @class IRelationalTransactionMgr IRelationalTransactionMgr.h
   *
   *  Abstract interface for a manager of relational database transactions.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-10
   */

  class IRelationalTransactionMgr {

    friend class RelationalTransaction;

  public:

    /// Destructor
    virtual ~IRelationalTransactionMgr() {};

    /// Is the transaction active?
    virtual bool isActive() = 0;

    virtual void setAutoTransactions( bool flag ) = 0;
    virtual bool autoTransactions() const = 0;

    //protected:

    /// Start a transaction
    virtual void start( bool readOnly ) = 0;

    /// Commit a transaction
    virtual void commit() = 0;

    /// Rollback a transaction
    virtual void rollback() = 0;

  };

}

#endif // RELATIONALCOOL_RELATIONALTRANSACTIONMGR_H
