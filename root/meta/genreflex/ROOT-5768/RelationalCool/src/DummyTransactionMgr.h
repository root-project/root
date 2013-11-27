// $Id: DummyTransactionMgr.h,v 1.10 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_DUMMYTRANSACTIONMGR_H
#define RELATIONALCOOL_DUMMYTRANSACTIONMGR_H

// Local include files
#include "IRelationalTransactionMgr.h"
#include "RelationalException.h"

namespace cool {

  /** @class RelationalTransactionMgr RelationalTransactionMgr.h
   *
   *  Dummy manager of relational database transactions.
   *
   *  This manager is only used in read-only connections: the only non-dummy
   *  operation it performs is to throw if a read-write transaction is open!
   *
   *  @author Andrea Valassi
   *  @date   2007-03-28
   */

  class DummyTransactionMgr : public IRelationalTransactionMgr
  {

  public:

    /// Destructor
    virtual ~DummyTransactionMgr();

    /// Standard constructor
    DummyTransactionMgr();

    void setAutoTransactions( bool ) {}
    bool autoTransactions() const { return true; }

  protected:

    /// Start a transaction
    void start( bool readOnly )
    {
      if ( !readOnly )
        throw DatabaseOpenInReadOnlyMode( "DummyTransactionManager" );
      m_isActive = true;
    };

    /// Commit a transaction
    void commit()
    {
      m_isActive = false;
    };

    /// Rollback a transaction
    void rollback()
    {
      m_isActive = false;
    };

    /// Is the transaction active?
    bool isActive()
    {
      return m_isActive;
    };

  private:

    /// Is the transaction active?
    bool m_isActive;

  };

}

#endif // RELATIONALCOOL_DUMMYTRANSACTIONMGR_H
