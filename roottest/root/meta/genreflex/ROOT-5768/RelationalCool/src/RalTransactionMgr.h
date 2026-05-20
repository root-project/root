// $Id: RalTransactionMgr.h,v 1.13 2012-01-30 17:06:03 avalassi Exp $
#ifndef RELATIONALCOOL_RALTRANSACTIONMGR_H
#define RELATIONALCOOL_RALTRANSACTIONMGR_H

// Include files
#include <boost/shared_ptr.hpp>
#include "RelationalAccess/ITransaction.h"

// Local include files
#include "IRelationalTransactionMgr.h"

namespace cool {

  // Forward declarations
  class ISessionMgr;

  /** @class RalTransactionMgr RalTransactionMgr.h
   *
   *  RAL implementation of a manager of relational database transactions.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-10
   */

  class RalTransactionMgr : public IRelationalTransactionMgr {

  public:

    /// Destructor
    ~RalTransactionMgr() override;

    /// Constructor from a RalSessionMgr
    RalTransactionMgr( const boost::shared_ptr<ISessionMgr>& sessionMgr,
                       bool autoTransactions = true );

  private:

    /// Standard constructor is private
    RalTransactionMgr();

    /// Copy constructor is private
    RalTransactionMgr( const RalTransactionMgr& rhs );

    /// Assignment operator is private
    RalTransactionMgr& operator=( const RalTransactionMgr& rhs );

  protected:

    /// Start a transaction
    void start( bool readOnly ) override;

    /// Commit a transaction
    void commit() override;

    /// Rollback a transaction
    void rollback() override;

    /// Is the transaction active?
    bool isActive() override;

    /// Is the transaction read-only?
    bool isReadOnly();

    /// Enable auto-transactions
    void setAutoTransactions( bool flag ) override { m_autoTransactions = flag; }

    /// Are auto-transactions enabled?
    bool autoTransactions() const override { return m_autoTransactions; }

    /// Get the CORAL transaction
    coral::ITransaction& coralTransaction( const std::string& source ) const;

  private:

    /// Handle to the RalSessionMgr (shared ownership)
    boost::shared_ptr<ISessionMgr> m_sessionMgr;

    /// Are auto-transactions enabled?
    bool m_autoTransactions;

  };

}

#endif // RELATIONALCOOL_RALTRANSACTIONMGR_H
