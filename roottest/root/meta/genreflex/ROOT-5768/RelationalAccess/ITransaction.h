#ifndef RELATIONALACCESS_ITRANSACTION_H
#define RELATIONALACCESS_ITRANSACTION_H

namespace coral {

  /**
   * Class ITransaction
   * Interface for the transaction control in an active session.
   */
  class ITransaction {
  public:
    /**
     * Starts a new transaction.
     * In case of failure a TransactionNotStartedException is thrown.
     *
     */
    virtual void start( bool readOnly = false ) = 0;

    /**
     * Commits the transaction.
     * In case of failure a TransactionNotCommittedException is thrown.
     */
    virtual void commit() = 0;

    /**
     * Aborts and rolls back a transaction.
     */
    virtual void rollback() = 0;

    /**
     * Returns the status of the transaction (if it is active or not)
     */
    virtual bool isActive() const = 0;

    /**
     * Returns the mode of the transaction (if it is read-only or not)
     */
    virtual bool isReadOnly() const = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITransaction() {}
  };

}

#endif
