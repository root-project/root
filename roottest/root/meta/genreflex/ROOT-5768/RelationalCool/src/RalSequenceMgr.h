// $Id: RalSequenceMgr.h,v 1.14 2012-01-30 17:06:03 avalassi Exp $
#ifndef RELATIONALCOOL_RALSEQUENCEMGR_H
#define RELATIONALCOOL_RALSEQUENCEMGR_H

// Local include files
#include "RelationalSequenceMgr.h"

namespace cool 
{

  // Forward declarations
  class ISessionMgr;

  /** @class RalSequenceMgr RalSequenceMgr.h
   *
   *  CORAL implementation of a manager of COOL relational 'sequences'.
   *
   *  Transactions are NOT handled by this class.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-10
   */

  class RalSequenceMgr : public RelationalSequenceMgr
  {

  public:

    /// Constructor from a RelationalQueryMgr and an ISessionMgr
    /// Inlined with the base class non-standard constructor as otherwise
    /// the compiler attempts to use the base class standard constructor
    /// (Windows compilation error C2248: standard constructor is private)
    RalSequenceMgr( const RelationalQueryMgr& queryMgr,
                    const ISessionMgr& sessionMgr )
      : RelationalSequenceMgr( queryMgr )
      , m_sessionMgr( sessionMgr )
    {
      initialize();
    }

    /// Destructor
    ~RalSequenceMgr() override;

    /// Create a new sequence (ownership of the C++ instance is shared).
    boost::shared_ptr<RelationalSequence>
    createSequence( const std::string& name ) override;

    /// Does this sequence exist?
    bool existsSequence( const std::string& name ) override;

    /// Get an existing sequence (ownership of the C++ instance is shared).
    boost::shared_ptr<RelationalSequence>
    getSequence( const std::string& name ) override;

    /// Drop an existing sequence
    void dropSequence( const std::string& name ) override;

    /// Init the sequence (fill with initial values)
    void initSequence( const std::string& name ) override;

  private:

    /// Initialize (complete non-standard constructor with non-inlined code)
    void initialize();

    /// Standard constructor is private
    RalSequenceMgr();

    /// Copy constructor is private
    RalSequenceMgr( const RalSequenceMgr& rhs );

    /// Assignment operator is private
    RalSequenceMgr& operator=( const RalSequenceMgr& rhs );

  private:

    /// Reference to ISessionMgr (owned via shared ptr by parent RalQueryMgr)
    const ISessionMgr& m_sessionMgr;

  };

}

#endif // RELATIONALCOOL_RALSEQUENCEMGR_H
