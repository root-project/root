// $Id: RelationalSequenceMgr.h,v 1.16 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALSEQUENCEMGR_H
#define RELATIONALCOOL_RELATIONALSEQUENCEMGR_H

// Include files
#include <memory>
#include <string>
#include <boost/shared_ptr.hpp>
#include "CoralBase/MessageStream.h"

namespace cool {

  // Forward declarations
  class RelationalQueryMgr;
  class RelationalSequence;

  /** @class RelationalSequenceMgr RelationalSequenceMgr.h
   *
   *  Abstract base class for a manager of COOL relational 'sequences'.
   *
   *  Transactions are NOT handled by this class.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-10
   */

  class RelationalSequenceMgr
  {

  public:

    /// Destructor
    virtual ~RelationalSequenceMgr();

    /// Create a new sequence (ownership of the C++ instance is shared).
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<RelationalSequence>
    createSequence( const std::string& name ) = 0;

    /// Does this sequence exist?
    /// PURE VIRTUAL method implemented in subclasses.
    virtual bool existsSequence( const std::string& name ) = 0;

    /// Get an existing sequence (ownership of the C++ instance is shared).
    /// PURE VIRTUAL method implemented in subclasses.
    virtual boost::shared_ptr<RelationalSequence>
    getSequence( const std::string& name ) = 0;

    /// Drop an existing sequence
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void dropSequence( const std::string& name ) = 0;

    /// Init the sequence (fill with initial values)
    /// PURE VIRTUAL method implemented in subclasses.
    virtual void initSequence( const std::string& name ) = 0;

    /// Instantiate a sequence (ownership of the C++ instance is shared).
    boost::shared_ptr<RelationalSequence>
    instantiateSequence( const std::string& name );

    /// Get the RelationalQueryMgr reference
    const RelationalQueryMgr& queryMgr() const { return m_queryMgr; }

  protected:

    /// Constructor from a RelationalQueryMgr reference
    RelationalSequenceMgr( const RelationalQueryMgr& queryMgr );

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const { return *m_log; }

  private:

    /// Standard constructor is private
    RelationalSequenceMgr();

    /// Copy constructor is private
    RelationalSequenceMgr( const RelationalSequenceMgr& rhs );

    /// Assignment operator is private
    RelationalSequenceMgr& operator=( const RelationalSequenceMgr& rhs );

  protected:

    /// Reference to the parent RelationalQueryMgr
    const RelationalQueryMgr& m_queryMgr;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RELATIONALSEQUENCEMGR_H
