#ifndef RELATIONALCOOL_RALCURSOR_H
#define RELATIONALCOOL_RALCURSOR_H

// Include files
#include <memory>
#include "RelationalAccess/ICursor.h"
#include "RelationalAccess/IQuery.h"

// Local include files
#include "IRelationalCursor.h"
//#include "RalQueryMgr.h" // for TimingReport

namespace cool
{

  /**
   * Class RalCursor
   *
   * Wrapper for a coral::ICursor.
   *
   */

  class RalCursor : public IRelationalCursor {

  public:

    /// Destructor
    ~RalCursor() override {}

    /// Constructor from a CORAL query.
    /// The constructor takes ownership of the query and executes it.
    RalCursor( std::auto_ptr<coral::IQuery> query )
      : m_query( query )
      , m_cursor( m_query->execute() ) {}
    //, m_cursor( RalQueryMgr::executeQuery(*m_query) ) {} // with TimingReport

    /// Positions the cursor to the next available row in the result set.
    /// If there are no more rows in the result set false is returned.
    bool next() override
    {
      return m_cursor.next();
      //return RalQueryMgr::cursorNext( m_cursor ); // with TimingReport
    }

    /// Returns a reference to the output buffer holding the last row fetched.
    const coral::AttributeList& currentRow() const override
    {
      return m_cursor.currentRow();
    }

    /// Explicitly closes the cursor, releasing the resources on the server.
    void close() override
    {
      return m_cursor.close();
    }

  private:

    RalCursor();
    RalCursor( const RalCursor& );
    RalCursor& operator=( const RalCursor& );

  private:

    std::auto_ptr<coral::IQuery> m_query;
    coral::ICursor& m_cursor;

  };

}
#endif
