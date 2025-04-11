// $Id: RelationalObjectIterator.h,v 1.22 2012-01-30 17:42:51 avalassi Exp $
#ifndef COOLKERNEL_RELATIONALOBJECTITERATOR_H
#define COOLKERNEL_RELATIONALOBJECTITERATOR_H

// Include files
#include <boost/shared_ptr.hpp>
#include <memory>
#include "CoolKernel/ChannelSelection.h"
#include "CoolKernel/IObjectIterator.h"
#include "CoolKernel/FolderVersioning.h"

// Local include files
#include "ConstRelationalObjectAdapter.h"
#include "RelationalPayloadQuery.h"

namespace cool
{

  // Forward declarations
  class IRecordSelection;
  class IRelationalCursor;
  class IRelationalQueryDefinition;
  class IRelationalTransactionMgr;
  class ISessionMgr;
  class RelationalFolder;
  class RelationalObjectIteratorTest;
  class RelationalObjectTable;
  class RelationalQueryMgr;
  class RelationalTagMgr;

  /** @class RelationalObjectIterator RelationalObjectIterator.h
   *
   *  RAL implementation of an object iterator.
   *
   *  The iterator can be used only ONCE to forward-iterate over a result set.
   *
   *  During its whole lifetime, the iterator is associated to a read-only RAL
   *  transaction (started in the constructor), a RAL query (created in the
   *  constructor) and a RAL cursor reference (obtained by processing this
   *  query in the constructor). All these resources are released in the
   *  destructor by a call to the close() method, which commits and deletes
   *  the transaction, and deletes the query. The close() method can also be
   *  called explicitly in the user code to release the resources: this moves
   *  the iterator to the Closed state, from which it cannot be reused.
   *
   *  For perfomance reasons the iterator doesn't query the number of rows,
   *  unless size() or isEmpty() are called.
   *
   *  @author Andrea Valassi
   *  @date   2007-03-29
   */

  class RelationalObjectIterator : public IObjectIterator 
  {

    // Test class needs acces to some private methods
    friend class cool::RelationalObjectIteratorTest;

  public:

    /// Destructor
    virtual ~RelationalObjectIterator();

    /// Constructor from all relevant query parameters
    RelationalObjectIterator
    ( const boost::shared_ptr<ISessionMgr>& sessionMgr,
      const RelationalQueryMgr& queryMgr,
      const boost::shared_ptr<IRelationalTransactionMgr>& transactionMgr,
      const RelationalTagMgr& tagMgr,
      const RelationalFolder& folder,
      const ValidityKey& since,
      const ValidityKey& until,
      const ChannelSelection& channels,
      const std::string& tagName,
      bool isUserTag,
      const IRecordSelection* payloadQuery = 0,
      const bool countOnly = false );

    /// Does the iterator have zero objects in the loop?
    bool isEmpty();

    /// Fetch the next object in the iterator loop.
    /// Return false if there is no next object.
    bool goToNext();

    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    const IObject& currentRef();

    /// Returns the 'length' of the iterator
    unsigned int size();

    /// Returns all objects in the iterator as a vector.
    /// Throws an exception if goToNext() has already retrieved one object:
    /// this method can only be called INSTEAD of the loop using goToNext().
    const IObjectVectorPtr fetchAllAsVector();

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    void close()
    {
      bool rollback = false;
      close( rollback );
    }

    /// Fetch the next object from the CORAL cursor
    bool fetchNext();

  private:

    /// Prefetch the number of rows that the iterator will return
    unsigned int getSize( const ValidityKey& since,
                          const ValidityKey& until,
                          const ChannelSelection& channels,
                          const std::string& tagName,
                          bool isUserTag,
                          const IRecordSelection* payloadQuery = 0 ) const;

    /// Get the appropriate query definition for this iterator
    std::auto_ptr<IRelationalQueryDefinition>
    getQueryDefinition( const ValidityKey& since,
                        const ValidityKey& until,
                        const ChannelSelection& channels,
                        const std::string& tagName,
                        bool isUserTag,
                        const IRecordSelection* payloadQuery = 0 );

    /// Is timing active? (hack to activate it at beginning of ctor)
    bool isTimingActive() const;

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    /// Commit/rollback the open transaction in case of success/failure.
    void close( bool rollback );


    /// returns true if fetching CLOBs is optimized by fetching them
    /// as char 4000
    bool optimizeClobs();

  private:

    /// Is timing active? (hack to activate it at beginning of ctor)
    bool m_isTimingActive;

    /// The IRelationalTransactionMgr referenced in the iterator counter
    /// NB This is only used as an index - you don't even need the header!
    const IRelationalTransactionMgr* m_transactionMgr;

    /// Is this iterator registered? (hack to do it at beginning of ctor)
    bool m_isRegistered;

    /// Handle to the ISessionMgr (shared ownership)
    boost::shared_ptr<ISessionMgr> m_sessionMgr;

    /// Relational query manager
    std::auto_ptr<RelationalQueryMgr> m_queryMgr;

    /// Object table for the relevant folder
    std::auto_ptr<RelationalObjectTable> m_objectTable;

    /// Versioning mode for the relevant folder
    FolderVersioning::Mode m_versioningMode;

    /// Data buffer used by the active cursor
    boost::shared_ptr<coral::AttributeList> m_dataBuffer;

    /// details about the query, needed to fetch the size of the vector
    /// if demanded.
    const ValidityKey m_since;
    const ValidityKey m_until;
    const ChannelSelection m_channels;
    const std::string m_tagName;
    bool m_isUserTag;

    /// Record selection for this iterator
    std::auto_ptr<IRecordSelection> m_selection;

    /// Payload query for this iterator
    std::auto_ptr<RelationalPayloadQuery> m_pq;

    /// Relational query definition
    std::auto_ptr<IRelationalQueryDefinition> m_queryDef;

    /// Is m_size known (it is queried on demand only)
    bool m_sizeKnown;

    /// number of rows that the iterator will return
    /// queried on demand by the size() method
    unsigned int m_size;

    /// FSM - current object in RelationalObjectIterator private cache
    unsigned int m_currentObject;

    /// Active cursor
    std::auto_ptr<IRelationalCursor> m_cursor;

    /// Adapter of const AttributeList& currentRow() to the IObject interface.
    std::auto_ptr<ConstRelationalObjectAdapter> m_currentRowAdapter;

    /// State of the iterator (active, end of rows, closed; or countonly)
    enum { ACTIVE, END_OF_ROWS, CLOSED, COUNTONLY } m_state;

    /// folder payload mode
    PayloadMode::Mode m_payloadMode;

  };

}
#endif
