// $Id: RelationalPayloadQuery.h,v 1.10 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALPAYLOADQUERY_H
#define RELATIONALCOOL_RELATIONALPAYLOADQUERY_H

// Include files
//#include <memory>
#include "CoolKernel/CompositeSelection.h"
#include "CoolKernel/FieldSelection.h"
#include "CoolKernel/IRecordSelection.h"
#include "CoolKernel/Record.h"

namespace cool {

  /** @class RelationalPayloadQuery RelationalPayloadQuery.h
   *
   *  @author Andrea Valassi, Martin Wache
   *  @date   2008-07-29
   */

  // AV Inheritance is probably not needed here?
  class RelationalPayloadQuery //: virtual public IRecordSelection
  {

  public:

    /// Destructor
    virtual ~RelationalPayloadQuery();

    /// Constructor from an IRecordSelection.
    /// Parameter tableName should already contain the trailing '.' so that it
    /// can be prepended directly to column names (e.g. 'table.' + 'column').
    RelationalPayloadQuery( const IRecordSelection& selection,
                            const std::string& tableName = "",
                            const std::string& technology = "" );

    /*
    /// Can the selection be applied to a record with the given specification?
    bool canSelect( const IRecordSpecification& spec ) const
    {
      return m_selection->canSelect( spec );
    }

    /// Apply the selection to the given record.
    bool select( const IRecord& record ) const
    {
      return m_selection->canSelect( spec );
    }

    /// Clone the record selection (and any objects referenced therein).
    IRecordSelection* clone() const;
    */

    /// Is this payload query trusted?
    bool isTrusted() const
    {
      return m_isTrusted;
    }

    /// Return the WHERE clause.
    /// Throw an exception if this payload query is not trusted.
    const std::string& whereClause() const;

    /// Return the WHERE data (bind variables).
    /// Throw an exception if this payload query is not trusted.
    const IRecord& whereData() const;

  private:

    /// Copy constructor is private
    RelationalPayloadQuery( const RelationalPayloadQuery& rhs );

    /// Assignment operator is private
    RelationalPayloadQuery& operator=( const RelationalPayloadQuery& rhs );

    /// Bind variable name for the i-th bind variable (i>0)
    const std::string bindVariableName( unsigned ibv ) const;

    /// Add WHERE clause and WHERE data for a generic IRecordSelection.
    /// The return value (true on success, false on failure)
    /// indicates whether the input record selection can be trusted.
    /// Keep track of how many variables have been used already.
    bool addSelection( const IRecordSelection& sel,
                       std::string& whereClause,
                       Record& whereData ) const;

    /// Add WHERE clause and WHERE data for a FieldSelection.
    /// The return value (true on success, false on failure)
    /// indicates whether the input record selection can be trusted.
    /// Keep track of how many variables have been used already.
    bool addFieldSelection( const FieldSelection& sel,
                            std::string& whereClause,
                            Record& whereData ) const;

    /// Add WHERE clause and WHERE data for a CompositeSelection.
    /// The return value (true on success, false on failure)
    /// indicates whether the input record selection can be trusted.
    /// Keep track of how many variables have been used already.
    bool addCompositeSelection( const CompositeSelection& sel,
                                std::string& whereClause,
                                Record& whereData ) const;

  private:

    /*
    // AV The clone is probably not needed here with no inheritance?
    /// Private clone of the user-supplied record selection
    std::auto_ptr<IRecordSelection> m_selection;
    */

    /// Table name
    const std::string m_tableName;

    /// Technology
    const std::string m_technology;

    /// Is this payload query truested?
    bool m_isTrusted;

    /// The WHERE clause (SQL fragment)
    std::string m_whereClause;

    /// The WHERE data (bind variables)
    Record m_whereData;

  };

}

#endif // RELATIONALCOOL_RELATIONALPAYLOADQUERY_H
