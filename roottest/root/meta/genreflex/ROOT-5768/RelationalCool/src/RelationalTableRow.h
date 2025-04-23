// $Id: RelationalTableRow.h,v 1.13 2010-03-30 10:41:05 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTABLEROW_H
#define RELATIONALCOOL_RELATIONALTABLEROW_H

// Local include files
#include "RelationalTableRowBase.h"

namespace cool 
{

  /** @class RelationalTableRow RelationalTableRow.h
   *
   *  Concrete implementation of a generic relational table row.
   *
   *  This is the generic relational table row created by fetch statements.
   *  It is the default implementation of a RelationalTableRowBase.
   *
   *  This class can be 'reinterpreted' as rows of specific tables by
   *  defining a derived class FooTableRow with one constructor from a
   *  RelationalTableRow. This class mainly exists to avoid constructing a
   *  FooTableRow from an instance of its base class RelationalTableRowBase.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2005-10-12
   */

  class RelationalTableRow : public RelationalTableRowBase 
  {

  public:

    /// Destructor.
    ~RelationalTableRow() override;

    /// Standard constructor.
    /// Creates a table row with empty AttributeList data.
    RelationalTableRow();

    /// Copy constructor.
    /// Performs a deep copy of the AttributeList values.
    RelationalTableRow( const RelationalTableRow& rhs );

    /// Assignment operator.
    /// Performs a deep copy of the AttributeList values.
    RelationalTableRow& operator=( const RelationalTableRow& rhs );

    /// Constructor from an AttributeList.
    /// Performs a deep copy of the AttributeList values.
    explicit RelationalTableRow( const coral::AttributeList& data );

  };

}
#endif // RELATIONALCOOL_RELATIONALTABLEROW_H
