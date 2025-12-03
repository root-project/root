// $Id: RelationalTableRowBase.h,v 1.3 2010-03-30 11:17:22 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTABLEROWBASE_H
#define RELATIONALCOOL_RELATIONALTABLEROWBASE_H

// Include files
#include "CoralBase/AttributeList.h"
#include "CoralBase/AttributeListSpecification.h"

namespace cool 
{

  /** @class RelationalTableRowBase RelationalTableRowBase.h
   *
   *  Base class for a relational table row.
   *  This class cannot be instantiated as all its constructors are protected.
   *
   *  This class mainly exists because of the known problems with the
   *  coral::AttributeList copy constructor and assignment operator.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2006-03-11
   */

  class RelationalTableRowBase 
  {

  public:

    /*
    /// Returns the row specification.
    const coral::AttributeListSpecification& rowSpecification() const
    {
      return m_data->attributeListSpecification();
    }
    */

    /// Returns the start const iterator for the fields.
    coral::AttributeList::const_iterator begin() const
    {
      return m_data.begin();
    }

    /// Returns the end const iterator for the fields.
    coral::AttributeList::const_iterator end() const
    {
      return m_data.end();
    }

    /// Proxy for the AttributeList [] access operator (non const version).
    coral::Attribute& operator[]( const std::string& name )
    {
      return m_data[name];
    }

    /// Proxy for the AttributeList [] access operator (const version).
    const coral::Attribute& operator[]( const std::string& name ) const
    {
      return m_data[name];
    }

    /// Accessor to the row data.
    const coral::AttributeList& data() const
    {
      return m_data;
    }

  protected:

    /// Destructor.
    virtual ~RelationalTableRowBase();

    /// Standard constructor.
    /// Creates a table row with empty AttributeList data.
    RelationalTableRowBase();

    /// Copy constructor.
    /// Performs a deep copy of the AttributeList values.
    RelationalTableRowBase( const RelationalTableRowBase& rhs );

    /// Constructor from an AttributeList.
    /// Performs a deep copy of the AttributeList values.
    explicit RelationalTableRowBase( const coral::AttributeList& data );

    /// Assignment operator.
    /// Performs a deep copy of the AttributeList values.
    /// [AV 14.03.06 - I would have left this private but this is needed by
    /// vector<RelationalObjectTable>::push_back()... check why!]
    RelationalTableRowBase& operator=( const RelationalTableRowBase& rhs );

  protected:

    // The relational row data.
    coral::AttributeList m_data;

  };

}

#endif // RELATIONALCOOL_RELATIONALTABLEROWBASE_H
