// $Id: AttributeTable.h,v 1.7 2009-12-16 17:17:36 avalassi Exp $
#ifndef RELATIONALCOOL_ATTRIBUTETABLE_H
#define RELATIONALCOOL_ATTRIBUTETABLE_H 1

// Include files
#include "AttributeList/AttributeList.h"
#include "AttributeList/AttributeListSpecification.h"

namespace cool {

  /** @class AttributeTable AttributeTable.h
   *
   *  Implementation of a vector of AttributeList's
   *  with a single AttributeListSpecification.
   *
   *  Each AttributeList is available to the user only as a reference.
   *  The lifetime of the references coincides with that of the AttributeTable.
   *
   *  @author Andrea Valassi
   *  @date   2005-02-01
   */

  class AttributeTable {

  public:

    /// Constructor from an AttributeListSpecification reference
    AttributeTable( const coral::AttributeListSpecification& spec ) {
      m_spec = new coral::AttributeListSpecification();
      coral::AttributeListSpecification::const_iterator itSpec;
      for ( itSpec = spec.begin(); itSpec != spec.end(); ++itSpec ) {
        m_spec->push_back( itSpec->name(), itSpec->type_name() );
      }
    }

    /// Destructor
    virtual ~AttributeTable() {
      std::vector< coral::AttributeList* >::const_iterator itList;
      for ( itList = m_lists.begin();
            itList != m_lists.end();
            itList++ ) {
        delete (*itList);
      }
      delete m_spec;
    }

    /// Insert a new row from an AttributeList reference
    /// Throws a RelationalException if a specification is invalid.
    void push_back( const coral::AttributeList& list ) {
      if ( list.attributeListSpecification() != *m_spec )
        throw RelationalException
          ( "Invalid specification", "AttributeTable" );
      coral::AttributeList* newList( new coral::AttributeList( *m_spec ) );
      coral::AttributeList::const_iterator itList;
      for ( itList=list.begin(); itList!=list.end(); ++itList ) {
        (*newList)[ itList->spec().name() ].shareData( *itList );
      }
      return m_lists.size();
    }

    /// Retrieve a row as an AttributeList reference
    coral::AttributeList& operator[] ( unsigned int iRow ) {
      return *(m_lists[iRow]);
    }

    /// Number of rows in the table (number of AttributeList's)
    unsigned int size() const {
      return m_lists.size();
    }

  private:

    /// Standard constructor is private
    AttributeTable();

    /// Copy constructor is private
    AttributeTable( const AttributeTable& rhs );

    /// Assignment operator is private
    AttributeTable& operator=( const AttributeTable& rhs );

  private:

    /// The attributeListSpecification for the AttributeTable
    const coral::AttributeListSpecification* m_spec;

    /// The vector of AttributeList's
    std::vector< coral::AttributeList* > m_lists;

  };

}

#endif // RELATIONALCOOL_ATTRIBUTETABLE_H
