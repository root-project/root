#ifndef CORALBASE_ATTRIBUTELISTSPECIFICATION_H
#define CORALBASE_ATTRIBUTELISTSPECIFICATION_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include <string>
#include <typeinfo>
#include <vector>
#include <map>

#ifdef __ICC
#pragma warning (push)
#pragma warning (disable: 522)
#endif

namespace coral
{

  class AttributeSpecification;

  /**
   * @class AttributeListSpecification AttributeListSpecification.h CoralBase/AttributeListSpecification.h
   *
   * The specification of an AttributeList
   */
  class AttributeListSpecification
  {
  public:

    /// Constructor
    AttributeListSpecification();

    /// Copy constructor.
    AttributeListSpecification( const AttributeListSpecification& rhs );

    /// Decrements the reference counter
    void release() const;

    /// Appends a new specification
    template<typename T> void extend( const std::string& name );
    void extend( const std::string& name,
                 const std::type_info& type );
    void extend( const std::string& name,
                 const std::string& typeName );

    /// Returns the size of the specification
    size_t size() const;

    /// Equal operator. Compares only types and values. Not the attribute names
    bool operator==( const AttributeListSpecification& rhs ) const;

    /// Comparison operator. Compares only types and values. Not the attribute names
    bool operator!=( const AttributeListSpecification& rhs ) const;

#ifdef CORAL240AL
    /// Does the attribute with the given name exist?
    bool exists( const std::string& name ) const;
#endif

    /// Returns the index of a specification given its name.
    /// If the name is not found it returns -1.
    int index( const std::string& name ) const;

    /// Bracket operator
    const AttributeSpecification& operator[]( size_t index ) const;

    /// Returns the specification given an index
    const AttributeSpecification& specificationForAttribute( int index ) const;

    /// The iterator class
    class const_iterator
    {
    public:
      ~const_iterator() {};
      const_iterator( const const_iterator& rhs );
      const_iterator& operator=( const const_iterator& rhs );
    private:
      friend class AttributeListSpecification;
      const_iterator( std::vector< AttributeSpecification* >::const_iterator theIterator );
    public:
      const AttributeSpecification* operator->() const;
      const AttributeSpecification& operator*() const;
      void operator++();
      bool operator==( const const_iterator& rhs ) const;
      bool operator!=( const const_iterator& rhs ) const;
    private:
      std::vector< AttributeSpecification* >::const_iterator m_iterator;
    };

    /// Returns a forward iterator
    const_iterator begin() const;
    const_iterator end() const;

  protected:

    /// Increments the reference counter only the AttributeList can call this method
    friend class AttributeList;
    void addRef() const;

    /// The destructor is private.
    ~AttributeListSpecification();

    /// No assignment operator
    AttributeListSpecification& operator=( const AttributeListSpecification& );

  private:

    /// The reference counter
    mutable int m_counter;

    /// The specifications
    std::vector< AttributeSpecification* > m_attributeSpecifications;

    /// The map of names to indices
    std::map< std::string, int >  m_mapOfNameToIndex;

  };

}


// Inline methods
template<typename T> void
coral::AttributeListSpecification::extend( const std::string& name )
{
  this->extend( name, typeid(T) );
}


inline size_t
coral::AttributeListSpecification::size() const
{
  return m_attributeSpecifications.size();
}


inline coral::AttributeListSpecification::const_iterator
coral::AttributeListSpecification::begin() const
{
  return coral::AttributeListSpecification::const_iterator( m_attributeSpecifications.begin() );
}


inline coral::AttributeListSpecification::const_iterator
coral::AttributeListSpecification::end() const
{
  return coral::AttributeListSpecification::const_iterator( m_attributeSpecifications.end() );
}


inline
coral::AttributeListSpecification::const_iterator::const_iterator( const coral::AttributeListSpecification::const_iterator& rhs ) :
  m_iterator( rhs.m_iterator )
{
}


inline coral::AttributeListSpecification::const_iterator&
coral::AttributeListSpecification::const_iterator::operator=( const coral::AttributeListSpecification::const_iterator& rhs )
{
#ifdef CORAL240CO
  if ( this == &rhs ) return *this;  // Fix Coverity SELF_ASSIGN
#endif
  m_iterator = rhs.m_iterator;
  return *this;
}


inline
coral::AttributeListSpecification::const_iterator::const_iterator( std::vector< AttributeSpecification* >::const_iterator theIterator ) :
  m_iterator( theIterator )
{
}


inline const coral::AttributeSpecification*
coral::AttributeListSpecification::const_iterator::operator->() const
{
  return *m_iterator;
}


inline const coral::AttributeSpecification&
coral::AttributeListSpecification::const_iterator::operator*() const
{
  return **m_iterator;
}


inline void
coral::AttributeListSpecification::const_iterator::operator++()
{
  m_iterator++;
}


inline bool
coral::AttributeListSpecification::const_iterator::operator==( const coral::AttributeListSpecification::const_iterator& rhs ) const
{
  return m_iterator == rhs.m_iterator;
}


inline bool
coral::AttributeListSpecification::const_iterator::operator!=( const coral::AttributeListSpecification::const_iterator& rhs ) const
{
  return m_iterator != rhs.m_iterator;
}


inline const coral::AttributeSpecification&
coral::AttributeListSpecification::operator[]( size_t index ) const
{
  return this->specificationForAttribute( static_cast<int>( index ) );
}


inline bool
coral::AttributeListSpecification::operator!=( const coral::AttributeListSpecification& rhs ) const
{
  return ( ! ( *this == rhs ) );
}


#ifdef __ICC
#pragma warning (pop)
#endif

#endif
