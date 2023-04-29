#ifndef CORALBASE_ATTRIBUTELIST_H
#define CORALBASE_ATTRIBUTELIST_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include <string>
#include <vector>
#include <typeinfo>
#include <iosfwd>

namespace coral
{

  // forward declarations
  class Attribute;
  class AttributeSpecification;
  class AttributeListSpecification;

  /**
   * @class AttributeList AttributeList.h CoralBase/AttributeList.h
   *
   *  A simple container of Attributes
   */

  class AttributeList
  {
  public:

    /// Default constructor
    AttributeList();

    /// Constructor with specification
    AttributeList( const AttributeListSpecification& spec,
                   bool sharedSpecification = false );

    /// Copy constructor. Copies specification and data.
    AttributeList( const AttributeList& rhs );

    /// Destructor
    ~AttributeList();

    /// Assignment operator. Copies specification and data.
    AttributeList& operator=( const AttributeList& rhs );

    /// Copies the data of an other attribute list. Performs type checking. The rhs can have longer specification. The first elements are used.
    void copyData( const AttributeList& rhs );

    /// Copies the data of an other attribute list without type checking. The rhs can have longer specification. The first elements are used.
    void fastCopyData( const AttributeList& rhs );

#ifdef CORAL240AS
    /// Returns the specification of the attribute list (bug #100873).
    const AttributeListSpecification& specification() const;
#endif

    /// Extends the AttributeList by one attribute, given a specification.
    /// In case the specification is shared, a copy is triggered.
    template<typename T> void extend( const std::string& name );
    void extend( const std::string& name,
                 const std::string& typeName );
    void extend( const std::string& name,
                 const std::type_info& type );

    /// Returns the size of the AttributeList
    size_t size() const;

    /// Equal operator. Compares only types and values. Not the attribute names
    bool operator==( const AttributeList& rhs ) const;

    /// Comparison operator. Compares only types and values. Not the attribute names
    bool operator!=( const AttributeList& rhs ) const;

#ifdef CORAL240AL
    /// Does the attribute with the given name exist (task #20089)?
    bool exists( const std::string& name ) const;
#endif

    /// Returns a reference to an attribute given its name
    Attribute& operator[]( const std::string name );
    const Attribute& operator[]( const std::string name ) const;

    /// Returns a reference to an attribute given its index
    Attribute& operator[]( unsigned int index );
    const Attribute& operator[]( unsigned int index ) const;

    /// Merges into and shares the contents of another attribute list
    AttributeList& merge( const AttributeList& rhs );

    /// Dumps output to an output stream
    std::ostream& toOutputStream( std::ostream& os ) const;


    class iterator_base
    {
    protected:
      iterator_base( std::vector< Attribute* >::const_iterator theIterator ) : m_iterator( theIterator ) {}
      iterator_base( const iterator_base& rhs ) : m_iterator( rhs.m_iterator ) {}
#ifdef CORAL240CO
      iterator_base& operator=( const iterator_base& rhs ) { if ( this == &rhs ) return *this; m_iterator = rhs.m_iterator; return *this; } // Fix Coverity SELF_ASSIGN
#else
      iterator_base& operator=( const iterator_base& rhs ) { m_iterator = rhs.m_iterator; return *this; }
#endif
      virtual ~iterator_base() {}
      std::vector< Attribute* >::const_iterator m_iterator;
    public:
      bool operator==( const iterator_base& rhs ) const { return m_iterator == rhs.m_iterator; }
      bool operator!=( const iterator_base& rhs ) const { return m_iterator != rhs.m_iterator; }
      void operator++() { ++m_iterator; }
    };


    class iterator : public iterator_base
    {
    public:
      ~iterator() override {};
      iterator( const iterator_base& rhs ) : iterator_base( rhs ) {}
      iterator& operator=( const iterator_base& rhs ) { iterator_base::operator=( rhs ); return *this; }
    private:
      friend class AttributeList;
      iterator( std::vector< Attribute* >::const_iterator theIterator ) : iterator_base( theIterator ) {}
    public:
      Attribute* operator->() { return const_cast<Attribute*>( *m_iterator ); }
      Attribute& operator*() { return const_cast<Attribute&>( **m_iterator ); }
    };


    /// Returns a forward iterator
    iterator begin();
    iterator end();

    /// The constant iterator
    class const_iterator : public iterator_base
    {
    public:
      ~const_iterator() override {};
      const_iterator( const iterator_base& rhs ) : iterator_base( rhs ) {}
      const_iterator& operator=( const iterator_base& rhs ) { iterator_base::operator=( rhs ); return *this; }
    private:
      friend class AttributeList;
      const_iterator( std::vector< Attribute* >::const_iterator theIterator ) : iterator_base( theIterator ) {}
    public:
      const Attribute* operator->() const { return *m_iterator; }
      const Attribute& operator*() const { return **m_iterator; }
    };

    /// Returns a constant forward iterator
    const_iterator begin() const;
    const_iterator end() const;

  private:

    /// The underlying specification
    AttributeListSpecification* m_specification;

    /// Flag indicating the ownership of the specification
    bool m_ownSpecification;

    /// Data members;
    std::vector< Attribute* > m_data;

  };

}


// Inline methods
template<typename T> void
coral::AttributeList::extend( const std::string& name )
{
  this->extend( name, typeid(T) );
}

inline size_t
coral::AttributeList::size() const
{
  return m_data.size();
}

inline coral::AttributeList::iterator
coral::AttributeList::begin()
{
  return coral::AttributeList::iterator( m_data.begin() );
}

inline coral::AttributeList::iterator
coral::AttributeList::end()
{
  return coral::AttributeList::iterator( m_data.end() );
}

inline coral::AttributeList::const_iterator
coral::AttributeList::begin() const
{
  return coral::AttributeList::const_iterator( m_data.begin() );
}

inline coral::AttributeList::const_iterator
coral::AttributeList::end() const
{
  return coral::AttributeList::const_iterator( m_data.end() );
}

inline bool
coral::AttributeList::operator!=( const coral::AttributeList& rhs ) const
{
  return ( ! ( *this == rhs ) );
}

#if ( !defined(__clang__) && !defined(CORAL240CL) )
inline std::ostream& operator << (std::ostream& os, const coral::AttributeList& x)
{
  x.toOutputStream(os);
  return os;
}
#else
namespace coral
{
  // Move operator<< for AttributeList inside coral namespace to fix clang32
  // compilation by Koenig lookup (CORAL bug #100663 aka ATLAS bug #100527)
  // See http://stackoverflow.com/questions/8363759
  inline std::ostream& operator<<( std::ostream& os, const coral::AttributeList& x )
  {
    x.toOutputStream(os);
    return os;
  }
}
#endif

#endif
