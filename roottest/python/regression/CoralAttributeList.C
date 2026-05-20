#ifndef CORALBASE_ATTRIBUTELIST_H
#define CORALBASE_ATTRIBUTELIST_H

#include <string>
#include <vector>
#include <typeinfo>
#include <iosfwd>

typedef std::string Attribute;

namespace coral_pyroot_regression {

  // forward declarations
  class AttributeSpecification;
  class AttributeListSpecification;

  class AttributeList
  {
  public:
    void extend( const std::string& name,
                 const std::string& typeName );

    size_t size() const;

    class iterator_base {
    protected:
      iterator_base() {}
      iterator_base( std::vector< Attribute* >::const_iterator theIterator ) : m_iterator( theIterator ) {}
      iterator_base( const iterator_base& rhs ): m_iterator( rhs.m_iterator ) {}
      iterator_base& operator=( const iterator_base& rhs ) { m_iterator = rhs.m_iterator; return *this; }
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
      ~iterator() {};

      iterator( const iterator_base& rhs ): iterator_base( rhs ) {}
      iterator& operator=( const iterator_base& rhs ) { iterator_base::operator=( rhs ); return *this; }

    private:
      friend class AttributeList;
      iterator( std::vector< Attribute* >::const_iterator theIterator ): iterator_base( theIterator ) {}

    public:
      Attribute* operator->() { return const_cast<Attribute*>( *m_iterator ); }
      Attribute& operator*() { return const_cast<Attribute&>( **m_iterator ); }
    };

    iterator begin();
    iterator end();


    class const_iterator : public iterator_base
    {
    public:
      ~const_iterator() {};

      const_iterator( const iterator_base& rhs ): iterator_base( rhs ) {}
      const_iterator& operator=( const iterator_base& rhs ) { iterator_base::operator=( rhs ); return *this; }

    private:
      friend class AttributeList;
      const_iterator( std::vector< Attribute* >::const_iterator theIterator ): iterator_base( theIterator ) {}

    public:
      const Attribute* operator->() const { return *m_iterator; }
      const Attribute& operator*() const { return **m_iterator; }
    };

    const_iterator begin() const;
    const_iterator end() const;

    bool useVar() { return m_specification && m_ownSpecification; }

  private:
    AttributeListSpecification* m_specification;

    bool m_ownSpecification;

    std::vector< Attribute* > m_data;
  };

}





// Inline methods
void
coral_pyroot_regression::AttributeList::extend( const std::string& name, const std::string& /* typeName */ )
{
  m_data.push_back( new std::string(name) );
}

inline size_t
coral_pyroot_regression::AttributeList::size() const
{
  return m_data.size();
}

inline coral_pyroot_regression::AttributeList::iterator
coral_pyroot_regression::AttributeList::begin()
{
  return coral_pyroot_regression::AttributeList::iterator( m_data.begin() );
}

inline coral_pyroot_regression::AttributeList::iterator
coral_pyroot_regression::AttributeList::end()
{
  return coral_pyroot_regression::AttributeList::iterator( m_data.end() );
}

inline coral_pyroot_regression::AttributeList::const_iterator
coral_pyroot_regression::AttributeList::begin() const
{
  return coral_pyroot_regression::AttributeList::const_iterator( m_data.begin() );
}

inline coral_pyroot_regression::AttributeList::const_iterator
coral_pyroot_regression::AttributeList::end() const
{
  return coral_pyroot_regression::AttributeList::const_iterator( m_data.end() );
}

#endif

