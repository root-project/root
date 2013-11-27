#ifndef CORALBASE_ATTRIBUTESPECIFICATION_H
#define CORALBASE_ATTRIBUTESPECIFICATION_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include <typeinfo>
#include <string>

namespace coral
{

  /**
     @class AttributeSpecification AttributeSpecification.h CoralBase/AttributeSpecification.h
     Specification of an Attribute
  */

  class AttributeSpecification
  {

  private:

    /// Only the AttributeListSpecification can create such objects
    friend class AttributeListSpecification;

    /// Constructor
    AttributeSpecification( const std::string& name,
                            const std::type_info& type );

    /// Constructor
    AttributeSpecification( const std::string& name,
                            const std::string& typeName );

    /// Copy constructor
    AttributeSpecification( const AttributeSpecification& rhs );

    /// Destructor
    ~AttributeSpecification();

  public:

    /// Assignment operator
    AttributeSpecification& operator=( const AttributeSpecification& rhs );

    /// Returns the attribute name
    std::string name() const;

    /// Returns the type
    const std::type_info& type() const;

    /// Returns the type name
    std::string typeName() const;

    /// Equal operator
    bool operator==( const AttributeSpecification& rhs ) const;

    /// Comparison operator
    bool operator!=( const AttributeSpecification& rhs ) const;

  public:

    /// Returns the type name given the type id
    static std::string typeNameForId( const std::type_info& type );

    /// Returns the type name for the given type
    template<typename T> static std::string typeNameForType() { return typeNameForId(typeid(T)); }
    template<typename T> static std::string typeNameForType( const T& ) { return typeNameForId(typeid(T)); }

    /// Returns the type given a type name
    static const std::type_info* typeIdForName( const std::string& typeName );

  private:

    /// Validates the type
    static const std::type_info* validateType( const std::type_info& type );

  private:

    /// The name
    std::string m_name;

    /// The type
    const std::type_info* m_type;

  };


  // Inline methods
  inline
  AttributeSpecification::AttributeSpecification( const std::string& name,
                                                  const std::string& typeName ) :
    m_name( name ),
    m_type( AttributeSpecification::typeIdForName( typeName ) )
  {
  }


  inline
  AttributeSpecification::AttributeSpecification( const AttributeSpecification& rhs ) :
    m_name( rhs.m_name ),
    m_type( rhs.m_type )
  {
  }


  inline AttributeSpecification&
  AttributeSpecification::operator=( const AttributeSpecification& rhs )
  {
#ifdef CORAL240CO
    if ( this == &rhs ) return *this;  // Fix Coverity SELF_ASSIGN
#endif
    m_name = rhs.m_name;
    m_type = rhs.m_type;
    return *this;
  }


  inline std::string
  AttributeSpecification::name() const
  {
    return m_name;
  }


  inline const std::type_info&
  AttributeSpecification::type() const
  {
    return *m_type;
  }


  inline std::string
  AttributeSpecification::typeName() const
  {
    return AttributeSpecification::typeNameForId( *m_type );
  }


  inline bool
  AttributeSpecification::operator==( const AttributeSpecification& rhs ) const
  {
    return ( this->m_name == rhs.m_name &&
             *( this->m_type ) == *( rhs.m_type ) );
  }


  inline bool
  AttributeSpecification::operator!=( const AttributeSpecification& rhs ) const
  {
    return ( this->m_name != rhs.m_name ||
             *( this->m_type ) != *( rhs.m_type ) );
  }

}
#endif
