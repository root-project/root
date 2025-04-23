// -*- C++ -*-
// $Id: Attribute.h,v 1.22 2013-03-02 09:38:52 avalassi Exp $
#ifndef CORALBASE_ATTRIBUTE_H
#define CORALBASE_ATTRIBUTE_H

#include "CoralBase/AttributeException.h"
#include "CoralBase/AttributeSpecification.h"
#include "CoralBase/VersionInfo.h"

#include <typeinfo>
#include <iosfwd>

namespace coral {

  // forward declarations
  class AttributeData;

  /**
     @class Attribute Attribute.h CoralBase/Attribute.h

     An attribute class which holds the data and a reference to its specification (type, name)
  */
  class Attribute {
  public:
    /// Assignment operator. Copies data.
    Attribute& operator=( const Attribute& rhs );

    /// Copies data without performing check on the type
    void fastCopy( const Attribute& rhs );

    /// Returns the specification of the attribute.
    const AttributeSpecification& specification() const;

    /// Returns a reference to the actual data.
    template<typename T> T& data();
    template<typename T> const T& data() const;

    /// Sets the value from an external variable. Performs type checking
    template<typename T> void setValue( const T& value );

    /// Binds safely to an external variable.
    template<typename T> void bind( const T& externalVariable );
    template<typename T> void bind( T& externalVariable );

    /// Copies data from an external address. By definition this is an unsafe operation.
    void setValueFromAddress( const void* externalAddress );

    /// Copies data to an external address. By definition this is an unsafe operation.
    void copyValueToAddress( void* externalAddress ) const;

    /// Returns the addess of the actual data.
    void* addressOfData();
    const void* addressOfData() const;

    /// Shares the data with another Attribute.
    void shareData( const Attribute& sourceAttribute );

    /// Binds to an external address (unsafe operation).
    void bindUnsafely( const void* externalAddress );
    void bindUnsafely( void* externalAddress );

    /// Sets the NULL-ness of the variable
    void setNull( bool isVariableNull = true );

    /// Retrieves the NULL-ness of the variable
    bool isNull() const;

    /// Returns the size of the variable
    int size() const;

    /// Equal operator compares data and type. Not the name!
    bool operator==( const Attribute& rhs ) const;

    /// Comparison operator compares data and type. Not the name!
    bool operator!=( const Attribute& rhs ) const;

    /// Dumps the attribute contents
    std::ostream& toOutputStream( std::ostream& os, bool valueOnly = false ) const;

    /// Check if the variable is producing a NOT NULL value in ALL of the backends supported.
    bool isValidData() const;

  private:
    /// Binds safely to an external variable.
    void bindVariable( const std::type_info& type, const void* externalAddress );
    void bindVariable( const std::type_info& type, void* externalAddress );

    /// Sets the value safely from an external source
    void setValue( const std::type_info& type, const void* externalAddress );

  private:
    /// Only the AttributeList can create Attributes
    friend class AttributeList;

    /// Constructor
    explicit Attribute( const AttributeSpecification& specification );

    /// Destructor
    ~Attribute();

    /// No copy constructor.
    Attribute( const Attribute& rhs );

  private:
    /// The specification
    const AttributeSpecification* m_specification;

    /// The data
    AttributeData* m_data;
  };


  // Inline methods
  inline const AttributeSpecification&
  Attribute::specification() const
  {
    return *m_specification;
  }


  template<typename T>
  const T& Attribute::data() const
  {
    if( this->specification().type() != typeid(T))
      throw coral::AttributeException( "Attempt to assign attribute \"" + this->specification().name() + "\" of type " + this->specification().typeName() + " with " + coral::AttributeSpecification::typeNameForId(typeid(T)) );
    if(this->isNull())
    {
      if ( typeid(T) != typeid(std::string) )
        throw coral::AttributeException( "Attempt to access data of NULL attribute \"" + this->specification().name()+ "\"" );
      else
        *( const_cast< std::string* >( static_cast< const std::string* >( this->addressOfData() ) ) ) = "";
    }
    return *( static_cast< const T* >( this->addressOfData() ) );
  }


  template<typename T>
  T& Attribute::data()
  {
    if(this->specification().type() != typeid(T))
      throw coral::AttributeException( "Attempt to assign attribute \"" + this->specification().name() + "\" of type " + this->specification().typeName() + " with " + coral::AttributeSpecification::typeNameForId(typeid(T)) );
    if(this->isNull())
    {
      if ( typeid(T) != typeid(std::string) )
        throw coral::AttributeException( "Attempt to access data of NULL attribute \"" + this->specification().name()+ "\"" );
      else
        *( static_cast< std::string* >( this->addressOfData() ) ) = "";
    }
    return *( static_cast< T* >( this->addressOfData() ) );
  }


  template<typename T>
  void Attribute::bind( const T& externalVariable )
  {
    this->bindVariable( typeid(T), &externalVariable );
  }


  template<typename T>
  void Attribute::bind( T& externalVariable )
  {
    this->bindVariable( typeid(T), &externalVariable );
  }


  template<typename T>
  void Attribute::setValue( const T& value )
  {
    this->setValue( typeid(T), &value );
  }
}

#if ( !defined(__clang__) && !defined(CORAL240CL) )
inline std::ostream& operator<< (std::ostream& os, const coral::Attribute& x)
{
  x.toOutputStream(os);
  return os;
}
#else
namespace coral
{
  // Move operator<< for Attribute inside coral namespace to fix clang
  // compilation by Koenig lookup (CORAL bug #100663 aka ATLAS bug #100527)
  // See http://stackoverflow.com/questions/8363759
  inline std::ostream& operator<<( std::ostream& os, const coral::Attribute& x )
  {
    x.toOutputStream(os);
    return os;
  }
}
#endif

#endif
