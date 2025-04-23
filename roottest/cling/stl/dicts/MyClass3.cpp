#include "MyClass3.h"
ClassImp(MyClass3)

MyClass3::MyClass3():
  m_id( 0 ),
  m_data()
{
  m_data.push_back( std::vector< double >() );
}

int
MyClass3::getId() const
{
  return m_id;
}

void
MyClass3::setId( int id )
{
  m_id = id;
}

void
MyClass3::insert( double value )
{
  m_data[0].push_back( value );
}

double
MyClass3::getValue( int id ) const
{
  return m_data[0][id];
}
