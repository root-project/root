#include "MyClass1.h"
ClassImp(MyClass1)

MyClass1::MyClass1():
  m_id( 0 ),
  m_vectorOfMaps()
{}

int
MyClass1::getId() const
{
  return m_id;
}

void
MyClass1::setId( int id )
{
  m_id = id;
}

void
MyClass1::insertMap()
{
  m_vectorOfMaps.push_back( std::map< int, std::string >() );
}

void
MyClass1::insert( int id, int key, const std::string& value )
{
  m_vectorOfMaps[id].insert( std::make_pair( key, value ) );
}

const std::string&
MyClass1::getValue( int id, int key ) const
{
  return m_vectorOfMaps[id].find(key)->second;
}
