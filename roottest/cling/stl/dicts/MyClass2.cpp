#include "MyClass2.h"
ClassImp(MyClass2)

MyClass2::MyClass2():
  m_id( 0 ),
  m_vectorOfMaps()
{}

int
MyClass2::getId() const
{
  return m_id;
}

void
MyClass2::setId( int id )
{
  m_id = id;
}

void
MyClass2::insert( int key, const std::string& value )
{
  m_vectorOfMaps.push_back( std::make_pair( key, value ) );
}

const std::string&
MyClass2::getValue( int id ) const
{
  return m_vectorOfMaps[id].second;
}
