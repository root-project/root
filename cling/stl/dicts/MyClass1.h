#ifndef MYCLASS_H
#define MYCLASS_H

#include <string>
#include <vector>
#include <map>

#include "Rtypes.h"

class MyClass1 {
public:
  MyClass1();
  virtual ~MyClass1() {}

  int getId() const;
  void setId( int );

  void insertMap();
  void insert( int id, int key, const std::string& value );

  const std::string& getValue( int id, int key ) const;

private:
  int m_id;
  std::vector< std::map< int, std::string > > m_vectorOfMaps;

  ClassDef(MyClass1,1) // analyse my data
};

#endif
