#ifndef MYCLASS_H
#define MYCLASS_H

#include <string>
#include <vector>
#include <map>

#include "Rtypes.h"
#include "TObject.h"

class MyClass2 : public TObject {
public:
  MyClass2();
  ~MyClass2() {}

  int getId() const;
  void setId( int );

  void insert( int key, const std::string& value );

  const std::string& getValue( int id ) const;

private:
  int m_id;
  std::vector< std::pair< int, std::string > > m_vectorOfMaps;

  ClassDef(MyClass2,1) // analyse my data
};

#endif
