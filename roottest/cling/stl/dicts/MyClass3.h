#ifndef MYCLASS_H
#define MYCLASS_H

#include <vector>

#include "Rtypes.h"
#include "TObject.h"

class MyClass3 : public TObject {
public:
  MyClass3();
  ~MyClass3() {}

  int getId() const;
  void setId( int );

  void insert( double value );

  double getValue( int x ) const;

private:
  int m_id;
  std::vector< std::vector< double > > m_data;

  ClassDef(MyClass3,1) // analyse my data
};

#endif
