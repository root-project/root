#ifndef TEST_H_
#define TEST_H_

#include <algorithm>
#include <vector>
#include <map>
#include <functional>
#include "TClonesArray.h"

class DataTObject : public TObject {
public:
  int i;    //
  float j;  //
  int k;    //!
  int l;    //!
  int m;    //!
  DataTObject() {}
  DataTObject(int ii,float ij) : i(ii),j(ij) {}
  ~DataTObject() override {}
  virtual void release() {}
  ClassDefOverride(DataTObject,1);
};

struct DataObject  {
public:
  int i;    //!
  int j;    //!
  int k;    //!
  int l;    //!
  int m;    //!
  DataObject() {}
  virtual ~DataObject() {}
  virtual void release() {}
};

struct IInterface  {
public:
  virtual ~IInterface() {}
  virtual void release() = 0;
};

template<class FROM, class TO> struct IRelation : public IInterface  {
public:
  ~IRelation() override {}
};

template<class FROM,class TO> struct RelationBase  {  
public:
   RelationBase() : m_tentries("DataTObject"),m_ptentries(new TClonesArray("DataTObject")) {}
   virtual ~RelationBase() {};
   std::vector<std::pair<int,float> > m_entries;
   TClonesArray  m_tentries;
   TClonesArray *m_ptentries;
};

template<class FROM,class TO> struct Relation : public IRelation<FROM,TO> {  
public:
  RelationBase<FROM,TO>   m_direct;
  Relation () : m_direct()      {}
  virtual ~Relation() {}
  virtual void release() {}
};

template<class FROM,class TO> struct Relation1D :
  public DataObject,
  public Relation<FROM,TO>
{
public:
  Relation1D ()   {  };
  ~Relation1D() override  {  }
};
#endif
