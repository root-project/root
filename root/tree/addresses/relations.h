#ifndef TEST_H_
#define TEST_H_

#include <algorithm>
#include <vector>
#include <map>
#include <functional>

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
  virtual void release() = 0;
};

template<class FROM, class TO> struct IRelation : public IInterface  {
public:
  virtual ~IRelation() {}
};

template<class FROM,class TO> struct RelationBase  {  
public:
  RelationBase()    {}
  virtual ~RelationBase() {};
  std::vector<std::pair<int,float> > m_entries;
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
  virtual ~Relation1D()  {  }
};
#endif
