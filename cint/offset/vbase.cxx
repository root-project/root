#include <typeinfo>

struct Top {
  int fValue;
  virtual ~Top() {}
};

struct Mid1 : public virtual Top
{
   int fMid1;
};

struct Mid2 : public virtual Top
{
   int fMid2;
};

struct Bottom : public Mid1, Mid2
{
   int fBottom;
};

const char *getname(Top *t)
{
  return typeid(*t).name();
}

#include <stdio.h>

void vbase()
{
Bottom *m = new Bottom;
//m;
Mid1 *m1; Mid2 *m2;
Top *t;
m1 = m;
m2 = m;
t = m1;
t = m2;
//m;
const char * c = getname(t); // typeid(*t).name();
fprintf(stderr,"found %s\n",c);
}
