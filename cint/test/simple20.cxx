/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <ertti.h>

class myclass {
public:
   int i;
};

void disp(G__ClassInfo& c)
{
   G__DataMemberInfo d(c);
   while(d.Next()) {
      G__TypeInfo *t = d.Type();
      if (t==0) {
         printf("type is null!\n");
      }
      const char *name = t->Name();
      printf("typename is %s\n",name);
      const char *dname = d.Name();
      printf("data name is %s\n",dname);
      printf("%s %s\n",d.Type()->Name(),d.Name());
   }
}

int main() {
  G__ClassInfo c1("myclass"); disp(c1);
}

