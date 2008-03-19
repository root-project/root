/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

class A {public: int a;};
class B {public: int a;};
class C {public: int a;};
class D : public A , public B , public C {public: int a;};

int main() {
   D d;
   A *pa = &d;
   B *pb = &d;
   C *pc = &d;
   D *pd = &d;
   int i;
   for(i=0; i<3; i++) {
      //cout << (void*)pa << " " << (void*)pb << " " << (void*)pc << " " << (void*)pd << " " << endl;
      //fprintf(stdout,"%p %p %p %p\n",pa,pb,pc,pd);
      cout << (pa==pd) << " ";
      cout << (pb==pd) << " ";
      cout << (pc==pd) << " ";
      cout << (pd==pd) << " ";

      cout << (pd==pa) << " ";
      cout << (pd==pb) << " ";
      cout << (pd==pc) << " ";
      cout << (pd==pd) << endl;

      if(pa==pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pb==pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pc==pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd==pd) cout << "true" << " ";
      else       cout << "false" << " ";

      if(pd==pa) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd==pb) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd==pc) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd==pd) cout << "true" << endl;
      else       cout << "false" << endl;
   }

   for(i=0; i<3; i++) {
      //cout << pa << " " << pb << " " << pc << " " << pd << " " << endl;
      cout << (pa!=pd) << " ";
      cout << (pb!=pd) << " ";
      cout << (pc!=pd) << " ";
      cout << (pd!=pd) << " ";

      cout << (pd!=pa) << " ";
      cout << (pd!=pb) << " ";
      cout << (pd!=pc) << " ";
      cout << (pd!=pd) << endl;

      if(pa!=pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pb!=pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pc!=pd) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd!=pd) cout << "true" << " ";
      else       cout << "false" << " ";

      if(pd!=pa) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd!=pb) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd!=pc) cout << "true" << " ";
      else       cout << "false" << " ";
      if(pd!=pd) cout << "true" << endl;
      else       cout << "false" << endl;
   }
   return 0;
}
