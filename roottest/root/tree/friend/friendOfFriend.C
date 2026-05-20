#include "TNtuple.h"

void friendOfFriend() {
   TNtuple *t1 = new TNtuple("t1","title1","var1");
   TNtuple *t2 = new TNtuple("t2","title2","var2");
   TNtuple *t3 = new TNtuple("t3","title3","var3");
   t1->Fill(1);
   t1->Fill(1);
   t2->Fill(2);
   t2->Fill(2);
   t3->Fill(3);
   t3->Fill(3);

   t2->AddFriend(t3);
   t1->AddFriend(t2);
   t2->Scan("var2:var3");
   t1->Scan("var1:var2:var3");
}