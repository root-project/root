#include "TList.h"
#include "TNamed.h"

#include "gtest/gtest.h"

TEST(TIter, Loop)
{
   TList lst;
   lst.SetOwner(kTRUE);
   lst.Add(new TNamed("first","title"));
   lst.Add(new TNamed("second","title"));
   lst.Add(new TNamed("third","title"));

   int cnt = 0;
   TIter iter(&lst);

   while (iter.Next()) cnt++;

   EXPECT_EQ(cnt, 3);
}


class TListIterTest : public TListIter {
public:
   Bool_t *fDestroyed{nullptr};

   TListIterTest(Bool_t *destroyed, TList *lst, Bool_t dir = kIterForward) : TListIter(lst, dir)
   {
      fDestroyed = destroyed;
   }
   virtual ~TListIterTest()
   {
      if (fDestroyed) *fDestroyed = kTRUE;
   }

   TIterator &operator=(const TIterator &rhs)
   {
      if (this != &rhs)
         TListIter::operator=(rhs);
      return *this;
   }

   TListIterTest &operator=(const TListIterTest &rhs)
   {
      if (this != &rhs)
         TListIter::operator=(rhs);
      return *this;
   }
};

TEST(TIter, Assign)
{
   TList lst;
   lst.SetOwner(kTRUE);
   lst.Add(new TNamed("first","title"));
   lst.Add(new TNamed("second","title"));
   lst.Add(new TNamed("third","title"));

   Bool_t flag = kFALSE;

   {
      TIter iter(&lst);
      Int_t cnt = 0;
      while (iter.Next()) cnt++;
      EXPECT_EQ(cnt, 3);

      // check that iterator assigned without copying
      iter = new TListIterTest(&flag, &lst);

      EXPECT_EQ(flag, kFALSE);

      cnt = 0;
      while (iter.Next()) cnt++;

      EXPECT_EQ(cnt, 3);
   }

   EXPECT_EQ(flag, kTRUE);
}
