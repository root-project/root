#include <vector>

#include "gtest/gtest.h"

#include "TFile.h"
#include "TKey.h"
#include "TNamed.h"
#include "TSystem.h"

TEST(TFile, WriteObjectTObject)
{
    auto filename{"tfile_writeobject_tobject.root"};
    auto tnamed_name{"mytnamed_name"};
    auto tnamed_title{"mytnamed_title"};

    {
        TNamed mytnamed{tnamed_name, tnamed_title};
        TFile f{filename, "recreate"};
        f.WriteObject(&mytnamed, mytnamed.GetName());
        f.Close();
    }

    TFile input{filename};
    auto named = input.Get<TNamed>(tnamed_name);
    auto keyptr = static_cast<TKey *>(input.GetListOfKeys()->At(0));

    EXPECT_STREQ(named->GetName(), tnamed_name);
    EXPECT_STREQ(named->GetTitle(), tnamed_title);
    EXPECT_STREQ(keyptr->GetName(), tnamed_name);
    EXPECT_STREQ(keyptr->GetTitle(), tnamed_title);

    input.Close();
    gSystem->Unlink(filename);
}

TEST(TFile, WriteObjectVector)
{
    auto filename{"tfile_writeobject_vector.root"};
    auto vec_name{"object name"}; // Decided arbitrarily
    auto vec_title{"object title"}; // Default title for non-TObject-derived instances

    {
        std::vector<int> myvec{1,2,3,4,5};
        TFile f{filename, "recreate"};
        f.WriteObject(&myvec, vec_name);
        f.Close();
    }

    TFile input{filename};
    auto retvecptr = input.Get<std::vector<int>>(vec_name);
    const auto &retvec = *retvecptr;
    auto retkey = static_cast<TKey *>(input.GetListOfKeys()->At(0));

    std::vector<int> expected{1,2,3,4,5};

    ASSERT_EQ(retvec.size(), expected.size());
    for (std::size_t i = 0; i < retvec.size(); ++i) {
        EXPECT_EQ(retvec[i], expected[i]);
    }

    EXPECT_STREQ(retkey->GetName(), vec_name);
    EXPECT_STREQ(retkey->GetTitle(), vec_title);

    input.Close();
    gSystem->Unlink(filename);
}

// Tests ROOT-9857
TEST(TFile, ReadFromSameFile)
{
   const auto filename = "ReadFromSameFile.root";
   const auto objname = "foo";
   const auto objpath = "./ReadFromSameFile.root/foo";
   {
      TFile f(filename, "RECREATE");
      TObject obj;
      f.WriteObject(&obj, objname);
   }

   TFile f1(filename);
   auto o1 = f1.Get(objname);

   TFile f2(filename);
   auto o2 = f2.Get(objpath);

   EXPECT_TRUE(o1 != o2) << "Same objects read from two different files have the same pointer!";
}
