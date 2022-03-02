// Tests for the RooAbsCollection and derived classes
// Authors: Stephan Hageboeck, CERN  05/2020
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooHelpers.h"

#include "gtest/gtest.h"

/// ROOT-10845 IsOnHeap() always returned false.
TEST(RooArgSet, IsOnHeap) {
  auto setp = new RooArgSet();
  EXPECT_TRUE(setp->IsOnHeap());

  RooArgSet setStack;
  EXPECT_FALSE(setStack.IsOnHeap());
}

TEST(RooArgSet_List, VariadicTemplateConstructor) {
  RooRealVar x("x", "x", 0.);
  RooRealVar y("y", "y", 0.);
  RooRealVar z("z", "z", 0.);

  RooArgSet theSet(x, "Hallo");
  EXPECT_EQ(theSet.size(), 1);
  EXPECT_STREQ(theSet.GetName(), "Hallo");

  RooArgSet theSet2(x, y, "Hallo2");
  EXPECT_EQ(theSet2.size(), 2);
  EXPECT_STREQ(theSet2.GetName(), "Hallo2");

  RooArgSet theSet3(x, y, z, "Hallo3");
  EXPECT_EQ(theSet3.size(), 3);
  EXPECT_STREQ(theSet3.GetName(), "Hallo3");


  RooArgList theList(x, "Hallo");
  EXPECT_EQ(theList.size(), 1);
  EXPECT_STREQ(theList.GetName(), "Hallo");

  RooArgList theList2(x, y, "Hallo2");
  EXPECT_EQ(theList2.size(), 2);
  EXPECT_STREQ(theList2.GetName(), "Hallo2");

  RooArgList theList3(x, y, z, "Hallo3");
  EXPECT_EQ(theList3.size(), 3);
  EXPECT_STREQ(theList3.GetName(), "Hallo3");

}

TEST(RooArgSet, SubscriptOperator) {
  RooRealVar x("x", "x", 0.);
  RooRealVar y("y", "y", 0.);
  RooArgSet theSet(x, y);

  EXPECT_EQ(theSet[0], &x);
  EXPECT_EQ(theSet[1], &y);

  const RooAbsArg& xRef = theSet["x"];
  const RooAbsArg& yRef = theSet["y"];
  EXPECT_EQ(&xRef, static_cast<RooAbsArg*>(&x));
  EXPECT_EQ(&yRef, static_cast<RooAbsArg*>(&y));

  // Silence error to come next:
  RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);
  EXPECT_THROW(theSet[nullptr], std::invalid_argument);
}

TEST(RooArgSet, FromVector) {
  std::vector<RooRealVar> vars;
  vars.emplace_back("x", "x", 0.);
  vars.emplace_back("y", "y", 0.);

  RooArgSet theSet(vars.begin(), vars.end(), "Hallo");
  EXPECT_EQ(theSet.size(), 2);
  EXPECT_STREQ(theSet.GetName(), "Hallo");
}

TEST(RooArgSet, InsertAndDeduplicate) {
  RooArgList list;
  RooArgList list2;
  for (unsigned int i = 0; i < 4; ++i) {
    char name[] = {char('a'+i), '\0'};
    list.addOwned(*(new RooRealVar(name, name, 0.)));
  }
  for (unsigned int i = 0; i < 5; ++i) {
    char name[] = {char('a'+i), '\0'};
    list2.addOwned(*(new RooRealVar(name, name, 0.)));
    name[0] = char('a'+ 4 - i);
    list2.addOwned(*(new RooRealVar(name, name, 0.)));
  }

  // Silence deduplication error messages
  RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);

  RooArgSet set{list};
  EXPECT_EQ(set.size(), list.size());

  RooArgSet set2{list2};
  EXPECT_EQ(set2.size(), list2.size()/2);

  RooArgSet set3{list};
  set3.add(list2);
  EXPECT_EQ(set3.size(), set.size() + 1);
  EXPECT_EQ(set3[set3.size()-1], & list2[1]);

  RooArgSet set4{list, list2};
  EXPECT_EQ(set4.size(), set.size() + 1);
}

TEST(RooArgSet, HashAssistedFind) {
  RooArgList list;
  RooArgList list2;
  for (unsigned int i = 0; i < 4; ++i) {
    char name[] = {char('a'+i), '\0'};
    list.addOwned(*(new RooRealVar(name, name, 0.)));
  }
  for (unsigned int i = 0; i < 5; ++i) {
    char name[] = {char('a'+i), '\0'};
    list2.addOwned(*(new RooRealVar(name, name, 0.)));
    name[0] = char('a'+ 4 - i);
    list2.addOwned(*(new RooRealVar(name, name, 0.)));
  }

  // Silence deduplication error messages
  RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments);

  RooArgSet set{list};
  set.useHashMapForFind(true);
  EXPECT_FALSE(set.add(RooRealVar("a", "a", 0)));
  EXPECT_EQ(set.size(), list.size());

  set.add(list2);
  EXPECT_EQ(set.size(), 5);

  // Renaming would invalidate the old hash map. Test that it gets regenerated correctly:
  list.useHashMapForFind(true);

  list[0].SetName("a'");
  EXPECT_EQ(set.find("a"), nullptr);
  EXPECT_EQ(set.find("a'"), & list[0]);

  EXPECT_EQ(list.find("a"), nullptr);
  EXPECT_EQ(list.find("a'"), & list[0]);
}
