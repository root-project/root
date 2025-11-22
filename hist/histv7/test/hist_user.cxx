#include "hist_test.hxx"

#include <type_traits>

// User-defined bin content type consisting of a single double, but can be much more complicated.
struct User {
   double fValue = 0;

   User &operator++()
   {
      fValue++;
      return *this;
   }

   User operator++(int)
   {
      User old = *this;
      operator++();
      return old;
   }

   User &operator+=(double w)
   {
      fValue += w;
      return *this;
   }

   User &operator+=(const User &rhs)
   {
      fValue += rhs.fValue;
      return *this;
   }

   User &operator*=(double factor)
   {
      fValue *= factor;
      return *this;
   }

   void AtomicInc() { ROOT::Experimental::Internal::AtomicInc(&fValue); }

   void AtomicAdd(double w) { ROOT::Experimental::Internal::AtomicAdd(&fValue, w); }

   void AtomicAdd(const User &rhs) { ROOT::Experimental::Internal::AtomicAdd(&fValue, rhs.fValue); }
};

static_assert(std::is_nothrow_move_constructible_v<RHistEngine<User>>);
static_assert(std::is_nothrow_move_assignable_v<RHistEngine<User>>);

static_assert(std::is_nothrow_move_constructible_v<RHist<User>>);
static_assert(std::is_nothrow_move_assignable_v<RHist<User>>);

TEST(RHistEngineUser, Add)
{
   // Addition uses operator+=(const User &)
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engineA({axis});
   RHistEngine<User> engineB({axis});

   engineA.Fill(8.5);
   engineB.Fill(9.5);

   engineA.Add(engineB);

   EXPECT_EQ(engineA.GetBinContent(RBinIndex(8)).fValue, 1);
   EXPECT_EQ(engineA.GetBinContent(RBinIndex(9)).fValue, 1);
}

TEST(RHistEngineUser, AddAtomic)
{
   // Addition with atomic instructions uses AtomicAdd(const User &)
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engineA({axis});
   RHistEngine<User> engineB({axis});

   engineA.Fill(8.5);
   engineB.Fill(9.5);

   engineA.AddAtomic(engineB);

   EXPECT_EQ(engineA.GetBinContent(RBinIndex(8)).fValue, 1);
   EXPECT_EQ(engineA.GetBinContent(RBinIndex(9)).fValue, 1);
}

TEST(RHistEngineUser, Clear)
{
   // Clearing assigns default-constructed objects.
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.Fill(8.5);
   engine.Fill(9.5);

   engine.Clear();

   EXPECT_EQ(engine.GetBinContent(RBinIndex(8)).fValue, 0);
   EXPECT_EQ(engine.GetBinContent(RBinIndex(9)).fValue, 0);
}

TEST(RHistEngineUser, Clone)
{
   // Cloning copy-assigns the objects.
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engineA({axis});

   engineA.Fill(8.5);

   RHistEngine<User> engineB = engineA.Clone();
   EXPECT_EQ(engineB.GetBinContent(8).fValue, 1);

   // Check that we can continue filling the clone.
   engineB.Fill(9.5);

   EXPECT_EQ(engineA.GetBinContent(9).fValue, 0);
   EXPECT_EQ(engineB.GetBinContent(9).fValue, 1);
}

TEST(RHistEngineUser, Fill)
{
   // Unweighted filling uses operator++(int)
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.Fill(8.5);
   engine.Fill(std::make_tuple(9.5));

   EXPECT_EQ(engine.GetBinContent(RBinIndex(8)).fValue, 1);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(engine.GetBinContent(indices).fValue, 1);
}

TEST(RHistEngineUser, FillWeight)
{
   // Weighted filling uses operator+=(double)
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.Fill(8.5, RWeight(0.8));
   engine.Fill(std::make_tuple(9.5), RWeight(0.9));

   EXPECT_EQ(engine.GetBinContent(RBinIndex(8)).fValue, 0.8);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(engine.GetBinContent(indices).fValue, 0.9);
}

TEST(RHistEngineUser, FillAtomic)
{
   // Unweighted filling with atomic instructions uses AtomicInc
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.FillAtomic(8.5);
   engine.FillAtomic(std::make_tuple(9.5));

   EXPECT_EQ(engine.GetBinContent(RBinIndex(8)).fValue, 1);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(engine.GetBinContent(indices).fValue, 1);
}

TEST(RHistEngineUser, FillAtomicWeight)
{
   // Weighted filling with atomic instructions uses AtomicAdd
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.FillAtomic(8.5, RWeight(0.8));
   engine.FillAtomic(std::make_tuple(9.5), RWeight(0.9));

   EXPECT_EQ(engine.GetBinContent(RBinIndex(8)).fValue, 0.8);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(engine.GetBinContent(indices).fValue, 0.9);
}

TEST(RHistEngineUser, Scale)
{
   // Scaling uses operator+=(double)
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, {0, Bins});
   RHistEngine<User> engine({axis});

   engine.Fill(8.5, RWeight(0.8));
   engine.Fill(9.5, RWeight(0.9));

   static constexpr double Factor = 0.8;
   engine.Scale(Factor);

   EXPECT_EQ(engine.GetBinContent(8).fValue, Factor * 0.8);
   EXPECT_EQ(engine.GetBinContent(9).fValue, Factor * 0.9);
}
