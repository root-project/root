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

   User &operator+=(const User &rhs)
   {
      fValue += rhs.fValue;
      return *this;
   }
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
