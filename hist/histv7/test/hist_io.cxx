#include "hist_test.hxx"

#include <ROOT/TestSupport.hxx>
#include <TMemFile.h>

#include <stdexcept>

template <typename T>
static void ExpectThrowOnWriteObject(const T &obj)
{
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(kWarning, "TKey::TKey", "no public constructor", /*matchFullMessage=*/false);

   TMemFile f("mem.root", "RECREATE");
   EXPECT_THROW(f.WriteObject(&obj, "o"), std::runtime_error);
}

TEST(RRegularAxis, Streamer)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);
   ExpectThrowOnWriteObject(axis);
}

TEST(RVariableBinAxis, Streamer)
{
   static constexpr std::size_t Bins = 20;
   std::vector<double> bins;
   for (std::size_t i = 0; i < Bins; i++) {
      bins.push_back(i);
   }
   bins.push_back(Bins);

   const RVariableBinAxis axis(bins);
   ExpectThrowOnWriteObject(axis);
}

TEST(RAxes, Streamer)
{
   static constexpr std::size_t BinsX = 20;
   const RRegularAxis regularAxis(BinsX, 0, BinsX);
   static constexpr std::size_t BinsY = 30;
   std::vector<double> bins;
   for (std::size_t i = 0; i < BinsY; i++) {
      bins.push_back(i);
   }
   bins.push_back(BinsY);
   const RVariableBinAxis variableBinAxis(bins);

   const RAxes axes({regularAxis, variableBinAxis});
   ExpectThrowOnWriteObject(axes);
}

TEST(RHistEngine, Streamer)
{
   static constexpr std::size_t Bins = 20;
   const RRegularAxis axis(Bins, 0, Bins);

   // We don't request a dictionary for RHistEngine<unsigned char>, and we generally don't recommend such narrow bin
   // content types. If used, the RAxes member will prevent streaming.
   const RHistEngine<unsigned char> engineC({axis});
   ExpectThrowOnWriteObject(engineC);

   const RHistEngine<int> engineI({axis});
   ExpectThrowOnWriteObject(engineI);

   const RHistEngine<long> engineL({axis});
   ExpectThrowOnWriteObject(engineL);

   const RHistEngine<long long> engineLL({axis});
   ExpectThrowOnWriteObject(engineLL);

   const RHistEngine<float> engineF({axis});
   ExpectThrowOnWriteObject(engineF);

   const RHistEngine<double> engineD({axis});
   ExpectThrowOnWriteObject(engineD);
}
