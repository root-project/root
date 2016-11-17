#include "TFile.h"
#include "TInterpreter.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"

#include "gtest/gtest.h"

#include "data.h"

#include <fstream>

TEST(TTreeReaderLeafs, LeafListCaseA) {
   // From "Case A" of the TTree class doc:
   const char* str = "This is a null-terminated string literal";
   signed char SChar = 2;
   unsigned char UChar = 3;
   signed short SShort = 4;
   unsigned short UShort = 5;
   signed int SInt = 6;
   unsigned int UInt = 7;
   float Float = 8.;
   double Double = 9.;
   long long SLL = 10;
   unsigned long long ULL = 11;
   bool Bool = true;

   TTree* tree = new TTree("T", "In-memory test tree");
   tree->Branch("C", &str, "C/C");
   tree->Branch("B", &SChar, "B/B");
   tree->Branch("b", &UChar, "b/b");
   tree->Branch("S", &SShort, "S/S");
   tree->Branch("s", &UShort, "s/s");
   tree->Branch("I", &SInt, "I/I");
   tree->Branch("i", &UInt, "i/i");
   tree->Branch("F", &Float, "F/F");
   tree->Branch("D", &Double, "D/D");
   tree->Branch("L", &SLL, "L/L");
   tree->Branch("l", &ULL, "l/l");
   tree->Branch("O", &Bool, "O/O");

   tree->Fill();
   tree->Fill();
   tree->Fill();

   TTreeReader TR(tree);
   //TTreeReaderValue<const char*> trStr(TR, "C");
   TTreeReaderValue<signed char> trSChar(TR, "B");
   TTreeReaderValue<unsigned char> trUChar(TR, "b");
   TTreeReaderValue<signed short> trSShort(TR, "S");
   TTreeReaderValue<unsigned short> trUShort(TR, "s");
   TTreeReaderValue<signed int> trSInt(TR, "I");
   TTreeReaderValue<unsigned int> trUInt(TR, "i");
   TTreeReaderValue<float> trFloat(TR, "F");
   TTreeReaderValue<double> trDouble(TR, "D");
   TTreeReaderValue<signed long long> trSLL(TR, "L");
   TTreeReaderValue<unsigned long long> trULL(TR, "l");
   TTreeReaderValue<bool> trBool(TR, "O");

   TR.SetEntry(1);

   //EXPECT_STREQ(str, *trStr);
   EXPECT_EQ(SChar, *trSChar);
   EXPECT_EQ(UChar, *trUChar);
   EXPECT_EQ(SShort, *trSShort);
   EXPECT_EQ(UShort, *trUShort);
   EXPECT_EQ(SInt, *trSInt);
   EXPECT_EQ(UInt, *trUInt);
   EXPECT_FLOAT_EQ(Float, *trFloat);
   EXPECT_DOUBLE_EQ(Double, *trDouble);
   EXPECT_EQ(SLL, *trSLL);
   EXPECT_EQ(ULL, *trULL);
   EXPECT_EQ(Bool, *trBool);
}



void WriteData() {
   gInterpreter->ProcessLine("#include \"data.h\"");

   TFile fout("data.root", "RECREATE");
   Data data;
   TTree* tree = new TTree("T", "test tree");
   tree->Branch("Data", &data);
   data.fArray = new double[4]{12., 13., 14., 15.};
   data.fSize = 4;
   data.fUArray = new float[2]{42., 43.};
   data.fUSize = 2;
   tree->Fill();
   tree->Fill();
   tree->Write();
}

TEST(TTreeReaderLeafs, LeafList) {
   WriteData();

   TFile fin("data.root");
   TTree* tree = 0;
   fin.GetObject("T", tree);
   TTreeReader tr(tree);
   TTreeReaderArray<double> arr(tr, "fArray");
   TTreeReaderArray<float> arrU(tr, "fUArray");

   tr.Next();
   EXPECT_FLOAT_EQ(4, arr.GetSize());
   EXPECT_FLOAT_EQ(2, arrU.GetSize());
   //FAILS EXPECT_FLOAT_EQ(13., arr[1]);
   //FAILS EXPECT_DOUBLE_EQ(43., arrU[1]);
   // T->Scan("fUArray") claims fUArray only has one instance per row.
}
