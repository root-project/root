
#include <stdio.h>

#include "Bytes.h"
#include "TBranch.h"
#include "TBufferFile.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "ROOT/TTreeReaderFast.hxx"
#include "ROOT/TTreeReaderValueFast.hxx"
#include "ROOT/TBulkBranchRead.hxx"
#include "ROOT/TIOFeatures.hxx"

#include "gtest/gtest.h"


class BulkApiVariableTest : public ::testing::Test {
public:
    static constexpr Long64_t fClusterSize = 1e5;
    static constexpr Long64_t fEventCount = 4e6;
    const std::string fFileName = "BulkApiTestVarLength.root";

protected:
    virtual void SetUp()
    {
        auto hfile = new TFile(fFileName.c_str(), "RECREATE", "TTree float micro benchmark ROOT file");
        hfile->SetCompressionLevel(0); // No compression at all.

        auto tree = new TTree("T", "A ROOT tree of variable-length primitive branches.");
        tree->SetBit(TTree::kOnlyFlushAtCluster);
        tree->SetAutoFlush(fClusterSize);
        ROOT::TIOFeatures features;
        features.Set(ROOT::Experimental::EIOFeatures::kGenerateOffsetMap);
        tree->SetIOFeatures(features);

        float f_counter = 0;
        float f[10];
        double d[10];
        int i[10];
        int myLen = 0;

        tree->Branch("myLen", &myLen, "myLen/I", 32000);
        auto branch2 = tree->Branch("f", &f, "f[myLen]/F", 32000);
        auto branch3 = tree->Branch("d", &d, "d[myLen]/D", 32000);
        auto branch4 = tree->Branch("i", &i, "i[myLen]/I", 32000);
        branch2->SetAutoDelete(kFALSE);
        branch3->SetAutoDelete(kFALSE);
        branch4->SetAutoDelete(kFALSE);
        for (Long64_t ev = 1; ev < fEventCount + 1; ev++) {

          for (Int_t idx = 0; idx < (ev % 10); idx++) {
            i[idx] = f_counter;
            f[idx] = f_counter++;
            d[idx] = f_counter + 1;
          }

          myLen = ev % 10;
          tree->Fill();
        } 
        hfile = tree->GetCurrentFile();
        hfile->Write();
        tree->Print();
        printf("Successful write of all events.\n");

        delete hfile;
    }
};
constexpr Long64_t BulkApiVariableTest::fClusterSize;
constexpr Long64_t BulkApiVariableTest::fEventCount;



TEST_F(BulkApiVariableTest, stdRead)
{
    auto hfile = TFile::Open(fFileName.c_str());
    printf("Starting read of file %s.\n", fFileName.c_str());
    TStopwatch sw;

    printf("Using standard read APIs.\n");

    TTreeReader myReader("T", hfile);
    TTreeReaderArray<float> myF(myReader, "f");
    TTreeReaderArray<double> myD(myReader, "d");
    TTreeReaderValue<int> myI(myReader, "myLen");
    Long64_t ev = 1;
    float idx_f = 0;
    double idx_g = 2;
    Int_t events = fEventCount;

    sw.Start();
    while (myReader.Next()) {
        if (R__unlikely(*myI != (ev % 10))) {
            printf("Incorrect number of entries on myLen branch: %d, expected %lld (event %lld)\n",
                   *myI, ev % 10, ev);
            ASSERT_TRUE(false);
        }         
        if (R__unlikely(myF.GetSize() != static_cast<size_t>(ev % 10))) {
            printf("Incorrect number of entries on float branch: %lu, expected %lld (event %lld)\n",
                   myF.GetSize(),
                   ev % 10,
                   ev);
            ASSERT_TRUE(false);
        }
        if (R__unlikely(myD.GetSize() != static_cast<size_t>(ev % 10))) {
            printf("Incorrect number of entries on double branch: %lu, expected %lld (event %lld)\n",
                   myD.GetSize(),
                   ev % 10,
                   ev);
            ASSERT_TRUE(false);
        }
        for (int idx = 0; idx < *myI; idx++) {
            float tree_f = myF[idx];
            double tree_d = myD[idx];
            if (R__unlikely((ev < 16000000) && (tree_f != idx_f))) {
                printf("Incorrect value on float branch: %f, expected %f (event %lld, entry %d)\n",
                       tree_f, idx_f, ev, idx);
                ASSERT_TRUE(false);
            }
            idx_f++;
            if (R__unlikely((ev < 1600000) && (tree_d != idx_g))) {
                printf("Incorrect value on double branch: %f, expected %f (event %lld, entry %d)\n",
                       tree_d, idx_g, ev, idx);
                ASSERT_TRUE(false);
            }
            idx_g++;
        }
        ev++;
    }
    ASSERT_EQ(ev, events+1);

    sw.Stop();
    printf("TTreeReader: Successful read of all events.\n");
    printf("TTreeReader: Total elapsed time (seconds) for standard APIs: %.2f\n", sw.RealTime());
}

TEST_F(BulkApiVariableTest, serializedRead)
{
    auto hfile = TFile::Open(fFileName.c_str());
    printf("Starting read of file %s.\n", fFileName.c_str());
    TStopwatch sw;

    printf("Using serialized bulk APIs.\n");

    auto tree = dynamic_cast<TTree*>(hfile->Get("T"));
    ASSERT_TRUE(tree);
    auto branchLen = tree->GetBranch("myLen");
    ASSERT_TRUE(branchLen);
    auto branchFloat = tree->GetBranch("f");
    ASSERT_TRUE(branchFloat);
    auto branchDouble = tree->GetBranch("d");
    ASSERT_TRUE(branchDouble);
    auto branchInt = tree->GetBranch("i");
    ASSERT_TRUE(branchInt);

    int idx_i = 0;
    float idx_f = 0;
    double idx_d = 2;
    Long64_t evt_idx = 0;
    Long64_t events = fEventCount;
    Int_t cluster_size = std::min(fClusterSize, fEventCount);
    TBufferFile floatBuf(TBuffer::kWrite, 32*1024);
    TBufferFile doubleBuf(TBuffer::kWrite, 32*1024);
    TBufferFile intBuf(TBuffer::kWrite, 32*1024);
    // Count buffer and len buffer should be identical.
    TBufferFile countBuf(TBuffer::kWrite, 32*1024);
    TBufferFile lenBuf(TBuffer::kWrite, 32*1024);

    while (events) {
        auto count = branchFloat->GetBulkRead().GetEntriesSerialized(evt_idx, floatBuf, &countBuf);
        ASSERT_EQ(count, cluster_size);
        count = branchDouble->GetBulkRead().GetEntriesSerialized(evt_idx, doubleBuf);
        ASSERT_EQ(count, cluster_size);
        count = branchInt->GetBulkRead().GetEntriesSerialized(evt_idx, intBuf);
        ASSERT_EQ(count, cluster_size);
        count = branchLen->GetBulkRead().GetEntriesSerialized(evt_idx, lenBuf);
        ASSERT_EQ(count, cluster_size);

        if (events > count) {
            events -= count;
        } else {
            events = 0;
        }
        char *float_buf = floatBuf.GetCurrent();
        char *double_buf = doubleBuf.GetCurrent();
        char *int_buf = intBuf.GetCurrent();
        int *entry_count_buf = reinterpret_cast<int*>(countBuf.GetCurrent());
        int *entry_len_buf = reinterpret_cast<int*>(countBuf.GetCurrent());
        for (Int_t idx = 0; idx<count; idx++) {
            int entry_count, entry_len;
            char *tmp_buf = reinterpret_cast<char*>(entry_count_buf + idx);
            frombuf(tmp_buf, &entry_count);
            tmp_buf = reinterpret_cast<char*>(entry_len_buf + idx);
            frombuf(tmp_buf, &entry_len);
            //printf("Event %lld has %d entries.\n", evt_idx + idx + 1, entry_count);
            // These should reference the same basket, just through different APIs.
            ASSERT_EQ(entry_count, entry_len);
            if (R__unlikely(entry_count != ((evt_idx + idx + 1) % 10))) {
                printf("Incorrect number of entries on myLen branch: %d, expected %lld (event %lld)\n",
                       entry_count, (evt_idx + idx + 1) % 10, evt_idx + idx);
                ASSERT_TRUE(false);
            }

            for (int entry_idx=0; entry_idx<entry_count; entry_idx++) {
                float entry_f;
                frombuf(float_buf, &entry_f);
                double entry_d;
                frombuf(double_buf, &entry_d);
                int entry_i;
                frombuf(int_buf, &entry_i);
                //printf("Event %lld entry %d (buffer %p) has int value %d\n", evt_idx + idx, entry_idx, int_buf, entry_i);
                if (R__unlikely((evt_idx < 1600000) && (entry_i != idx_i))) {
                    printf("Incorrect value on int branch: %d, expected %d (event %lld, entry %d)\n", entry_i, idx_i, evt_idx + idx, entry_idx);
                    ASSERT_TRUE(false);
                }
                idx_i++;
                //printf("Event %lld entry %d (buffer %p) has float value %f\n", evt_idx + idx, entry_idx, float_buf, entry_f);
                if (R__unlikely((evt_idx < 1600000) && (entry_f != idx_f))) {
                    printf("Incorrect value on float branch: %f, expected %f (diff %f, event %lld)\n", entry_f, idx_f, fabs(entry_f - idx_f), evt_idx + idx);
                    ASSERT_TRUE(false);
                }
                idx_f++;
                if (R__unlikely((evt_idx < 1600000) && (entry_d != idx_d))) {
                    printf("Incorrect value on double branch: %f, expected %f (diff %f, event %lld)\n", entry_d, idx_d, fabs(entry_d - idx_d), evt_idx + idx);
                    ASSERT_TRUE(false);
                }
                idx_d++;
            }

        }
        evt_idx += count;
    }
    events = fEventCount;
    ASSERT_EQ(evt_idx, events);

    sw.Stop();
    printf("Bulk Serialized API: Successful read of all events.\n");
    printf("Bulk Serialized API: Total elapsed time (seconds) for API: %.2f\n", sw.RealTime());
}
