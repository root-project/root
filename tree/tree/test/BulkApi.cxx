
#include <stdio.h>

#include "TBranch.h"
#include "TBufferFile.h"
#include "TFile.h"
#include "TTree.h"
#include "TStopwatch.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "ROOT/TTreeReaderFast.hxx"
#include "ROOT/TTreeReaderValueFast.hxx"
#include "ROOT/TBulkBranchRead.hxx"


#include "gtest/gtest.h"


class BulkApiTest : public ::testing::Test {
public:
    static constexpr Int_t fEventCount = 1e7;
    const std::string fFileName = "BulkApiTest.root";

protected:
    virtual void SetUp()
    {
        auto hfile = new TFile(fFileName.c_str(), "RECREATE", "TTree float micro benchmark ROOT file");
        hfile->SetCompressionLevel(0); // No compression at all.

        // Otherwise, we keep with the current ROOT defaults.
        auto tree = new TTree("T", "A ROOT tree of floats.");
        float f = 2;
        TBranch *branch2 = tree->Branch("myFloat", &f, 320000, 1);
        branch2->SetAutoDelete(kFALSE);
        for (Long64_t ev = 0; ev < fEventCount; ev++) {
          tree->Fill();
          f ++;
        }
        hfile = tree->GetCurrentFile();
        hfile->Write();
        tree->Print();
        printf("Successful write of all events.\n");
        hfile->Close();

        delete hfile;
    }
};


TEST_F(BulkApiTest, stdRead)
{
    auto hfile = TFile::Open(fFileName.c_str());
    printf("Starting read of file %s.\n", fFileName.c_str());
    TStopwatch sw;

    printf("Using standard read APIs.\n");
    // Read via standard APIs.
    TTreeReader myReader("T", hfile);
    TTreeReaderValue<float> myF(myReader, "myFloat");
    Long64_t idx = 0;
    float idx_f = 1;
    Int_t events = fEventCount;
    sw.Start();
    while (myReader.Next()) {
        if (R__unlikely(idx == events)) {break;}
        idx_f++;
        if (R__unlikely((idx < 16000000) && (*myF != idx_f))) {
            printf("Incorrect value on myFloat branch: %f, expected %f (event %lld)\n", *myF, idx_f, idx);
            ASSERT_TRUE(false);
        }
        idx++;
    }
    sw.Stop();
    printf("TTreeReader: Successful read of all events.\n");
    printf("TTreeReader: Total elapsed time (seconds) for standard APIs: %.2f\n", sw.RealTime());
}

TEST_F(BulkApiTest, simpleRead)
{
    auto hfile = TFile::Open(fFileName.c_str());
    printf("Starting read of file %s.\n", fFileName.c_str());
    TStopwatch sw;

    printf("Using inline bulk read APIs.\n");
    TBufferFile branchbuf(TBuffer::kWrite, 32*1024);
    TTree *tree = dynamic_cast<TTree*>(hfile->Get("T"));
    ASSERT_TRUE(tree);

    TBranch *branchF = tree->GetBranch("myFloat");
    ASSERT_TRUE(branchF);

    Int_t events = fEventCount;
    float idx_f = 1;
    Long64_t evt_idx = 0;
    while (events) {
        auto count = branchF->GetBulkRead().GetEntriesSerialized(evt_idx, branchbuf);
        ASSERT_GE(count, 0);
        events = events > count ? (events - count) : 0;

        float *entry = reinterpret_cast<float*>(branchbuf.GetCurrent());
        for (Int_t idx=0; idx<count; idx++) {
            idx_f++;
            Int_t *buf = reinterpret_cast<Int_t*>(&entry[idx]);
            *buf = __builtin_bswap32(*buf);

            if (R__unlikely((evt_idx < 16000000) && (entry[idx] != idx_f))) {
                printf("Incorrect value on myFloat branch: %f (event %lld)\n", entry[idx], evt_idx + idx);
                ASSERT_TRUE(false);
            }
        }
        evt_idx += count;
    }
    sw.Stop();
    printf("GetEntriesSerialized: Successful read of all events.\n");
    printf("GetEntriesSerialized: Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
}
    /*else {
            printf("Using bulk read APIs.\n");
            // Read using bulk APIs.
            TBufferFile branchbuf(TBuffer::kWrite, 32*1024);
            TTree *tree = dynamic_cast<TTree*>(hfile->Get("T"));
            if (!tree) {
                std::cout << "Failed to fetch tree named 'T' from input file.\n";
                return 1;
            }
            TBranch *branchF = tree->GetBranch("myFloat");
            if (!branchF) {
                std::cout << "Unable to find branch 'myFloat' in tree 'T'\n";
                return 1;
            }
            sw.Start();
            float idx_f = 1;
            Long64_t evt_idx = 0;
            while (events) {
                auto count = branchF->GetBulkRead().GetEntriesFast(evt_idx, branchbuf);
                if (R__unlikely(count < 0)) {
                    printf("Failed to get entries via the 'fast' method for index %d.\n", evt_idx);
                    return 1;
                }
                if (events > count) {
                    events -= count;
                } else {
                    events = 0;
                }
                float *entry = reinterpret_cast<float*>(branchbuf.GetCurrent());
                for (Int_t idx=0; idx<count; idx++) {
                    idx_f++;
//
//                    Int_t *buf = reinterpret_cast<Int_t*>(&entry[idx]);
//                    *buf = __builtin_bswap32(*buf);
//

                    if (R__unlikely((evt_idx < 16000000) && (entry[idx] != idx_f))) {
                        printf("Incorrect value on myFloat branch: %f (event %ld)\n", entry[idx], evt_idx + idx);
                        return 1;
                    }
                }
                evt_idx += count;
            }
        }
        sw.Stop();
        printf("Successful read of all events.\n");
        printf("Total elapsed time (seconds) for bulk APIs: %.2f\n", sw.RealTime());
    } else {
        if (!do_std) {
            printf("There are currently no bulk APIs for writing.\n");
            return 1;
        }
        hfile = new TFile(fname, "RECREATE", "TTree float micro benchmark ROOT file");
        if (do_lz4) {
            hfile->SetCompressionLevel(7);  // High enough to get L4Z-HC
            hfile->SetCompressionAlgorithm(4);  // Enable LZ4 codec.
        } else if (do_uncompressed) {
            hfile->SetCompressionLevel(0); // No compression at all.
        } else if (do_zip) {
            hfile->SetCompressionLevel(6);
            hfile->SetCompressionAlgorithm(1);
        } else if (do_lzma) {
            hfile->SetCompressionLevel(6);
            hfile->SetCompressionAlgorithm(2); // LZMA
        }
        // Otherwise, we keep with the current ROOT defaults.
        tree = new TTree("T", "A ROOT tree of floats.");
        float f = 2;
        TBranch *branch2 = tree->Branch("myFloat", &f, 320000, 1);
        branch2->SetAutoDelete(kFALSE);
        for (Long64_t ev = 0; ev < events; ev++) {
          tree->Fill();
          f ++;
        }
        hfile = tree->GetCurrentFile();
        hfile->Write();
        tree->Print();
        printf("Successful write of all events.\n");
    }
    hfile->Close();

    return 0;
}
*/
