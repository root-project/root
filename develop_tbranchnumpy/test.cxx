#include <iostream>
#include <deque>

#include "TFile.h"
#include "TTree.h"
#include "TBufferFile.h"
#include <ROOT/TBulkBranchRead.hxx>

// void test() {
//   std::cout << "start" << std::endl;

//   TFile *file = new TFile("TTJets_13TeV_amcatnloFXFX_pythia8_2_77.root");
//   TTree *tree;
//   file->GetObject("Events", tree);

// // tree->Print();

//   TBufferFile buf_evtNum(TBuffer::kWrite, 32*1024);
//   TBufferFile buf_nPU(TBuffer::kWrite, 32*1024);
//   TBufferFile buf_pvx(TBuffer::kWrite, 32*1024);

//   TBranch *evtNum = tree->GetBranch("evtNum");
//   TBranch *nPU = tree->GetBranch("nPU");
//   TBranch *pvx = tree->GetBranch("pvx");

//   Long64_t evt_evtNum = 0;
//   Long64_t evt_nPU = 0;
//   Long64_t evt_pvx= 0;
//   while (0 <= evt_evtNum  &&  evt_evtNum < tree->GetEntries()) {
//     auto count_evtNum = evtNum->GetBulkRead().GetEntriesSerialized(evt_evtNum, buf_evtNum);
//     auto count_nPU = nPU->GetBulkRead().GetEntriesSerialized(evt_nPU, buf_nPU);
//     auto count_pvx = pvx->GetBulkRead().GetEntriesSerialized(evt_pvx, buf_pvx);

//     evt_evtNum += count_evtNum;
//     evt_nPU += count_nPU;
//     evt_pvx += count_pvx;

//     std::cout << evt_evtNum << " " << evt_nPU << " " << evt_pvx << std::endl;

//     if (count_evtNum <= 0  ||  count_nPU <= 0  ||  count_pvx <= 0)
//       break;


//     // float *entry = reinterpret_cast<float*>(branchbuf.GetCurrent());
//     // for (Int_t idx = 0;  idx < count;  idx++) {
//     //   Int_t *buf = reinterpret_cast<Int_t*>(&entry[idx]);
//     //   *buf = __builtin_bswap32(*buf);
//     // }
//   }

//   std::cout << "done" << std::endl;
// }

// void test() {
//   std::cout << "start" << std::endl;

//   TFile *file = new TFile("TrackResonanceNtuple.root");
//   TTree *tree;
//   file->GetObject("TrackResonanceNtuple/twoMuon", tree);

//   tree->Print();

//   TBufferFile buf_mass_mumu(TBuffer::kWrite, 32*1024);
//   TBufferFile buf_px(TBuffer::kWrite, 32*1024);
//   TBufferFile buf_py(TBuffer::kWrite, 32*1024);
//   TBufferFile buf_pz(TBuffer::kWrite, 32*1024);

//   TBranch *mass_mumu = tree->GetBranch("mass_mumu");
//   TBranch *px = tree->GetBranch("px");
//   TBranch *py = tree->GetBranch("py");
//   TBranch *pz = tree->GetBranch("pz");

//   Long64_t evt_mass_mumu = 0;
//   Long64_t evt_px = 0;
//   Long64_t evt_py = 0;
//   Long64_t evt_pz= 0;
//   while (evt_mass_mumu < tree->GetEntries()) {
//     evt_mass_mumu += mass_mumu->GetBulkRead().GetEntriesSerialized(evt_mass_mumu, buf_mass_mumu);
//     evt_px += px->GetBulkRead().GetEntriesSerialized(evt_px, buf_px);
//     evt_py += py->GetBulkRead().GetEntriesSerialized(evt_py, buf_py);
//     evt_pz += pz->GetBulkRead().GetEntriesSerialized(evt_pz, buf_pz);

//     std::cout << evt_mass_mumu << " " << evt_px << " " << evt_py << " " << evt_pz << std::endl;

//     // float *entry = reinterpret_cast<float*>(branchbuf.GetCurrent());
//     // for (Int_t idx = 0;  idx < count;  idx++) {
//     //   Int_t *buf = reinterpret_cast<Int_t*>(&entry[idx]);
//     //   *buf = __builtin_bswap32(*buf);
//     // }
//   }

//   std::cout << "done" << std::endl;
// }

class BasketBuffer {
public:
  Long64_t entry_start;
  Long64_t entry_end;
  TBufferFile buffer;

  BasketBuffer() : entry_start(0), entry_end(0), buffer(TBuffer::kWrite, 32*1024) {}

  void consume(Long64_t entry, TBranch* branch) {
    entry_start = entry;
    entry_end = entry_start + branch->GetBulkRead().GetEntriesSerialized(entry, buffer);
    if (entry_end <= entry_start)
      entry_end = -1;
  }
};

void test() {
  std::cout << "start" << std::endl;

  TFile *file = new TFile("TrackResonanceNtuple.root");
  TTree *tree;
  file->GetObject("TrackResonanceNtuple/twoMuon", tree);

  TBranch *mbranch = tree->GetBranch("mass_mumu");
  TBranch *pbranch = tree->GetBranch("px");

  std::deque<BasketBuffer*> mbuffer;
  std::deque<BasketBuffer*> pbuffer;

  mbuffer.push_back(new BasketBuffer);
  pbuffer.push_back(new BasketBuffer);

  Long64_t entry_start = 0;
  Long64_t entry_end = 0;

  do {
    // increment the buffers that are at the forefront
    if (mbuffer.back()->entry_end == entry_start) mbuffer.back()->consume(entry_start, mbranch);
    if (pbuffer.back()->entry_end == entry_start) pbuffer.back()->consume(entry_start, pbranch);

    // check for error conditions
    if (mbuffer.back()->entry_end < 0  ||  pbuffer.back()->entry_end < 0)
      break;

    // find maximum entry_end
    entry_end = -1;
    if (mbuffer.back()->entry_end > entry_end) entry_end = mbuffer.back()->entry_end;
    if (pbuffer.back()->entry_end > entry_end) entry_end = pbuffer.back()->entry_end;

    // bring all others up to at least entry_end

    while (mbuffer.back()->entry_end < entry_end) {
      BasketBuffer* buf;

      if (mbuffer.front()->entry_end <= entry_start) {
        // the front is no longer needed; move it to the back and reuse it
        buf = mbuffer.front();
        mbuffer.pop_front();
      }
      else {
        // the front is still needed; add a new buffer to the back
        buf = new BasketBuffer;
      }

      buf->consume(mbuffer.back()->entry_end, mbranch);
      mbuffer.push_back(buf);
    }

    while (pbuffer.back()->entry_end < entry_end) {
      BasketBuffer* buf;

      if (pbuffer.front()->entry_end <= entry_start) {
        // the front is no longer needed; move it to the back and reuse it
        buf = pbuffer.front();
        pbuffer.pop_front();
      }
      else {
        // the front is still needed; add a new buffer to the back
        buf = new BasketBuffer;
      }

      buf->consume(pbuffer.back()->entry_end, pbranch);
      pbuffer.push_back(buf);
    }

    std::cout << "entries " << entry_start << ":" << entry_end << " (size is " << (entry_end - entry_start) << ", max is " << tree->GetEntries() << ")" << std::endl;
    std::cout << "mbuffer ";
    for (int i = 0;  i < mbuffer.size();  i++)
      std::cout << mbuffer[i]->entry_start << ":" << mbuffer[i]->entry_end << " ";
    std::cout << std::endl;
    std::cout << "pbuffer ";
    for (int i = 0;  i < pbuffer.size();  i++)
      std::cout << pbuffer[i]->entry_start << ":" << pbuffer[i]->entry_end << " ";
    std::cout << std::endl << std::endl;

    entry_start = entry_end;
  } while (entry_end < tree->GetEntries());

  std::cout << "done" << std::endl;
}
