
#include "TProfile.h"
#include "TROOT.h"
#include "TObject.h"
#include <thread>
#include <memory>
#include <vector>
#include <atomic>
#include <sstream>

int main()
{
  std::atomic<bool> canStart{false};
  std::vector<std::unique_ptr<TProfile>> profiles;
  std::vector<std::thread> threads;

  TH1::AddDirectory(kFALSE);

  //Tell Root we want to be multi-threaded
  ROOT::EnableThreadSafety();
  //When threading, also have to keep ROOT from logging all TObjects into a list
  TObject::SetObjectStat(false);

  for(unsigned int i=0; i<20; ++i) {
    std::ostringstream s;
    s<<"Dummy"<<i;
    profiles.push_back(std::unique_ptr<TProfile>(new TProfile(s.str().c_str(),s.str().c_str(), 100,10,11,0,10)));
    profiles.back()->SetCanExtend(TH1::kAllAxes);
    auto profile = profiles.back().get();
    threads.emplace_back([profile,&canStart]() {
        while(!canStart) {}
        for(int x=10; x>0; --x) {
          for(int y=0; y<20; ++y) {
            profile->Fill(double(x), double(y),1.);
          }
        }
      });
  }
  canStart = true;

  for(auto& thread: threads) {
    thread.join();
  }

  // Print Stats
  for (auto&& profile : profiles) {
    TProfile *p = profile.get();
    printf("\n--------Profile\n");
    printf("- Num entries: %f\n",p->GetEntries());
    printf("- Mean x: %f\n",p->GetMean());
    printf("- RMS x: %f\n",p->GetRMS());
    printf("- Mean y: %f\n",p->GetMean(2));
    printf("- RMS y: %f\n",p->GetRMS(2));
  }

  return 0;
}
