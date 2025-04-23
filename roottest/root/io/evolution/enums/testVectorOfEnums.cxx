#include "classes.hxx"

#include <TEnum.h>
#include <TFile.h>
#include <TTree.h>
#include <TClass.h>
#include <TStreamerInfo.h>
#include <TError.h>

#include <TClass.h>
#include <TVirtualStreamerInfo.h>
#include <TStreamerElement.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>


void printData(const CalArray<PadFlags> *data)
{
   for(auto e : data->mFlags)
      std::cout << "  Value: " << static_cast<int>( e ) << '\n';
   std::cout << "  PadSubset: " << +static_cast<std::underlying_type_t<PadSubset>>(data->mPadSubset) << '\n';
}

void printData(const std::vector<CalArray<PadFlags>> *vec)
{
   std::cout << "  CalArrays\n";
   for(const auto &c : *vec)
      printData(&c);
}

void printData(const std::vector<CalArray2<PadFlags>> *vec)
{
   std::cout << "  CalArrays\n";
   for(const auto &c : *vec)
      printData(&c);
}

void printEvent(Event *ev)
{
   std::cout << "First flag: " << static_cast<std::underlying_type_t<PadFlags>>(ev->mMainFlag) << '\n';
   std::cout << "Canary (1): " << std::dec << static_cast<int>(ev->mCanary1) << '\n';
   std::cout << "Second flag: " << static_cast<std::underlying_type_t<PadFlags>>(ev->mSecondFlag) << '\n';
   std::cout << "Canary (2): " <<  std::dec << static_cast<int>(ev->mCanary2) << '\n';
   std::cout << "mFlags size: " << ev->mFlags.size() << std::endl;
   for(auto e : ev->mFlags)
      std::cout << "Value: " << static_cast<int>( e ) << '\n';
   printData(&ev->mData);
}

void fillEvent(Event *ev)
{
   // Content: 0, 0, 2, 1
   ev->mFlags = std::vector<PadFlags>(2, PadFlags::kConst);
   ev->mFlags.push_back(PadFlags::kTwo);
   ev->mFlags.push_back(PadFlags::kOne);

   // Set transient members;
   ev->mCanary1 = 10;
   ev->mCanary2 = 20;

   CalArray<PadFlags> c;
   c.mFlags.push_back(PadFlags::kThree);
   c.mFlags.push_back(PadFlags::kFour);
   c.mPadSubset = PadSubset::kFive;

   ev->mData.push_back(c);
   ev->mData.push_back(c);

   std::cout << "After initial filling:\n";
   printEvent(ev);
}

void printEvent(UnevenEvent *ev)
{
   std::cout << "When filled expecting:\n";
   std::cout << "  mPad: 4,3,2,1,3\n";
   std::cout << "  mChar: 5,4,3,2,1\n";
   for(auto e : ev->mPad)
      std::cout << "mPad: " << static_cast<int>( e ) << '\n';
   for(auto e : ev->mChar)
      std::cout << "mChar: " << static_cast<int>( e ) << '\n';
}

void fillEvent(UnevenEvent *ev)
{
  ev->mPad.clear();
  ev->mPad.push_back(PadFlags::kFour);
  ev->mPad.push_back(PadFlags::kThree);
  ev->mPad.push_back(PadFlags::kTwo);
  ev->mPad.push_back(PadFlags::kOne);
  ev->mPad.push_back(PadFlags::kThree);
  
  ev->mChar.clear();
  ev->mChar.push_back(PadSubset::kFive);
  ev->mChar.push_back(PadSubset::kFour);
  ev->mChar.push_back(PadSubset::kThree);
  ev->mChar.push_back(PadSubset::kTwo);
  ev->mChar.push_back(PadSubset::kOne);
};

void writeObject(const char *filename)
{
   TClass::GetClass("CalArray<PadFlags>")->GetStreamerInfo()->ls();

   std::cout << "Size of PadFlags: " << sizeof(PadFlags) << std::endl;

   auto en = TEnum::GetEnum("PadFlags");
   std::cout << "Enum underlying type: " << en->GetUnderlyingType() << std::endl;

   auto f = TFile::Open(filename, "RECREATE");
   Event *ev = new Event();
   fillEvent(ev);

   f->WriteObject(ev, "event");

   UnevenEvent uev;
   fillEvent(&uev);
   printEvent(&uev);
   
   f->WriteObject(&uev, "uneven");

   f->Write();
   delete f;
   delete ev;
}

void readObject(const char *filename, bool broken)
{
   auto f = TFile::Open(filename);
   if (!f || f->IsZombie())
      Fatal("Enum test readObject", "Cannot open file %s\n", filename);
   if (broken) {
      // Check that the file was indeed written by an older version of ROOT.
      if (f->GetVersion() > 63402)
         Fatal("Enum test readObject",
               "File %s was written by version of ROOT (%d) which it too recent."
               " We need v6.34.02 or older\n", f->GetName(), f->GetVersion());
   }

   auto ev = f->Get<Event>("event");
   if (!ev)
     Fatal("Enum test readobject", "Missing 'event' in %s\n", f->GetName());

   std::cout << "After reading:\n";
   printEvent(ev);

   auto uev = f->Get<UnevenEvent>("uneven");
   if (!uev) 
     Fatal("Enum test readobject", "Missing 'uneven' in %s\n", f->GetName());
   printEvent(uev);
}

void writeTree(const char *filename)
{
   auto f = TFile::Open(filename, "RECREATE");
   Event *ev = new Event();
   fillEvent(ev);

   auto t = new TTree("t", "t");
   t->Branch("ev.", &ev);
   t->Fill();

   f->Write();
   delete f;
   delete ev;
}

void readTree(const char *filename, bool broken)
{
   Event *ev = new Event();

   auto f = TFile::Open(filename, "READ");
   if (!f || f->IsZombie())
      Fatal("Enum test readTree", "Cannot open file %s\n", filename);
   if (broken) {
      // Check that the file was indeed written by an older version of ROOT.
      if (f->GetVersion() > 63402)
         Fatal("Enum test readobject", "File %s was written by version of ROOT (%d) which it too recent."
               " We need v6.34.02 or older\n", f->GetName(), f->GetVersion());
   }
  auto t = f->Get<TTree>("t");

   // t->Scan("*");
   // printEvent(ev);

   t->SetBranchAddress("ev.", &ev);
   t->GetEntry(0);

   printEvent(ev);

   delete f;
}

int main(int argc, char **argv)
{
   const bool needwrite = (argc > 1 && strcmp(argv[1], "w") == 0);
   const bool needtree = (argc > 1 && strcmp(argv[1], "t") == 0);

   const char *objfile = "test.root";
   const char *treefile = "treetest.root";
   const char *objfile_v634 = "test_v634.root";
   const char *treefile_v634 = "treetest_v634.root";

   if (needwrite) {
       writeObject(objfile);
       writeTree(treefile);
   } else {
      if (1) {
         // Test old files rules used
         readObject(objfile_v634, true);
         if (needtree) 
            readTree(treefile_v634, true);
      }
      if (1) {
         // Test new files, rules no used.
         readObject(objfile, false);
         if (needtree) 
            readTree(treefile, false);
      }
   }


   return 0;
}


