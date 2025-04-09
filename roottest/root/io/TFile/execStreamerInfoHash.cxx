#include "TFile.h"
#include "TError.h"
#include "TLine.h"

/*
* 7cf9d5dc8c - (1 year, 5 months ago) fix hashing of streamer info — Josh Bendavid
* eda6328ce3 - (4 years, 8 months ago) Fix ROOT-9694, GetStreamerInfoList infinite recursion. — Philippe Canal
* b6523df224 - (4 years, 8 months ago) Fix ROOT-9662 race condition in TStreamerInfo loading. — Philippe Canal
* 66dfb08bd7 - (5 years ago) [IO] Do not process the streamerinfo record of a file if we read the si already — Danilo Piparo
*/

static const char * const gFileName = "sihash.root";

bool CreateFile(const char *filename = gFileName)
{
   Printf("Creating file %s", filename);
   std::unique_ptr<TFile> file(TFile::Open(filename, "RECREATE"));
   if (!file || file->IsZombie()) {
      Error("CreateFile", "Could not create the file %s", filename);
      return false;
   }
   TNamed n("content", "This is the content of the file");
   file->WriteTObject(&n);
   file->Write();
   return true;
}

bool CheckFile(TFile &file, const char *mode) {
   auto siList = file.GetStreamerInfoList();
   if (!siList) {
      Error("CheckFile",
            "The file %s has no StreamerInfo list during \"%s\".",
            file.GetName(), mode);
      return false;
   }
   auto si = siList->FindObject("TNamed");
   if (!si) {
      Error("CheckFile",
            "The file %s has no StreamerInfo for the TNamed class during \"%s\".",
            file.GetName(), mode);
      siList->ls();
      return false;
   }
   return true;
}

bool UpdateFile(const char *filename = gFileName)
{
   Printf("Updating file %s", filename);
   std::unique_ptr<TFile> file(TFile::Open(filename, "UPDATE"));
   if (!file || file->IsZombie()) {
      Error("UpdateFile", "Could not open the file %s", filename);
      return false;
   }
   bool result = CheckFile(*file, "update");
   TLine obj;
   // We need at least one StreamerInfo recorded to trigger a written of
   // the StreamerInfo record (which was incorrect before the correction
   // for issue https://github.com/root-project/root/issues/12842
   file->WriteTObject(&obj);
   file->Write();
   return result;
}

bool CheckFile(const char *filename = gFileName) {
   Printf("Checking file %s", filename);
   std::unique_ptr<TFile> file(TFile::Open(filename, "READ"));
   if (!file || file->IsZombie()) {
      Error("CheckFile", "Could not create the file %s", filename);
      return false;
   }
   return CheckFile(*file, "read");
}

int execStreamerInfoHash(const char *filename = gFileName)
{
   if (! CreateFile(filename))
      return 1;
   if (! CheckFile(filename))
      return 2;
   if (! UpdateFile(filename))
      return 3;
   if (! CheckFile(filename))
      return 4;
   return 0;
}
