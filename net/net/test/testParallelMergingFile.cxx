#include <gtest/gtest.h>

#include <thread>
#include <memory>
#include <mutex>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/TestSupport.hxx>

#include "TParallelMergingFile.h"
#include "TServerSocket.h"
#include "TFileMerger.h"

enum EStatusKind {
   // These values are dictated by TParallelMergingFile::UploadAndReset().
   kStartConnection = 0,
   kProtocolVersion = 1,
   kProtocol = 1,
};

static std::mutex gMutex;
static std::atomic_int gNReceived;

static void Server(std::unique_ptr<TServerSocket> ss, const std::string &outFile)
{
   // NOTE: only accepts a single client.
   auto client = std::unique_ptr<TSocket>(ss->Accept());
   ASSERT_NE(client, nullptr);

   // initial handshake
   UInt_t clientIdx = 0;
   client->Send(clientIdx, kStartConnection);
   client->Send(kProtocolVersion, kProtocol);

   TFileMerger merger{/* isLocal = */ false};
   merger.SetMergeOptions(TString("rntuple.MergingMode=Union"));
   {
      bool ok = merger.OutputFile(outFile.c_str(), "RECREATE");
      ASSERT_TRUE(ok);
   }

   // Client loop
   bool alive = true;
   while (alive) {
      TMessage *msg = nullptr;
      client->Recv(msg);
      ASSERT_NE(msg, nullptr);

      switch (msg->What()) {
      case kMESS_STRING: {
         // This is how the TParallelMergingFile terminates its connection (it sends a message containing
         // the string "Finished").
         char str[64];
         msg->ReadString(str, sizeof(str));
         alive = false;
      } break;

      case kMESS_ANY: {
         // Receive and merge data
         Int_t clientId;
         TString filename;
         Long64_t length;
         msg->ReadInt(clientId);
         msg->ReadTString(filename);
         msg->ReadLong64(length);

         // XXX: this lock is here to work around https://github.com/root-project/root/issues/20641
         // Note that we don't care about minimizing its scope since we are not testing multithreaded scaling here
         // and it's fine if the client's and server's operations get serialized by the mutexes.
         std::scoped_lock<std::mutex> lock(gMutex);
         auto input = std::make_unique<TMemFile>((std::string("server_") + filename), msg->Buffer() + msg->Length(),
                                                 length, "READ");

         msg->SetBufferOffset(msg->Length() + length);

         EXPECT_NE(input->Get<ROOT::RNTuple>("ntpl"), nullptr);

         merger.AddFile(input.get());
         bool mergeOk = merger.PartialMerge(TFileMerger::kAllIncremental | TFileMerger::kKeepCompression);
         EXPECT_TRUE(mergeOk);
         ++gNReceived;

      } break;

      default: std::cout << "ignoring message of kind " << msg->What() << "\n";
      }

      delete msg;
   }
}

TEST(TParallelMergingFile, UploadAndResetNonTObject)
{
   constexpr auto sockPath = "/tmp/parallelMergeTest.sock";

   // Start server
   gSystem->Unlink(sockPath);
   auto ss = std::make_unique<TServerSocket>(sockPath);
   ASSERT_TRUE(ss->IsValid());

   struct Cleanup {
      const char *fSockPath;
      Cleanup(const char *sockPath) : fSockPath(sockPath) {}
      ~Cleanup() { gSystem->Unlink(fSockPath); }
   } cleanup(sockPath);

   ROOT::TestSupport::FileRaii fileGuardMerged("parallelFileMerged.root");

   std::thread serverThread(Server, std::move(ss), fileGuardMerged.GetPath());

   // Create "client-side" file
   auto *file = dynamic_cast<TParallelMergingFile *>(
      TFile::Open((std::string("parallelMergeTest.root?pmerge=") + sockPath).c_str(), "RECREATE"));
   ASSERT_NE(file, nullptr);
   file->SetCacheWrite(nullptr);

   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kWarning, "TParallelMergingFile::ResetObjects", "can not be ResetAfterMerge", false);
   diags.requiredDiag(kWarning, "TFileMerger::MergeRecursive", "Merging RNTuples is experimental");

   constexpr auto kNEntries = 10000;
   constexpr auto kNUploads = 4;
   {
      auto model = ROOT::RNTupleModel::CreateBare();
      model->MakeField<float>("px");
      model->MakeField<float>("py");
      auto writer = ROOT::RNTupleWriter::Append(model->Clone(), "ntpl", *file);
      auto entry = writer->CreateEntry();
      auto px = entry->GetPtr<float>("px");
      auto py = entry->GetPtr<float>("py");
      int nSent = 0;
      for (int i = 0; i < kNEntries; ++i) {
         *px = i;
         *py = 2 * i;
         writer->Fill(*entry);
         if (i > 0 && (i % (kNEntries / kNUploads)) == 0) {
            {
               // XXX: this lock is here to work around https://github.com/root-project/root/issues/20641
               std::scoped_lock<std::mutex> lock(gMutex);
               // Force UploadAndReset() by destroying the RNTupleWriter, which in turn will call CommitDataset()
               // which ultimately calls file->Write().
               // IMPORTANT: writer.reset() must be called explicitly, since we must not create the new RNTuple
               // before destroying the old one (otherwise the header will get a stale address and end up corrupted).
               writer.reset();
               ++nSent;
            }

            while (gNReceived < nSent) {
               // Wait for the server to process the reset.
               std::this_thread::yield();
            }

            {
               std::scoped_lock<std::mutex> lock(gMutex);
               writer = ROOT::RNTupleWriter::Append(model->Clone(), "ntpl", *file);
            }
            entry = writer->CreateEntry();
            px = entry->GetPtr<float>("px");
            py = entry->GetPtr<float>("py");
         }
      }
   }

   delete file;

   serverThread.join();

   {
      auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuardMerged.GetPath());
      EXPECT_EQ(reader->GetNEntries(), kNEntries);

      auto px = reader->GetModel().GetDefaultEntry().GetPtr<float>("px");
      auto py = reader->GetModel().GetDefaultEntry().GetPtr<float>("py");
      for (auto idx : reader->GetEntryRange()) {
         reader->LoadEntry(idx);
         EXPECT_FLOAT_EQ(*px, idx);
         EXPECT_FLOAT_EQ(*py, 2 * idx);
      }
   }
}
