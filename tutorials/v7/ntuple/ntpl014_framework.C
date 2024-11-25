/// \file
/// \ingroup tutorial_ntuple
/// \notebook
///
/// Example of framework usage for writing RNTuples:
/// 1. Creation of (bare) RNTupleModels and RFieldTokens.
/// 2. Creation of RNTupleWriter and RNTupleParallelWriter when appending to a single TFile.
/// 3. Creation of RNTupleFillContext and (bare) REntry per thread, and usage of BindRawPtr.
/// 4. Usage of FillNoFlush(), RNTupleFillStatus::ShouldFlushCluster(), FlushColumns(), and FlushCluster().
///
/// Please note that this tutorial has very simplified versions of classes that could be found in a framework, such as
/// DataProduct, FileService, ParallelOutputter, and SerializingOutputter. They try to mimick the usage in a framework
/// (for example, Outputters are agnostic of the data written, which is encapsulated in std::vector<DataProduct>), but
/// are not meant for production usage!
///
/// Also note that this tutorial uses std::thread and std::mutex directly instead of a task scheduling library such as
/// Threading Building Blocks (TBB). For that reason, turning on ROOT's implicit multithreading (IMT) would not be very
/// efficient with the simplified code in this tutorial because a thread blocking to acquire a std::mutex cannot "help"
/// the other thread that is currently in the critical section by executing its tasks. If that is wanted, the framework
/// should use synchronization methods provided by TBB directly (which goes beyond the scope of this tutorial).
///
/// \macro_code
///
/// \date September 2024
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/REntry.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RNTupleFillStatus.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleParallelWriter.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>

#include <cassert>
#include <cstddef>    // for std::size_t
#include <cstdint>    // for std::uint32_t
#include <functional> // for std::ref
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <utility> // for std::pair
#include <vector>

// Import classes from Experimental namespace for the time being
using ROOT::Experimental::REntry;
using ROOT::Experimental::RNTupleFillContext;
using ROOT::Experimental::RNTupleFillStatus;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleParallelWriter;
using ROOT::Experimental::RNTupleWriteOptions;
using ROOT::Experimental::RNTupleWriter;

using ModelTokensPair = std::pair<std::unique_ptr<RNTupleModel>, std::vector<REntry::RFieldToken>>;

// A DataProduct associates an arbitrary address to an index in the model.
struct DataProduct {
   std::size_t index;
   void *address;

   DataProduct(std::size_t i, void *a) : index(i), address(a) {}
};

// The FileService opens a TFile and provides synchronization.
class FileService {
   std::unique_ptr<TFile> fFile;
   std::mutex fMutex;

public:
   FileService(std::string_view url, std::string_view options = "")
   {
      fFile.reset(TFile::Open(std::string(url).c_str(), std::string(options).c_str()));
      // The file is automatically closed when destructing the std::unique_ptr.
   }

   TFile &GetFile() { return *fFile; }
   std::mutex &GetMutex() { return fMutex; }
};

// An Outputter provides the interface to fill DataProducts into an RNTuple.
class Outputter {
public:
   virtual ~Outputter() = default;

   virtual void InitSlot(unsigned slot) = 0;
   virtual void Fill(unsigned slot, const std::vector<DataProduct> &products) = 0;
};

// A ParallelOutputter uses an RNTupleParallelWriter to append an RNTuple to a TFile.
class ParallelOutputter final : public Outputter {
   FileService &fFileService;
   std::unique_ptr<RNTupleParallelWriter> fParallelWriter;
   std::vector<REntry::RFieldToken> fTokens;

   struct SlotData {
      std::shared_ptr<RNTupleFillContext> fillContext;
      std::unique_ptr<REntry> entry;
   };
   std::vector<SlotData> fSlots;

public:
   ParallelOutputter(ModelTokensPair modelTokens, FileService &fileService, std::string_view ntupleName,
                     const RNTupleWriteOptions &options)
      : fFileService(fileService), fTokens(std::move(modelTokens.second))
   {
      auto &model = modelTokens.first;

      std::lock_guard g(fileService.GetMutex());
      fParallelWriter = RNTupleParallelWriter::Append(std::move(model), ntupleName, fFileService.GetFile(), options);
   }

   void InitSlot(unsigned slot) final
   {
      if (slot >= fSlots.size()) {
         fSlots.resize(slot + 1);
      }
      // Create an RNTupleFillContext and REntry that are used for all fills from this slot. We recommend creating a
      // bare entry if binding all fields.
      fSlots[slot].fillContext = fParallelWriter->CreateFillContext();
      fSlots[slot].entry = fSlots[slot].fillContext->GetModel().CreateBareEntry();
   }

   void Fill(unsigned slot, const std::vector<DataProduct> &products) final
   {
      assert(slot < fSlots.size());
      auto &fillContext = *fSlots[slot].fillContext;
      auto &entry = *fSlots[slot].entry;

      // Use the field tokens to bind the products' raw pointers.
      for (auto &&product : products) {
         entry.BindRawPtr(fTokens[product.index], product.address);
      }

      // Fill the entry without triggering an implicit flush.
      RNTupleFillStatus status;
      fillContext.FillNoFlush(entry, status);
      if (status.ShouldFlushCluster()) {
         // If we are asked to flush, first try to do as much work as possible outside of the critical section:
         // FlushColumns() will flush column data and trigger compression, but not actually write to storage.
         // (A framework may of course also decide to flush more often.)
         fillContext.FlushColumns();

         {
            // FlushCluster() will flush data to the underlying TFile, so it requires synchronization.
            std::lock_guard g(fFileService.GetMutex());
            fillContext.FlushCluster();
         }
      }
   }
};

// A SerializingOutputter uses a sequential RNTupleWriter to append an RNTuple to a TFile and a std::mutex to
// synchronize multiple threads. Note that ROOT's implicit multithreading would not be very efficient with this
// implementation because a thread blocking to acquire a std::mutex cannot "help" the other thread that is currently
// in the critical section by executing its tasks. See also the note at the top of the file.
class SerializingOutputter final : public Outputter {
   FileService &fFileService;
   std::unique_ptr<RNTupleWriter> fWriter;
   std::mutex fWriterMutex;
   std::vector<REntry::RFieldToken> fTokens;

   struct SlotData {
      std::unique_ptr<REntry> entry;
   };
   std::vector<SlotData> fSlots;

public:
   SerializingOutputter(ModelTokensPair modelTokens, FileService &fileService, std::string_view ntupleName,
                        const RNTupleWriteOptions &options)
      : fFileService(fileService), fTokens(std::move(modelTokens.second))
   {
      auto &model = modelTokens.first;

      std::lock_guard g(fileService.GetMutex());
      fWriter = RNTupleWriter::Append(std::move(model), ntupleName, fileService.GetFile(), options);
   }

   void InitSlot(unsigned slot) final
   {
      if (slot >= fSlots.size()) {
         fSlots.resize(slot + 1);
      }
      // Create an REntry that is used for all fills from this slot. We recommend creating a bare entry if binding all
      // fields.
      fSlots[slot].entry = fWriter->GetModel().CreateBareEntry();
   }

   void Fill(unsigned slot, const std::vector<DataProduct> &products) final
   {
      assert(slot < fSlots.size());
      auto &entry = *fSlots[slot].entry;

      // Use the field tokens to bind the products' raw pointers.
      for (auto &&product : products) {
         entry.BindRawPtr(fTokens[product.index], product.address);
      }

      {
         // Fill the entry without triggering an implicit flush. This requires synchronization with other threads using
         // the same writer, but not (yet) with the underlying TFile.
         std::lock_guard g(fWriterMutex);
         RNTupleFillStatus status;
         fWriter->FillNoFlush(entry, status);
         if (status.ShouldFlushCluster()) {
            // If we are asked to flush, first try to do as much work as possible outside of the critical section:
            // FlushColumns() will flush column data and trigger compression, but not actually write to storage.
            // (A framework may of course also decide to flush more often.)
            fWriter->FlushColumns();

            {
               // FlushCluster() will flush data to the underlying TFile, so it requires synchronization.
               std::lock_guard g(fFileService.GetMutex());
               fWriter->FlushCluster();
            }
         }
      }
   }
};

// === END OF TUTORIAL FRAMEWORK CODE ===

// Simple structs to store events
struct Track {
   float eta;
   float mass;
   float pt;
   float phi;
};

struct ChargedTrack : public Track {
   std::int8_t charge;
};

struct Event {
   std::uint32_t eventId;
   std::uint32_t runId;
   std::vector<ChargedTrack> electrons;
   std::vector<Track> photons;
   std::vector<ChargedTrack> muons;
};

// RNTupleModel for Events; in a real framework, this would likely be dynamic.
ModelTokensPair CreateEventModel()
{
   // We recommend creating a bare model if the default entry is not used.
   auto model = RNTupleModel::CreateBare();
   // For more efficient access, also create field tokens.
   std::vector<REntry::RFieldToken> tokens;

   model->MakeField<decltype(Event::eventId)>("eventId");
   tokens.push_back(model->GetToken("eventId"));

   model->MakeField<decltype(Event::runId)>("runId");
   tokens.push_back(model->GetToken("runId"));

   model->MakeField<decltype(Event::electrons)>("electrons");
   tokens.push_back(model->GetToken("electrons"));

   model->MakeField<decltype(Event::photons)>("photons");
   tokens.push_back(model->GetToken("photons"));

   model->MakeField<decltype(Event::muons)>("muons");
   tokens.push_back(model->GetToken("muons"));

   return {std::move(model), std::move(tokens)};
}

// DataProducts with addresses that point into the Event object.
std::vector<DataProduct> CreateEventDataProducts(Event &event)
{
   std::vector<DataProduct> products;
   // The indices have to match the order of std::vector<REntry::RFieldToken> above.
   products.emplace_back(0, &event.eventId);
   products.emplace_back(1, &event.runId);
   products.emplace_back(2, &event.electrons);
   products.emplace_back(3, &event.photons);
   products.emplace_back(4, &event.muons);
   return products;
}

// Simple struct to store runs
struct Run {
   std::uint32_t runId;
   std::uint32_t nEvents;
};

// RNTupleModel for Runs; in a real framework, this would likely be dynamic.
ModelTokensPair CreateRunModel()
{
   // We recommend creating a bare model if the default entry is not used.
   auto model = RNTupleModel::CreateBare();
   // For more efficient access, also create field tokens.
   std::vector<REntry::RFieldToken> tokens;

   model->MakeField<decltype(Run::runId)>("runId");
   tokens.push_back(model->GetToken("runId"));

   model->MakeField<decltype(Run::nEvents)>("nEvents");
   tokens.push_back(model->GetToken("nEvents"));

   return {std::move(model), std::move(tokens)};
}

// DataProducts with addresses that point into the Run object.
std::vector<DataProduct> CreateRunDataProducts(Run &run)
{
   std::vector<DataProduct> products;
   // The indices have to match the order of std::vector<REntry::RFieldToken> above.
   products.emplace_back(0, &run.runId);
   products.emplace_back(1, &run.nEvents);
   return products;
}

constexpr unsigned kNRunsPerThread = 100;
constexpr unsigned kMeanNEventsPerRun = 400;
constexpr unsigned kStddevNEventsPerRun = 100;
constexpr unsigned kMeanNTracks = 5;

void ProcessRunsAndEvents(unsigned threadId, Outputter &eventOutputter, Outputter &runOutputter)
{
   std::mt19937 gen(threadId);
   std::normal_distribution<double> nEventsDist(kMeanNEventsPerRun, kStddevNEventsPerRun);
   std::poisson_distribution<> nTracksDist(kMeanNTracks);
   std::uniform_real_distribution<float> floatDist;

   for (std::uint32_t runId = threadId * kNRunsPerThread; runId < (threadId + 1) * kNRunsPerThread; runId++) {
      double nEventsD = nEventsDist(gen);
      std::uint32_t nEvents = 0;
      if (nEventsD > 0) {
         nEvents = static_cast<std::uint32_t>(nEventsD);
      }

      // Process events, reusing a single Event object.
      Event event;
      event.runId = runId;
      auto eventProducts = CreateEventDataProducts(event);
      for (std::uint32_t eventId = 0; eventId < nEvents; eventId++) {
         event.eventId = eventId;

         // Produce some data; eta, phi, and pt are just filled with uniformly distributed data.
         event.electrons.resize(nTracksDist(gen));
         for (auto &electron : event.electrons) {
            electron.eta = floatDist(gen);
            electron.mass = 0.511 /* MeV */;
            electron.phi = floatDist(gen);
            electron.pt = floatDist(gen);
            electron.charge = (gen() % 2 ? 1 : -1);
         }
         event.photons.resize(nTracksDist(gen));
         for (auto &photon : event.photons) {
            photon.eta = floatDist(gen);
            photon.mass = 0;
            photon.phi = floatDist(gen);
            photon.pt = floatDist(gen);
         }
         event.muons.resize(nTracksDist(gen));
         for (auto &muon : event.muons) {
            muon.eta = floatDist(gen);
            muon.mass = 105.658 /* MeV */;
            muon.phi = floatDist(gen);
            muon.pt = floatDist(gen);
            muon.charge = (gen() % 2 ? 1 : -1);
         }

         eventOutputter.Fill(threadId, eventProducts);
      }

      // Fill the Run data.
      Run run;
      run.runId = runId;
      run.nEvents = nEvents;

      auto runProducts = CreateRunDataProducts(run);
      runOutputter.Fill(threadId, runProducts);
   }
}

constexpr unsigned kNThreads = 4;

void ntpl014_framework()
{
   FileService fileService("ntpl014_framework.root", "RECREATE");

   RNTupleWriteOptions options;
   // Parallel writing requires buffered writing; force it on (even if it is the default).
   options.SetUseBufferedWrite(true);
   // For demonstration purposes, reduce the cluster size to 2 MiB.
   options.SetApproxZippedClusterSize(2 * 1024 * 1024);
   ParallelOutputter eventOutputter(CreateEventModel(), fileService, "Events", options);

   // SerializingOutputter also relies on buffered writing; force it on (even if it is the default).
   options.SetUseBufferedWrite(true);
   // For demonstration purposes, reduce the cluster size for the very simple Run data to 1 KiB.
   options.SetApproxZippedClusterSize(1024);
   SerializingOutputter runOutputter(CreateRunModel(), fileService, "Runs", options);

   // Initialize slots in the two Outputters.
   for (unsigned i = 0; i < kNThreads; i++) {
      eventOutputter.InitSlot(i);
      runOutputter.InitSlot(i);
   }

   std::vector<std::thread> threads;
   for (unsigned i = 0; i < kNThreads; i++) {
      threads.emplace_back(ProcessRunsAndEvents, i, std::ref(eventOutputter), std::ref(runOutputter));
   }
   for (unsigned i = 0; i < kNThreads; i++) {
      threads[i].join();
   }
}
