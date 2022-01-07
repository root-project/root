#ifndef ROOT_RDF_COUNTERHELPER
#define ROOT_RDF_COUNTERHELPER

#include <ROOT/RDataFrame.hxx>

#include <atomic>
#include <functional>
#include <memory>

class CounterHelper : public ROOT::Detail::RDF::RActionImpl<CounterHelper> {
   std::shared_ptr<std::atomic_uint> fNCalls; // final result
public:
   CounterHelper() : fNCalls(std::make_shared<std::atomic_uint>(0u)) {}
   CounterHelper(CounterHelper &&) = default;
   CounterHelper(const CounterHelper &) = default;

   using Result_t = std::atomic_uint;
   std::shared_ptr<std::atomic_uint> GetResultPtr() const { return fNCalls; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}
#ifndef COUNTERHELPER_CALLBACKMODE
   void Exec(unsigned int) { ++(*fNCalls); }
#else
   void Exec(unsigned int) {}
   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int, const ROOT::RDF::RSampleInfo &) mutable { ++(*this->fNCalls); };
   }
#endif
   void Finalize() {}

   std::string GetActionName() { return "ThreadSafeCounter"; }
};

#endif
