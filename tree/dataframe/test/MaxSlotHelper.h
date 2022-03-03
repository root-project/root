#include <ROOT/RDataFrame.hxx>

#include <memory>

class MaxSlotHelper : public ROOT::Detail::RDF::RActionImpl<MaxSlotHelper> {
   const std::shared_ptr<unsigned int> fMaxSlot; // final result
   std::vector<unsigned int> fMaxSlots;          // per-thread partial results
public:
   MaxSlotHelper(unsigned int nSlots)
      : fMaxSlot(std::make_shared<unsigned int>(std::numeric_limits<unsigned int>::lowest())),
        fMaxSlots(nSlots, std::numeric_limits<unsigned int>::lowest())
   {
   }
   MaxSlotHelper(MaxSlotHelper &&) = default;
   MaxSlotHelper(const MaxSlotHelper &) = delete;
   using Result_t = unsigned int;
   std::shared_ptr<unsigned int> GetResultPtr() const { return fMaxSlot; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, unsigned int /*slot2*/) { fMaxSlots[slot] = std::max(fMaxSlots[slot], slot); }
   void Finalize() { *fMaxSlot = *std::max_element(fMaxSlots.begin(), fMaxSlots.end()); }

   std::string GetActionName() { return "MaxSlot"; }
};
