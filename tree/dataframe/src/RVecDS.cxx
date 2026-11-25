#include <ROOT/RVecDS.hxx>
#include <ROOT/RDF/RMaskedEntryRange.hxx>

void *ROOT::Internal::RDF::RVecDSColumnReader::GetImpl(std::size_t entryInBulk)
{
   return *fCachedArrayBegin + entryInBulk * fValueSize;
}
void ROOT::Internal::RDF::RVecDSColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   fLastEntry = mask.GetFirstEntry();
   fSetEntryFn(fSlot, fLastEntry);
}

ROOT::Internal::RDF::RVecDSColumnReader::RVecDSColumnReader(unsigned int slot,
                                                            std::function<bool(unsigned int, ULong64_t)> fn,
                                                            std::byte **arrayBeginPtr, std::size_t valueSize)
   : fSlot(slot),
     fSetEntryFn(fn),
     fCachedArrayBegin(arrayBeginPtr),
     fValueSize(valueSize),
     fLastEntry(std::numeric_limits<std::uint64_t>::max())
{
}
