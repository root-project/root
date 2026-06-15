/// \file ROOT/RNTupleProcessor.hxx
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-03-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleProcessor
#define ROOT_RNTupleProcessor

#include <ROOT/RNTupleComposer.hxx>

namespace ROOT {
namespace Experimental {

class RNTupleProcessor {
private:
   RNTupleComposer *fComposer;
   ROOT::NTupleSize_t fNEntriesProcessed = 0;

public:
   RNTupleProcessor(RNTupleComposer &composer) : fComposer(&composer) {}

   ROOT::NTupleSize_t GetNEntriesProcessed() const { return fNEntriesProcessed; }

   RNTupleComposer &GetComposer() { return *fComposer; }
   const RNTupleComposer &GetComposer() const { return *fComposer; }

   // clang-format off
   /**
   \class ROOT::Experimental::RNTupleComposer::RIterator
   \ingroup NTuple
   \brief Iterator over the entries of a composed RNTuple.
   */
   // clang-format on
   class RIterator {
   private:
      RNTupleProcessor &fProcessor;
      RNTupleComposer &fComposer;
      ROOT::NTupleSize_t fCurrentEntryNumber;

   public:
      using iterator_category = std::input_iterator_tag;
      using iterator = RIterator;
      using value_type = ROOT::NTupleSize_t;
      using difference_type = std::ptrdiff_t;
      using pointer = ROOT::NTupleSize_t *;
      using reference = ROOT::NTupleSize_t &;

      RIterator(RNTupleProcessor &processor, ROOT::NTupleSize_t entryNumber)
         : fProcessor(processor), fComposer(fProcessor.GetComposer()), fCurrentEntryNumber(entryNumber)
      {
         if (!fComposer.fEntry) {
            fCurrentEntryNumber = ROOT::kInvalidNTupleIndex;
         }
         // This constructor is called with kInvalidNTupleIndex for RNTupleComposer::end(). In that case, we already
         // know there is nothing to load.
         if (fCurrentEntryNumber != ROOT::kInvalidNTupleIndex) {
            fComposer.Connect(fComposer.fEntry->GetFieldIndices(), Internal::RNTupleProcessorProvenance(),
                              /*updateFields=*/false);
            fCurrentEntryNumber = fComposer.LoadEntry(fCurrentEntryNumber);
            if (fCurrentEntryNumber != ROOT::kInvalidNTupleIndex)
               fProcessor.fNEntriesProcessed++;
         }
      }

      iterator operator++()
      {
         fCurrentEntryNumber = fComposer.LoadEntry(fCurrentEntryNumber + 1);
         if (fCurrentEntryNumber != ROOT::kInvalidNTupleIndex)
            fProcessor.fNEntriesProcessed++;
         return *this;
      }

      iterator operator++(int)
      {
         auto obj = *this;
         ++(*this);
         return obj;
      }

      reference operator*() { return fCurrentEntryNumber; }

      friend bool operator!=(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryNumber != rh.fCurrentEntryNumber;
      }
      friend bool operator==(const iterator &lh, const iterator &rh)
      {
         return lh.fCurrentEntryNumber == rh.fCurrentEntryNumber;
      }
   };

   RIterator begin() { return RIterator(*this, 0); }
   RIterator end() { return RIterator(*this, ROOT::kInvalidNTupleIndex); }
};
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleProcessor
