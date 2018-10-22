// @(#)root/roofit:$Id$
// Author: Stephan Hageboeck, CERN, 10/2018
/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** Memory pool for RooArgSet and RooDataSet.
 * \class MemPoolForRooSets
 * \ingroup roofitcore
 * RooArgSet and RooDataSet were using a mempool that guarantees that allocating,
 * de-allocating and re-allocating a set does not yield the same pointer. Since
 * both were using the same logic, the functionality has been put in this class.
 * This class solves RooFit's static destruction order problems by intentionally leaking
 * arenas of the mempool that still contain live objects at the end of the program.
 */

#ifndef ROOFIT_ROOFITCORE_SRC_MEMPOOLFORROOSETS_H_
#define ROOFIT_ROOFITCORE_SRC_MEMPOOLFORROOSETS_H_

#include <vector>
#include <algorithm>

template <class RooSet_t, std::size_t POOLSIZE>
class MemPoolForRooSets {

  struct Arena {
    Arena()
      : memBegin{static_cast<RooSet_t *>(::operator new(POOLSIZE * sizeof(RooSet_t)))}, nextItem{memBegin},
        memEnd{memBegin + POOLSIZE}, refCount{0}, teardownMode{false}
    {
    }

    Arena(const Arena &) = delete;
    Arena(Arena && other)
      : memBegin{other.memBegin}, nextItem{other.nextItem}, memEnd{other.memEnd}, refCount{other.refCount},
	teardownMode{other.teardownMode}
#ifndef NDEBUG
      , deletedElements { std::move(other.deletedElements) }
#endif
    {
      // Needed for unique ownership
      other.memBegin = nullptr;
      other.refCount = 0;
    }

    Arena & operator=(const Arena &) = delete;
    Arena & operator=(Arena && other)
    {
      memBegin = other.memBegin;
      nextItem = other.nextItem;
      memEnd   = other.memEnd;
#ifndef NDEBUG
      deletedElements = std::move(other.deletedElements);
#endif
      refCount     = other.refCount;
      teardownMode = other.teardownMode;

      other.memBegin = nullptr;
      other.refCount = 0;

      return *this;
    }

    // If there is any user left, the arena shouldn't be deleted.
    // If this happens, nevertheless, one has an order of destruction problem.
    ~Arena()
    {
      if (!memBegin) return;

      if (refCount != 0) {
        std::cerr << __FILE__ << ":" << __LINE__ << "Deleting arena " << memBegin << " with use count " << refCount
                  << std::endl;
        assert(false);
      }

      ::operator delete(memBegin);
    }

    bool inPool(void * ptr) const
    {
      auto       thePtr = static_cast<RooSet_t *>(ptr);
      const bool inPool = memBegin <= thePtr && thePtr < memBegin + POOLSIZE;
      return inPool;
    }

    bool isFull() const { return nextItem >= memBegin + POOLSIZE; }

    bool empty() const { return refCount == 0; }

    void * tryAllocate()
    {
      if (isFull()) return nullptr;

      ++refCount;
      return nextItem++;
    }

    bool tryDeallocate(void * ptr)
    {
      if (inPool(ptr)) {
        --refCount;
#ifndef NDEBUG
        const std::size_t index = static_cast<RooSet_t *>(ptr) - memBegin;
        assert(deletedElements.count(index) == 0);
        deletedElements.insert(index);
#endif
        return true;
      } else
        return false;
    }

    RooSet_t * memBegin;
    RooSet_t * nextItem;
    RooSet_t * memEnd;
    std::size_t refCount;
    bool        teardownMode;
#ifndef NDEBUG
    std::set<std::size_t> deletedElements;
#endif
  };

  private:
  std::vector<Arena> fArenas;
  bool               fTeardownMode{false};

  public:
  /// Create empty mem pool.
  MemPoolForRooSets() : fArenas{} {}

  MemPoolForRooSets(const MemPoolForRooSets &) = delete;
  MemPoolForRooSets(MemPoolForRooSets &&)      = delete;
  MemPoolForRooSets & operator=(const MemPoolForRooSets &) = delete;
  MemPoolForRooSets & operator=(MemPoolForRooSets &&) = delete;

  /// Destructor. Should not be called when RooArgSets or RooDataSets are still alive.
  ~MemPoolForRooSets()
  {
    if (!empty()) {
      std::cerr << __PRETTY_FUNCTION__ << " The mem pool being deleted is not empty. This will lead to crashes."
                << std::endl;
      assert(false);
    }
  }

  /// Allocate memory for the templated set type. Fails if bytes != sizeof(RooSet_t).
  void * allocate(std::size_t bytes)
  {
    assert(bytes == sizeof(RooSet_t));

    if (fArenas.empty() || fArenas.back().isFull()) {
      prune();
      fArenas.emplace_back();
    }

    void * ptr = fArenas.back().tryAllocate();
    assert(ptr != nullptr);

    return ptr;
  }

  /// Deallocate memory for the templated set type if in pool.
  ///\return True if element was in pool.
  bool deallocate(void * ptr)
  {
    bool deallocSuccess = false;
    for (auto & arena : fArenas) {
      if (arena.tryDeallocate(ptr)) {
        deallocSuccess = true;
        break;
      }
    }

    if (fTeardownMode) {
      // Try pruning after each dealloc because we are tearing down
      prune();
    }

    return deallocSuccess;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Delete arenas that don't have space and no users.
  /// In fTeardownMode, it will also delete the arena that still has space.
  ///
  void prune()
  {
    bool doTeardown = fTeardownMode;

    auto shouldFree = [doTeardown](Arena & arena) -> bool {
      if (arena.refCount == 0 && (doTeardown || arena.isFull())) {
        return true;
      }

      return false;
    };

    fArenas.erase(std::remove_if(fArenas.begin(), fArenas.end(), shouldFree), fArenas.end());
  }

  /// Test if pool is empty.
  bool empty() const
  {
    return std::all_of(fArenas.begin(), fArenas.end(), [](const Arena & ar) { return ar.empty(); });
  }

  /// Set pool to teardown mode (at program end).
  /// Will prune all empty arenas. Non-empty arenas will survive until all contained elements
  /// are deleted. They may therefore leak if not all elements are destructed.
  void teardown()
  {
    fTeardownMode = true;
    for (auto & arena : fArenas) {
      arena.teardownMode = true;
    }

    prune();
  }
};

#endif /* ROOFIT_ROOFITCORE_SRC_MEMPOOLFORROOSETS_H_ */
