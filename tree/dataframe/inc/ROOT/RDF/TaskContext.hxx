// Author: Axel Naumann CERN  2019-09-18

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFTASKCONTEXT
#define ROOT_RDFTASKCONTEXT

#include <algorithm>
#include <cassert>
#include <exception>
#include <memory>
#include <vector>

/// \cond HIDDEN_SYMBOLS

/// page-per-task, collection of those for all tasks
/// collection of those, for all pages == pool

namespace ROOT {
namespace Internal {
namespace RDF {

class PageForTasks {
private:
   constexpr static size_t kPageSize = 4*1024;
   size_t fFreeBytes; /// free bytes, identical for all tasks' pages.

   /// Memory pages for each task.
   /// Even if realloc'ed, the addresses of the pages' memory are stable.
   std::vector<std::unique_ptr<char>> fPageForEachTask;

   static size_t AtLeastPageSize(size_t reserve)
   {
      return (reserve < kPageSize) ? kPageSize : reserve;
   }

   void AllocatePages(size_t nbytes)
   {
      // TODO: consider combining allocs into one single big chunk.
      for (auto &&pageForATask: fPageForEachTask)
         pageForATask.reset(new char[nbytes]);
   }

public:
   PageForTasks(size_t nTasks, size_t reserve = kPageSize):
      fFreeBytes(AtLeastPageSize(reserve)), fPageForEachTask(nTasks)
   {
      AllocatePages(fFreeBytes);
   }

   size_t NumFreeBytes() const { return fFreeBytes; }

   std::vector<char *> Allocate(size_t nBytes)
   {
      if (NumFreeBytes() < nBytes)
         throw std::runtime_error("Slab too small!");
      fFreeBytes -= nBytes;

      std::vector<char *> ret;
      ret.reserve(fPageForEachTask.size());
      for (auto &&pageForTask: fPageForEachTask)
         ret.emplace_back(pageForTask.get() + NumFreeBytes()); // Allocating backwards!
      return ret;
   }

   size_t size() const { return fPageForEachTask.size(); }
};

class TaskContextStorage {
   std::vector<PageForTasks> fPagesForTasks; /// pool for each task

   PageForTasks &GetPageForBytes(size_t nBytes)
   {
      auto pageWithEnoughSpace
         = std::find_if(fPagesForTasks.begin(), fPagesForTasks.end(),
                        [nBytes](const PageForTasks& pageForTasks) {
                           return pageForTasks.NumFreeBytes() > nBytes; }
                        );
      if (pageWithEnoughSpace != fPagesForTasks.end())
         return *pageWithEnoughSpace;

      fPagesForTasks.emplace_back(size(), nBytes);
      return fPagesForTasks.back();
   }

public:
   /// Construct TaskContextStorage, providing the estimated number of tasks.
   /// Constructs the first storage page for each task.
   TaskContextStorage(size_t expectedNumTasks)
   {
      PageForTasks elem{expectedNumTasks};
      fPagesForTasks.emplace_back(std::move(elem));
   }

   std::vector<char *> Allocate(size_t nbytes) {
      auto &pageWithEnoughSpace = GetPageForBytes(nbytes);
      return pageWithEnoughSpace.Allocate(nbytes);
   }

   size_t size() const { return fPagesForTasks[0].size(); }
};

/// This adaptor accesses per-task data of type `Element`.
/// The data is distributed such that it is non-consecutive across
/// tasks, but instead kept in a pool per task, to avoid false sharing.
template <class Element>
class FSVector {
   std::vector<char *> fAddresses;

   template <class... Args>
   void Construct(const Args&... args)
   {
      for (auto &&mem: fAddresses)
         new (mem) Element(args...);
   }

   void Destroy()
   {
      for (auto &&mem: fAddresses)
         reinterpret_cast<Element*>(mem)->~Element();
   }

public:
   template<class ElementPtrIter>
   class iter_impl {
   public:
      using value_type = Element;
      using reference = Element &;
      using pointer = Element *;
      using difference_type = std::ptrdiff_t;
      using iterator_category	= std::random_access_iterator_tag;

      ElementPtrIter fUnderlyingIter;

      reference operator*() { return *reinterpret_cast<Element*>(*fUnderlyingIter); }
      pointer operator->() { return reinterpret_cast<Element*>(*fUnderlyingIter); }

      iter_impl& operator++()
      {
         ++fUnderlyingIter;
         return *this;
      }

      iter_impl operator++(int)
      {
         const auto ret = *this;
         ++*this;
         return ret;
      }

      friend bool operator==(const iter_impl &lhs, const iter_impl &rhs)
      {
         return lhs.fUnderlyingIter == rhs.fUnderlyingIter;
      }
      friend bool operator!=(const iter_impl &lhs, const iter_impl &rhs)
      {
         return !(lhs==rhs);
      }

      friend size_t operator-(const iter_impl &lhs, const iter_impl &rhs)
      {
         return lhs.fUnderlyingIter - rhs.fUnderlyingIter;
      }
   };

   using iterator = iter_impl<typename std::vector<char *>::iterator>;
   using const_iterator = iter_impl<typename std::vector<char *>::const_iterator>;

   FSVector() = default;
   template<class...Args>
   FSVector(TaskContextStorage& storage, Args&&... args):
      fAddresses(storage.Allocate(sizeof(Element)))
   {
      Construct(std::forward<Args>(args)...);
   }

   FSVector(FSVector &other) = default;

   FSVector(FSVector &&other): fAddresses(std::move(other.fAddresses))
   {
      other.fAddresses.clear();
   }

   ~FSVector() {
      Destroy();
   }

   FSVector &operator=(FSVector &other) = default;

   FSVector &operator=(FSVector &&other) {
      fAddresses = std::move(other.fAddresses);
      other.fAddresses.clear();
   }

   Element &operator[](size_t taskID) {
      assert(taskID < fAddresses.size() && "Cannot extend storage pool for new tasks yet!");
      return *reinterpret_cast<Element*>(fAddresses[taskID]);
   }

   const Element &operator[](size_t taskID) const {
      assert(taskID < fAddresses.size() && "Cannot extend storage pool for new tasks yet!");
      return *reinterpret_cast<const Element*>(fAddresses[taskID]);
   }

   size_t size() const { return fAddresses.size(); }
   iterator begin() { return {fAddresses.begin()}; }
   const_iterator begin() const { return {fAddresses.begin()}; }
   iterator end() { return {fAddresses.end()}; }
   const_iterator end() const { return {fAddresses.end()}; }

   void clear() {
      Destroy();
      fAddresses.clear(); // empty once, cannot resize afterwards!
   }
};

} // end of NS RDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif // ROOT_RDFTASKCONTEXT
