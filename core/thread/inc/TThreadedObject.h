#ifndef ROOT_TTHREADEDOBJECT
#define ROOT_TTHREADEDOBJECT

#include <vector>
#include <functional>
#include <memory>

#ifndef ROOT_ThreadIndex
#include "ThreadIndex.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TError
#include "TError.h"
#endif

namespace ROOT {

   namespace Internal {

      namespace TThreadedObjectUtils {

         /// Return a copy of the object or a "Clone" if the copy constructor is not implemented.
         template<class T, bool isCopyConstructible = std::is_copy_constructible<T>::value>
         struct Cloner {
            static T *Clone(const T &obj) {
               return new T(obj);
            }
         };

         template<class T>
         struct Cloner<T, false> {
            static T *Clone(const T &obj) {
               return (T*)obj.Clone();
            }
         };

      } // End of namespace TThreadedObjectUtils
   } // End of namespace Internals

   namespace TThreadedObjectUtils {

      template<class T>
      using MergeFunctionType = std::function<void(std::shared_ptr<T>, std::vector<std::shared_ptr<T>>&)>;
      /// Merge TObjects
      template<class T>
      void MergeTObjects(std::shared_ptr<T> target, std::vector<std::shared_ptr<T>> &objs)
      {
         if (!target) return;
         TList objTList;
         // Cannot do better than this
         for (auto obj : objs) {
            if (obj && obj != target) objTList.Add(obj.get());
         }
         target->Merge(&objTList);
      }
   } // end of namespace TThreadedObjectUtils

   /**
    \class ROOT::TThreadedObject
    \brief A wrapper to make object instances thread private, lazily.
    \tparam T Class of the object to be made thread private (e.g. TH1F)
    \ingroup Multicore

    A wrapper which makes objects thread private. The methods of the underlying
    object can be invoked via the the arrow operator. The object is created in
    a specific thread lazily, i.e. upon invocation of one of its methods.
    */
   template<class T>
   class TThreadedObject {
   public:
      /// Construct the TThreaded object and the "model" of the thread private
      /// objects.
      template<class ...ARGS>
      TThreadedObject(ARGS... args):
      fModel(std::forward<ARGS>(args)...), fObjPointers(fMaxSlots, nullptr)  {};

      /// Access a particular slot which corresponds to a single thread.
      std::shared_ptr<T> GetAtSlot(unsigned i)
      {
         if ( i >= fMaxSlots) {
            Warning("TThreadedObject::Merge", "Maximum number of slots reached.");
            return nullptr;
         }
         auto objPointer = fObjPointers[i];
         if (!objPointer) {
            objPointer.reset(Internal::TThreadedObjectUtils::Cloner<T>::Clone(fModel));
            fObjPointers[i] = objPointer;
         }
         return objPointer;
      }

      /// Access the wrapped object and allow to call its methods
      T *operator->()
      {
         return GetAtSlot(fThrIndexer.GetThreadIndex()).get();
      }

      /// Merge all the thread private objects. Can be called once: it does not
      /// create any new object but destroys the present bookkeping.
      std::shared_ptr<T> Merge(TThreadedObjectUtils::MergeFunctionType<T> mergeFunction = TThreadedObjectUtils::MergeTObjects<T>)
      {
         // We do not return if we already merged.
         if (fIsMerged) {
            Warning("TThreadedObject::Merge", "This object was already merged. Returning the previous result.");
            return fObjPointers[0];
         }
         mergeFunction(fObjPointers[0], fObjPointers);
         fIsMerged = true;
         return fObjPointers[0];
      }

      /// Merge all the thread private objects. Can be called many times. It
      /// does create a new instance of class T to represent the "Sum" object.
      /// This method is not thread safe: correct or acceptable behaviours
      /// depend on the nature of T and of the merging function.
      std::unique_ptr<T> SnapshotMerge(TThreadedObjectUtils::MergeFunctionType<T> mergeFunction = TThreadedObjectUtils::MergeTObjects<T>)
      {
         if (fIsMerged) {
            Warning("TThreadedObject::SnapshotMerge", "This object was already merged. Returning the previous result.");
            return std::unique_ptr<T>(Internal::TThreadedObjectUtils::Cloner<T>::Clone(*fObjPointers[0].get()));
         }
         auto targetPtr = Internal::TThreadedObjectUtils::Cloner<T>::Clone(fModel);
         std::shared_ptr<T> targetPtrShared (targetPtr, [](T*){});
         mergeFunction(targetPtrShared, fObjPointers);
         return std::unique_ptr<T>(targetPtr);
      }

   private:
      static const unsigned fMaxSlots = 128;
      const T fModel;                               ///< Use to store a "model" of the object
      std::vector<std::shared_ptr<T>> fObjPointers; ///< A pointer per thread is kept.
      ROOT::Internal::ThreadIndexer fThrIndexer;    ///< Get the slot index for the threads
      bool fIsMerged = false;                       ///< Remember if the objects have been merged already
   };



   ////////////////////////////////////////////////////////////////////////////////
   /// Obtain a TThreadedObject instance
   /// \tparam T Class of the object to be made thread private (e.g. TH1F)
   /// \tparam ARGS Arguments of the constructor
   template<class T, class ...ARGS>
   TThreadedObject<T> MakeThreaded(ARGS &&... args)
   {
      return TThreadedObject<T>(std::forward<ARGS>(args)...);
   }

} // End ROOT namespace

#endif
