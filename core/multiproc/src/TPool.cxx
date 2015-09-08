#include "TPool.h"

//////////////////////////////////////////////////////////////////////////
/// Class constructor.
/// nWorkers is the number of times this ROOT session will be forked, i.e.
/// the number of workers that will be spawned.
TPool::TPool(unsigned nWorkers) : TMPClient(nWorkers)
{
   Reset();
}


//////////////////////////////////////////////////////////////////////////
/// Reset TPool's state.
void TPool::Reset()
{
   fNProcessed = 0;
   fNToProcess = 0;
   fWithArg = false;
}


//////////////////////////////////////////////////////////////////////////
/// Merge collection of TObjects.
/// This function looks for an implementation of the Merge method
/// (e.g. TH1F::Merge) and calls it on the objects contained in objs.
/// If Merge is not found, a null pointer is returned.
TObject* PoolUtils::ReduceObjects(const std::vector<TObject *>& objs)
{
   //get first object from objs
   TObject *obj = objs[0];
   //get merge function
   ROOT::MergeFunc_t merge = obj->IsA()->GetMerge();
   if(!merge) {
      std::cerr << "could not find merge method for TObject*\n. Aborting operation.";
      return nullptr;
   }

   //put the rest of the objs in a list
   TList mergelist;
   unsigned NObjs = objs.size();
   for(unsigned i=1; i<NObjs; ++i) //skip first object
      mergelist.Add(objs[i]);

   //call merge
   merge(obj, &mergelist, nullptr);
   mergelist.Delete();

   //return result
   return obj;
}


//////////////////////////////////////////////////////////////////////////
/// Reply to a worker who just sent a result.
/// If another argument to process exists, tell the worker. Otherwise
/// send a shutdown order.
void TPool::ReplyToResult(TSocket *s)
{
   if (fNProcessed < fNToProcess) {
      if (fWithArg)
         MPSend(s, EPoolCode::kExecFuncWithArg, fNProcessed);
      else
         MPSend(s, EPoolCode::kExecFunc);
      ++fNProcessed;
   } else
      MPSend(s, EMPCode::kShutdownOrder);
}
