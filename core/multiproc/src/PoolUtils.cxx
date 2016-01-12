#include "PoolUtils.h"
#include "TClass.h"
#include "TList.h"
#include <iostream>

//////////////////////////////////////////////////////////////////////////
/// Merge collection of TObjects.
/// This function looks for an implementation of the Merge method
/// (e.g. TH1F::Merge) and calls it on the objects contained in objs.
/// If Merge is not found, a null pointer is returned.
TObject* PoolUtils::ReduceObjects(const std::vector<TObject *>& objs)
{
   if(objs.size() == 0)
      return nullptr;

   if(objs.size() == 1)
      return objs[0];

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
