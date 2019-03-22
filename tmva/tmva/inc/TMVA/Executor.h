// @(#)root/tmva $Id$   
// Author: Lorenzo Moneta
/*************************************************************************
 * Copyright (C) 2019, ROOT/TMVA                                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//
//  Defining Executor classes to be used in TMVA
// wrapping the functionality of the ROOT TThreadExecutor and
// ROOT TSequential Executor 
//  
/////////////////////////////////////////////////////////////////////////
#ifndef ROOT_TMVA_Executor
#define ROOT_TMVA_Executor

#include <memory>

#include <ROOT/TSequentialExecutor.hxx>
#ifdef R__USE_IMT
#include <ROOT/TThreadExecutor.hxx>
#endif


namespace TMVA {


/// Base Excutor class
class Executor {

public:

   template< class F, class... T>
   using noReferenceCond = typename std::enable_if<"Function can't return a reference" &&
                                                   !(std::is_reference<typename std::result_of<F(T...)>::type>::value)>::type;

  

   //////////////////////////////////////
   /// constructor of TMVA Executor class
   /// if ROOT::EnableIMplicitMT has not been called then by default a serial executor will be created
   /// A user can create a thread pool and enable multi-thread excution by calling TMVA::Config::Instance()::EnableMT(nthreads)
   /// FOr releasing the thread pool used by TMVA one can do it by calling  TMVA::Config::Instance()::DisableMT() or
   /// calling TMVA::Config::Instance()::EnableMT with only one thread 
   ////////////////////////////////////////////
   explicit Executor(int nthreads  = 0) {
      if (ROOT::IsImplicitMTEnabled() || nthreads > 1 ) {
#ifdef R__USE_IMT 
         fMTExecImpl = (nthreads == 0) ? std::unique_ptr< ROOT::TThreadExecutor>(new ROOT::TThreadExecutor()) :
            std::unique_ptr< ROOT::TThreadExecutor>(new ROOT::TThreadExecutor(nthreads));
#else
         ::Error("Executor","Cannot have multi threads when ROOT is built without IMT");
#endif
      }
      // case of single thread usage
      if (!fMTExecImpl)
         fSeqExecImpl = std::unique_ptr<ROOT::TSequentialExecutor>(new ROOT::TSequentialExecutor());
   }

   ROOT::TThreadExecutor * GetMultiThreadExecutor() {
      if (fMTExecImpl) return fMTExecImpl.get();
      else {
         fMTExecImpl =  std::unique_ptr< ROOT::TThreadExecutor>(new ROOT::TThreadExecutor());
         Info("GetThreadExecutor","Creating a TThread executor with a pool with a defult size of %d",fMTExecImpl->GetPoolSize()); 
         return fMTExecImpl.get();
      }
   }

   unsigned int GetPoolSize() const {
      if (!fMTExecImpl) return 1;
      return fMTExecImpl->GetPoolSize(); 
   }

   /// wrap TExecutor::Foreach
   template<class Function> 
   void Foreach(Function func, unsigned int nTimes, unsigned nChunks = 0) {
      if (fMTExecImpl) fMTExecImpl->Foreach(func,nTimes, nChunks);
      else fSeqExecImpl->Foreach(func,nTimes); 
   }
   template<class Function, class T> 
   void Foreach(Function func, std::vector<T> & args, unsigned nChunks = 0) {
      if (fMTExecImpl) fMTExecImpl->Foreach(func,args, nChunks);
      else fSeqExecImpl->Foreach(func, args); 
   }
   template<class Function, class INTEGER>
   void Foreach(Function func, ROOT::TSeq<INTEGER> args, unsigned nChunks = 0){
      if (fMTExecImpl) fMTExecImpl->Foreach(func,args, nChunks);
      else fSeqExecImpl->Foreach(func, args); 
   }

   /// Wrap TExecutor::Map functions
   template<class F, class Cond = noReferenceCond<F>>
   auto Map(F func, unsigned nTimes) -> std::vector<typename std::result_of<F()>::type>  {
      if (fMTExecImpl) return fMTExecImpl->Map(func,nTimes);
      else return fSeqExecImpl->Map(func, nTimes); 
   }
   template<class F, class INTEGER, class Cond = noReferenceCond<F, INTEGER>>
   auto Map(F func, ROOT::TSeq<INTEGER> args) -> std::vector<typename std::result_of<F(INTEGER)>::type> { 
      if (fMTExecImpl) return fMTExecImpl->Map(func,args);
      else return fSeqExecImpl->Map(func, args); 
   }

   /// Wrap TExecutor::MapReduce functions
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc) -> typename std::result_of<F(INTEGER)>::type {
      if (fMTExecImpl) return fMTExecImpl->MapReduce(func, args, redfunc);
      else return fSeqExecImpl->MapReduce(func, args, redfunc); 
   }
   template<class F, class INTEGER, class R, class Cond = noReferenceCond<F, INTEGER>>
   auto MapReduce(F func, ROOT::TSeq<INTEGER> args, R redfunc, unsigned nChunks) -> typename std::result_of<F(INTEGER)>::type {
      if (fMTExecImpl) return fMTExecImpl->MapReduce(func, args, redfunc, nChunks);
      else return fSeqExecImpl->MapReduce(func, args, redfunc); 
   }
  
   

   ///Wrap Reduce function
   template<class T, class R>
   auto Reduce(const std::vector<T> &objs, R redfunc) -> decltype(redfunc(objs)) {
      if (fMTExecImpl) return fMTExecImpl->Reduce(objs, redfunc);
      else return fSeqExecImpl->Reduce(objs, redfunc); 
   }
   //template<class T> T* Reduce(const std::vector<T*> &mergeObjs);
   
#ifdef R__USE_IMT
   std::unique_ptr<ROOT::TThreadExecutor>  fMTExecImpl;
#else
   std::unique_ptr<ROOT::TSequentialExecutor> fMTExecImpl; // if not using MT the two pointers will be of same type 
#endif
   std::unique_ptr<ROOT::TSequentialExecutor> fSeqExecImpl; 
};

}  // end namespace TMVA

#endif
