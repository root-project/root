// @(#)root/io:$Id$
// Author: Philippe Canal 05/2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TStreamerInfo.h"
#include "TStreamerInfoActions.h"
#include "TROOT.h"
#include "TStreamerElement.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"
#include "TError.h"
#include "TVirtualArray.h"
#include "TBufferFile.h"
#include "TMemberStreamer.h"
#include "TError.h"
#include "TClassEdit.h"
#include "TVirtualCollectionIterators.h"

static const Int_t kRegrouped = TStreamerInfo::kOffsetL;

// More possible optimizations:
// Avoid call the virtual version of TBuffer::ReadInt and co.
// Merge the Reading of the version and the looking up or the StreamerInfo
// Avoid if (bytecnt) inside the CheckByteCount routines and avoid multiple (mostly useless nested calls)
// Try to avoid if statement on onfile class being set (TBufferFile::ReadClassBuffer).

using namespace TStreamerInfoActions;

namespace TStreamerInfoActions 
{
   void TConfiguration::AddToOffset(Int_t delta)
   {
      // Add the (potentially negative) delta to all the configuration's offset.  This is used by
      // TBranchElement in the case of split sub-object.

      fOffset += delta;
   }

   void TConfiguredAction::PrintDebug(TBuffer &buf, void *addr) const
   {
      // Inform the user what we are about to stream.

      // Idea, we should find a way to print the name of the function
      if (fConfiguration) fConfiguration->PrintDebug(buf,addr);
   }

   void TConfiguration::Print() const
   {
      // Inform the user what we are about to stream.

      // Idea: We should print the name of the action function.
      TStreamerInfo *info = (TStreamerInfo*)fInfo;
      TStreamerElement *aElement = (TStreamerElement*)info->GetElems()[fElemId];

      printf("StreamerInfoAction, class:%s, name=%s, fType[%d]=%d,"
             " %s, offset=%d\n",
             info->GetClass()->GetName(), aElement->GetName(), fElemId, info->GetTypes()[fElemId],
             aElement->ClassName(), fOffset);
   }

   void TConfiguration::PrintDebug(TBuffer &buf, void *addr) const
   {
      // Inform the user what we are about to stream.

      if (gDebug > 1) {
         // Idea: We should print the name of the action function.
         TStreamerInfo *info = (TStreamerInfo*)fInfo;
         TStreamerElement *aElement = (TStreamerElement*)info->GetElems()[fElemId];

         printf("StreamerInfoAction, class:%s, name=%s, fType[%d]=%d,"
                " %s, bufpos=%d, arr=%p, offset=%d\n",
                info->GetClass()->GetName(), aElement->GetName(), fElemId, info->GetTypes()[fElemId],
                aElement->ClassName(), buf.Length(), addr, fOffset);
      }
   }

   void TLoopConfiguration::Print() const
   {
      // Inform the user what we are about to stream.

      printf("TLoopConfiguration: unconfigured\n");
   }
   

   struct TGenericConfiguration : TConfiguration {
      // Configuration of action using the legacy code.
      // Mostly to cancel out the PrintDebug.
   public:
      TGenericConfiguration(TVirtualStreamerInfo *info, UInt_t id, Int_t offset = 0) : TConfiguration(info,id,offset) {};
      void PrintDebug(TBuffer &, void *) const {
         // Since we call the old code, it will print the debug statement.
      }
   };


   Int_t GenericReadAction(TBuffer &buf, void *addr, const TConfiguration *config) 
   {
      char *obj = (char*)addr;
      TGenericConfiguration *conf = (TGenericConfiguration*)config;
      return ((TStreamerInfo*)conf->fInfo)->ReadBuffer(buf, &obj, conf->fElemId, 1, config->fOffset, 2);
   }

   Int_t GenericWriteAction(TBuffer &buf, void *addr, const TConfiguration *config) 
   {
      char *obj = (char*)addr;
      TGenericConfiguration *conf = (TGenericConfiguration*)config;
      return ((TStreamerInfo*)conf->fInfo)->WriteBufferAux(buf, &obj, conf->fElemId, 1, config->fOffset, 2);
   }

   template <typename T> 
   inline Int_t ReadBasicType(TBuffer &buf, void *addr, const TConfiguration *config)
   {
      T *x = (T*)( ((char*)addr) + config->fOffset );
      // Idea: Implement buf.ReadBasic/Primitive to avoid the return value
      buf >> *x;
      return 0;
   }

   template <typename T> 
   inline Int_t WriteBasicType(TBuffer &buf, void *addr, const TConfiguration *config)
   {
      T *x = (T*)( ((char*)addr) + config->fOffset );
      // Idea: Implement buf.ReadBasic/Primitive to avoid the return value
      buf << *x;
      return 0;
   }

   class TConfWithFactor : public TConfiguration {
      // Configuration object for the Float16/Double32 where a factor has been specified.
   public:
      Double_t fFactor;
      Double_t fXmin;
      TConfWithFactor(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, Double_t factor, Double_t xmin) : TConfiguration(info,id,offset),fFactor(factor),fXmin(xmin) {};
      virtual TConfiguration *Copy() { return new TConfWithFactor(*this); }
   };

   template <typename T>
   inline Int_t ReadBasicType_WithFactor(TBuffer &buf, void *addr, const TConfiguration *config)
   {
      // Stream a Float16 or Double32 where a factor has been specified.
      //a range was specified. We read an integer and convert it back to a double.

      TConfWithFactor *conf = (TConfWithFactor *)config;
      buf.ReadWithFactor((T*)( ((char*)addr) + config->fOffset ), conf->fFactor, conf->fXmin);
      return 0;
   }

   class TConfNoFactor : public TConfiguration {
      // Configuration object for the Float16/Double32 where a factor has been specified.
   public:
      Int_t fNbits;
      TConfNoFactor(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, Int_t nbits) : TConfiguration(info,id,offset),fNbits(nbits) {};
      virtual TConfiguration *Copy() { return new TConfNoFactor(*this); }
   };

   template <typename T>
   inline Int_t ReadBasicType_NoFactor(TBuffer &buf, void *addr, const TConfiguration *config)
   {
      // Stream a Float16 or Double32 where a factor has not been specified.

      TConfNoFactor *conf = (TConfNoFactor *)config;
      Int_t nbits = conf->fNbits;

      buf.ReadWithNbits( (T*)( ((char*)addr) + config->fOffset ), nbits );
      return 0;
   }

   inline Int_t ReadTString(TBuffer &buf, void *addr, const TConfiguration *config) 
   {
      // Read in a TString object.

      // Idea: We could separate the TString Streamer in its two parts and 
      // avoid the if (buf.IsReading()) and try having it inlined.
      ((TString*)(((char*)addr)+config->fOffset))->TString::Streamer(buf);
      return 0;
   }

   inline Int_t ReadTObject(TBuffer &buf, void *addr, const TConfiguration *config) 
   {
      // Read in a TObject object part.

      // Idea: We could separate the TObject Streamer in its two parts and 
      // avoid the if (buf.IsReading()).
      ((TObject*)(((char*)addr)+config->fOffset))->TObject::Streamer(buf);
      return 0;
   }

   inline Int_t ReadTNamed(TBuffer &buf, void *addr, const TConfiguration *config) 
   {
      // Read in a TNamed object part.
      // Since the TNamed streamer is solely delegating back to the StreamerInfo we 
      // can skip the streamer.

      // Idea: We could extract the code from ReadClassBuffer and avoid one function
      // code.
      static const TClass *TNamed_cl = TNamed::Class();
      return buf.ReadClassBuffer(TNamed_cl,(((char*)addr)+config->fOffset));
   }

   class TConfigSTL : public TConfiguration {
      // Configuration object for the kSTL case
   private:
      void Init() {
         TVirtualCollectionProxy *proxy = fNewClass->GetCollectionProxy();
         if (proxy) {
            fCreateIterators = proxy->GetFunctionCreateIterators();
            fCopyIterator = proxy->GetFunctionCopyIterator();
            fDeleteIterator = proxy->GetFunctionDeleteIterator();
            fDeleteTwoIterators = proxy->GetFunctionDeleteTwoIterators();
         }
      }

   public:
      TClass          *fOldClass;   // Class of the content on file
      TClass          *fNewClass;   // Class of the content in memory.
      TMemberStreamer *fStreamer;
      const char      *fTypeName;   // Type name of the member as typed by ther user.
      bool             fIsSTLBase;  // aElement->IsBase() && aElement->IsA()!=TStreamerBase::Class()

      TVirtualCollectionProxy::CreateIterators_t    fCreateIterators;
      TVirtualCollectionProxy::CopyIterator_t       fCopyIterator;
      TVirtualCollectionProxy::DeleteIterator_t     fDeleteIterator;
      TVirtualCollectionProxy::DeleteTwoIterators_t fDeleteTwoIterators;

      TConfigSTL(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, UInt_t length, TClass *oldClass, const char *type_name, Bool_t isbase) : 
         TConfiguration(info,id,offset,length), fOldClass(oldClass), fNewClass(oldClass), fStreamer(0), fTypeName(type_name), fIsSTLBase(isbase),
         fCreateIterators(0), fCopyIterator(0), fDeleteIterator(0), fDeleteTwoIterators(0) { Init(); }

      TConfigSTL(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, UInt_t length, TClass *oldClass, TClass *newClass, const char *type_name, Bool_t isbase) : 
         TConfiguration(info,id,offset,length), fOldClass(oldClass), fNewClass(newClass), fStreamer(0), fTypeName(type_name), fIsSTLBase(isbase),
         fCreateIterators(0), fCopyIterator(0), fDeleteIterator(0), fDeleteTwoIterators(0) { Init(); }

      TConfigSTL(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, UInt_t length, TClass *oldClass, TMemberStreamer* streamer, const char *type_name, Bool_t isbase) : 
         TConfiguration(info,id,offset,length), fOldClass(oldClass), fNewClass(oldClass), fStreamer(streamer), fTypeName(type_name), fIsSTLBase(isbase),
         fCreateIterators(0), fCopyIterator(0), fDeleteIterator(0), fDeleteTwoIterators(0) { Init(); }

      TConfigSTL(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, UInt_t length, TClass *oldClass, TClass *newClass, TMemberStreamer* streamer, const char *type_name, Bool_t isbase) : 
         TConfiguration(info,id,offset,length), fOldClass(oldClass), fNewClass(newClass), fStreamer(streamer), fTypeName(type_name), fIsSTLBase(isbase),
         fCreateIterators(0), fCopyIterator(0), fDeleteIterator(0), fDeleteTwoIterators(0) { Init(); }

      virtual TConfiguration *Copy() { return new TConfigSTL(*this); }
   };

   class TVectorLoopConfig : public TLoopConfiguration {
      // Base class of the Configurations used in member wise streaming.
   protected:
   public:
      Long_t fIncrement; // Either a value to increase the cursor by and 
   public:
      TVectorLoopConfig(Long_t increment) : fIncrement(increment) {};
      //virtual void PrintDebug(TBuffer &buffer, void *);
      virtual ~TVectorLoopConfig() {};
      void Print() const
      {
         printf("TVectorLoopConfig: increment=%ld\n",fIncrement);
      }

      void* GetFirstAddress(void *start, const void * /* end */) const
      {
         // Return the address of the first element of the collection.
         
         return start;
      }

      virtual TLoopConfiguration* Copy() { return new TVectorLoopConfig(*this); }
   };

   class TAssocLoopConfig : public TLoopConfiguration {
      // Base class of the Configurations used in member wise streaming.
   protected:
   public:
      TVirtualCollectionProxy *fProxy;
   public:
      TAssocLoopConfig(TVirtualCollectionProxy *proxy) : fProxy(proxy) {};
      //virtual void PrintDebug(TBuffer &buffer, void *);
      virtual ~TAssocLoopConfig() {};
      void Print() const
      {
         printf("TAssocLoopConfig: proxy=%s\n",fProxy->GetCollectionClass()->GetName());
      }
      virtual TLoopConfiguration* Copy() { return new TAssocLoopConfig(*this); }

      void* GetFirstAddress(void *start, const void * /* end */) const
      {
         // Return the address of the first element of the collection.
         
         R__ASSERT(0);
//         char iterator[TVirtualCollectionProxy::fgIteratorArenaSize];
//         void *iter = genloopconfig->fCopyIterator(&iterator,start_collection);
//         arr0 = genloopconfig->fNext(iter,end_collection);
//         if (iter != &iterator[0]) {
//            genloopconfig->fDeleteIterator(iter);
//         }
         return start;
      }
   };

   class TGenericLoopConfig : public TLoopConfiguration {
      // Configuration object for the generic case of member wise streaming looping.
   private:
      void Init() {
         if (fProxy) {
            if (fProxy->HasPointers()) {
               fNext = TVirtualCollectionPtrIterators::Next;
               fCopyIterator = TVirtualCollectionPtrIterators::CopyIterator;
               fDeleteIterator = TVirtualCollectionPtrIterators::DeleteIterator;               
            } else {
               fNext = fProxy->GetFunctionNext();
               fCopyIterator = fProxy->GetFunctionCopyIterator();
               fDeleteIterator = fProxy->GetFunctionDeleteIterator();
            }
         }
      }
   public:
      TVirtualCollectionProxy                      *fProxy;
      TVirtualCollectionProxy::Next_t               fNext;
      TVirtualCollectionProxy::CopyIterator_t       fCopyIterator;
      TVirtualCollectionProxy::DeleteIterator_t     fDeleteIterator;
      
      TGenericLoopConfig(TVirtualCollectionProxy *proxy) : fProxy(proxy), fNext(0), fCopyIterator(0), fDeleteIterator(0)
      { 
         Init(); 
      }
      virtual ~TGenericLoopConfig() {};
      void Print() const
      {
         printf("TGenericLoopConfig: proxy=%s\n",fProxy->GetCollectionClass()->GetName());
      }
      virtual TLoopConfiguration* Copy() { return new TGenericLoopConfig(*this); }

      void* GetFirstAddress(void *start_collection, const void *end_collection) const
      {
         // Return the address of the first element of the collection.
         
         char iterator[TVirtualCollectionProxy::fgIteratorArenaSize];
         void *iter = fCopyIterator(&iterator,start_collection);
         void *arr0 = fNext(iter,end_collection);
         if (iter != &iterator[0]) {
            fDeleteIterator(iter);
         }
         return arr0;
      }
   };

   inline void ReadSTLMemberWiseSameClass(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers)
   {
      // Collection was saved member-wise

      TConfigSTL *config = (TConfigSTL*)conf;
      vers &= ~( TBufferFile::kStreamedMemberWise );

      if( vers >= 8 ) {

         TClass *oldClass = config->fOldClass;   

         TClass *valueClass = oldClass->GetCollectionProxy()->GetValueClass();
         Version_t vClVersion = buf.ReadVersionForMemberWise( valueClass );

         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
         TVirtualCollectionProxy::TPushPop helper( oldProxy, (char*)addr );
         Int_t nobjects;
         buf.ReadInt(nobjects);
         void* alternative = oldProxy->Allocate(nobjects,true);
         if (nobjects) {
            TActionSequence *actions = oldProxy->GetReadMemberWiseActions( vClVersion );

            char startbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
            char endbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
            void *begin = &(startbuf[0]);
            void *end = &(endbuf[0]);
            config->fCreateIterators(alternative, &begin, &end );
            // We can not get here with a split vector of pointer, so we can indeed assume
            // that actions->fConfiguration != null.
            buf.ApplySequence(*actions, begin, end);
            if (begin != &(startbuf[0])) {
               // assert(end != endbuf);
               config->fDeleteTwoIterators(begin,end);
            }
         }
         oldProxy->Commit(alternative);

      } else {

         TClass *oldClass = config->fOldClass;

         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();

         TVirtualCollectionProxy::TPushPop helper( oldProxy, (char*)addr );
         Int_t nobjects;
         buf.ReadInt(nobjects);
         void* env = oldProxy->Allocate(nobjects,true);

         if (nobjects || vers < 7 ) {
            // coverity[dereference] since this is a member streaming action by definition the collection contains objects. 
            TStreamerInfo *subinfo = (TStreamerInfo*)oldProxy->GetValueClass()->GetStreamerInfo( 0 );

            if (subinfo->IsOptimized()) {
               subinfo->SetBit(TVirtualStreamerInfo::kCannotOptimize);
               subinfo->Compile();
            }
            subinfo->ReadBuffer(buf, *oldProxy, -1, nobjects, 0, 1);
         }
         oldProxy->Commit(env);
      }
   }

   inline void ReadArraySTLMemberWiseSameClass(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers)
   {
      // Collection was saved member-wise

      TConfigSTL *config = (TConfigSTL*)conf;
      vers &= ~( TBufferFile::kStreamedMemberWise );

      if( vers >= 8 ) {

         TClass *oldClass = config->fOldClass;   

         TClass *valueClass = oldClass->GetCollectionProxy()->GetValueClass();
         Version_t vClVersion = buf.ReadVersionForMemberWise( valueClass );

         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();
         TActionSequence *actions = oldProxy->GetReadMemberWiseActions( vClVersion );

         int objectSize = oldClass->Size();
         char *obj = (char*)addr;
         char *endobj = obj + conf->fLength*objectSize;

         for(; obj<endobj; obj+=objectSize) {            
            Int_t nobjects;
            buf.ReadInt(nobjects);
            TVirtualCollectionProxy::TPushPop helper( oldProxy, (char*)obj );
            void* alternative = oldProxy->Allocate(nobjects,true);
            if (nobjects) {
               char startbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
               char endbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
               void *begin = &(startbuf[0]);
               void *end = &(endbuf[0]);
               config->fCreateIterators(alternative, &begin, &end );
               // We can not get here with a split vector of pointer, so we can indeed assume
               // that actions->fConfiguration != null.
               buf.ApplySequence(*actions, begin, end);
               if (begin != &(startbuf[0])) {
                  // assert(end != endbuf);
                  config->fDeleteTwoIterators(begin,end);
               }
            }
            oldProxy->Commit(alternative);
         }

      } else {

         TClass *oldClass = config->fOldClass;

         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();

         int objectSize = oldClass->Size();
         char *obj = (char*)addr;
         char *endobj = obj + conf->fLength*objectSize;

         for(; obj<endobj; obj+=objectSize) {            
            TVirtualCollectionProxy::TPushPop helper( oldProxy, (char*)obj );
            Int_t nobjects;
            buf.ReadInt(nobjects);
            void* env = oldProxy->Allocate(nobjects,true);

            if (nobjects || vers < 7 ) {
               // coverity[dereference] since this is a member streaming action by definition the collection contains objects. 
               TStreamerInfo *subinfo = (TStreamerInfo*)oldProxy->GetValueClass()->GetStreamerInfo( 0 );

               if (subinfo->IsOptimized()) {
                  subinfo->SetBit(TVirtualStreamerInfo::kCannotOptimize);
                  subinfo->Compile();
               }
               subinfo->ReadBuffer(buf, *oldProxy, -1, nobjects, 0, 1);
            }
            oldProxy->Commit(env);
         }
      }
   }

   inline void ReadSTLMemberWiseChangedClass(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers)
   {
      // Collection was saved member-wise

      TConfigSTL *config = (TConfigSTL*)conf;

      vers &= ~( TBufferFile::kStreamedMemberWise );

      TClass *newClass = config->fNewClass;
      TClass *oldClass = config->fOldClass;

      if( vers < 8 ) {
         Error( "ReadSTLMemberWiseChangedClass", "Unfortunately, version %d of TStreamerInfo (used in %s) did not record enough information to convert a %s into a %s.",
               vers, buf.GetParent() ? buf.GetParent()->GetName() : "memory/socket", oldClass->GetName(), newClass->GetName() );
      } else {

         Version_t vClVersion = buf.ReadVersionForMemberWise( oldClass->GetCollectionProxy()->GetValueClass() );

         TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();
         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();

         TVirtualCollectionProxy::TPushPop helper( newProxy, (char*)addr );
         Int_t nobjects;
         buf.ReadInt(nobjects);
         void* alternative = newProxy->Allocate(nobjects,true);
         if (nobjects) {
            TActionSequence *actions = newProxy->GetConversionReadMemberWiseActions( oldProxy->GetValueClass(), vClVersion );
            char startbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
            char endbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
            void *begin = &(startbuf[0]);
            void *end = &(endbuf[0]);
            config->fCreateIterators( alternative, &begin, &end );
            // We can not get here with a split vector of pointer, so we can indeed assume
            // that actions->fConfiguration != null.
            buf.ApplySequence(*actions, begin, end);
            if (begin != &(startbuf[0])) {
               // assert(end != endbuf);
               config->fDeleteTwoIterators(begin,end);
            }
         }
         newProxy->Commit(alternative);
      }
   }

   inline void ReadArraySTLMemberWiseChangedClass(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers)
   {
      // Collection was saved member-wise

      TConfigSTL *config = (TConfigSTL*)conf;

      vers &= ~( TBufferFile::kStreamedMemberWise );

      TClass *newClass = config->fNewClass;
      TClass *oldClass = config->fOldClass;

      if( vers < 8 ) {
         Error( "ReadSTLMemberWiseChangedClass", "Unfortunately, version %d of TStreamerInfo (used in %s) did not record enough information to convert a %s into a %s.",
               vers, buf.GetParent() ? buf.GetParent()->GetName() : "memory/socket", oldClass->GetName(), newClass->GetName() );
      } else {

         Version_t vClVersion = buf.ReadVersionForMemberWise( oldClass->GetCollectionProxy()->GetValueClass() );

         TVirtualCollectionProxy *newProxy = newClass->GetCollectionProxy();
         TVirtualCollectionProxy *oldProxy = oldClass->GetCollectionProxy();

         int objectSize = newClass->Size();
         char *obj = (char*)addr;
         char *endobj = obj + conf->fLength*objectSize;

         for(; obj<endobj; obj+=objectSize) {            
            TVirtualCollectionProxy::TPushPop helper( newProxy, (char*)obj );
            Int_t nobjects;
            buf.ReadInt(nobjects);
            void* alternative = newProxy->Allocate(nobjects,true);
            if (nobjects) {
               TActionSequence *actions = newProxy->GetConversionReadMemberWiseActions( oldProxy->GetValueClass(), vClVersion );
               char startbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
               char endbuf[TVirtualCollectionProxy::fgIteratorArenaSize];
               void *begin = &(startbuf[0]);
               void *end = &(endbuf[0]);
               config->fCreateIterators( alternative, &begin, &end );
               // We can not get here with a split vector of pointer, so we can indeed assume
               // that actions->fConfiguration != null.
               buf.ApplySequence(*actions, begin, end);
               if (begin != &(startbuf[0])) {
                  // assert(end != endbuf);
                  config->fDeleteTwoIterators(begin,end);
               }
            }
            newProxy->Commit(alternative);
         }
      }
   }


   inline void ReadSTLObjectWiseFastArray(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t /* vers */, UInt_t /* start */)
   {
      TConfigSTL *config = (TConfigSTL*)conf;
      // Idea: This needs to be unrolled, it currently calls the TGenCollectionStreamer ....
      buf.ReadFastArray(addr,config->fNewClass,conf->fLength,(TMemberStreamer*)0,config->fOldClass);
   }
   inline void ReadSTLObjectWiseStreamer(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t /* vers */, UInt_t /* start */)
   {
      TConfigSTL *config = (TConfigSTL*)conf;
      (*config->fStreamer)(buf,addr,conf->fLength);
   }
   inline void ReadSTLObjectWiseFastArrayV2(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers, UInt_t start)
   {
      // case of old TStreamerInfo

      TConfigSTL *config = (TConfigSTL*)conf;
      //  Backward compatibility. Some TStreamerElement's where without
      //  Streamer but were not removed from element list
      if (config->fIsSTLBase || vers == 0) {
         buf.SetBufferOffset(start);  //there is no byte count
      }
      // Idea: This needs to be unrolled, it currently calls the TGenCollectionStreamer ....
      buf.ReadFastArray(addr,config->fNewClass,conf->fLength,(TMemberStreamer*)0,config->fOldClass);
   }
   inline void ReadSTLObjectWiseStreamerV2(TBuffer &buf, void *addr, const TConfiguration *conf, Version_t vers, UInt_t start)
   {
      // case of old TStreamerInfo

      TConfigSTL *config = (TConfigSTL*)conf;
      //  Backward compatibility. Some TStreamerElement's where without
      //  Streamer but were not removed from element list
      if (config->fIsSTLBase || vers == 0) {
         buf.SetBufferOffset(start);  //there is no byte count
      }
      (*config->fStreamer)(buf,addr,conf->fLength);
   }

   template <void (*memberwise)(TBuffer&,void *,const TConfiguration*, Version_t), 
             void (*objectwise)(TBuffer&,void *,const TConfiguration*, Version_t, UInt_t)>
   inline Int_t ReadSTL(TBuffer &buf, void *addr, const TConfiguration *conf)
   {
      TConfigSTL *config = (TConfigSTL*)conf;
      UInt_t start, count;
      Version_t vers = buf.ReadVersion(&start, &count, config->fOldClass);
      if ( vers & TBufferFile::kStreamedMemberWise ) {
         memberwise(buf,((char*)addr)+config->fOffset,config, vers);
      } else {
         objectwise(buf,((char*)addr)+config->fOffset,config, vers, start);
      }
      buf.CheckByteCount(start,count,config->fTypeName);
      return 0;
   }

   template <typename From, typename To>
   inline Int_t ConvertBasicType(TBuffer &buf, void *addr, const TConfiguration *config)
   {
      // Simple conversion from a 'From' on disk to a 'To' in memory.
      From temp;
      buf >> temp;
      *(To*)( ((char*)addr) + config->fOffset ) = (To)temp;
      return 0;
   }

   class TConfigurationUseCache : public TConfiguration {
      // Configuration object for the UseCache case.
   public:
      TConfiguredAction fAction;
      Bool_t            fNeedRepeat;
      
      TConfigurationUseCache(TVirtualStreamerInfo *info, TConfiguredAction &action, Bool_t repeat) : 
              TConfiguration(info,action.fConfiguration->fElemId,action.fConfiguration->fOffset),fAction(action),fNeedRepeat(repeat) {};
      virtual void PrintDebug(TBuffer &b, void *addr) const
      {
         if (gDebug > 1) {
            // Idea: We should print the name of the action function.
            TStreamerInfo *info = (TStreamerInfo*)fInfo;
            TStreamerElement *aElement = (TStreamerElement*)info->GetElems()[fElemId];
            fprintf(stdout,"StreamerInfoAction, class:%s, name=%s, fType[%d]=%d,"
                   " %s, bufpos=%d, arr=%p, eoffset=%d, Redirect=%p\n",
                   info->GetClass()->GetName(),aElement->GetName(),fElemId,info->GetTypes()[fElemId],
                   aElement->ClassName(),b.Length(),addr, 0,b.PeekDataCache()->GetObjectAt(0));
         }            

      }
      virtual ~TConfigurationUseCache() {};
      virtual TConfiguration *Copy() { 
         TConfigurationUseCache *copy = new TConfigurationUseCache(*this);
         fAction.fConfiguration = copy->fAction.fConfiguration->Copy(); // since the previous allocation did a 'move' of fAction we need to fix it.
         return copy;
      }
   };


   inline Int_t UseCache(TBuffer &b, void *addr, const TConfiguration *conf) 
   {
      TConfigurationUseCache *config = (TConfigurationUseCache*)conf;
      
      Int_t bufpos = b.Length();
      TVirtualArray *cached = b.PeekDataCache();
      if (cached==0) {
         TStreamerElement *aElement = (TStreamerElement*)conf->fInfo->GetElems()[conf->fElemId];
         TStreamerInfo *info = (TStreamerInfo*)conf->fInfo;
         Warning("ReadBuffer","Skipping %s::%s because the cache is missing.",info->GetName(),aElement->GetName());
         char *ptr = (char*)addr;
         info->ReadBufferSkip(b,&ptr,config->fElemId,info->GetTypes()[config->fElemId]+TStreamerInfo::kSkip,aElement,1,0);
      } else {
         config->fAction(b, (*cached)[0]);
      }
      // Idea: Factor out this 'if' to a UseCacheRepeat function
      if (config->fNeedRepeat) {
         b.SetBufferOffset(bufpos); 
      }
      return 0;
   }

   inline Int_t UseCacheVectorPtrLoop(TBuffer &b, void *start, const void *end, const TConfiguration *conf) 
   {
      TConfigurationUseCache *config = (TConfigurationUseCache*)conf;
      Int_t bufpos = b.Length();

      TVirtualArray *cached = b.PeekDataCache();
      if (cached==0) {
         TStreamerElement *aElement = (TStreamerElement*)config->fInfo->GetElems()[config->fElemId];
         TStreamerInfo *info = (TStreamerInfo*)config->fInfo;
         Warning("ReadBuffer","Skipping %s::%s because the cache is missing.",info->GetName(),aElement->GetName());
         char *ptr = (char*)start;
         UInt_t n = (((void**)end)-((void**)start));
         info->ReadBufferSkip(b,&ptr,config->fElemId,info->GetTypes()[config->fElemId]+TStreamerInfo::kSkip,aElement,n,0);
      } else {
         TVectorLoopConfig cached_config( cached->fClass->Size() );
         void *cached_start = (*cached)[0];
         void *cached_end = ((char*)cached_start) + cached->fSize * cached_config.fIncrement;
         config->fAction(b,cached_start,cached_end,&cached_config);         
      }
      // Idea: Factor out this 'if' to a UseCacheRepeat function
      if (config->fNeedRepeat) {
         b.SetBufferOffset(bufpos); 
      }
      return 0;
   }

   inline Int_t UseCacheVectorLoop(TBuffer &b, void *start, const void *end, const TLoopConfiguration *loopconf, const TConfiguration *conf) 
   {
      TConfigurationUseCache *config = (TConfigurationUseCache*)conf;

      Int_t bufpos = b.Length();
      TVirtualArray *cached = b.PeekDataCache();
      if (cached==0) {
         TStreamerElement *aElement = (TStreamerElement*)config->fInfo->GetElems()[config->fElemId];
         TStreamerInfo *info = (TStreamerInfo*)config->fInfo;         
         Warning("ReadBuffer","Skipping %s::%s because the cache is missing.",info->GetName(),aElement->GetName());
         char *ptr = (char*)start;
         UInt_t n = (((char*)end)-((char*)start))/((TVectorLoopConfig*)loopconf)->fIncrement;
         info->ReadBufferSkip(b,&ptr,config->fElemId,info->GetTypes()[config->fElemId]+TStreamerInfo::kSkip,aElement,n,0);
      } else {
         TVectorLoopConfig cached_config( cached->fClass->Size() );
         void *cached_start = (*cached)[0];
         void *cached_end = ((char*)cached_start) + cached->fSize * cached_config.fIncrement;
         config->fAction(b,cached_start,cached_end,&cached_config);
      }
      // Idea: Factor out this 'if' to a UseCacheRepeat function
      if (config->fNeedRepeat) {
         b.SetBufferOffset(bufpos); 
      }
      return 0;
   }

   inline Int_t UseCacheGenericCollection(TBuffer &b, void *, const void *, const TLoopConfiguration *loopconfig, const TConfiguration *conf) 
   {
      TConfigurationUseCache *config = (TConfigurationUseCache*)conf;

      Int_t bufpos = b.Length();
      TVirtualArray *cached = b.PeekDataCache();
      if (cached==0) {
         TStreamerElement *aElement = (TStreamerElement*)config->fInfo->GetElems()[config->fElemId];
         TStreamerInfo *info = (TStreamerInfo*)config->fInfo;
         
         TVirtualCollectionProxy *proxy = ((TGenericLoopConfig*)loopconfig)->fProxy;
         Warning("ReadBuffer","Skipping %s::%s because the cache is missing.",info->GetName(),aElement->GetName());
         UInt_t n = proxy->Size();
         info->ReadBufferSkip(b, *proxy,config->fElemId,info->GetTypes()[config->fElemId]+TStreamerInfo::kSkip,aElement,n,0);
      } else {
         TVectorLoopConfig cached_config( cached->fClass->Size() );
         void *cached_start = (*cached)[0];
         void *cached_end = ((char*)cached_start) + cached->fSize * cached_config.fIncrement;
         config->fAction(b,cached_start,cached_end,&cached_config);
      }
      // Idea: Factor out this 'if' to a UseCacheRepeat function
      if (config->fNeedRepeat) {
         b.SetBufferOffset(bufpos); 
      }
      return 0;
   }

   // Support for collections.
   
   Int_t ReadLoopInvalid(TBuffer &, void *, const void *, const TConfiguration *config)
   {
      Fatal("ApplySequence","The sequence of actions to read %s:%d member-wise was not initialized.",config->fInfo->GetName(),config->fInfo->GetClassVersion());
      return 0;
   }

   Int_t WriteLoopInvalid(TBuffer &, void *, const void *, const TConfiguration *config)
   {
      Fatal("ApplySequence","The sequence of actions to write %s:%d member-wise was not initialized.",config->fInfo->GetName(),config->fInfo->GetClassVersion());
      return 0;
   }

   Int_t GenericVectorPtrReadAction(TBuffer &buf, void *iter, const void *end, const TConfiguration *config) 
   {
      Int_t n = ( ((void**)end) - ((void**)iter) );
      char **arr = (char**)iter;
      return ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, arr, config->fElemId, n, config->fOffset, 1|2 );
   }

   Int_t GenericVectorPtrWriteAction(TBuffer &buf, void *iter, const void *end, const TConfiguration *config) 
   {
      Int_t n = ( ((void**)end) - ((void**)iter) );
      char **arr = (char**)iter;
      return ((TStreamerInfo*)config->fInfo)->WriteBufferAux(buf, arr, config->fElemId, n, config->fOffset, 1|2 );
   }

   Int_t ReadVectorBase(TBuffer &buf, void *start, const void *end, const TLoopConfiguration * loopconfig, const TConfiguration *config) 
   {
      // Well the implementation is non trivial since we do not have a proxy for the container of _only_ the base class.  For now
      // punt.

      UInt_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      UInt_t n = (((char*)end)-((char*)start))/incr;
      char **arrptr = new char*[n];
      UInt_t i = 0;
      for(void *iter = start; iter != end; iter = (char*)iter + incr, ++i ) {
         arrptr[i] = (char*)iter;
      }
      ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, arrptr, config->fElemId, n, config->fOffset, 1|2 );
      delete [] arrptr;

//      // Idea: need to cache this result!
//      TStreamerInfo *info = (TStreamerInfo*)config->fInfo;
//      TStreamerElement *aElement = (TStreamerElement*)info->GetElems()[config->fElemId];
//      Int_t clversion = ((TStreamerBase*)aElement)->GetBaseVersion();
//      TClass *cle = aElement->GetNewBaseClass();
//      TSequence *actions = CreateReadMemberWiseActions( cle->GetStreamerInfo(clversion), ???? );
//      actions->ReadBuffer(b,start,end);
//      delete actions;

//      const Int_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
//      for(void *iter = start; iter != end; iter = (char*)iter + incr ) 
//      {
//         Int_t clversion = ((TStreamerBase*)aElement)->GetBaseVersion();
//         ((TStreamerInfo*)cle->GetStreamerInfo(clversion))->ReadBuffer(b,arr,-1,narr,ioffset,arrayMode);
//
//         ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, (char**)&iter, config->fElemId, 1, config->fOffset, 1|2 );
//      }
      return 0;
   }

   Int_t ReadVectorWrapping(TBuffer &buf, void *start, const void *end, const TLoopConfiguration * loopconfig, const TConfiguration *config) 
   {
      // Well the implementation is non trivial. For now punt.

      UInt_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      UInt_t n = (((char*)end)-((char*)start))/incr;
      char **arrptr = new char*[n];
      UInt_t i = 0;
      for(void *iter = start; iter != end; iter = (char*)iter + incr, ++i ) {
         arrptr[i] = (char*)iter;
      }
      ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, arrptr, config->fElemId, n, config->fOffset, 1|2 );
      delete [] arrptr;
      return 0;
   }

   Int_t WriteVectorWrapping(TBuffer &buf, void *start, const void *end, const TLoopConfiguration * loopconfig, const TConfiguration *config) 
   {
      // Well the implementation is non trivial. For now punt.

      UInt_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      UInt_t n = (((char*)end)-((char*)start))/incr;
      char **arrptr = new char*[n];
      UInt_t i = 0;
      for(void *iter = start; iter != end; iter = (char*)iter + incr, ++i ) {
         arrptr[i] = (char*)iter;
      }
      ((TStreamerInfo*)config->fInfo)->WriteBufferAux(buf, arrptr, config->fElemId, n, config->fOffset, 1|2 );
      delete [] arrptr;
      return 0;
   }

   Int_t GenericVectorReadAction(TBuffer &buf, void *start, const void *end, const TLoopConfiguration * loopconfig, const TConfiguration *config) 
   {
      const Int_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      for(void *iter = start; iter != end; iter = (char*)iter + incr ) {
         void **iter_ptr = &iter;
         ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, (char**)iter_ptr, config->fElemId, 1, config->fOffset, 1|2 );
      }
      return 0;
   }

   Int_t GenericCollectionReadAction(TBuffer &buf, void *, const void *, const TLoopConfiguration * loopconf, const TConfiguration *config) 
   {
      TGenericLoopConfig *loopconfig = (TGenericLoopConfig*)loopconf;
      TVirtualCollectionProxy *proxy = loopconfig->fProxy;
      return ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, *proxy, config->fElemId, proxy->Size(), config->fOffset, 1|2 );
   }

   Int_t GenericCollectionWriteAction(TBuffer &buf, void *, const void *, const TLoopConfiguration * loopconf, const TConfiguration *config) 
   {
      TGenericLoopConfig *loopconfig = (TGenericLoopConfig*)loopconf;
      TVirtualCollectionProxy *proxy = loopconfig->fProxy;
      return ((TStreamerInfo*)config->fInfo)->WriteBufferAux(buf, *proxy, config->fElemId, proxy->Size(), config->fOffset, 1|2 );
   }

   Int_t GenericAssocCollectionReadAction(TBuffer &buf, void *, const void *, const TLoopConfiguration *loopconf, const TConfiguration *config) 
   {
      TAssocLoopConfig *loopconfig = (TAssocLoopConfig*)loopconf;
      TVirtualCollectionProxy *proxy = loopconfig->fProxy;
      return ((TStreamerInfo*)config->fInfo)->ReadBuffer(buf, *proxy, config->fElemId, proxy->Size(), config->fOffset, 1|2 );
   }

   template <typename T> 
   Int_t ReadBasicTypeVectorLoop(TBuffer &buf, void *iter, const void *end, const TLoopConfiguration *loopconfig, const TConfiguration *config)
   {
      const Int_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      iter = (char*)iter + config->fOffset;
      end = (char*)end + config->fOffset;
      for(; iter != end; iter = (char*)iter + incr ) {
         T *x = (T*) ((char*) iter);
         buf >> *x;
      }
      return 0;
   }

   template <typename T> 
   Int_t WriteBasicTypeVectorLoop(TBuffer &buf, void *iter, const void *end, const TLoopConfiguration *loopconfig, const TConfiguration *config)
   {
      const Int_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      iter = (char*)iter + config->fOffset;
      end = (char*)end + config->fOffset;
      for(; iter != end; iter = (char*)iter + incr ) {
         T *x = (T*) ((char*) iter);
         buf << *x;
      }
      return 0;
   }

   template <typename T> 
   Int_t ReadBasicTypeGenericLoop(TBuffer &buf, void *start, const void *end, const TLoopConfiguration *loopconf, const TConfiguration *config)
   {
      TGenericLoopConfig *loopconfig = (TGenericLoopConfig*)loopconf;

      // const Int_t offset = config->fOffset;
      Next_t next = loopconfig->fNext;
      const Int_t offset = config->fOffset;

      char iterator[TVirtualCollectionProxy::fgIteratorArenaSize];
      void *iter = loopconfig->fCopyIterator(iterator,start);
      void *addr;
      while( (addr = next(iter,end)) ) {
         T *x =  (T*)( ((char*)addr) + offset );
         buf >> *x;
      }
      if (iter != &iterator[0]) {
         loopconfig->fDeleteIterator(iter);
      }
      return 0;
   }

   template <typename T> 
   Int_t WriteBasicTypeGenericLoop(TBuffer &buf, void *start, const void *end, const TLoopConfiguration *loopconf, const TConfiguration *config)
   {
      TGenericLoopConfig *loopconfig = (TGenericLoopConfig*)loopconf;

      // const Int_t offset = config->fOffset;
      Next_t next = loopconfig->fNext;
      const Int_t offset = config->fOffset;

      char iterator[TVirtualCollectionProxy::fgIteratorArenaSize];
      void *iter = loopconfig->fCopyIterator(iterator,start);
      void *addr;
      while( (addr = next(iter,end)) ) {
         T *x =  (T*)( ((char*)addr) + offset );
         buf << *x;
      }
      if (iter != &iterator[0]) {
         loopconfig->fDeleteIterator(iter);
      }
      return 0;
   }

   template <typename T> 
   Int_t ReadBasicTypeVectorPtrLoop(TBuffer &buf, void *iter, const void *end, const TConfiguration *config)
   {
      const Int_t offset = config->fOffset;

      for(; iter != end; iter = (char*)iter + sizeof(void*) ) {
         T *x = (T*)( ((char*) (*(void**)iter) ) + offset );
         buf >> *x;
      }
      return 0;
   }

   template <typename T> 
   Int_t WriteBasicTypeVectorPtrLoop(TBuffer &buf, void *iter, const void *end, const TConfiguration *config)
   {
      const Int_t offset = config->fOffset;

      for(; iter != end; iter = (char*)iter + sizeof(void*) ) {
         T *x = (T*)( ((char*) (*(void**)iter) ) + offset );
         buf << *x;
      }
      return 0;
   }

   template <Int_t (*action)(TBuffer&,void *,const TConfiguration*)>
   Int_t VectorPtrLooper(TBuffer &buf, void *start, const void *end, const TConfiguration *config) 
   {     
      for(void *iter = start; iter != end; iter = (char*)iter + sizeof(void*) ) {
         action(buf, *(void**)iter, config);
      }
      return 0;
   }

   template <Int_t (*action)(TBuffer&,void *,const TConfiguration*)>
   Int_t VectorLooper(TBuffer &buf, void *start, const void *end, const TLoopConfiguration *loopconfig, const TConfiguration *config) 
   {     
      const Int_t incr = ((TVectorLoopConfig*)loopconfig)->fIncrement;
      //Idea: can we factor out the addition of fOffset
      //  iter = (char*)iter + config->fOffset;
      for(void *iter = start; iter != end; iter = (char*)iter + incr ) {
         action(buf, iter, config);
      }
      return 0;
   }

   template <Int_t (*action)(TBuffer&,void *,const TConfiguration*)>
   Int_t GenericLooper(TBuffer &buf, void *start, const void *end, const TLoopConfiguration *loopconf, const TConfiguration *config) 
   {
      TGenericLoopConfig *loopconfig = (TGenericLoopConfig*)loopconf;

      // const Int_t offset = config->fOffset;
      Next_t next = loopconfig->fNext;

      char iterator[TVirtualCollectionProxy::fgIteratorArenaSize];
      void *iter = loopconfig->fCopyIterator(&iterator,start);
      void *addr;
      while( (addr = next(iter,end)) ) {
         action(buf, addr, config);
      }
      if (iter != &iterator[0]) {
         loopconfig->fDeleteIterator(iter);
      }
      return 0;
   }

}

static TConfiguredAction GetVectorReadAction(TVirtualStreamerInfo *info, TStreamerElement *element, Int_t type, UInt_t i, Int_t offset)
{
   switch (type) {
         // read basic types
      case TStreamerInfo::kBool:    return TConfiguredAction( ReadBasicTypeVectorLoop<Bool_t>, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kChar:    return TConfiguredAction( ReadBasicTypeVectorLoop<Char_t>, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kShort:   return TConfiguredAction( ReadBasicTypeVectorLoop<Short_t>, new TConfiguration(info,i,offset) );   break;
      case TStreamerInfo::kInt:     return TConfiguredAction( ReadBasicTypeVectorLoop<Int_t>, new TConfiguration(info,i,offset) );     break;
      case TStreamerInfo::kLong:    return TConfiguredAction( ReadBasicTypeVectorLoop<Long_t>, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kLong64:  return TConfiguredAction( ReadBasicTypeVectorLoop<Long64_t>, new TConfiguration(info,i,offset) );  break;
      case TStreamerInfo::kFloat:   return TConfiguredAction( ReadBasicTypeVectorLoop<Float_t>, new TConfiguration(info,i,offset) );   break;
      case TStreamerInfo::kDouble:  return TConfiguredAction( ReadBasicTypeVectorLoop<Double_t>, new TConfiguration(info,i,offset) );  break;
      case TStreamerInfo::kUChar:   return TConfiguredAction( ReadBasicTypeVectorLoop<UChar_t>, new TConfiguration(info,i,offset) );   break;
      case TStreamerInfo::kUShort:  return TConfiguredAction( ReadBasicTypeVectorLoop<UShort_t>, new TConfiguration(info,i,offset) );  break;
      case TStreamerInfo::kUInt:    return TConfiguredAction( ReadBasicTypeVectorLoop<UInt_t>, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kULong:   return TConfiguredAction( ReadBasicTypeVectorLoop<ULong_t>, new TConfiguration(info,i,offset) );   break;
      case TStreamerInfo::kULong64: return TConfiguredAction( ReadBasicTypeVectorLoop<ULong64_t>, new TConfiguration(info,i,offset) ); break;
      case TStreamerInfo::kFloat16: {
         if (element->GetFactor() != 0) {
            return TConfiguredAction( VectorLooper<ReadBasicType_WithFactor<float> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
         } else {
            Int_t nbits = (Int_t)element->GetXmin();
            if (!nbits) nbits = 12;
            return TConfiguredAction( VectorLooper<ReadBasicType_NoFactor<float> >, new TConfNoFactor(info,i,offset,nbits) );               
         }
         break;
      }
      case TStreamerInfo::kDouble32: {
         if (element->GetFactor() != 0) {
            return TConfiguredAction( VectorLooper<ReadBasicType_WithFactor<double> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
         } else {
            Int_t nbits = (Int_t)element->GetXmin();
            if (!nbits) {
               return TConfiguredAction( VectorLooper<ConvertBasicType<float,double> >, new TConfiguration(info,i,offset) );
            } else {
               return TConfiguredAction( VectorLooper<ReadBasicType_NoFactor<double> >, new TConfNoFactor(info,i,offset,nbits) );
            }
         }
         break;
      }
      case TStreamerInfo::kTNamed:  return TConfiguredAction( VectorLooper<ReadTNamed >, new TConfiguration(info,i,offset) );    break;
         // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
         // Streamer alltogether.
      case TStreamerInfo::kTObject: return TConfiguredAction( VectorLooper<ReadTObject >, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kTString: return TConfiguredAction( VectorLooper<ReadTString >, new TConfiguration(info,i,offset) );    break;
      case TStreamerInfo::kArtificial:
      case TStreamerInfo::kCacheNew:
      case TStreamerInfo::kCacheDelete:
      case TStreamerInfo::kSTL:  return TConfiguredAction( ReadVectorWrapping, new TConfiguration(info,i,0 /* 0 because we call the legacy code */) ); break;
      case TStreamerInfo::kBase: return TConfiguredAction( ReadVectorBase, new TConfiguration(info,i,0 /* 0 because we call the legacy code */) ); break;
      default:
         return TConfiguredAction( ReadVectorWrapping, new TConfiguration(info,i,0 /* 0 because we call the legacy code */) );
         // return TConfiguredAction( GenericVectorAction, new TConfigSTL(info,i,0 /* the offset will be used from TStreamerInfo */,0,proxy->GetCollectionClass(),0,0) );
         break;
   }
   R__ASSERT(0); // We should never be here
   return TConfiguredAction();
}

static TConfiguredAction GetVectorWriteAction(TVirtualStreamerInfo *info, TStreamerElement * /*element*/, Int_t type, UInt_t i, Int_t offset) {
  switch (type) {
        // read basic types
     case TStreamerInfo::kBool:    return TConfiguredAction( WriteBasicTypeVectorLoop<Bool_t>, new TConfiguration(info,i,offset) );    break;
     case TStreamerInfo::kChar:    return TConfiguredAction( WriteBasicTypeVectorLoop<Char_t>, new TConfiguration(info,i,offset) );    break;
     case TStreamerInfo::kShort:   return TConfiguredAction( WriteBasicTypeVectorLoop<Short_t>, new TConfiguration(info,i,offset) );   break;
     case TStreamerInfo::kInt:     return TConfiguredAction( WriteBasicTypeVectorLoop<Int_t>, new TConfiguration(info,i,offset) );     break;
     case TStreamerInfo::kLong:    return TConfiguredAction( WriteBasicTypeVectorLoop<Long_t>, new TConfiguration(info,i,offset) );    break;
     case TStreamerInfo::kLong64:  return TConfiguredAction( WriteBasicTypeVectorLoop<Long64_t>, new TConfiguration(info,i,offset) );  break;
     case TStreamerInfo::kFloat:   return TConfiguredAction( WriteBasicTypeVectorLoop<Float_t>, new TConfiguration(info,i,offset) );   break;
     case TStreamerInfo::kDouble:  return TConfiguredAction( WriteBasicTypeVectorLoop<Double_t>, new TConfiguration(info,i,offset) );  break;
     case TStreamerInfo::kUChar:   return TConfiguredAction( WriteBasicTypeVectorLoop<UChar_t>, new TConfiguration(info,i,offset) );   break;
     case TStreamerInfo::kUShort:  return TConfiguredAction( WriteBasicTypeVectorLoop<UShort_t>, new TConfiguration(info,i,offset) );  break;
     case TStreamerInfo::kUInt:    return TConfiguredAction( WriteBasicTypeVectorLoop<UInt_t>, new TConfiguration(info,i,offset) );    break;
     case TStreamerInfo::kULong:   return TConfiguredAction( WriteBasicTypeVectorLoop<ULong_t>, new TConfiguration(info,i,offset) );   break;
     case TStreamerInfo::kULong64: return TConfiguredAction( WriteBasicTypeVectorLoop<ULong64_t>, new TConfiguration(info,i,offset) ); break;
     default:
         return TConfiguredAction( WriteVectorWrapping, new TConfiguration(info,i,0 /* 0 because we call the legacy code */) );  
  }
  R__ASSERT(0); // We should never be here
  return TConfiguredAction();
}

//______________________________________________________________________________
void TStreamerInfo::Compile()
{
   // loop on the TStreamerElement list
   // regroup members with same type
   // Store predigested information into local arrays. This saves a huge amount
   // of time compared to an explicit iteration on all elements.

   R__LOCKGUARD(gCINTMutex);

   // fprintf(stderr,"Running Compile for %s %d %d req=%d,%d\n",GetName(),fClassVersion,fOptimized,CanOptimize(),TestBit(kCannotOptimize));

   // if (IsCompiled() && (!fOptimized || (CanOptimize() && !TestBit(kCannotOptimize)))) return;

   fOptimized = kFALSE;
   fNdata = 0;

   TObjArray* infos = (TObjArray*) gROOT->GetListOfStreamerInfo();
   if (fNumber >= infos->GetSize()) {
      infos->AddAtAndExpand(this, fNumber);
   } else {
      if (!infos->At(fNumber)) {
         infos->AddAt(this, fNumber);
      }
   }

   delete[] fType;
   fType = 0;
   delete[] fNewType;
   fNewType = 0;
   delete[] fOffset;
   fOffset = 0;
   delete[] fLength;
   fLength = 0;
   delete[] fElem;
   fElem = 0;
   delete[] fMethod;
   fMethod = 0;
   delete[] fComp;
   fComp = 0;

   if (fReadObjectWise) {
      fReadObjectWise->fActions.clear();
   }
   if (fWriteObjectWise) {
      fWriteObjectWise->fActions.clear();
   }
   Int_t ndata = fElements->GetEntries();

   fOffset = new Int_t[ndata+1];
   fType   = new Int_t[ndata+1];

   SetBit(kIsCompiled);
   if (!fReadObjectWise) fReadObjectWise = new TStreamerInfoActions::TActionSequence(this,ndata);
   if (!fWriteObjectWise) fWriteObjectWise = new TStreamerInfoActions::TActionSequence(this,ndata);

   if (!ndata) {
      // This may be the case for empty classes (e.g., TAtt3D).
      // We still need to properly set the size of emulated classes (i.e. add the virtual table)
      if (fClass->TestBit(TClass::kIsEmulation) && fNVirtualInfoLoc!=0) {
         fSize = sizeof(TStreamerInfo*);
      }
      return;
   }


   fComp = new TCompInfo[ndata];
   fNewType = new Int_t[ndata];
   fLength = new Int_t[ndata];
   fElem = new ULong_t[ndata];
   fMethod = new ULong_t[ndata];

   TStreamerElement* element;
   TStreamerElement* previous = 0;
   Int_t keep = -1;
   Int_t i;

   if (!CanOptimize()) {
      SetBit(kCannotOptimize);
   }

   Bool_t isOptimized = kFALSE;

   for (i = 0; i < ndata; ++i) {
      element = (TStreamerElement*) fElements->At(i);
      if (!element) {
         break;
      }
      if (element->GetType() < 0) {
         // -- Skip an ignored TObject base class.
         // Note: The only allowed negative value here is -1,
         // and signifies that Build() has found a TObject
         // base class and TClass::IgnoreTObjectStreamer() was
         // called.  In this case the compiled version of the
         // elements omits the TObject base class element,
         // which has to be compensated for by TTree::Bronch()
         // when it is making branches for a split object.
         continue;
      }
      if (TestBit(kCannotOptimize) && element->IsBase()) 
      {
         // Make sure the StreamerInfo for the base class is also
         // not optimized.
         TClass *bclass = element->GetClassPointer();
         Int_t clversion = ((TStreamerBase*)element)->GetBaseVersion();
         TStreamerInfo *binfo = ((TStreamerInfo*)bclass->GetStreamerInfo(clversion));
         binfo->SetBit(kCannotOptimize);
         if (binfo->IsOptimized())
         {
            // Optimizing does not work with splitting.
            binfo->Compile();
         }      
      }
      Int_t asize = element->GetSize();
      if (element->GetArrayLength()) {
         asize /= element->GetArrayLength();
      }
      fType[fNdata] = element->GetType();
      fNewType[fNdata] = element->GetNewType();
      fOffset[fNdata] = element->GetOffset();
      fLength[fNdata] = element->GetArrayLength();
      fElem[fNdata] = (ULong_t) element;
      fMethod[fNdata] = element->GetMethod();
      // try to group consecutive members of the same type
      if (!TestBit(kCannotOptimize) 
          && (keep >= 0) 
          && (element->GetType() < 10) 
          && (fType[fNdata] == fNewType[fNdata]) 
          && (fMethod[keep] == 0) 
          && (element->GetType() > 0) 
          && (element->GetArrayDim() == 0) 
          && (fType[keep] < kObject) 
          && (fType[keep] != kCharStar) /* do not optimize char* */ 
          && (element->GetType() == (fType[keep]%kRegrouped)) 
          && ((element->GetOffset()-fOffset[keep]) == (fLength[keep])*asize)
          && ((fOldVersion<6) || !previous || /* In version of TStreamerInfo less than 6, the Double32_t were merged even if their annotation (aka factor) were different */
              ((element->GetFactor() == previous->GetFactor())
               && (element->GetXmin() == previous->GetXmin())
               && (element->GetXmax() == previous->GetXmax())
               )
              )
          && (element->TestBit(TStreamerElement::kCache) == previous->TestBit(TStreamerElement::kCache))
          ) 
      {
         if (fLength[keep] == 0) {
            fLength[keep]++;
         }
         fLength[keep]++;
         fType[keep] = element->GetType() + kRegrouped;
         isOptimized = kTRUE;
      } else {
         if (fNewType[fNdata] != fType[fNdata]) {
            if (fNewType[fNdata] > 0) {
               if ( (fNewType[fNdata] == kObjectp || fNewType[fNdata] == kAnyp
                     || fNewType[fNdata] == kObject || fNewType[fNdata] == kAny
                     || fNewType[fNdata] == kTObject || fNewType[fNdata] == kTNamed || fNewType[fNdata] == kTString)
                   && (fType[fNdata] == kObjectp || fType[fNdata] == kAnyp
                       || fType[fNdata] == kObject || fType[fNdata] == kAny
                       || fType[fNdata] == kTObject || fType[fNdata] == kTNamed || fType[fNdata] == kTString )
                   ) {
                  fType[fNdata] = fNewType[fNdata];
               } else if (fType[fNdata] != kCounter) {
                  fType[fNdata] += kConv;
               }
            } else {
               if (fType[fNdata] == kCounter) {
                  Warning("Compile", "Counter %s should not be skipped from class %s", element->GetName(), GetName());
               }
               fType[fNdata] += kSkip;
            }
         }
         keep = fNdata;
         if (fLength[keep] == 0) {
            fLength[keep] = 1;
         }
         fNdata++;
      }
      previous = element;
   }

   if (fReadMemberWise) {
      fReadMemberWise->fActions.clear();
   } else {
      fReadMemberWise = new TStreamerInfoActions::TActionSequence(this,ndata);
   }
   if (fWriteMemberWise) {
      fWriteMemberWise->fActions.clear();
   } else {
      fWriteMemberWise = new TStreamerInfoActions::TActionSequence(this,ndata);
   }
   if ( isOptimized ) {
      fReadMemberWise->AddAction( ReadLoopInvalid, new TConfiguration(this,0,0) );
      fWriteMemberWise->AddAction( WriteLoopInvalid, new TConfiguration(this,0,0) );
   }

   for (i = 0; i < fNdata; ++i) {
      element = (TStreamerElement*) fElem[i];
      if (!element) {
         continue;
      }
      fComp[i].fClass = element->GetClassPointer();
      fComp[i].fNewClass = element->GetNewClass();
      fComp[i].fClassName = TString(element->GetTypeName()).Strip(TString::kTrailing, '*');
      fComp[i].fStreamer = element->GetStreamer();
      AddReadAction(i,element);
      AddWriteAction(i,element);
   }
   ComputeSize();

   fOptimized = isOptimized;

   if (gDebug > 0) {
      ls();
   }
}

//______________________________________________________________________________
void TStreamerInfo::AddReadAction(Int_t i, TStreamerElement* element)
{
   // Add a read action for the given element.

   switch (fType[i]) {
      // read basic types
      case TStreamerInfo::kBool:    fReadObjectWise->AddAction( ReadBasicType<Bool_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kChar:    fReadObjectWise->AddAction( ReadBasicType<Char_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kShort:   fReadObjectWise->AddAction( ReadBasicType<Short_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kInt:     fReadObjectWise->AddAction( ReadBasicType<Int_t>, new TConfiguration(this,i,fOffset[i]) );     break;
      case TStreamerInfo::kLong:    fReadObjectWise->AddAction( ReadBasicType<Long_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kLong64:  fReadObjectWise->AddAction( ReadBasicType<Long64_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kFloat:   fReadObjectWise->AddAction( ReadBasicType<Float_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kDouble:  fReadObjectWise->AddAction( ReadBasicType<Double_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kUChar:   fReadObjectWise->AddAction( ReadBasicType<UChar_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kUShort:  fReadObjectWise->AddAction( ReadBasicType<UShort_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kUInt:    fReadObjectWise->AddAction( ReadBasicType<UInt_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kULong:   fReadObjectWise->AddAction( ReadBasicType<ULong_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kULong64: fReadObjectWise->AddAction( ReadBasicType<ULong64_t>, new TConfiguration(this,i,fOffset[i]) ); break;
      case TStreamerInfo::kFloat16: {
         if (element->GetFactor() != 0) {
            fReadObjectWise->AddAction( ReadBasicType_WithFactor<float>, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
         } else {
            Int_t nbits = (Int_t)element->GetXmin();
            if (!nbits) nbits = 12;
            fReadObjectWise->AddAction( ReadBasicType_NoFactor<float>, new TConfNoFactor(this,i,fOffset[i],nbits) );               
         }
         break;
      }
      case TStreamerInfo::kDouble32: {
         if (element->GetFactor() != 0) {
            fReadObjectWise->AddAction( ReadBasicType_WithFactor<double>, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
         } else {
            Int_t nbits = (Int_t)element->GetXmin();
            if (!nbits) {
               fReadObjectWise->AddAction( ConvertBasicType<float,double>, new TConfiguration(this,i,fOffset[i]) );
            } else {
               fReadObjectWise->AddAction( ReadBasicType_NoFactor<double>, new TConfNoFactor(this,i,fOffset[i],nbits) );
            }
         }
         break;
      }
      case TStreamerInfo::kTNamed:  fReadObjectWise->AddAction( ReadTNamed, new TConfiguration(this,i,fOffset[i]) );    break;
         // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
         // Streamer alltogether.
      case TStreamerInfo::kTObject: fReadObjectWise->AddAction( ReadTObject, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kTString: fReadObjectWise->AddAction( ReadTString, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kSTL: {
         TClass *newClass = element->GetNewClass();
         TClass *oldClass = element->GetClassPointer();
         Bool_t isSTLbase = element->IsBase() && element->IsA()!=TStreamerBase::Class();
         
         if (element->GetArrayLength() <= 1) {
            if (fOldVersion<3){   // case of old TStreamerInfo
               if (newClass && newClass != oldClass) {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseChangedClass,ReadSTLObjectWiseStreamerV2>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseChangedClass,ReadSTLObjectWiseFastArrayV2>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetTypeName(),isSTLbase));                     
                  }
               } else {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseSameClass,ReadSTLObjectWiseStreamerV2>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseSameClass,ReadSTLObjectWiseFastArrayV2>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetTypeName(),isSTLbase));
                  }                  
               }
            } else {
               if (newClass && newClass != oldClass) {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseChangedClass,ReadSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseChangedClass,ReadSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetTypeName(),isSTLbase));                     
                  }
               } else {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseSameClass,ReadSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadSTLMemberWiseSameClass,ReadSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetTypeName(),isSTLbase));
                  }
               }               
            }
         } else {
            if (fOldVersion<3){   // case of old TStreamerInfo
               if (newClass && newClass != oldClass) {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseChangedClass,ReadSTLObjectWiseStreamerV2>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseChangedClass,ReadSTLObjectWiseFastArrayV2>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetTypeName(),isSTLbase));                     
                  }
               } else {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseSameClass,ReadSTLObjectWiseStreamerV2>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseSameClass,ReadSTLObjectWiseFastArrayV2>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetTypeName(),isSTLbase));
                  }                  
               }
            } else {
               if (newClass && newClass != oldClass) {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseChangedClass,ReadSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseChangedClass,ReadSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetTypeName(),isSTLbase));                     
                  }
               } else {
                  if (element->GetStreamer()) {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseSameClass,ReadSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
                  } else {
                     fReadObjectWise->AddAction(ReadSTL<ReadArraySTLMemberWiseSameClass,ReadSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetTypeName(),isSTLbase));
                  }
               }
            }
         }
         break;
      }
      default:
         fReadObjectWise->AddAction( GenericReadAction, new TGenericConfiguration(this,i) );
         break;
   }
   if (element->TestBit(TStreamerElement::kCache)) {
      TConfiguredAction action( fReadObjectWise->fActions.back() );  // Action is moved, we must pop it next.
      fReadObjectWise->fActions.pop_back();
      fReadObjectWise->AddAction( UseCache, new TConfigurationUseCache(this,action,element->TestBit(TStreamerElement::kRepeat)) );
   }            
   if (fReadMemberWise) {
      // This is for streaming via a TClonesArray.
      
      if (element->TestBit(TStreamerElement::kCache)) {
         TConfiguredAction action( GetVectorReadAction(this,element,fType[i],i,fOffset[i]) );
         fReadMemberWise->AddAction( UseCacheVectorPtrLoop, new TConfigurationUseCache(this,action,element->TestBit(TStreamerElement::kRepeat)) );
      } else {
         switch (fType[i]) {
            // read basic types
            case TStreamerInfo::kBool:    fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Bool_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kChar:    fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Char_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kShort:   fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Short_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kInt:     fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Int_t>, new TConfiguration(this,i,fOffset[i]) );     break;
            case TStreamerInfo::kLong:    fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Long_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kLong64:  fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Long64_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kFloat:   fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Float_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kDouble:  fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<Double_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kUChar:   fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<UChar_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kUShort:  fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<UShort_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kUInt:    fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<UInt_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kULong:   fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<ULong_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kULong64: fReadMemberWise->AddAction( ReadBasicTypeVectorPtrLoop<ULong64_t>, new TConfiguration(this,i,fOffset[i]) ); break;
            case TStreamerInfo::kFloat16: {
               if (element->GetFactor() != 0) {
                  fReadMemberWise->AddAction( VectorPtrLooper<ReadBasicType_WithFactor<float> >, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
               } else {
                  Int_t nbits = (Int_t)element->GetXmin();
                  if (!nbits) nbits = 12;
                  fReadMemberWise->AddAction( VectorPtrLooper<ReadBasicType_NoFactor<float> >, new TConfNoFactor(this,i,fOffset[i],nbits) );               
               }
               break;
            }
            case TStreamerInfo::kDouble32: {
               if (element->GetFactor() != 0) {
                  fReadMemberWise->AddAction( VectorPtrLooper<ReadBasicType_WithFactor<double> >, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
               } else {
                  Int_t nbits = (Int_t)element->GetXmin();
                  if (!nbits) {
                     fReadMemberWise->AddAction( VectorPtrLooper<ConvertBasicType<float,double> >, new TConfiguration(this,i,fOffset[i]) );
                  } else {
                     fReadMemberWise->AddAction( VectorPtrLooper<ReadBasicType_NoFactor<double> >, new TConfNoFactor(this,i,fOffset[i],nbits) );
                  }
               }
               break;
            }
            case TStreamerInfo::kTNamed:  fReadMemberWise->AddAction( VectorPtrLooper<ReadTNamed >, new TConfiguration(this,i,fOffset[i]) );    break;
               // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
               // Streamer alltogether.
            case TStreamerInfo::kTObject: fReadMemberWise->AddAction( VectorPtrLooper<ReadTObject >, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kTString: fReadMemberWise->AddAction( VectorPtrLooper<ReadTString >, new TConfiguration(this,i,fOffset[i]) );    break;
            default:
               fReadMemberWise->AddAction( GenericVectorPtrReadAction, new TGenericConfiguration(this,i) );
               break;
         }
      }
   }
}

//______________________________________________________________________________
void TStreamerInfo::AddWriteAction(Int_t i, TStreamerElement* /*element*/ )
{
   switch (fType[i]) {
      // write basic types
      case TStreamerInfo::kBool:    fWriteObjectWise->AddAction( WriteBasicType<Bool_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kChar:    fWriteObjectWise->AddAction( WriteBasicType<Char_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kShort:   fWriteObjectWise->AddAction( WriteBasicType<Short_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kInt:     fWriteObjectWise->AddAction( WriteBasicType<Int_t>, new TConfiguration(this,i,fOffset[i]) );     break;
      case TStreamerInfo::kLong:    fWriteObjectWise->AddAction( WriteBasicType<Long_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kLong64:  fWriteObjectWise->AddAction( WriteBasicType<Long64_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kFloat:   fWriteObjectWise->AddAction( WriteBasicType<Float_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kDouble:  fWriteObjectWise->AddAction( WriteBasicType<Double_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kUChar:   fWriteObjectWise->AddAction( WriteBasicType<UChar_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kUShort:  fWriteObjectWise->AddAction( WriteBasicType<UShort_t>, new TConfiguration(this,i,fOffset[i]) );  break;
      case TStreamerInfo::kUInt:    fWriteObjectWise->AddAction( WriteBasicType<UInt_t>, new TConfiguration(this,i,fOffset[i]) );    break;
      case TStreamerInfo::kULong:   fWriteObjectWise->AddAction( WriteBasicType<ULong_t>, new TConfiguration(this,i,fOffset[i]) );   break;
      case TStreamerInfo::kULong64: fWriteObjectWise->AddAction( WriteBasicType<ULong64_t>, new TConfiguration(this,i,fOffset[i]) ); break;
      /*case TStreamerInfo::kFloat16: {
         if (element->GetFactor() != 0) {
            fWriteObjectWise->AddAction( WriteBasicType_WithFactor<float>, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
         } else {
            Int_t nbits = (Int_t)element->GetXmin();
            if (!nbits) nbits = 12;
            fWriteObjectWise->AddAction( WriteBasicType_NoFactor<float>, new TConfNoFactor(this,i,fOffset[i],nbits) );               
         }
         break;
      } */
     /*case TStreamerInfo::kDouble32: {
        if (element->GetFactor() != 0) {
           fWriteObjectWise->AddAction( WriteBasicType_WithFactor<double>, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
        } else {
           Int_t nbits = (Int_t)element->GetXmin();
           if (!nbits) {
              fWriteObjectWise->AddAction( ConvertBasicType<float,double>, new TConfiguration(this,i,fOffset[i]) );
           } else {
              fWriteObjectWise->AddAction( WriteBasicType_NoFactor<double>, new TConfNoFactor(this,i,fOffset[i],nbits) );
           }
        }
        break;
     } */
     //case TStreamerInfo::kTNamed:  fWriteObjectWise->AddAction( WriteTNamed, new TConfiguration(this,i,fOffset[i]) );    break;
        // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
        // Streamer alltogether.
     //case TStreamerInfo::kTObject: fWriteObjectWise->AddAction( WriteTObject, new TConfiguration(this,i,fOffset[i]) );    break;
     //case TStreamerInfo::kTString: fWriteObjectWise->AddAction( WriteTString, new TConfiguration(this,i,fOffset[i]) );    break;
     /*case TStreamerInfo::kSTL: {
        TClass *newClass = element->GetNewClass();
        TClass *oldClass = element->GetClassPointer();
        Bool_t isSTLbase = element->IsBase() && element->IsA()!=TStreamerBase::Class();
        
        if (element->GetArrayLength() <= 1) {
           if (newClass && newClass != oldClass) {
              if (element->GetStreamer()) {
                 fWriteObjectWise->AddAction(WriteSTL<WriteSTLMemberWiseChangedClass,WriteSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
              } else {
                 fWriteObjectWise->AddAction(WriteSTL<WriteSTLMemberWiseChangedClass,WriteSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],1,oldClass,newClass,element->GetTypeName(),isSTLbase));                     
              }
           } else {
              if (element->GetStreamer()) {
                 fWriteObjectWise->AddAction(WriteSTL<WriteSTLMemberWiseSameClass,WriteSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
              } else {
                 fWriteObjectWise->AddAction(WriteSTL<WriteSTLMemberWiseSameClass,WriteSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],1,oldClass,element->GetTypeName(),isSTLbase));
              }                  
           }                  
        } else {
           if (newClass && newClass != oldClass) {
              if (element->GetStreamer()) {
                 fWriteObjectWise->AddAction(WriteSTL<WriteArraySTLMemberWiseChangedClass,WriteSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
              } else {
                 fWriteObjectWise->AddAction(WriteSTL<WriteArraySTLMemberWiseChangedClass,WriteSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,newClass,element->GetTypeName(),isSTLbase));                     
              }
           } else {
              if (element->GetStreamer()) {
                 fWriteObjectWise->AddAction(WriteSTL<WriteArraySTLMemberWiseSameClass,WriteSTLObjectWiseStreamer>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetStreamer(),element->GetTypeName(),isSTLbase));
              } else {
                 fWriteObjectWise->AddAction(WriteSTL<WriteArraySTLMemberWiseSameClass,WriteSTLObjectWiseFastArray>, new TConfigSTL(this,i,fOffset[i],element->GetArrayLength(),oldClass,element->GetTypeName(),isSTLbase));
              }                  
           }                                    
        }
        break;
     } */
      default:
         fWriteObjectWise->AddAction( GenericWriteAction, new TGenericConfiguration(this,i) );
         break;
   }
#if defined(CDJ_NO_COMPILE)
   if (element->TestBit(TStreamerElement::kCache)) {
      TConfiguredAction action( fWriteObjectWise->fActions.back() );  // Action is moved, we must pop it next.
      fWriteObjectWise->fActions.pop_back();
      fWriteObjectWise->AddAction( UseCache, new TConfigurationUseCache(this,action,element->TestBit(TStreamerElement::kRepeat)) );
   }
#endif
   if (fWriteMemberWise) {
      // This is for streaming via a TClonesArray.

#if defined(CDJ_NO_COMPILE)
      if (element->TestBit(TStreamerElement::kCache)) {
         TConfiguredAction action( GetVectorWriteAction(this,element,fType[i],i,fOffset[i]) );
         fWriteMemberWise->AddAction( UseCacheVectorPtrLoop, new TConfigurationUseCache(this,action,element->TestBit(TStreamerElement::kRepeat)) );
      } else {
         switch (fType[i]) {
            // read basic types
            case TStreamerInfo::kBool:    fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Bool_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kChar:    fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Char_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kShort:   fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Short_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kInt:     fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Int_t>, new TConfiguration(this,i,fOffset[i]) );     break;
            case TStreamerInfo::kLong:    fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Long_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kLong64:  fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Long64_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kFloat:   fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Float_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kDouble:  fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<Double_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kUChar:   fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<UChar_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kUShort:  fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<UShort_t>, new TConfiguration(this,i,fOffset[i]) );  break;
            case TStreamerInfo::kUInt:    fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<UInt_t>, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kULong:   fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<ULong_t>, new TConfiguration(this,i,fOffset[i]) );   break;
            case TStreamerInfo::kULong64: fWriteMemberWise->AddAction( WriteBasicTypeVectorPtrLoop<ULong64_t>, new TConfiguration(this,i,fOffset[i]) ); break;
            case TStreamerInfo::kFloat16: {
               if (element->GetFactor() != 0) {
                  fWriteMemberWise->AddAction( VectorPtrLooper<WriteBasicType_WithFactor<float> >, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
               } else {
                  Int_t nbits = (Int_t)element->GetXmin();
                  if (!nbits) nbits = 12;
                  fWriteMemberWise->AddAction( VectorPtrLooper<WriteBasicType_NoFactor<float> >, new TConfNoFactor(this,i,fOffset[i],nbits) );               
               }
               break;
            }
            case TStreamerInfo::kDouble32: {
               if (element->GetFactor() != 0) {
                  fWriteMemberWise->AddAction( VectorPtrLooper<WriteBasicType_WithFactor<double> >, new TConfWithFactor(this,i,fOffset[i],element->GetFactor(),element->GetXmin()) );
               } else {
                  Int_t nbits = (Int_t)element->GetXmin();
                  if (!nbits) {
                     fWriteMemberWise->AddAction( VectorPtrLooper<ConvertBasicType<float,double> >, new TConfiguration(this,i,fOffset[i]) );
                  } else {
                     fWriteMemberWise->AddAction( VectorPtrLooper<WriteBasicType_NoFactor<double> >, new TConfNoFactor(this,i,fOffset[i],nbits) );
                  }
               }
               break;
            }
            case TStreamerInfo::kTNamed:  fWriteMemberWise->AddAction( VectorPtrLooper<WriteTNamed >, new TConfiguration(this,i,fOffset[i]) );    break;
               // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
               // Streamer alltogether.
            case TStreamerInfo::kTObject: fWriteMemberWise->AddAction( VectorPtrLooper<WriteTObject >, new TConfiguration(this,i,fOffset[i]) );    break;
            case TStreamerInfo::kTString: fWriteMemberWise->AddAction( VectorPtrLooper<WriteTString >, new TConfiguration(this,i,fOffset[i]) );    break;
            default:
               fWriteMemberWise->AddAction( GenericVectorPtrWriteAction, new TGenericConfiguration(this,i) );
               break;
         }
      }
#else
      fWriteMemberWise->AddAction( GenericVectorPtrWriteAction, new TGenericConfiguration(this,i) );
#endif
  }
}

//______________________________________________________________________________
TStreamerInfoActions::TActionSequence *TStreamerInfoActions::TActionSequence::CreateReadMemberWiseActions(TVirtualStreamerInfo *info, TVirtualCollectionProxy &proxy)
{
   // Create the bundle of the actions necessary for the streaming memberwise of the content described by 'info' into the collection described by 'proxy'

   if (info == 0) {
      return new TStreamerInfoActions::TActionSequence(0,0);
   }

   if (info->IsOptimized()) {
      // For now insures that the StreamerInfo is not optimized
      info->SetBit(TVirtualStreamerInfo::kCannotOptimize);
      info->Compile();
   }
   UInt_t ndata = info->GetElements()->GetEntries();
   TStreamerInfoActions::TActionSequence *sequence = new TStreamerInfoActions::TActionSequence(info,ndata);
   if ( (proxy.GetCollectionType() == TClassEdit::kVector) || (proxy.GetProperties() & TVirtualCollectionProxy::kIsEmulated) ) 
   {
      if (proxy.HasPointers()) {
         // Instead of the creating a new one let's copy the one from the StreamerInfo.
         delete sequence;
         
         sequence = static_cast<TStreamerInfo*>(info)->GetReadMemberWiseActions(kTRUE)->CreateCopy();
         
         return sequence;
      }
      
      // We can speed up the iteration in case of vector.  We also know that all emulated collection are stored internally as a vector.
      Long_t increment = proxy.GetIncrement();
      sequence->fLoopConfig = new TVectorLoopConfig(increment);
   } else if (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
              || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) 
   {
      Long_t increment = proxy.GetIncrement();
      sequence->fLoopConfig = new TVectorLoopConfig(increment);
      // sequence->fLoopConfig = new TAssocLoopConfig(proxy);
   } else {
      sequence->fLoopConfig = new TGenericLoopConfig(&proxy);
   }
   for (UInt_t i = 0; i < ndata; ++i) {
      TStreamerElement *element = (TStreamerElement*) info->GetElements()->At(i);
      if (!element) {
         break;
      }
      if (element->GetType() < 0) {
         // -- Skip an ignored TObject base class.
         // Note: The only allowed negative value here is -1, and signifies that Build() has found a TObject
         // base class and TClass::IgnoreTObjectStreamer() was called.  In this case the compiled version of the
         // elements omits the TObject base class element, which has to be compensated for by TTree::Bronch()
         // when it is making branches for a split object.
         continue;
      }
      Int_t asize = element->GetSize();
      if (element->GetArrayLength()) {
         asize /= element->GetArrayLength();
      }
      Int_t oldType = element->GetType();
      Int_t newType = element->GetNewType();

      Int_t offset = element->GetOffset();
      if (newType != oldType) {
         if (newType > 0) {
            if (oldType != TVirtualStreamerInfo::kCounter) {
               oldType += TVirtualStreamerInfo::kConv;
            }
         } else {
            oldType += TVirtualStreamerInfo::kSkip;
         }
      }
      if ( (proxy.GetCollectionType() == TClassEdit::kVector) || (proxy.GetProperties() & TVirtualCollectionProxy::kIsEmulated)  
            || (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
            || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) )
      {

         // We can speed up the iteration in case of vector.  We also know that all emulated collection are stored internally as a vector.
         if (element->TestBit(TStreamerElement::kCache)) {
            TConfiguredAction action( GetVectorReadAction(info,element,oldType,i,offset) );
            sequence->AddAction( UseCacheVectorLoop,  new TConfigurationUseCache(info,action,element->TestBit(TStreamerElement::kRepeat)) );
         } else {            
            sequence->AddAction(GetVectorReadAction(info,element,oldType,i,offset));
         }
         
//         } else if (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
//                    || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) {
//            sequence->AddAction( GenericAssocCollectionAction, new TConfigSTL(info,i,offset,0,proxy.GetCollectionClass(),0,0) );
      } else {
         // The usual collection case.
         if (element->TestBit(TStreamerElement::kCache)) {
            TConfiguredAction action( GetVectorReadAction(info,element,oldType,i,offset) );
            sequence->AddAction( UseCacheGenericCollection, new TConfigurationUseCache(info,action,element->TestBit(TStreamerElement::kRepeat)) );
         } else {
            switch (oldType) {
                  // read basic types
               case TVirtualStreamerInfo::kBool:    sequence->AddAction( ReadBasicTypeGenericLoop<Bool_t>, new TConfiguration(info,i,offset) );    break;
               case TVirtualStreamerInfo::kChar:    sequence->AddAction( ReadBasicTypeGenericLoop<Char_t>, new TConfiguration(info,i,offset) );    break;
               case TVirtualStreamerInfo::kShort:   sequence->AddAction( ReadBasicTypeGenericLoop<Short_t>, new TConfiguration(info,i,offset) );   break;
               case TVirtualStreamerInfo::kInt:     sequence->AddAction( ReadBasicTypeGenericLoop<Int_t>, new TConfiguration(info,i,offset) );     break;
               case TVirtualStreamerInfo::kLong:    sequence->AddAction( ReadBasicTypeGenericLoop<Long_t>, new TConfiguration(info,i,offset) );    break;
               case TVirtualStreamerInfo::kLong64:  sequence->AddAction( ReadBasicTypeGenericLoop<Long64_t>, new TConfiguration(info,i,offset) );  break;
               case TVirtualStreamerInfo::kFloat:   sequence->AddAction( ReadBasicTypeGenericLoop<Float_t>, new TConfiguration(info,i,offset) );   break;
               case TVirtualStreamerInfo::kDouble:  sequence->AddAction( ReadBasicTypeGenericLoop<Double_t>, new TConfiguration(info,i,offset) );  break;
               case TVirtualStreamerInfo::kUChar:   sequence->AddAction( ReadBasicTypeGenericLoop<UChar_t>, new TConfiguration(info,i,offset) );   break;
               case TVirtualStreamerInfo::kUShort:  sequence->AddAction( ReadBasicTypeGenericLoop<UShort_t>, new TConfiguration(info,i,offset) );  break;
               case TVirtualStreamerInfo::kUInt:    sequence->AddAction( ReadBasicTypeGenericLoop<UInt_t>, new TConfiguration(info,i,offset) );    break;
               case TVirtualStreamerInfo::kULong:   sequence->AddAction( ReadBasicTypeGenericLoop<ULong_t>, new TConfiguration(info,i,offset) );   break;
               case TVirtualStreamerInfo::kULong64: sequence->AddAction( ReadBasicTypeGenericLoop<ULong64_t>, new TConfiguration(info,i,offset) ); break;
               case TVirtualStreamerInfo::kFloat16: {
                  if (element->GetFactor() != 0) {
                     sequence->AddAction( GenericLooper<ReadBasicType_WithFactor<float> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
                  } else {
                     Int_t nbits = (Int_t)element->GetXmin();
                     if (!nbits) nbits = 12;
                     sequence->AddAction( GenericLooper<ReadBasicType_NoFactor<float> >, new TConfNoFactor(info,i,offset,nbits) );               
                  }
                  break;
               }
               case TVirtualStreamerInfo::kDouble32: {
                  if (element->GetFactor() != 0) {
                     sequence->AddAction( GenericLooper<ReadBasicType_WithFactor<double> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
                  } else {
                     Int_t nbits = (Int_t)element->GetXmin();
                     if (!nbits) {
                        sequence->AddAction( GenericLooper<ConvertBasicType<float,double> >, new TConfiguration(info,i,offset) );
                     } else {
                        sequence->AddAction( GenericLooper<ReadBasicType_NoFactor<double> >, new TConfNoFactor(info,i,offset,nbits) );
                     }
                  }
                  break;
               }
               case TVirtualStreamerInfo::kTNamed:  sequence->AddAction( GenericLooper<ReadTNamed >, new TConfiguration(info,i,offset) );    break;
                  // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
                  // Streamer alltogether.
               case TVirtualStreamerInfo::kTObject: sequence->AddAction( GenericLooper<ReadTObject >, new TConfiguration(info,i,offset) );    break;
               case TVirtualStreamerInfo::kTString: sequence->AddAction( GenericLooper<ReadTString >, new TConfiguration(info,i,offset) );    break;
               default:
                  sequence->AddAction( GenericCollectionReadAction, new TConfigSTL(info,i,0 /* the offset will be used from TStreamerInfo */,0,proxy.GetCollectionClass(),0,0) );
                  break;
            }
         }
      }
   }
   return sequence;
}

//______________________________________________________________________________
TStreamerInfoActions::TActionSequence *TStreamerInfoActions::TActionSequence::CreateWriteMemberWiseActions(TVirtualStreamerInfo *info, TVirtualCollectionProxy &proxy)
{
      // Create the bundle of the actions necessary for the streaming memberwise of the content described by 'info' into the collection described by 'proxy'

      if (info == 0) {
         return new TStreamerInfoActions::TActionSequence(0,0);
      }

      if (info->IsOptimized()) {
         // For now insures that the StreamerInfo is not optimized
         info->SetBit(TVirtualStreamerInfo::kCannotOptimize);
         info->Compile();
      }
      UInt_t ndata = info->GetElements()->GetEntries();
      TStreamerInfoActions::TActionSequence *sequence = new TStreamerInfoActions::TActionSequence(info,ndata);
      if ( (proxy.GetCollectionType() == TClassEdit::kVector) || (proxy.GetProperties() & TVirtualCollectionProxy::kIsEmulated) ) 
      {
         if (proxy.HasPointers()) {
            // Instead of the creating a new one let's copy the one from the StreamerInfo.
            delete sequence;

            sequence = static_cast<TStreamerInfo*>(info)->GetWriteMemberWiseActions(kTRUE)->CreateCopy();

            return sequence;
         }

         // We can speed up the iteration in case of vector.  We also know that all emulated collection are stored internally as a vector.
         Long_t increment = proxy.GetIncrement();
         sequence->fLoopConfig = new TVectorLoopConfig(increment);
      /*} else if (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
                 || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) 
      {
         Long_t increment = proxy.GetIncrement();
         sequence->fLoopConfig = new TVectorLoopConfig(increment);
         // sequence->fLoopConfig = new TAssocLoopConfig(proxy); */
      } else {        
         sequence->fLoopConfig = new TGenericLoopConfig(&proxy);
      }
      for (UInt_t i = 0; i < ndata; ++i) {
         TStreamerElement *element = (TStreamerElement*) info->GetElements()->At(i);
         if (!element) {
            break;
         }
         if (element->GetType() < 0) {
            // -- Skip an ignored TObject base class.
            // Note: The only allowed negative value here is -1, and signifies that Build() has found a TObject
            // base class and TClass::IgnoreTObjectStreamer() was called.  In this case the compiled version of the
            // elements omits the TObject base class element, which has to be compensated for by TTree::Bronch()
            // when it is making branches for a split object.
            continue;
         }
         Int_t asize = element->GetSize();
         if (element->GetArrayLength()) {
            asize /= element->GetArrayLength();
         }
         Int_t oldType = element->GetType();
         Int_t offset = element->GetOffset();
#if defined(CDJ_NO_COMPILE)
         Int_t newType = element->GetNewType();

         if (newType != oldType) {
            if (newType > 0) {
               if (oldType != TVirtualStreamerInfo::kCounter) {
                  oldType += TVirtualStreamerInfo::kConv;
               }
            } else {
               oldType += TVirtualStreamerInfo::kSkip;
            }
         }
         if ( (proxy.GetCollectionType() == TClassEdit::kVector) || (proxy.GetProperties() & TVirtualCollectionProxy::kIsEmulated)  
               /*|| (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
               || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) */ )
         {

            // We can speed up the iteration in case of vector.  We also know that all emulated collection are stored internally as a vector.
            if (element->TestBit(TStreamerElement::kCache)) {
               TConfiguredAction action( GetVectorWriteAction(info,element,oldType,i,offset) );
               sequence->AddAction( UseCacheVectorLoop,  new TConfigurationUseCache(info,action,element->TestBit(TStreamerElement::kRepeat)) );
            } else {            
               sequence->AddAction(GetVectorWriteAction(info,element,oldType,i,offset));
            }

   //         } else if (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
   //                    || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap) {
   //            sequence->AddAction( GenericAssocCollectionAction, new TConfigSTL(info,i,offset,0,proxy.GetCollectionClass(),0,0) );
         } else {
            // The usual collection case.
            if (element->TestBit(TStreamerElement::kCache)) {
               TConfiguredAction action( GetVectorWriteAction(info,element,oldType,i,offset) );
               sequence->AddAction( UseCacheGenericCollection, new TConfigurationUseCache(info,action,element->TestBit(TStreamerElement::kRepeat)) );
            } else {
               switch (oldType) {
                     // read basic types
                  case TVirtualStreamerInfo::kBool:    sequence->AddAction( WriteBasicTypeGenericLoop<Bool_t>, new TConfiguration(info,i,offset) );    break;
                  case TVirtualStreamerInfo::kChar:    sequence->AddAction( WriteBasicTypeGenericLoop<Char_t>, new TConfiguration(info,i,offset) );    break;
                  case TVirtualStreamerInfo::kShort:   sequence->AddAction( WriteBasicTypeGenericLoop<Short_t>, new TConfiguration(info,i,offset) );   break;
                  case TVirtualStreamerInfo::kInt:     sequence->AddAction( WriteBasicTypeGenericLoop<Int_t>, new TConfiguration(info,i,offset) );     break;
                  case TVirtualStreamerInfo::kLong:    sequence->AddAction( WriteBasicTypeGenericLoop<Long_t>, new TConfiguration(info,i,offset) );    break;
                  case TVirtualStreamerInfo::kLong64:  sequence->AddAction( WriteBasicTypeGenericLoop<Long64_t>, new TConfiguration(info,i,offset) );  break;
                  case TVirtualStreamerInfo::kFloat:   sequence->AddAction( WriteBasicTypeGenericLoop<Float_t>, new TConfiguration(info,i,offset) );   break;
                  case TVirtualStreamerInfo::kDouble:  sequence->AddAction( WriteBasicTypeGenericLoop<Double_t>, new TConfiguration(info,i,offset) );  break;
                  case TVirtualStreamerInfo::kUChar:   sequence->AddAction( WriteBasicTypeGenericLoop<UChar_t>, new TConfiguration(info,i,offset) );   break;
                  case TVirtualStreamerInfo::kUShort:  sequence->AddAction( WriteBasicTypeGenericLoop<UShort_t>, new TConfiguration(info,i,offset) );  break;
                  case TVirtualStreamerInfo::kUInt:    sequence->AddAction( WriteBasicTypeGenericLoop<UInt_t>, new TConfiguration(info,i,offset) );    break;
                  case TVirtualStreamerInfo::kULong:   sequence->AddAction( WriteBasicTypeGenericLoop<ULong_t>, new TConfiguration(info,i,offset) );   break;
                  case TVirtualStreamerInfo::kULong64: sequence->AddAction( WriteBasicTypeGenericLoop<ULong64_t>, new TConfiguration(info,i,offset) ); break;
                  case TVirtualStreamerInfo::kFloat16: {
                     if (element->GetFactor() != 0) {
                        sequence->AddAction( GenericLooper<WriteBasicType_WithFactor<float> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
                     } else {
                        Int_t nbits = (Int_t)element->GetXmin();
                        if (!nbits) nbits = 12;
                        sequence->AddAction( GenericLooper<WriteBasicType_NoFactor<float> >, new TConfNoFactor(info,i,offset,nbits) );               
                     }
                     break;
                  }
                  case TVirtualStreamerInfo::kDouble32: {
                     if (element->GetFactor() != 0) {
                        sequence->AddAction( GenericLooper<WriteBasicType_WithFactor<double> >, new TConfWithFactor(info,i,offset,element->GetFactor(),element->GetXmin()) );
                     } else {
                        Int_t nbits = (Int_t)element->GetXmin();
                        if (!nbits) {
                           sequence->AddAction( GenericLooper<ConvertBasicType<float,double> >, new TConfiguration(info,i,offset) );
                        } else {
                           sequence->AddAction( GenericLooper<WriteBasicType_NoFactor<double> >, new TConfNoFactor(info,i,offset,nbits) );
                        }
                     }
                     break;
                  }
                  case TVirtualStreamerInfo::kTNamed:  sequence->AddAction( GenericLooper<WriteTNamed >, new TConfiguration(info,i,offset) );    break;
                     // Idea: We should calculate the CanIgnoreTObjectStreamer here and avoid calling the
                     // Streamer alltogether.
                  case TVirtualStreamerInfo::kTObject: sequence->AddAction( GenericLooper<WriteTObject >, new TConfiguration(info,i,offset) );    break;
                  case TVirtualStreamerInfo::kTString: sequence->AddAction( GenericLooper<WriteTString >, new TConfiguration(info,i,offset) );    break;
                  default:
                     sequence->AddAction( GenericCollectionWriteAction, new TConfigSTL(info,i,0 /* the offset will be used from TStreamerInfo */,0,proxy.GetCollectionClass(),0,0) );
                     break;
               }
            }
         }
#else
         if ( (proxy.GetCollectionType() == TClassEdit::kVector) || (proxy.GetProperties() & TVirtualCollectionProxy::kIsEmulated)  
               /*|| (proxy.GetCollectionType() == TClassEdit::kSet || proxy.GetCollectionType() == TClassEdit::kMultiSet
               || proxy.GetCollectionType() == TClassEdit::kMap || proxy.GetCollectionType() == TClassEdit::kMultiMap)*/ )
         {
           sequence->AddAction(GetVectorWriteAction(info,element,oldType,i,offset));
         } else {
           sequence->AddAction( GenericCollectionWriteAction, new TConfigSTL(info,i,0 /* the offset will be used from TStreamerInfo */,0,proxy.GetCollectionClass(),0,0) );
         }
#endif
      }
      return sequence;
}
void TStreamerInfoActions::TActionSequence::AddToOffset(Int_t delta)
{
   // Add the (potentially negative) delta to all the configuration's offset.  This is used by
   // TTBranchElement in the case of split sub-object.

   TStreamerInfoActions::ActionContainer_t::iterator end = fActions.end();
   for(TStreamerInfoActions::ActionContainer_t::iterator iter = fActions.begin();
       iter != end;
       ++iter) 
   {
      iter->fConfiguration->AddToOffset(delta);
   }
}

TStreamerInfoActions::TActionSequence *TStreamerInfoActions::TActionSequence::CreateCopy()
{
   // Create a copy of this sequence.
   
   TStreamerInfoActions::TActionSequence *sequence = new TStreamerInfoActions::TActionSequence(fStreamerInfo,fActions.size());

   sequence->fLoopConfig = fLoopConfig ? fLoopConfig->Copy() : 0;

   TStreamerInfoActions::ActionContainer_t::iterator end = fActions.end();
   for(TStreamerInfoActions::ActionContainer_t::iterator iter = fActions.begin();
       iter != end;
       ++iter) 
   {
      TConfiguration *conf = iter->fConfiguration->Copy();
      sequence->AddAction( iter->fAction, conf );
   }
   return sequence;
}

TStreamerInfoActions::TActionSequence *TStreamerInfoActions::TActionSequence::CreateSubSequence(const std::vector<Int_t> &element_ids, size_t offset)
{
   // Create a sequence containing the subset of the action corresponding to the SteamerElement whose ids is contained in the vector.
   // 'offset' is the location of this 'class' within the object (address) that will be pass to ReadBuffer when using this sequence.
   
   TStreamerInfoActions::TActionSequence *sequence = new TStreamerInfoActions::TActionSequence(fStreamerInfo,element_ids.size());
   
   sequence->fLoopConfig = fLoopConfig ? fLoopConfig->Copy() : 0;
   
   for(UInt_t id = 0; id < element_ids.size(); ++id) {
      if ( element_ids[id] < 0 ) {
         TStreamerInfoActions::ActionContainer_t::iterator end = fActions.end();
         for(TStreamerInfoActions::ActionContainer_t::iterator iter = fActions.begin();
             iter != end;
             ++iter) 
         {
            TConfiguration *conf = iter->fConfiguration->Copy();
            conf->AddToOffset(offset);
            sequence->AddAction( iter->fAction, conf );
         }
      } else {
         TStreamerInfoActions::ActionContainer_t::iterator end = fActions.end();
         for(TStreamerInfoActions::ActionContainer_t::iterator iter = fActions.begin();
             iter != end;
             ++iter) {
            if ( iter->fConfiguration->fElemId == (UInt_t)element_ids[id] ) {
               TConfiguration *conf = iter->fConfiguration->Copy();
               conf->AddToOffset(offset);
               sequence->AddAction( iter->fAction, conf );
            }
         }
      }
   }
   return sequence;
}

void TStreamerInfoActions::TActionSequence::Print(Option_t *) const
{
   // Add the (potentially negative) delta to all the configuration's offset.  This is used by
   // TTBranchElement in the case of split sub-object.

   if (fLoopConfig) {
      fLoopConfig->Print();
   }
   TStreamerInfoActions::ActionContainer_t::const_iterator end = fActions.end();
   for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = fActions.begin();
       iter != end;
       ++iter) 
   {
      iter->fConfiguration->Print();
   }
}


