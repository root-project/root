// @(#)root/io:$Id$
// Author: Philippe Canal 05/2010

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TStreamerInfoActions
#define ROOT_TStreamerInfoActions

#include <vector>

#include "TStreamerInfo.h"

namespace TStreamerInfoActions {

   class TConfiguration {
      // Base class of the Configurations.
   protected:
   public:
      TVirtualStreamerInfo *fInfo;    // TStreamerInfo form which the action is derived
      UInt_t                fElemId;  // Identifier of the TStreamerElement
      Int_t                 fOffset;  // Offset within the object
      UInt_t                fLength;  // Number of element in a fixed length array.
   public:
      TConfiguration(TVirtualStreamerInfo *info, UInt_t id, Int_t offset) : fInfo(info), fElemId(id), fOffset(offset),fLength(1) {};
      TConfiguration(TVirtualStreamerInfo *info, UInt_t id, Int_t offset, UInt_t length) : fInfo(info), fElemId(id), fOffset(offset),fLength(length) {};
      virtual ~TConfiguration() {};
      
      virtual void AddToOffset(Int_t delta);

      virtual TConfiguration *Copy() { return new TConfiguration(*this); }

      virtual void Print() const;
      virtual void PrintDebug(TBuffer &buffer, void *object) const;
   };

   class TLoopConfiguration {
      // Base class of the Configurations for the member wise looping routines.
   public:
      TLoopConfiguration() {};
      // virtual void PrintDebug(TBuffer &buffer, void *object) const;
      virtual ~TLoopConfiguration() {};
      virtual void Print() const;
      virtual void *GetFirstAddress(void *start, const void *end) const = 0;
      virtual TLoopConfiguration* Copy() = 0; // { return new TLoopConfiguration(*this); }
   };
      
   typedef TVirtualCollectionProxy::Next_t Next_t;

   typedef Int_t (*TStreamerInfoAction_t)(TBuffer &buf, void *obj, const TConfiguration *conf);
   typedef Int_t (*TStreamerInfoVecPtrLoopAction_t)(TBuffer &buf, void *iter, const void *end, const TConfiguration *conf);
   typedef Int_t (*TStreamerInfoLoopAction_t)(TBuffer &buf, void *iter, const void *end, const TLoopConfiguration *loopconf, const TConfiguration *conf);
   
   class TConfiguredAction : public TObject {
   public:
      union {
         TStreamerInfoAction_t           fAction;
         TStreamerInfoVecPtrLoopAction_t fVecPtrLoopAction;
         TStreamerInfoLoopAction_t       fLoopAction;
      };
      TConfiguration              *fConfiguration;
   public:
      TConfiguredAction() : fAction(0), fConfiguration(0) {}
      TConfiguredAction(const TConfiguredAction &rval) : TObject(rval), fAction(rval.fAction), fConfiguration(rval.fConfiguration)
      {
         // Technically this is a move constructor ...
         const_cast<TConfiguredAction&>(rval).fConfiguration = 0;
      }
      TConfiguredAction(TStreamerInfoAction_t action, TConfiguration *conf) : fAction(action), fConfiguration(conf)
      {
         // Usual constructor.
      }
      TConfiguredAction(TStreamerInfoVecPtrLoopAction_t action, TConfiguration *conf) : fVecPtrLoopAction(action), fConfiguration(conf)
      {
         // Usual constructor.
      }
      TConfiguredAction(TStreamerInfoLoopAction_t action, TConfiguration *conf) : fLoopAction(action), fConfiguration(conf)
      {
         // Usual constructor.
      }
      ~TConfiguredAction() {
         // Usual destructor.
         // Idea: the configuration ownership might be moved to a single list so that
         // we can shared them between the optimized and non-optimized list of actions.
         delete fConfiguration; 
      }
      void PrintDebug(TBuffer &buffer, void *object) const;
                      
      inline Int_t operator()(TBuffer &buffer, void *object) const {
         return fAction(buffer, object, fConfiguration);
      }

      inline Int_t operator()(TBuffer &buffer, void *start_collection, const void *end_collection) const {
         return fVecPtrLoopAction(buffer, start_collection, end_collection, fConfiguration);
      }
      
      inline Int_t operator()(TBuffer &buffer, void *start_collection, const void *end_collection, const TLoopConfiguration *loopconf) const {
         return fLoopAction(buffer, start_collection, end_collection, loopconf, fConfiguration);
      }
      
      ClassDef(TConfiguredAction,0); // A configured action
   };
   
   typedef std::vector<TConfiguredAction> ActionContainer_t;
   class TActionSequence : public TObject {
      TActionSequence() {};
   public:
      TActionSequence(TVirtualStreamerInfo *info, UInt_t maxdata) : fStreamerInfo(info), fLoopConfig(0) { fActions.reserve(maxdata); };
      ~TActionSequence() { 
         delete fLoopConfig; 
      }

      template <typename action_t> 
      void AddAction( action_t action, TConfiguration *conf ) {
         fActions.push_back( TConfiguredAction(action, conf) );
      }
      void AddAction(const TConfiguredAction &action ) {
         fActions.push_back( action );
      }
      
      TVirtualStreamerInfo *fStreamerInfo; // StreamerInfo used to derive these actions.
      TLoopConfiguration   *fLoopConfig;   // If this is a bundle of memberwise streaming action, this configures the looping
      ActionContainer_t     fActions;

      void AddToOffset(Int_t delta);
      
      TActionSequence *CreateCopy();      
      static TActionSequence *CreateReadMemberWiseActions(TVirtualStreamerInfo *info, TVirtualCollectionProxy &proxy);
      TActionSequence *CreateSubSequence(const std::vector<Int_t> &element_ids, size_t offset);      
      
      void Print(Option_t * = "") const;

      ClassDef(TActionSequence,0);
   };

}

#endif // ROOT_TStreamerInfoActions


