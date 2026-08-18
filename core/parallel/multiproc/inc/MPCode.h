/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_MPCode
#define ROOT_MPCode

/////////////////////////////////////////////////////////////////////////
/// This namespace prevents conflicts between MPCode::kError and
/// ELogLevel::kError
namespace MPCode {

   //////////////////////////////////////////////////////////////////////////
   ///
   /// An enumeration of the message codes handled by TProcessExecutor,
   /// TTreeProcessorMP, TMPWorker, TMPWorkerTree and by the low level
   /// classes TMPClient and TMPWorker.
   ///
   //////////////////////////////////////////////////////////////////////////

   enum EMPCode : unsigned {
   //not an enum class because we want to be able to easily cast back and forth from unsigned
      /* TProcessExecutor::Map */
      kExecFunc = 0,    ///< Execute function without arguments
      kExecFuncWithArg, ///< Execute function with the argument contained in the message
      kFuncResult,      ///< The message contains the result of a function execution
      /* TProcessExecutor::MapReduce */
      kIdling = 100,    ///< We are ready for the next task
      kSendResult,      ///< Ask for a kFuncResult/kProcResult
      /* TTreeProcessorMP::Process */
      kProcFile = 200,  ///< Tell a TMPWorkerTree which tree to process. The object sent is a TreeInfo
      kProcRange,       ///< Tell a TMPWorkerTree which tree and entries range to process. The object sent is a TreeRangeInfo
      kProcTree,        ///< Tell a TMPWorkerTree to process the tree that was passed to it at construction time
      kProcSelector,    ///< Tell a TMPWorkerTree to process the tree using the selector passed to it at construction time
      kProcResult,      ///< The message contains the result of the processing of a TTree
      kProcEnded,       ///< Tell the client we are done processing (i.e. we have reached the target number of entries to process)
      kProcError,       ///< Tell the client there was an error while processing
      /* Generic messages, including errors */
      kMessage = 1000,  ///< Generic message
      kError,           ///< Error message
      kFatalError,      ///< Fatal error: whoever sends this message is terminating execution
      kShutdownOrder,   ///< Used by the client to tell servers to shutdown
      kShutdownNotice,  ///< Used by the workers to notify client of shutdown
      kRecvError        ///< Error while reading from the socket
   };
}

#endif
