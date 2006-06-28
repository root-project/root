// @(#)root/meta:$Name:  $: $Id: TVirtualRefProxy.h,v 1.1 2006/06/28 10:06:50 pcanal Exp $
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualRefProxy
#define ROOT_TVirtualRefProxy

// Framework include files
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif  // ROOT_Rtypes

// Forward declarations
class TClass;
class TFormLeafInfoReference;

//______________________________________________________________________________
//  
//   Abstract proxy definition to follow reference objects.
//   The type of the referenced object must either be known at compilation
//      time or it must be possible to guess it reading the first TTree entry.
//      In this case the following condiitons must be met:
//      - GetValueClass() must return the TClass to the referenced
//        objects (or a base class)
//      - GetObject() must return the pointer to the referenced
//        object. TTreeFormula then figures out how to access the
//        value to be plotted. 
//      This is typically the case for generic references like 
//      they are used in POOL or Gaudi.
// 
//______________________________________________________________________________
class TVirtualRefProxy  {
public:
   // Virtual Destructor
   virtual ~TVirtualRefProxy() {};

   // Release the reference proxy (virtual destructor)
   virtual void Release() = 0;

   // Clone the reference proxy (virtual constructor)
   virtual TVirtualRefProxy* Clone() const = 0;

   // Setter of reference class (executed when the proxy is adopted)
   // Setup the reference when it is adopted by the TClass structure
   //
   // classptr [IN]    Pointer to the reference class.
   virtual void SetClass(TClass *classptr) = 0;

   // Getter of reference class. 
   // The function returns the class description of the reference class
   // ie. in the case of TRef TRef::Class
   virtual TClass * GetClass() const = 0;

   // Access to the target class.
   // In the event the value class cannot be specified from the reference
   // itself, because the object behind the reference requires a cast,
   // the return value maust be NULL.
   //
   // data   [IN]   Resolved pointer to the referenced object
   virtual TClass* GetValueClass(void* data) const = 0;

   // Update (and propagate) cached information
   virtual Bool_t Update() = 0;

   // Flag to indicate if this is a container reference
   virtual Bool_t HasCounter()  const = 0;

   // Access to container size (if container reference (ie TRefArray) etc) 
   //
   // info    [IN]   Pointer to the structure called by TTree::Draw
   //                to extract the required object information.
   // data    [IN]   Pointer to the reference object
   //
   //  return value: The prepared pointer to the reference.
   virtual Int_t  GetCounterValue(TFormLeafInfoReference* info, void *data) = 0;

   // Prepare reused reference object (e.g. ZERO data pointers)
   // Because TTree::Draw reuses objects some reference implementations
   // require setup. For example the pointer to the object the reference points to
   // needs to be ZEROed.
   //
   // data   [IN]   Pointer to the reference object
   //
   // return value: The prepared pointer to the reference.
   virtual void* GetPreparedReference(void* data) = 0;

   // Access referenced object(-data)
   //
   // info     [IN]   Pointer to the structure called by TTree::Draw
   //                 to extract the required object information.
   // data     [IN]   Pointer to the reference object
   // instance [IN]   Item number if ref collection  
   //
   // return value: Pointer to the requested information
   virtual void* GetObject(TFormLeafInfoReference* info, void* data, int instance) = 0;
};
#endif // ROOT_TVirtualRefProxy
