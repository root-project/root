// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TStructViewer.h"
#include "TStructNodeProperty.h"
#include "TStructViewerGUI.h"
#include "TStructNode.h"

#include <TDataMember.h>
#include <TVirtualCollectionProxy.h>

ClassImp(TStructViewer);

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//
// TStructViewer viewer represents class, struct or other type as an object in 3D space.
// At the top of the scene we can see objects which is our pointer. Under it we see 
// pointers and collection elements. Collection must inherit from TCollection 
// or be STL collecion. 
// 
// We can change the number of visible levels or objects on the scene with the  GUI or 
// methods. The size of geometry objects is proportional to the memory taken by this object
// or to the number of members inside this object. 
// 
// An easy way to find some class in the viewer is to change the color of the type.
// We can connect for example a TF2 class with red color or connect all classes 
// inheriting from TF2 by adding plus to name. For example typename "TF2+" tells us 
// that all classes inheriting from TF2 will be red.
// 
// Navigation in viewer is very simple like in usual GLViewer. When you put the mouse over 
// some object you can see some information about it (e.g. name, size, actual level).
// When you double click this object, it becames top object on scene.
// Undo and redo operation are supported. 
//
//
//////////////////////////////////////////////////////////////////////////


//________________________________________________________________________
TStructViewer::TStructViewer(TObject* ptr)
{
   // Default constructor. An argument "ptr" is a main pointer to TObject, which should be shown in the viewer

   fTopNode = NULL;

   // add default color 
   fColors.Add(new TStructNodeProperty("+", 17));

   fPointer = ptr;
   Prepare();

   // creating GUI
   fGUI = new TStructViewerGUI(this, fTopNode, &fColors);
}

//________________________________________________________________________
TStructViewer::~TStructViewer()
{
   // Destructor. Clean all object after closing the viewer

   Reset();
}

//________________________________________________________________________
void TStructViewer::AddNode(TStructNode* node, ULong_t size)
{
   // Find list with nodes on specified level and add node to this list and increment list of sizes and list of members

   TList* list = (TList*)fLevelArray[node->GetLevel()];
   // if list doesn't exist -> create one
   if(!list) {
      fLevelArray[node->GetLevel()] = list = new TList();
   }
   list->Add(node);

   // increase number of members on this level
   fLevelMembersCount(node->GetLevel())++;
   // increase size of this level
   fLevelSize(node->GetLevel()) += size;
}

//________________________________________________________________________
void TStructViewer::CountMembers(TClass* cl, TStructNode* parent, void* pointer)
{
   // Count allocated memory, increase member counters, find child nodes

   if(!cl) {
      return;
   }

   if (cl->InheritsFrom("TClass")) {
      return;
   }

   //////////////////////////////////////////////////////////////////////////
   // DATA MEMBERS
   //////////////////////////////////////////////////////////////////////////
   // Set up list of RealData so TClass doesn't create a new object itself
   cl->BuildRealData(parent->GetPointer());
   TIter it(cl->GetListOfDataMembers());
   TDataMember* dm;
   while ((dm = (TDataMember*) it() ))
   {
      // increase counters in parent node
      parent->SetAllMembersCount(parent->GetAllMembersCount() + 1);
      parent->SetMembersCount(parent->GetMembersCount() + 1);

      if (dm->Property() & kIsStatic) {
         continue;
      }

      ENodeType type;
      if(dm->GetDataType()) {   // pointer to basic type
         type = kBasic;
      } else if (dm->IsSTLContainer() == TDataMember::kVector) {
         type = kSTLCollection;

//             it works only for pointer in std object (not pointer)
//             TClass* stlClass = TClass::GetClass(dm->GetTypeName());
//             if (!stlClass) {
//                continue;
//             }
//             
//             TVirtualCollectionProxy* proxy = stlClass->GetCollectionProxy();
//             if (!proxy) {
//                continue;
//             }
//             void* ptr = (void*)((ULong_t)pointer + dm->GetOffset());
//             TVirtualCollectionProxy::TPushPop helper(proxy, ptr);
// 
//             UInt_t count = proxy->Size();
//             parent->SetMembersCount(parent->GetMembersCount() + count);
//    
//             if(!proxy->HasPointers() || proxy->GetType()) { // only objects or pointers to basic type
//                parent->SetTotalSize(parent->GetTotalSize() + count * proxy->Sizeof());
//                parent->SetAllMembersCount(parent->GetAllMembersCount() + count);
//             } else {
//                void* element;
//                for (UInt_t i = 0; i < count ; i++) {
//                   element = *(void**)proxy->At(i);
//    
//                   if (!element) {
//                      continue;
//                   }
//    
//                   // get size of element
//                   ULong_t size = 0;
//                   TClass* clProxy = proxy->GetValueClass();
//                   const char * name = "name";
//                   if (clProxy) {
//                      if (clProxy->InheritsFrom("TNamed")) {
//                         name = ((TNamed*) element)->GetName();
//                      } else {
//                         name = clProxy->GetName();
//                      }
//                      
//                      size = clProxy->Size();
//                      
//                   }
//    
//                   // if there is no dictionary
//                   if (size == 0) {
//                      size = proxy->Sizeof();
//                   }
//    
//                   // create node
//                   TStructNode* node = new TStructNode(name, name, element, parent, size, kClass);
//                   // add addition information
//                   AddNode(node, size);
//                   // increase parents counter
//                   parent->SetMembersCount(parent->GetMembersCount() + 1);
//    
//                   //CountMembers(clProxy, node);
//    
//                   parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize());
//                   parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount());
//                }
//             }
      } else {
         type = kClass;
      }

      if(dm->IsaPointer()) {
         TString trueTypeName = dm->GetTrueTypeName();

         // skip if pointer to pointer
         if(trueTypeName.CountChar('*') > 1) {
            continue;
         }

         if (!pointer) {
            continue;
         }

         void** pptr = (void**)((ULong_t)pointer + dm->GetOffset());
         void* ptr = *pptr;

         if (!ptr) {
            continue;
         }

         if(fPointers.GetValue((ULong_t)ptr)) {
            continue;
         } else {
            fPointers.Add((ULong_t)ptr, (ULong_t)ptr);
         }

         ULong_t size = 0;
         if (TClass* cl2 = TClass::GetClass(dm->GetTypeName())) {
            size = cl2->Size();
         }

         if(size == 0) {
            size = dm->GetUnitSize();
         }
            
         // creating TStructNode
         TStructNode* node = new TStructNode(dm->GetName(), dm->GetTypeName(), ptr, parent, size, type);
         AddNode(node, size);
         
         CountMembers(TClass::GetClass(dm->GetTypeName()), node, ptr);

         // total size = size of parent + size of nodes daughters
         parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize() - size);
         // all members of node = all nodes of parent + nodes of daughter - 1 because node is added twice
         parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount() - 1);
      } else {
         void* ptr = (void*)((ULong_t)pointer + dm->GetOffset());

         if (!ptr) {
            continue;
         }
         CountMembers(TClass::GetClass(dm->GetTypeName()), parent, ptr);
      }
   }

   //////////////////////////////////////////////////////////////////////////
   // COLLECTION
   //////////////////////////////////////////////////////////////////////////
   // if our parent node is collection
   if(cl->InheritsFrom("TCollection")) {
      // we change type of node to collection
      parent->SetNodeType(kCollection);
   
      // return if invalid pointer to collection
      if (!pointer) {
         return;
      }

      TIter it2((TCollection*)pointer);
      TObject* item;
      // loop through all elements in collection
      while((item = it2())) {
         // get size of element
         ULong_t size = 0;
         if (TClass* cl3 = item->IsA()){
            size = cl3->Size();
         }
         
         // if there is no dictionary
         if (size == 0) {
            size = sizeof(item);
         }

         // create node
         TStructNode* node = new TStructNode(item->GetName(), item->ClassName(), item, parent, size, kClass);
         // add addition information
         AddNode(node, size);
         // increase parents counter
         parent->SetMembersCount(parent->GetMembersCount() + 1);
         
         CountMembers(item->IsA(), node, item);
         
         parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize());
         parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount());
      }
   }
}

//________________________________________________________________________
void TStructViewer::Draw(Option_t *option)
{
   // Draw object if there is valid pointer
   TString opt(option);
   if(opt == "count") {
      
   } else if (opt == "size") {

   }


   if (fTopNode) {
      fGUI->SetNodePtr(fTopNode);
   } else {
      
   }
}

//________________________________________________________________________
TCanvas* TStructViewer::GetCanvas()
{
   // Returns canvas used to keep TGeoVolumes

   return fGUI->GetCanvas();
}

//________________________________________________________________________
TGMainFrame* TStructViewer::GetFrame()
{
   // Returns pointer to main window
  
   return fGUI;
}
//________________________________________________________________________
TObject* TStructViewer::GetPointer() const
{
   // Return main pointer

   return fPointer;
}

//________________________________________________________________________
TExMap TStructViewer::GetLevelMembersCount() const
{
   // Returns TExMap with pairs <level number, number of objects>

   return fLevelMembersCount;
}

//________________________________________________________________________
TExMap TStructViewer::GetLevelSize() const
{
   // Returns TExMap with pairs <level number, size of level in bytes>

   return fLevelSize;
}

//________________________________________________________________________
Bool_t TStructViewer::GetLinksVisibility() const
{
   // Get visibility of links between objects

   return fGUI->GetLinksVisibility();
}

//________________________________________________________________________
void TStructViewer::Prepare()
{
   // Create top node and find all member nodes
   if (fTopNode) {
      Reset();
   }

   ULong_t size = 0;
   if (TClass * cl = fPointer->IsA()) {
      size = cl->Size();
   }
   
   if(size == 0) {
      size = sizeof(fPointer);
   }

   fTopNode = new TStructNode(fPointer->GetName(), fPointer->ClassName(), fPointer, NULL, size, kClass);
   AddNode(fTopNode, size);
   CountMembers(fPointer->IsA(), fTopNode, fPointer);
}

//________________________________________________________________________
void TStructViewer::Reset()
{
   // Deleting nodes, maps and array

   TList* lst;
   TIter it(&fLevelArray);
   while ((lst = (TList*) it() )) {
      lst->SetOwner();
      lst->Clear();
   }

   // deleting maps and array
   fLevelMembersCount.Clear();
   fLevelSize.Clear();
   fPointers.Clear();
   fLevelArray.Clear();
   fColors.SetOwner();
   fColors.Clear();
   
   fTopNode = NULL;
}

//________________________________________________________________________
void TStructViewer::SetColor(TString name, Int_t color)
{
   // Sets color for the class "name" to color "color"

   TIter it(&fColors);
   TStructNodeProperty* prop;
   while ((prop = (TStructNodeProperty*) it() )) {
      if (name == prop->GetName()) {
         prop->SetColor(TColor::GetColor(color));
         fGUI->Update();
         
         return;
      }
   }

   // add color
   prop = new TStructNodeProperty(name.Data(), color);
   fColors.Add(prop);
   fColors.Sort();
}

//________________________________________________________________________
void TStructViewer::SetLinksVisibility(Bool_t val)
{
   // ISets links visibility

   fGUI->SetLinksVisibility(val);
}

//________________________________________________________________________
void TStructViewer::SetPointer(TObject* ptr)
{
   // Set main pointer

   if (ptr) {
      fPointer = ptr;
      Prepare();
      fGUI->SetNodePtr(fTopNode);
   }
}

//________________________________________________________________________
TColor TStructViewer::GetColor(const char* typeName)
{
   // Returns color associated with type "typeName"

   TIter it(&fColors);
   TStructNodeProperty* prop;
   while((prop = (TStructNodeProperty*) it())) {
      if (!strcmp(prop->GetName(), typeName)) {
         return prop->GetColor();
      }
   }

   return TColor();
}
