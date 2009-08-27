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
#include <TColor.h>
#include <TVirtualCollectionProxy.h>

ClassImp(TStructViewer);

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//
// TStructViewer is viewer which represent class, struct or other type as an 
// object in 3D space.
// At the top of scene we can see objects which is our pointer. Uder it we see 
// pointers and collection elements. Colection must inherit from TCollection 
// or be STL collecion. 
// 
// We can change number of visible level or objects on the scene by GUI or 
// methods. Size of object is proportional to memory taken by this object
// or to the number of members inside this object. 
// 
// Easy way to find some class in our viewer is to change color of type.
// We can connect for example TF2 class with red color or connect all classes 
// inherit from TF2 by adding plus to name. For example typename "TF2+" tell us 
// that all classes inherit from TF2 will be red.
// 
// Navigation in viewer is very simple like in usual GLViewer. When you put mouse over 
// some object you can see some information aobut it (e.g. name, size, actual level).
// When you double click this object, it becames top object on scene.
// Undo and redo operation are supported. 
//
//////////////////////////////////////////////////////////////////////////


//________________________________________________________________________
TStructViewer::TStructViewer(TObject* ptr)
{
   // Default constructor. An argument "ptr" is a main pointer to TObject, which should be shown in the viewer

   fGUI = NULL;
   fTopNode = NULL;

   // add default color 
   fColors.Add(new TStructNodeProperty("+", 17));

   SetPointer(ptr);
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
void TStructViewer::CountMembers(TClass* cl, TStructNode* parent)
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

      if(dm->IsaPointer()) {   
         TString trueTypeName = dm->GetTrueTypeName();

         // skip if pointer to pointer
         if(trueTypeName.CountChar('*') > 1) {
            continue;
         }

         if (!parent->GetPointer()) {
            continue;
         }

         void** pptr = (void**)((ULong_t)(parent->GetPointer()) + dm->GetOffset());
         void* pointer = *pptr;

         if (!pointer) {
            continue;
         }

         if(fPointers.GetValue((ULong_t)pointer)) {
            continue;
         } else {
            fPointers.Add((ULong_t)pointer, (ULong_t)pointer);
         }

         ULong_t size = 0;
         if (TClass* cl = TClass::GetClass(dm->GetTypeName())) {
            size = cl->Size();
         }

         if(size == 0) {
            size = dm->GetUnitSize();
         }
            
         ENodeType type;
         if(dm->GetDataType()) {   // pointer to basic type
            type = kBasic;
         } else if (dm->IsSTLContainer() == TDataMember::kVector) {
            type = kSTLCollection;
         } else {
            type = kClass;
         }

         // creating TStructNode
         TStructNode* node = new TStructNode(dm->GetName(), dm->GetTypeName(), pointer, parent, size, type);
         AddNode(node, size);
         
         CountMembers(TClass::GetClass(dm->GetTypeName()), node);

         // total size = size of parent + size of nodes daughters
         parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize() - size);
         // all members of node = all nodes of parent + nodes of daughter - 1 because node is added twice
         parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount() - 1);
      } else {
         CountMembers(TClass::GetClass(dm->GetTypeName()), parent);
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
      if (!parent->GetPointer()) {
         return;
      }

      TIter it((TCollection*)parent->GetPointer());
      TObject* item;
      // loop through all elements in collection
      while((item = it())) {
         // get size of element
         ULong_t size = 0;
         if (TClass* cl = item->IsA()){
            size = cl->Size();
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
         
         CountMembers(item->IsA(), node);
         
         parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize());
         parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount());
      }
   }
   //////////////////////////////////////////////////////////////////////////
   // STL
   //////////////////////////////////////////////////////////////////////////
   if (parent->GetNodeType() == kSTLCollection) {
      TVirtualCollectionProxy* proxy = cl->GetCollectionProxy();
      
      UInt_t size = proxy->Size();
      parent->SetMembersCount(parent->GetMembersCount() + size);

      if(!proxy->HasPointers() || proxy->GetType()) { // only objects or pointers to basic type
         parent->SetTotalSize(parent->GetTotalSize() + size * proxy->Sizeof());
         parent->SetAllMembersCount(parent->GetAllMembersCount() + size);
      } else {
         void* element;
         for (UInt_t i = 0; i < size ; i++) {
            element = proxy->At(i);

            if (!element) {
               continue;
            }

            // get size of element
            ULong_t size = 0;
            TClass* cl = proxy->GetValueClass();
            const char * name = "name";
            if (cl) {
               size = cl->Size();
               name = cl->GetName();
            }

            // if there is no dictionary
            if (size == 0) {
               size = proxy->Sizeof();
            }
            
            // create node
            TStructNode* node = new TStructNode(name, name, element, parent, size, kClass);
            // add addition information
            AddNode(node, size);
            // increase parents counter
            parent->SetMembersCount(parent->GetMembersCount() + 1);

            CountMembers(cl, node);

            parent->SetTotalSize(parent->GetTotalSize() + node->GetTotalSize());
            parent->SetAllMembersCount(parent->GetAllMembersCount() + node->GetAllMembersCount());
         }
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
      if (fGUI) {
         fGUI->SetNodePtr(fTopNode);
      } else {
         fGUI = new TStructViewerGUI(this, fTopNode, &fColors);
      }
   } else {
      
   }
}

//________________________________________________________________________
TCanvas* TStructViewer::GetCanvas()
{
   // Returns canvas used to keep TGeoVolumes

   if(fGUI) {
      return fGUI->GetCanvas();
   }
   return NULL;
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

   if (fGUI) {
      return fGUI->GetLinksVisibility();
   } else {
      return false;
   }
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
   CountMembers(fPointer->IsA(), fTopNode);
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
         if(fGUI) {
            fGUI->Update();
         }

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
   // If GUI is created, set links visibility

   if (fGUI) {
      fGUI->SetLinksVisibility(val);
   }
}

//________________________________________________________________________
void TStructViewer::SetPointer(TObject* ptr)
{
   // Set main pointer

   if (ptr) {
      fPointer = ptr;
      Prepare();

      if (fGUI) {
         fGUI->SetNodePtr(fTopNode);
      }
   }
}
