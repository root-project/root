// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TStructNode.h"
#include <TList.h>
#include <TGeoManager.h>

ClassImp(TStructNode);

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
//
// TStructNode - class which represent a node. Node has all information
// about some pointer. It keeps information such as name of object, type,
// size of pointers class, size of node and daughter nodes, number of child
// nodes. It is also used to store information needed to draw TGeoVolume.
// It is for example x, y and z coordinates.
// Condition fVisible tells us that node is visible and should be drawn.
// fCollapsed tells us that we can see daughter nodes.
//
//////////////////////////////////////////////////////////////////////////

EScalingType TStructNode::fgScalBy = kMembers;

//________________________________________________________________________
TStructNode::TStructNode(TString name, TString typeName, void* pointer, TStructNode* parent, ULong_t size, ENodeType type)
{
   // Constructs node with name "name" of class "typeName" and given parent "parent" which represents pointer "pointer".
   // Size of node is set to "size" and type is set to "type"

   fName = name;
   fTypeName = typeName;
   fTotalSize = fSize = size;
   fMembers = new TList();
   fMembersCount = fAllMembersCount = 1;
   fLevel = 1;
   fX = fY = fWidth = fHeight = 0;
   fParent = parent;
   if (parent) {
      fLevel = parent->GetLevel()+1;
      parent->fMembers->Add(this);
   }

   fNodeType = type;
   fPointer = pointer;
   fCollapsed = false;
   fVisible = false;
   fMaxLevel = 3;
   fMaxObjects = 100;
}

//________________________________________________________________________
TStructNode::~TStructNode()
{
   // Destructs list of nodes

   delete fMembers;
}

//________________________________________________________________________
Int_t TStructNode::Compare(const TObject* obj) const
{
   // Overrided method. Compare to objects of TStructNode class.

   TStructNode* node = (TStructNode*)obj;

   if (GetVolume() < node->GetVolume()) {
      return -1;
   }
   if(GetVolume() > node->GetVolume()) {
      return 1;
   }

   if (this > node) {
      return 1;
   }
   if (this < node) {
      return -1;
   }

   return 0;
}

//________________________________________________________________________
ULong_t TStructNode::GetAllMembersCount() const
{
   // Returns number of all members in node

   return fAllMembersCount;
}

//________________________________________________________________________
Float_t TStructNode::GetCenter() const
{
   // Returns center of outlining box on x-axis
   return (fX + fWidth /2);
}

//________________________________________________________________________
Float_t TStructNode::GetHeight() const
{
   // Returns height of outlining box

   return fHeight;
}

//________________________________________________________________________
UInt_t TStructNode::GetLevel() const
{
   // Returns actual level of node

   return fLevel;
}

//________________________________________________________________________
const char* TStructNode::GetName() const
{
   // Returns name of object
   return fName.Data();
}

//________________________________________________________________________
ENodeType TStructNode::GetNodeType() const
{
   // Returns type of node

   return fNodeType;
}

//________________________________________________________________________
UInt_t TStructNode::GetMaxLevel() const
{
   // Returns maximum number of leves displayed when the node is top node on scene

   return fMaxLevel;
}

//________________________________________________________________________
UInt_t TStructNode::GetMaxObjects() const
{
   // Returns maximum number of objects displayed when the node is top node on scene
   return fMaxObjects;
}

//________________________________________________________________________
TList* TStructNode::GetMembers() const
{
   // Returns list with pointers to daughter nodes.

   return fMembers;
}

//________________________________________________________________________
ULong_t TStructNode::GetMembersCount() const
{
   // Returns numbers of members of node

   return fMembersCount;
}

//________________________________________________________________________
Float_t TStructNode::GetMiddle() const
{
   // Returns center of outlining box on y-axis

   return (fY + fHeight/2);
}

//________________________________________________________________________
TStructNode* TStructNode::GetParent() const
{
   // Returns pointer to parent node

   return fParent;
}

//________________________________________________________________________
void* TStructNode::GetPointer() const
{
   // Returns main pointer
   return fPointer;
}

//________________________________________________________________________
ULong_t TStructNode::GetRelativeMembersCount() const
{
   // Returns relative numbers of members. If node is collapsed, then method returns number of all members,
   // it's node and its daughters, otherwise it returns number of members of node

   if (fCollapsed) {
      return fAllMembersCount;
   }
   return fMembersCount;
}

//________________________________________________________________________
ULong_t TStructNode::GetRelativeSize() const
{
   // Returns relative size of node. If node is collapsed, then function returns size of node and dauthers,
   // otherwise returns size of node only.

   if (fCollapsed) {
      return fTotalSize;
   }
   return fSize;
}

//________________________________________________________________________
ULong_t TStructNode::GetRelativeVolume() const
{
   // Returns size or number of members. If ScaleBy is set to kMembers and node is collapsed, then it
   // returns all number of members. If node isn't collapsed it returns number of members.
   // If Scaleby is set to kSize and node is collapsed, then it returns total size of node and daughters,
   // else it returns size of node, otherwise it returns 0.

   if (fgScalBy == kMembers) {
      if (fCollapsed) {
         return GetAllMembersCount();
      } else {
         return GetMembersCount();
      }
   } else if (fgScalBy == kSize) {
      if (fCollapsed) {
         return GetTotalSize();
      } else {
         return GetSize();
      }
   } else {
      return 0;
   }
}

//________________________________________________________________________
Float_t TStructNode::GetRelativeVolumeRatio()
{
   // Returns ratio - relative volume to area taken by utlining box.

   return ((Float_t)(GetRelativeVolume())/(fWidth*fHeight));
}

//________________________________________________________________________
ULong_t TStructNode::GetSize() const
{
   // Returns size of node

   return fSize;
}

//________________________________________________________________________
ULong_t TStructNode::GetTotalSize() const
{
   // Returns total size of allocated memory in bytes

   return fTotalSize;
}

//________________________________________________________________________
TString TStructNode::GetTypeName() const
{
   // Returns name of class

   return fTypeName;
}

//________________________________________________________________________
ULong_t TStructNode::GetVolume() const
{
   // Returns size or number of members. If ScaleBy is set to kMembers it returns all number of members.
   // If Scaleby is set to kSize then it returns total size of node and daughters, otherwise it returns 0.

   if (fgScalBy == kMembers) {
      return GetAllMembersCount();
   } else if (fgScalBy == kSize) {
      return GetTotalSize();
   } else {
      return 0;
   }

}

//________________________________________________________________________
Float_t TStructNode::GetVolumeRatio()
{
   // Returns ratio - volme of node to area taken by outlining box

   return ((Float_t)(GetVolume())/(fWidth*fHeight));
}

//________________________________________________________________________
Float_t TStructNode::GetWidth() const
{
   // Returns width of outlining box

   return fWidth;
}

//________________________________________________________________________
Float_t TStructNode::GetX() const
{
   // Returns X coordinate

   return fX;
}

//________________________________________________________________________
Float_t TStructNode::GetY() const
{
   // Returns Y coordinate

   return fY;
}

//________________________________________________________________________
Bool_t TStructNode::IsCollapsed() const
{
   // Returns true if node is colllapsed

   return fCollapsed;
}

//________________________________________________________________________
Bool_t TStructNode::IsSortable() const
{
   // Returns true, because we have overrided method Compare

   return kTRUE;
}

//________________________________________________________________________
bool TStructNode::IsVisible() const
{
   // Returns true if node is visible

   return fVisible;
}

//________________________________________________________________________
void TStructNode::SetAllMembersCount(ULong_t number)
{
   // Sets numbers of all members to "number"

   fAllMembersCount = number;
}

//________________________________________________________________________
void TStructNode::SetCollapsed(Bool_t collapse)
{
   // Sets collapsing of node to "collapse"

   fCollapsed = collapse;
}

//________________________________________________________________________
void TStructNode::SetHeight(Float_t val)
{
   // Sets width of outlining box to "w"

   fHeight = val;
}

//________________________________________________________________________
void TStructNode::SetMaxLevel(UInt_t level)
{
   // Sets maximum number of leves displayed when the node is top node on scene

   fMaxLevel = level;
}

//________________________________________________________________________
void TStructNode::SetMaxObjects(UInt_t max)
{
   // Sets maximum number of objects displayed when the node is top node on scene

   fMaxObjects = max;
}

//________________________________________________________________________
void TStructNode::SetMembers(TList* list)
{
   // Sets list of dauther nodes to "list"

   fMembers = list;
}

//________________________________________________________________________
void TStructNode::SetMembersCount(ULong_t number)
{
   // Sets number of members to "number"
   fMembersCount = number;
}

//________________________________________________________________________
void TStructNode::SetNodeType(ENodeType type)
{
   // Sets type of node to "type"

   fNodeType = type;
}

//________________________________________________________________________
void TStructNode::SetPointer(void* pointer)
{
   // Sets main pointer to "pointer"

   fPointer = pointer;
}

//________________________________________________________________________
void TStructNode::SetScaleBy(EScalingType type)
{
   // Sets scaling by to "type"

   fgScalBy = type;
}

//________________________________________________________________________
void TStructNode::SetSize(ULong_t size)
{
   // Sets size of node to "size"

   fSize = size;
}

//________________________________________________________________________
void TStructNode::SetTotalSize(ULong_t size)
{
   // Sets total size  of allocated memory in bytes to value "size"

   fTotalSize = size;
}

//________________________________________________________________________
void TStructNode::SetVisible(bool visible)
{
   // Sets visibility of node to "visible"

   fVisible = visible;
}

//________________________________________________________________________
void TStructNode::SetWidth(Float_t w)
{
   // Sets width of outlining box to "w"

   fWidth = w;
}

//________________________________________________________________________
void TStructNode::SetX(Float_t x)
{
   // Sets X coordinate to "x"

   fX = x;
}

//________________________________________________________________________
void TStructNode::SetY(Float_t y)
{
   // Sets Y coordinate to "y"

   fY = y;
}
