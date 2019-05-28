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

////////////////////////////////////////////////////////////////////////////////
/// Constructs node with name "name" of class "typeName" and given parent "parent" which represents pointer "pointer".
/// Size of node is set to "size" and type is set to "type"

TStructNode::TStructNode(TString name, TString typeName, void* pointer, TStructNode* parent, ULong_t size, ENodeType type)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Destructs list of nodes

TStructNode::~TStructNode()
{
   delete fMembers;
}

////////////////////////////////////////////////////////////////////////////////
/// Overrided method. Compare to objects of TStructNode class.

Int_t TStructNode::Compare(const TObject* obj) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns number of all members in node

ULong_t TStructNode::GetAllMembersCount() const
{
   return fAllMembersCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns center of outlining box on x-axis

Float_t TStructNode::GetCenter() const
{
   return (fX + fWidth /2);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns height of outlining box

Float_t TStructNode::GetHeight() const
{
   return fHeight;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns actual level of node

UInt_t TStructNode::GetLevel() const
{
   return fLevel;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of object

const char* TStructNode::GetName() const
{
   return fName.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns type of node

ENodeType TStructNode::GetNodeType() const
{
   return fNodeType;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns maximum number of leves displayed when the node is top node on scene

UInt_t TStructNode::GetMaxLevel() const
{
   return fMaxLevel;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns maximum number of objects displayed when the node is top node on scene

UInt_t TStructNode::GetMaxObjects() const
{
   return fMaxObjects;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns list with pointers to daughter nodes.

TList* TStructNode::GetMembers() const
{
   return fMembers;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns numbers of members of node

ULong_t TStructNode::GetMembersCount() const
{
   return fMembersCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns center of outlining box on y-axis

Float_t TStructNode::GetMiddle() const
{
   return (fY + fHeight/2);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns pointer to parent node

TStructNode* TStructNode::GetParent() const
{
   return fParent;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns main pointer

void* TStructNode::GetPointer() const
{
   return fPointer;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns relative numbers of members. If node is collapsed, then method returns number of all members,
/// it's node and its daughters, otherwise it returns number of members of node

ULong_t TStructNode::GetRelativeMembersCount() const
{
   if (fCollapsed) {
      return fAllMembersCount;
   }
   return fMembersCount;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns relative size of node. If node is collapsed, then function returns size of node and dauthers,
/// otherwise returns size of node only.

ULong_t TStructNode::GetRelativeSize() const
{
   if (fCollapsed) {
      return fTotalSize;
   }
   return fSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns size or number of members. If ScaleBy is set to kMembers and node is collapsed, then it
/// returns all number of members. If node isn't collapsed it returns number of members.
/// If Scaleby is set to kSize and node is collapsed, then it returns total size of node and daughters,
/// else it returns size of node, otherwise it returns 0.

ULong_t TStructNode::GetRelativeVolume() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Returns ratio - relative volume to area taken by utlining box.

Float_t TStructNode::GetRelativeVolumeRatio()
{
   return ((Float_t)(GetRelativeVolume())/(fWidth*fHeight));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns size of node

ULong_t TStructNode::GetSize() const
{
   return fSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns total size of allocated memory in bytes

ULong_t TStructNode::GetTotalSize() const
{
   return fTotalSize;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of class

TString TStructNode::GetTypeName() const
{
   return fTypeName;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns size or number of members. If ScaleBy is set to kMembers it returns all number of members.
/// If Scaleby is set to kSize then it returns total size of node and daughters, otherwise it returns 0.

ULong_t TStructNode::GetVolume() const
{
   if (fgScalBy == kMembers) {
      return GetAllMembersCount();
   } else if (fgScalBy == kSize) {
      return GetTotalSize();
   } else {
      return 0;
   }

}

////////////////////////////////////////////////////////////////////////////////
/// Returns ratio - volme of node to area taken by outlining box

Float_t TStructNode::GetVolumeRatio()
{
   return ((Float_t)(GetVolume())/(fWidth*fHeight));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns width of outlining box

Float_t TStructNode::GetWidth() const
{
   return fWidth;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns X coordinate

Float_t TStructNode::GetX() const
{
   return fX;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns Y coordinate

Float_t TStructNode::GetY() const
{
   return fY;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if node is colllapsed

Bool_t TStructNode::IsCollapsed() const
{
   return fCollapsed;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true, because we have overrided method Compare

Bool_t TStructNode::IsSortable() const
{
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns true if node is visible

bool TStructNode::IsVisible() const
{
   return fVisible;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets numbers of all members to "number"

void TStructNode::SetAllMembersCount(ULong_t number)
{
   fAllMembersCount = number;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets collapsing of node to "collapse"

void TStructNode::SetCollapsed(Bool_t collapse)
{
   fCollapsed = collapse;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets width of outlining box to "w"

void TStructNode::SetHeight(Float_t val)
{
   fHeight = val;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets maximum number of leves displayed when the node is top node on scene

void TStructNode::SetMaxLevel(UInt_t level)
{
   fMaxLevel = level;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets maximum number of objects displayed when the node is top node on scene

void TStructNode::SetMaxObjects(UInt_t max)
{
   fMaxObjects = max;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets list of dauther nodes to "list"

void TStructNode::SetMembers(TList* list)
{
   fMembers = list;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets number of members to "number"

void TStructNode::SetMembersCount(ULong_t number)
{
   fMembersCount = number;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets type of node to "type"

void TStructNode::SetNodeType(ENodeType type)
{
   fNodeType = type;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets main pointer to "pointer"

void TStructNode::SetPointer(void* pointer)
{
   fPointer = pointer;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets scaling by to "type"

void TStructNode::SetScaleBy(EScalingType type)
{
   fgScalBy = type;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets size of node to "size"

void TStructNode::SetSize(ULong_t size)
{
   fSize = size;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets total size  of allocated memory in bytes to value "size"

void TStructNode::SetTotalSize(ULong_t size)
{
   fTotalSize = size;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets visibility of node to "visible"

void TStructNode::SetVisible(bool visible)
{
   fVisible = visible;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets width of outlining box to "w"

void TStructNode::SetWidth(Float_t w)
{
   fWidth = w;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets X coordinate to "x"

void TStructNode::SetX(Float_t x)
{
   fX = x;
}

////////////////////////////////////////////////////////////////////////////////
/// Sets Y coordinate to "y"

void TStructNode::SetY(Float_t y)
{
   fY = y;
}
