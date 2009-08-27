// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TStructNode
#define ROOT_TStructNode

#include <TObject.h>
#include <TString.h>
#include <TGeoVolume.h>

class TGeoMedium;

enum ENodeType {
   kUnknown = 1,  // Unknown type
   kClass,        // Class or structure
   kCollection,   // TCollection (e.g. TList, TExMap, TMap etc.)
   kBasic,        // Basic type (e.g. char, float, double)
   kSTLCollection // STL collection
};
enum ESortingType {
   kSize,         // Objects are proportional to allocated memory
   kMembers       // Objects are proportional to number of members
};

//________________________________________________________________________
//
// Logical node with informatioon about class

class TStructNode : public TObject {

private:
   static ESortingType  fSortBy;
   static TGeoMedium   *fMedium;          // Material and medium
   TString              fName;            // Name of node
   TString              fTypeName;        // Name of type
   ULong_t              fSize;            // Memory allocated by class without pointers and list elements
   ULong_t              fTotalSize;       // Total allocated memory
   TStructNode         *fParent;          // Pointer to parent node, NULL if not exist
   UInt_t               fLevel;           // Level number
   ULong_t              fMembersCount;    // Number of members in class
   ULong_t              fAllMembersCount; // Number of all members (class and its daughters)
   void*                fPointer;         // Pointer to data (address of variable)
   Bool_t               fCollapsed;       // Condition - true if node is collapsed (we don't see dauthers)
   Bool_t               fVisible;         // Condition - true if node is visible
   TList*               fMembers;         // List of daughter nodes
   Float_t              fX;               // X coordinate in 3D space
   Float_t              fY;               // Y coordinate in 3D space
   Float_t              fZ;               // Z coordinate in 3D space
   Float_t              fWidth;           // Width of outlining box
   Float_t              fHeight;          // Height of outlining box
   ENodeType            kNodeType;        // Type of node
   UInt_t               fMaxLevel;        // Number of levels displayed when the node is top node on scene
   UInt_t               fMaxObjects;      // Number of objects displayed when the node is top node on scene

public:
   TStructNode(TString name, TString typeName, void* pointer, TStructNode* parent, ULong_t size, ENodeType type);
   ~TStructNode();
   
   virtual Int_t  Compare(const TObject* obj) const;
   ULong_t        GetAllMembersCount() const;
   Float_t        GetCenter() const;
   Float_t        GetHeight() const;
   UInt_t         GetLevel() const;
   UInt_t         GetMaxLevel() const;
   UInt_t         GetMaxObjects() const;
   static TGeoMedium* GetMedium();
   TList*         GetMembers() const;
   ULong_t        GetMembersCount() const;
   Float_t        GetMiddle() const;
   const char*    GetName() const;
   ENodeType      GetNodeType() const;
   TStructNode   *GetParent() const;
   void*          GetPointer() const;
   ULong_t        GetRelativeMembersCount() const;
   ULong_t        GetRelativeSize() const;
   ULong_t        GetRelativeVolume() const;
   Float_t        GetRelativeVolumeRatio();
   ULong_t        GetSize() const;
   ULong_t        GetTotalSize() const;
   TString        GetTypeName() const;
   ULong_t        GetVolume() const;
   Float_t        GetVolumeRatio();
   Float_t        GetWidth() const;
   Float_t        GetX() const;
   Float_t        GetY() const;
   Float_t        GetZ() const;
   Bool_t         IsCollapsed() const;
   virtual Bool_t IsSortable() const;
   bool           IsVisible() const;
   void           SetAllMembersCount(ULong_t count);
   void           SetCollapsed(Bool_t collapsed);
   void           SetHeight(Float_t h);
   void           SetMaxLevel(UInt_t level);
   void           SetMaxObjects(UInt_t max);
   static void    SetMedium(TGeoMedium* val);
   void           SetMembers(TList* list);
   void           SetMembersCount(ULong_t count);
   void           SetNodeType(ENodeType type);
   void           SetPointer(void* pointer);
   void           SetSize(ULong_t size);
   static void    SetSortBy(ESortingType type);
   void           SetTotalSize(ULong_t size);
   void           SetVisible(bool visible);
   void           SetWidth(Float_t w);
   void           SetX(Float_t x);
   void           SetY(Float_t y);
   void           SetZ(Float_t z);

   ClassDef(TStructNode,0); // Node with information about class
};

#endif
