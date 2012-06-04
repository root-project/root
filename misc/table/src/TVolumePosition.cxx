// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "Riostream.h"

#include "TCernLib.h"
#include "TVolumePosition.h"
#include "TVolume.h"

#include "TROOT.h"
#include "TClass.h"
#include "TVirtualPad.h"
#include "TGeometry.h"
#include "TRotMatrix.h"
#include "TBrowser.h"
#include "X3DBuffer.h"

#include "TTablePadView3D.h"

//R__EXTERN  Size3D gSize3D;

ClassImp(TVolumePosition)

//______________________________________________________________________________
//*-*-*-*-*-*-*-*-* S T N O D E P O S I T I O N   description *-*-*-*-*-*-*-*-*-
//*-*               ===========================
//*-*
//*-*    A TVolumePosition object is used to build the geometry hierarchy (see TGeometry).
//*-*    A node may contain other nodes.
//*-*
//*-*    A geometry node has attributes:
//*-*      - name and title
//*-*      - pointer to the referenced shape (see TShape).
//*-*      - x,y,z offset with respect to the mother node.
//*-*      - pointer to the rotation matrix (see TRotMatrix).
//*-*
//*-*    A node can be drawn.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


//______________________________________________________________________________
TVolumePosition::TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const char *matrixname)
: fMatrix(0),fNode(node),fId(0)
{
//*-*-*-*-*-*-*-*-*-*-*Node normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================
//*-*
//*-*    name    is the name of the node
//*-*    title   is title
//*-*    x,y,z   are the offsets of the volume with respect to his mother
//*-*    matrixname  is the name of the rotation matrix
//*-*
//*-*    This new node is added into the list of sons of the current node
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   SetMatrixOwner(kFALSE);
   fX[0] = x; fX[1] =y; fX[2] = z;
   if (!node) return;
   static Int_t counter = 0;
   counter++;
   if(!(counter%1000))std::cout<<"TVolumePosition count="<<counter<<" name="<<node->GetName()<<std::endl;

   if (!gGeometry) new TGeometry;
   if (matrixname && strlen(matrixname)) fMatrix = gGeometry->GetRotMatrix(matrixname);
   if (!fMatrix) fMatrix = TVolume::GetIdentity();
}


//______________________________________________________________________________
TVolumePosition::TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, TRotMatrix *matrix)
               : fMatrix(matrix),fNode(node),fId(0)
{
//*-*-*-*-*-*-*-*-*-*-*Node normal constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                  ================================
//*-*
//*-*    name    is the name of the node
//*-*    title   is title
//*-*    x,y,z   are the offsets of the volume with respect to his mother
//*-*    matrix  is the pointer to the rotation matrix
//*-*
//*-*    This new node is added into the list of sons of the current node
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   SetMatrixOwner(kFALSE);
   if (!gGeometry) new TGeometry;
   fX[0] = x; fX[1] = y; fX[2] = z;
   if (!fMatrix) fMatrix = TVolume::GetIdentity();
}
//______________________________________________________________________________
TVolumePosition::TVolumePosition(const TVolumePosition* oldPosition, const TVolumePosition* curPosition){
   // Pick the "old" position by pieces
   fMatrix = 0;
   SetMatrixOwner(kFALSE);
   TVolume *curNode = 0;
   UInt_t curPositionId    = 0;
   TRotMatrix *curMatrix = 0;
   if (curPosition) {
      curNode       = curPosition->GetNode();
      curPositionId = curPosition->GetId();
      curMatrix     = (TRotMatrix *) curPosition->GetMatrix();
   }
   TRotMatrix *oldMatrix = 0;
   fX[0] = 0; fX[1] = 0; fX[2] = 0;
   Double_t oldTranslation[] = { 0, 0, 0 };
   if (oldPosition) {
      oldMatrix         = (TRotMatrix *) oldPosition->GetMatrix();
      oldTranslation[0] = oldPosition->GetX();
      oldTranslation[1] = oldPosition->GetY();
      oldTranslation[2] = oldPosition->GetZ();
   }

   // Pick the "current" position by pieces

   // Create a new position
   Double_t newMatrix[9];

   if(oldMatrix && curMatrix && curPosition)  {
      TGeometry::UpdateTempMatrix(oldTranslation,oldMatrix->GetMatrix(),
                                 curPosition->GetX(),curPosition->GetY(),curPosition->GetZ(),
                                 curMatrix->GetMatrix(),
                                 fX,newMatrix);
      Int_t num = gGeometry->GetListOfMatrices()->GetSize();
      Char_t anum[100];
      snprintf(anum,100,"%d",num+1);
      fMatrix = new TRotMatrix(anum,"NodeView",newMatrix);
      SetMatrixOwner(kTRUE);
   } else {
      if (curPosition) {
         fX[0] = oldTranslation[0] + curPosition->GetX();
         fX[1] = oldTranslation[1] + curPosition->GetY();
         fX[2] = oldTranslation[2] + curPosition->GetZ();
         fMatrix = curMatrix;
      }
   }
   fId = curPositionId;
   fNode = curNode;
}
//______________________________________________________________________________
//______________________________________________________________________________
TVolumePosition::TVolumePosition(const TVolumePosition&pos): TObject()
      , fMatrix(((TVolumePosition &)pos).GetMatrix()),fNode(pos.GetNode()),fId(pos.GetId())
{
   //to be documented
   for (int i=0;i<3;i++) fX[i] = pos.GetX(i);
   // Transferring the ownership.
   // The last created object owns the matrix if any.
   // The source object gives up its ownership in favour of the destination object

   SetMatrixOwner(pos.IsMatrixOwner());
   // !!! We have to break the "const'ness" at this point to take the ownerships
   ((TVolumePosition &)pos).SetMatrixOwner(kFALSE);
}

//______________________________________________________________________________
TVolumePosition::~TVolumePosition()
{
   //to be documented
   DeleteOwnMatrix();
}
//______________________________________________________________________________
void TVolumePosition::Browse(TBrowser *b)
{
   //to be documented
   if (GetNode()) {
      TShape *shape = GetNode()->GetShape();
      b->Add(GetNode(),shape?shape->GetName():GetNode()->GetName());
   } else {
      Draw();
      gPad->Update();
   }
}

//______________________________________________________________________________
Int_t TVolumePosition::DistancetoPrimitive(Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a Node*-*-*-*-*-*
//*-*                  ===========================================
//*-*  Compute the closest distance of approach from point px,py to this node.
//*-*  The distance is computed in pixels units.
//*-*
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   return 99999;
}

//______________________________________________________________________________
void TVolumePosition::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*-*Draw Referenced node with current parameters*-*-*-*
//*-*                   =============================================
   TVolume *node = GetNode();
   if (node) node->Draw(option);
}


//______________________________________________________________________________
void TVolumePosition::ExecuteEvent(Int_t, Int_t, Int_t)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//*-*  This member function must be implemented to realize the action
//*-*  corresponding to the mouse click on the object in the window
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//   if (gPad->GetView())
//             gPad->GetView()->ExecuteRotateView(event, px, py);

//   if (!gPad->GetListOfPrimitives()->FindObject(this)) gPad->SetCursor(kCross);
   gPad->SetCursor(kHand);
}

//______________________________________________________________________________
const Char_t *TVolumePosition::GetName() const
{
   //return VolumePosition name
   return GetNode()?GetNode()->GetName():IsA()->GetName();
}

//______________________________________________________________________________
char *TVolumePosition::GetObjectInfo(Int_t, Int_t) const
{
   //to be documented
   if (!gPad) return 0;
   if (!GetNode()) return 0;
   static char info[64];
   snprintf(info,64,"%s/%s, shape=%s/%s",GetNode()->GetName(),GetNode()->GetTitle(),GetNode()->GetShape()->GetName(),GetNode()->GetShape()->ClassName());
   return info;
}

//______________________________________________________________________________
Double_t *TVolumePosition::Errmx2Master(const Double_t *localError, Double_t *masterError) const
{
   //to be documented
   Double_t error[6];
   TCL::vzero(&error[1],4);
   error[0] = localError[0]; error[2] = localError[1]; error[5] = localError[2];
   return Cormx2Master(error, masterError);
}

//______________________________________________________________________________
Float_t *TVolumePosition::Errmx2Master(const Float_t *localError, Float_t *masterError) const
{
   //to be documented
   Float_t error[6];
   TCL::vzero(&error[1],4);
   error[0] = localError[0]; error[2] = localError[1]; error[5] = localError[2];
   return Cormx2Master(error, masterError);
}

//______________________________________________________________________________
Double_t *TVolumePosition::Cormx2Master(const Double_t *localCorr, Double_t *masterCorr)const
{
   //to be documented
   Double_t *res = 0;
   const TRotMatrix *rm = GetMatrix();
   double *m = 0;
   if (rm && ( m = ((TRotMatrix *)rm)->GetMatrix()) )
      res = TCL::trasat(m,(Double_t *)localCorr,masterCorr,3,3);
   else
      res = TCL::ucopy(localCorr,masterCorr,6);
   return res;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Cormx2Master(const Float_t *localCorr, Float_t *masterCorr) const
{
   //to be documented
   Float_t *res = 0;
   const TRotMatrix *rm = GetMatrix();
   Double_t *m = 0;
   if (rm && (m = ((TRotMatrix *)rm)->GetMatrix()) ) {
      double corLocal[6], corGlobal[6];
      TCL::ucopy(localCorr,corLocal,6);
      TCL::trasat(m,corLocal,corGlobal,3,3);
      res =  TCL::ucopy(corGlobal,masterCorr,6);
   } else
      res =  TCL::ucopy(localCorr,masterCorr,6);
   return res;
}
//______________________________________________________________________________
Double_t *TVolumePosition::Errmx2Local(const Double_t *masterError, Double_t *localError) const
{
   //to be documented
   Double_t error[6];
   TCL::vzero(&error[1],4);
   error[0] = masterError[0]; error[2] = masterError[1]; error[5] = masterError[2];
   return Cormx2Local(error, localError);
}
//______________________________________________________________________________
Float_t *TVolumePosition::Errmx2Local(const Float_t *masterError, Float_t *localError) const
{
   //to be documented
   Float_t error[6];
   TCL::vzero(&error[1],4);
   error[0] = masterError[0]; error[2] = masterError[1]; error[5] = masterError[2];
   return Cormx2Local(error, localError);
}
//______________________________________________________________________________
Double_t *TVolumePosition::Cormx2Local(const Double_t *localCorr, Double_t *masterCorr) const
{
   //to be documented
   Double_t *res = 0;
   TRotMatrix *rm = (TRotMatrix *) GetMatrix();
   double *m = 0;
   if (rm && ( m = rm->GetMatrix()) )
      res = TCL::tratsa(m,(Double_t *)localCorr,masterCorr,3,3);
   else
      res = TCL::ucopy(localCorr,masterCorr,6);
   return res;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Cormx2Local(const Float_t *localCorr, Float_t *masterCorr) const
{
   //to be documented
   Float_t *res = 0;
   TRotMatrix *rm = (TRotMatrix *) GetMatrix();
   Double_t *m = 0;
   if (rm && (m = rm->GetMatrix()) ) {
      double corLocal[6], corGlobal[6];
      TCL::ucopy(localCorr,corLocal,6);
      TCL::tratsa(m,corLocal,corGlobal,3,3);
      res =  TCL::ucopy(corGlobal,masterCorr,6);
   }
   else
      res =  TCL::ucopy(localCorr,masterCorr,6);
   return res;
}

//______________________________________________________________________________
Double_t *TVolumePosition::Local2Master(const Double_t *local, Double_t *master, Int_t nPoints) const
{
//*-*-*-*-*Convert one point from local system to master reference system*-*-*
//*-*      ==============================================================
//
//  Note that before invoking this function, the global rotation matrix
//  and translation vector for this node must have been computed.
//  This is automatically done by the Paint functions.
//  Otherwise TVolumePosition::UpdateMatrix should be called before.
   Double_t *matrix = 0;
   Double_t *trans = 0;
   if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = ((TRotMatrix *)fMatrix)->GetMatrix()) )  {
      trans = master;
      for (int i =0; i < nPoints; i++,local += 3, master += 3) TCL::vadd(local,fX,master,3);
   } else {
      trans = master;
      for (int i =0; i < nPoints; i++, local += 3, master += 3) {
         TCL::mxmpy2(matrix,local,master,3,3,1);
         TCL::vadd(master,fX,master,3);
      }
   }
   return trans;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Local2Master(const Float_t *local, Float_t *master, Int_t nPoints) const
{
   //*-*-*-*Convert nPoints points from local system to master reference system*-*-*
   //*-*      ==============================================================
   //
   //  Note that before invoking this function, the global rotation matrix
   //  and translation vector for this node must have been computed.
   //  This is automatically done by the Paint functions.
   //  Otherwise TVolumePosition::UpdateMatrix should be called before.
   //
   Double_t *matrix = 0;
   Float_t *trans = 0;
   if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = ((TRotMatrix *)fMatrix)->GetMatrix()) )
   {
      trans = master;
      for (int i =0; i < nPoints; i++,local += 3, master += 3) TCL::vadd(local,fX,master,3);
   } else {
      trans = master;
      for (int i =0; i < nPoints; i++, local += 3, master += 3) {
         Double_t dlocal[3];   Double_t dmaster[3];
         TCL::ucopy(local,dlocal,3);
         TCL::mxmpy2(matrix,dlocal,dmaster,3,3,1);
         TCL::vadd(dmaster,fX,dmaster,3);
         TCL::ucopy(dmaster,master,3);
      }
   }
   return trans;
}
//______________________________________________________________________________
Double_t *TVolumePosition::Master2Local(const Double_t *master, Double_t *local, Int_t nPoints) const
{
   //*-*-*-*-*Convert one point from master system to local reference system*-*-*
   //*-*      ==============================================================
   //
   //  Note that before invoking this function, the global rotation matrix
   //  and translation vector for this node must have been computed.
   //  This is automatically done by the Paint functions.
   //  Otherwise TVolumePosition::UpdateMatrix should be called before.
   Double_t *matrix = 0;
   Double_t *trans = 0;
   if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = ((TRotMatrix *)fMatrix)->GetMatrix()) ){
      trans = local;
      for (int i =0; i < nPoints; i++,master += 3, local += 3) TCL::vsub(master,fX,local,3);
   } else {
      trans = local;
      for (int i =0; i < nPoints; i++, master += 3, local += 3) {
         Double_t dlocal[3];
         TCL::vsub(master,fX,dlocal,3);
         TCL::mxmpy(matrix,dlocal,local,3,3,1);
      }
   }
   return trans;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Master2Local(const Float_t *master, Float_t *local, Int_t nPoints) const
{
   //*-*-*-*Convert nPoints points from master system to local reference system*-*-*
   //*-*      ==============================================================
   //
   //  Note that before invoking this function, the global rotation matrix
   //  and translation vector for this node must have been computed.
   //  This is automatically done by the Paint functions.
   //  Otherwise TVolumePosition::UpdateMatrix should be called before.
   //
   Double_t *matrix = 0;
   Float_t *trans = 0;
   if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = ((TRotMatrix *)fMatrix)->GetMatrix()) ){
      trans = local;
      for (int i =0; i < nPoints; i++,master += 3, local += 3) TCL::vsub(master,fX,local,3);
   } else {
      trans = local;
      for (int i =0; i < nPoints; i++, master += 3, local += 3) {
         Double_t dmaster[3];   Double_t dlocal[3];
         TCL::ucopy(master,dmaster,3);
         TCL::vsub(dmaster,fX,dmaster,3);
         TCL::mxmpy(matrix,dmaster,dlocal,3,3,1);
         TCL::ucopy(dlocal,local,3);
      }
   }
   return trans;
}
//______________________________________________________________________________
void TVolumePosition::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*-*Paint Referenced node with current parameters*-*-*-*
//*-*                   ==============================================
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   Error("Paint","Position can not be painted");
}

//_______________________________________________________________________
void TVolumePosition::Print(Option_t *) const
{
   //to be documented
   std::cout << *this << std::endl;
}

//______________________________________________________________________________
TVolumePosition *TVolumePosition::Reset(TVolume *node,Double_t x, Double_t y, Double_t z, TRotMatrix *matrix)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-* Reset this position *-*-*-*-*-*-*-*-*-*-*
//*-*                           ===================
//*-*    x,y,z   are the offsets of the volume with respect to his mother
//*-*    matrix  is the pointer to the rotation matrix
//*-*
//*-*    This method is to re-use the memory this object without delete/create steps
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

// This method has to be protected since it doesn't set properly kIsOwn bit.

   fNode = node;
   SetPosition(x,y,z);
   SetMatrix(matrix);
   if (!fMatrix) fMatrix = TVolume::GetIdentity();
   return this;
}

//_______________________________________________________________________
void TVolumePosition::SavePrimitive(std::ostream &, Option_t * /*= ""*/)
{
   //to be documented
#if 0
   out << "TVolumePosition *CreatePosition() { " << std::endl;
   out << "  TVolumePosition *myPosition = 0;    " << std::endl;
   Double_t x = GetX();
   Double_t y = GetY();
   Double_t z = GetZ();
   TRotMatrix *matrix =
   myPosition =  new TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const char *matrixname)
   : fNode(node),fX(x),fY(y),fZ(z),fMatrix(0)
{
/
   out << "  return myPosition; "                << std::endl;
   out << "} "                                   << std::endl;
#endif

}
//______________________________________________________________________________
void   TVolumePosition::SetLineAttributes()
{
   //to be documented
   TVolume *thisNode = GetNode();
   if (thisNode) thisNode->SetLineAttributes();
}
//_______________________________________________________________________
void TVolumePosition::SetMatrix(TRotMatrix *matrix)
{
   //to be documented
   if (matrix != fMatrix) {
      DeleteOwnMatrix();
      fMatrix = matrix;
   }
}
//_______________________________________________________________________
void TVolumePosition::UpdatePosition(Option_t *)
{
   //to be documented
   TTablePadView3D *view3D=(TTablePadView3D *)gPad->GetView3D();
//*-*- Update translation vector and rotation matrix for new level
   if (gGeometry->GeomLevel() && fMatrix) {
      gGeometry->UpdateTempMatrix(fX[0],fX[1],fX[2]
                                ,((TRotMatrix *)fMatrix)->GetMatrix()
                                ,fMatrix->IsReflection());
      if (view3D)
         view3D->UpdatePosition(fX[0],fX[1],fX[2],((TRotMatrix *)fMatrix));
   }
}

//______________________________________________________________________________
void TVolumePosition::SetVisibility(Int_t vis)
{
   //to be documented
   TVolume *node = GetNode();
   if (node) node->SetVisibility(TVolume::ENodeSEEN(vis));
}
//______________________________________________________________________________
TVolumePosition &TVolumePosition::Mult(const TVolumePosition &curPosition) {

   // This method mupltiply the position of this object to the position of the
   // curPosition object.
   // It doesn't change Id of either object involved.


   // Pick the "old" position by pieces
   TVolume *curNode = 0;
 //  UInt_t curPositionId    = 0;
   curNode       = curPosition.GetNode();
 //     curPositionId = curPosition.GetId();
   const TRotMatrix *oldMatrix = 0;
   Double_t oldTranslation[] = { 0, 0, 0 };
   oldMatrix         = GetMatrix();
   oldTranslation[0] = GetX();
   oldTranslation[1] = GetY();
   oldTranslation[2] = GetZ();

   // Pick the "current" position by pieces
   const TRotMatrix *curMatrix        = curPosition.GetMatrix();

   // Create a new position
   Double_t newTranslation[3];
   Double_t newMatrix[9];
   if(oldMatrix){
      TGeometry::UpdateTempMatrix(oldTranslation,((TRotMatrix *)oldMatrix)->GetMatrix()
                       ,curPosition.GetX(),curPosition.GetY(),curPosition.GetZ(),
                       ((TRotMatrix *)curMatrix)->GetMatrix()
                       ,newTranslation,newMatrix);
      Int_t num = gGeometry->GetListOfMatrices()->GetSize();
      Char_t anum[100];
      snprintf(anum,100,"%d",num+1);
      SetMatrixOwner();
      Reset(curNode
                           ,newTranslation[0],newTranslation[1],newTranslation[2]
                           ,new TRotMatrix(anum,"NodeView",newMatrix));
      SetMatrixOwner(kTRUE);
   } else {
      newTranslation[0] = oldTranslation[0] + curPosition.GetX();
      newTranslation[1] = oldTranslation[1] + curPosition.GetY();
      newTranslation[2] = oldTranslation[2] + curPosition.GetZ();
      Reset(curNode,newTranslation[0],newTranslation[1],newTranslation[2]);
   }
//    SetId(curPositionId);
   return *this;
}

//______________________________________________________________________________
void TVolumePosition::SetXYZ(Double_t *xyz)
{
   //to be documented
   if (xyz)  memcpy(fX,xyz,sizeof(fX));
   else      memset(fX,0,sizeof(fX));
}

//______________________________________________________________________________
void TVolumePosition::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVolumePosition.
   TRotMatrix     *save = fMatrix;
   if (R__b.IsReading()) {
      fMatrix = 0;
      R__b.ReadClassBuffer(TVolumePosition::Class(), this);
      if (!fMatrix) fMatrix = save;
   } else {
      if (save == TVolume::GetIdentity() ) fMatrix = 0;
      R__b.WriteClassBuffer(TVolumePosition::Class(), this);
      fMatrix = save;
   }
}
//______________________________________________________________________________
std::ostream& operator<<(std::ostream& s,const TVolumePosition &target)
{
   //to be documented
   s << " Node: ";
   if (target.GetNode()) s <<  target.GetNode()->GetName() << std::endl;
   else                  s << "NILL" << std::endl;
   s << Form(" Position: x=%10.5f : y=%10.5f : z=%10.5f\n", target.GetX(), target.GetY(), target.GetZ());
   TRotMatrix *rot = (TRotMatrix *) target.GetMatrix();
   if (rot){
      s << rot->IsA()->GetName() << "\t" << rot->GetName() << "\t" << rot->GetTitle() << std::endl;
      Double_t *matrix = rot->GetMatrix();
      Int_t i = 0;
      for (i=0;i<3;i++) {
         for (Int_t j=0;j<3;j++) s << Form("%10.5f:", *matrix++);
         s << std::endl;
      }
   }
   return s;
}
