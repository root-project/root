// @(#)root/star:$Name$:$Id$
// Author: Valery Fine(fine@bnl.gov)   25/12/98
// $Id: TNodePosition.cxx,v 1.23 1999/12/21 18:57:14 fine Exp $

#include <iostream.h>
#include <iomanip.h>

#include "TCL.h"
#include "TVolumePosition.h"
#include "TVolume.h"

#include "TROOT.h"
#include "TClass.h"
#include "TVirtualPad.h"
#include "TView.h"
#include "TGeometry.h"
#include "TRotMatrix.h"
#include "TBrowser.h"
#include "X3DBuffer.h"

#include "TPadView3D.h"

R__EXTERN  Size3D gSize3D;

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
TVolumePosition::TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const Text_t *matrixname)
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
   fX[0] = x; fX[1] =y; fX[2] = z;
   if (!node) return;
   static Int_t counter = 0;
   counter++;
   if(!(counter%1000))cout<<"TVolumePosition count="<<counter<<" name="<<node->GetName()<<endl;

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
   if (!gGeometry) new TGeometry;
   fX[0] = x; fX[1] = y; fX[2] = z;
   if (!fMatrix) fMatrix = TVolume::GetIdentity();
}

//______________________________________________________________________________
void TVolumePosition::Browse(TBrowser *b)
{
#ifndef WIN32
   Inspect();
#endif
   if (GetNode()) {
        TShape *shape = GetNode()->GetShape();
        b->Add(GetNode(),shape?shape->GetName():GetNode()->GetName());
   }
   else {
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
Text_t *TVolumePosition::GetObjectInfo(Int_t, Int_t)
{
   if (!gPad) return "";
   if (!GetNode()) return "";
   static char info[64];
   sprintf(info,"%s/%s, shape=%s/%s",GetNode()->GetName(),GetNode()->GetTitle(),GetNode()->GetShape()->GetName(),GetNode()->GetShape()->ClassName());
   return info;
}

//______________________________________________________________________________
Double_t *TVolumePosition::Errmx2Master(const Double_t *localError, Double_t *masterError)
{
  Double_t error[6];
  TCL::vzero(&error[1],4);
  error[0] = localError[0]; error[2] = localError[1]; error[5] = localError[2];
  return Cormx2Master(error, masterError);
}

//______________________________________________________________________________
Float_t *TVolumePosition::Errmx2Master(const Float_t *localError, Float_t *masterError)
{
  Float_t error[6];
  TCL::vzero(&error[1],4);
  error[0] = localError[0]; error[2] = localError[1]; error[5] = localError[2];
  return Cormx2Master(error, masterError);
}

//______________________________________________________________________________
Double_t *TVolumePosition::Cormx2Master(const Double_t *localCorr, Double_t *masterCorr)
{
  Double_t *res = 0;
  TRotMatrix *rm = GetMatrix();
  double *m = 0;
  if (rm && ( m = rm->GetMatrix()) )
    res = TCL::trasat(m,(Double_t *)localCorr,masterCorr,3,3);
  else
    res = TCL::ucopy(localCorr,masterCorr,6);
  return res;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Cormx2Master(const Float_t *localCorr, Float_t *masterCorr)
{
 Float_t *res = 0;
 TRotMatrix *rm = GetMatrix();
 Double_t *m = 0;
 if (rm && (m = rm->GetMatrix()) ) {
    double corLocal[6], corGlobal[6];
    TCL::ucopy(localCorr,corLocal,6);
    TCL::trasat(m,corLocal,corGlobal,3,3);
    res =  TCL::ucopy(corGlobal,masterCorr,6);
 }
 else
    res =  TCL::ucopy(localCorr,masterCorr,6);
 return res;
}


//______________________________________________________________________________
Double_t *TVolumePosition::Local2Master(const Double_t *local, Double_t *master, Int_t nPoints)
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
  if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = fMatrix->GetMatrix()) )
  {
    trans = master;
    for (int i =0; i < nPoints; i++,local += 3, master += 3) TCL::vadd(local,fX,master,3);
  }
  else
  {
    trans = master;
    for (int i =0; i < nPoints; i++, local += 3, master += 3) {
      TCL::mxmpy(matrix,local,master,3,3,1);
      TCL::vadd(master,fX,master,3);
    }
  }
  return trans;
}

//______________________________________________________________________________
Float_t *TVolumePosition::Local2Master(const Float_t *local, Float_t *master, Int_t nPoints)
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
  if (!fMatrix ||  fMatrix == TVolume::GetIdentity() || !(matrix = fMatrix->GetMatrix()) )
  {
    trans = master;
    for (int i =0; i < nPoints; i++,local += 3, master += 3) TCL::vadd(local,fX,master,3);
  }
  else
  {
    trans = master;
    for (int i =0; i < nPoints; i++, local += 3, master += 3) {
      Double_t dlocal[3];   Double_t dmaster[3];
      TCL::ucopy(local,dlocal,3);
      TCL::mxmpy(matrix,dlocal,dmaster,3,3,1);
      TCL::vadd(dmaster,fX,dmaster,3);
      TCL::ucopy(dmaster,master,3);
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
void TVolumePosition::Print(Option_t *)
{
  cout << " Node: " <<   GetNode()->GetName() << endl;
  cout << " Position: x=" <<
          GetX() << " : y=" <<
          GetY() << " : z=" <<
          GetZ() << endl;

  if (fMatrix){
      fMatrix->Print();
      Double_t *matrix = fMatrix->GetMatrix();
      Int_t i = 0;
      cout << setw(4) <<" " ;
      for (i=0;i<3;i++) cout << setw(3) << i+1 << setw(3) << ":" ;
      cout << endl;
      for (i=0;i<3;i++) {
        cout << i+1 << ". ";
        for (Int_t j=0;j<3;j++)
           cout << setw(6) << *matrix++ << " : " ;
        cout << endl;
      }
  }
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
   fNode = node;
   SetPosition(x,y,z);
   SetMatrix(matrix);
   if (!fMatrix) fMatrix = TVolume::GetIdentity();
   return this;
}

//_______________________________________________________________________
void TVolumePosition::SavePrimitive(ofstream &, Option_t *)
{
#if 0
  out << "TVolumePosition *CreatePosition() { " << endl;
  out << "  TVolumePosition *myPosition = 0;    " << endl;
  Double_t x = GetX();
  Double_t y = GetY();
  Double_t z = GetZ();
  TRotMatrix *matrix =
   myPosition =  new TVolumePosition(TVolume *node,Double_t x, Double_t y, Double_t z, const Text_t *matrixname)
: fNode(node),fX(x),fY(y),fZ(z),fMatrix(0)
{
/
  out << "  return myPosition; "                << endl;
  out << "} "                                   << endl;
#endif

}
//______________________________________________________________________________
void   TVolumePosition::SetLineAttributes()
{
  TVolume *thisNode = GetNode();
  if (thisNode) thisNode->SetLineAttributes();
}

//_______________________________________________________________________
void TVolumePosition::UpdatePosition(Option_t *)
{
  TPadView3D *view3D=(TPadView3D *)gPad->GetView3D();
//*-*- Update translation vector and rotation matrix for new level
  if (gGeometry->GeomLevel() && fMatrix) {
     gGeometry->UpdateTempMatrix(fX[0],fX[1],fX[2]
                                ,fMatrix->GetMatrix()
                                ,fMatrix->IsReflection());
     if (view3D)
        view3D->UpdatePosition(fX[0],fX[1],fX[2],fMatrix);
  }
}

//______________________________________________________________________________
void TVolumePosition::SetVisibility(Int_t vis)
{
 TVolume *node = GetNode();
 if (node) node->SetVisibility(TVolume::ENodeSEEN(vis));
}
