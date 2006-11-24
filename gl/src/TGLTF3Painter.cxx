#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "Buttons.h"
#include "TColor.h"
#include "TH1.h"
#include "TF3.h"

#include "TGLOrthoCamera.h"
#include "TGLTF3Painter.h"
#include "TGLIncludes.h"

ClassImp(TGLTF3Painter)

//______________________________________________________________________________
TGLTF3Painter::TGLTF3Painter(TF3 *fun, TH1 *hist, TGLOrthoCamera *camera, 
                             TGLPlotCoordinates *coord, Int_t ctx)
                  : TGLPlotPainter(hist, camera, coord, ctx, kFALSE, kFALSE, kFALSE),
                    fStyle(kDefault),
                    fF3(fun)
{
   // Constructor.
}
                    
//______________________________________________________________________________
char *TGLTF3Painter::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   //Coords for point on surface under cursor.
   return "fun3";
}

namespace {
   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY, 
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const TF3 *fun, std::vector<TGLTF3Painter::TriFace_t> &mesh);
}

//______________________________________________________________________________
Bool_t TGLTF3Painter::InitGeometry()
{
   //Create mesh.
   fCoord->SetCoordType(kGLCartesian);

   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   fCamera->SetViewVolume(fBackBox.Get3DBox());

   //Build mesh for TF3 surface
   fMesh.clear();

   const Int_t nX = fHist->GetNbinsX();
   const Int_t nY = fHist->GetNbinsY();
   const Int_t nZ = fHist->GetNbinsZ();

   const Double_t xMin = fXAxis->GetBinLowEdge(fXAxis->GetFirst());
   const Double_t xStep = (fXAxis->GetBinUpEdge(fXAxis->GetLast()) - xMin) / nX;
   const Double_t yMin = fYAxis->GetBinLowEdge(fYAxis->GetFirst());
   const Double_t yStep = (fYAxis->GetBinUpEdge(fYAxis->GetLast()) - yMin) / nY;
   const Double_t zMin = fZAxis->GetBinLowEdge(fZAxis->GetFirst());
   const Double_t zStep = (fZAxis->GetBinUpEdge(fZAxis->GetLast()) - zMin) / nZ;

   for (Int_t i = 0; i < nX; ++i) {
      for (Int_t j= 0; j < nY; ++j) {
         for (Int_t k = 0; k < nZ; ++k) {
            MarchingCube(xMin + i * xStep, yMin + j * yStep, zMin + k * zStep,
                         xStep, yStep, zStep, fCoord->GetXScale(), fCoord->GetYScale(), 
                         fCoord->GetZScale(), fF3, fMesh);
         }
      }
   }

   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      fCoord->ResetModified();
   }


   return kTRUE;
}

//______________________________________________________________________________
void TGLTF3Painter::StartPan(Int_t px, Int_t py)
{
   //User clicks right mouse button (in a pad).
   if (fBoxCut.IsActive() && fSelectedPart == 7) {
      fMousePosition.fX = px;
      fMousePosition.fY = fCamera->GetHeight() - py;
      fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
   } else
      fCamera->StartPan(px, py);
}

//______________________________________________________________________________
void TGLTF3Painter::Pan(Int_t px, Int_t py)
{
   //User's moving mouse cursor, with middle mouse button pressed (for pad).
   //Calculate 3d shift related to 2d mouse movement.
   if (!MakeGLContextCurrent())
      return;

   if (fSelectedPart) {
      if (fBoxCut.IsActive() && fSelectedPart == 7)
         fBoxCut.MoveBox(px, fCamera->GetHeight() - py);
      else
         fCamera->Pan(px, py);
   }
   
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLTF3Painter::AddOption(const TString &/*option*/)
{
   //No options for tf3
}

//______________________________________________________________________________
void TGLTF3Painter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{
   //Change color sheme.
   if (event == kKeyPress) {
      if (py == kKey_s || py == kKey_S) {
         fStyle < kMaple2 ? fStyle = ETF3Style(fStyle + 1) : fStyle = kDefault;
         //gGLManager->PaintSingleObject(this);
      } else if (py == kKey_c || py == kKey_C) {
         if (fHighColor)
            Info("ProcessEvent", "Cut box does not work in high color, please, switch to true color");
         else {
            fBoxCut.TurnOnOff();
            fUpdateSelection = kTRUE;
         }
      } else if (py == kKey_x || py == kKey_X) {
         if (fBoxCut.IsActive())
            fBoxCut.SetDirectionX();
      } else if (py == kKey_y || py == kKey_Y) {
         if (fBoxCut.IsActive())
            fBoxCut.SetDirectionY();
      } else if (py == kKey_z || py == kKey_Z) {
         if (fBoxCut.IsActive())
            fBoxCut.SetDirectionZ();
      }
   } else if (event == kButton1Double && fBoxCut.IsActive()) {
      fBoxCut.TurnOnOff();
      gGLManager->PaintSingleObject(this);
   }
}

//______________________________________________________________________________
void TGLTF3Painter::InitGL()const
{
   //Initialize OpenGL state variables.
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGLTF3Painter::ClearBuffers()const
{
   //Clears gl buffers (possibly with pad's background color).
   Float_t rgb[3] = {1.f, 1.f, 1.f};
   if (GetPadColor())
      GetPadColor()->GetRGB(rgb[0], rgb[1], rgb[2]);
   glClearColor(rgb[0], rgb[1], rgb[2], 1.);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

namespace {
   void GetColor(TGLVector3 &rfColor, const TGLVector3 &normal);
}

//______________________________________________________________________________
void TGLTF3Painter::DrawPlot()const
{
   //Draw mesh.
   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   //Draw TF3 surface
   if (!fSelectionPass)
      fStyle > kDefault ? glDisable(GL_LIGHTING) : SetSurfaceColor();//[0

   if (fStyle == kMaple1) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[1
      glPolygonOffset(1.f, 1.f);
   } else if (fStyle == kMaple2)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[2

   if (!fBoxCut.IsActive()) {
      glBegin(GL_TRIANGLES);

      TGLVector3 color;

      if (!fSelectionPass) {
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glNormal3dv(fMesh[i].fNormals[0].CArr());
            GetColor(color, fMesh[i].fNormals[0]);
            glColor3dv(color.CArr());
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glNormal3dv(fMesh[i].fNormals[1].CArr());
            GetColor(color, fMesh[i].fNormals[1]);
            glColor3dv(color.CArr());
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glNormal3dv(fMesh[i].fNormals[2].CArr());
            GetColor(color, fMesh[i].fNormals[2]);
            glColor3dv(color.CArr());
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }
      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }
      }

      glEnd();
      
      if (fStyle == kMaple1 && !fSelectionPass) {
         glDisable(GL_POLYGON_OFFSET_FILL);//1]
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[3
         glColor3d(0., 0., 0.);

         glBegin(GL_TRIANGLES);

         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }

         glEnd();
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//3]
      } else if (fStyle == kMaple2)
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//2]

      if (fStyle > kDefault && !fSelectionPass)
         glEnable(GL_LIGHTING); //0]
   } else {
      glBegin(GL_TRIANGLES);

      TGLVector3 color;

      if (!fSelectionPass) {
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            
            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            glNormal3dv(tri.fNormals[0].CArr());
            GetColor(color, tri.fNormals[0]);
            glColor3dv(color.CArr());
            glVertex3dv(tri.fXYZ[0].CArr());
            glNormal3dv(tri.fNormals[1].CArr());
            GetColor(color, tri.fNormals[1]);
            glColor3dv(color.CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glNormal3dv(tri.fNormals[2].CArr());
            GetColor(color, tri.fNormals[2]);
            glColor3dv(color.CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }
      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            
            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;
            glVertex3dv(tri.fXYZ[0].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }
      }

      glEnd();
      if (fStyle == kMaple1 && !fSelectionPass) {
         glDisable(GL_POLYGON_OFFSET_FILL);//1]
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[3
         glColor3d(0., 0., 0.);

         glBegin(GL_TRIANGLES);

         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            
            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;
            glVertex3dv(tri.fXYZ[0].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }

         glEnd();
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//3]
      } else if (fStyle == kMaple2) 
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//2]

      if (fStyle > kDefault && !fSelectionPass)
         glEnable(GL_LIGHTING); //0]
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);
   }

}

//______________________________________________________________________________
void TGLTF3Painter::SetSurfaceColor()const
{
   //Set color for surface.
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.65f};

   if (fF3->GetFillColor() != kWhite)
      if (const TColor *c = gROOT->GetColor(fF3->GetFillColor()))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   
   glMaterialfv(GL_BACK, GL_DIFFUSE, diffColor);
   diffColor[0] /= 2, diffColor[1] /= 2, diffColor[2] /= 2;
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionXOZ()const
{
   //XOZ parallel section.
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionYOZ()const
{
   //YOZ parallel section.
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionXOY()const
{
   //XOY parallel section
}

/*
TF3's based on a small, nice and neat implementation of marching cubes by Cory Bloyd (corysama at yahoo.com)
(many thanks!!!). All possible errors in code are mine - I've modified original code. (tpochep)
*/


namespace {
   //These tables are used so that everything can be done in little loops that you can look at all at once
   // rather than in pages and pages of unrolled code.
   //gA2VertexOffset lists the positions, relative to vertex0, of each of the 8 vertices of a cube
   const Double_t gA2VertexOffset[8][3] = 
   {
      {0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.},
      {0., 1., 0.}, {0., 0., 1.}, {1., 0., 1.},
      {1., 1., 1.}, {0., 1., 1.}
   };
   //gA2EdgeConnection lists the index of the endpoint vertices for each of the 12 edges of the cube
   const Int_t gA2EdgeConnection[12][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {4, 5}, {5, 6}, {6, 7}, {7, 4},
      {0,4}, {1,5}, {2,6}, {3,7}
   };
   //gA2EdgeDirection lists the direction vector (vertex1-vertex0) for each edge in the cube
   const Double_t gA2EdgeDirection[12][3] = 
   {
      {1., 0., 0.}, {0., 1., 0.}, {-1., 0., 0.},
      {0., -1., 0.}, {1., 0., 0.}, {0., 1., 0.},
      {-1., 0., 0.}, {0., -1., 0.}, {0., 0., 1.},
      {0., 0., 1.}, { 0., 0., 1.}, {0., 0., 1.}
   };

   const Float_t gTargetValue = 0.2f;
   //GetOffset finds the approximate point of intersection of the surface
   // between two points with the values fValue1 and fValue2
   Double_t GetOffset(Double_t val1, Double_t val2, Double_t valDesired)
   {
      Double_t delta = val2 - val1;

      if (!delta)
         return 0.5;

      return (valDesired - val1) / delta;
   }

   //GetColor generates a color from a given normal
   void GetColor(TGLVector3 &rfColor, const TGLVector3 &normal)
   {
      Double_t x = normal.X();
      Double_t y = normal.Y();
      Double_t z = normal.Z();
      rfColor.X() = (x > 0. ? x : 0.) + (y < 0. ? -0.5 * y : 0.) + (z < 0. ? -0.5 * z : 0.);
      rfColor.Y() = (y > 0. ? y : 0.) + (z < 0. ? -0.5 * z : 0.) + (x < 0. ? -0.5 * x : 0.);
      rfColor.Z() = (z > 0. ? z : 0.) + (x < 0. ? -0.5 * x : 0.) + (y < 0. ? -0.5 * y : 0.);
   }

   void GetNormal(TGLVector3 &normal, Double_t x, Double_t y, Double_t z, const TF3 *fun)
   {
      normal.X() = fun->Eval(x - 0.01, y, z) - fun->Eval(x + 0.01, y, z);
      normal.Y() = fun->Eval(x, y - 0.01, z) - fun->Eval(x, y + 0.01, z);
      normal.Z() = fun->Eval(x, y, z - 0.01) - fun->Eval(x, y, z + 0.01);
      normal.Normalise();
   }

   extern Int_t gCubeEdgeFlags[256];
   extern Int_t gTriangleConnectionTable[256][16];

   //MarchingCube performs the Marching Cubes algorithm on a single cube
   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY, 
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const TF3 *fun, std::vector<TGLTF3Painter::TriFace_t> &mesh)
   {
      Double_t afCubeValue[8] = {0.};
      TGLVector3 asEdgeVertex[12];
      TGLVector3 asEdgeNorm[12];

      //Make a local copy of the values at the cube's corners
      for (Int_t iVertex = 0; iVertex < 8; ++iVertex) {
         afCubeValue[iVertex] = fun->Eval(x + gA2VertexOffset[iVertex][0] * stepX,
                                          y + gA2VertexOffset[iVertex][1] * stepY,
                                          z + gA2VertexOffset[iVertex][2] * stepZ);
      }

      //Find which vertices are inside of the surface and which are outside
      Int_t iFlagIndex = 0;

      for (Int_t iVertexTest = 0; iVertexTest < 8; ++iVertexTest) {
         if(afCubeValue[iVertexTest] <= gTargetValue) 
            iFlagIndex |= 1<<iVertexTest;
      }

      //Find which edges are intersected by the surface
      Int_t iEdgeFlags = gCubeEdgeFlags[iFlagIndex];
      //If the cube is entirely inside or outside of the surface, then there will be no intersections
      if (!iEdgeFlags) return;
      //Find the point of intersection of the surface with each edge
      //Then find the normal to the surface at those points
      for (Int_t iEdge = 0; iEdge < 12; ++iEdge) {
         //if there is an intersection on this edge
         if (iEdgeFlags & (1<<iEdge)) {
            Double_t offset = GetOffset(afCubeValue[ gA2EdgeConnection[iEdge][0] ], 
                                       afCubeValue[ gA2EdgeConnection[iEdge][1] ],
                                       gTargetValue);

            asEdgeVertex[iEdge].X() = x + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][0]  +  offset * gA2EdgeDirection[iEdge][0]) * stepX;
            asEdgeVertex[iEdge].Y() = y + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][1]  +  offset * gA2EdgeDirection[iEdge][1]) * stepY;
            asEdgeVertex[iEdge].Z() = z + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][2]  +  offset * gA2EdgeDirection[iEdge][2]) * stepZ;

            GetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].X(), asEdgeVertex[iEdge].Y(), asEdgeVertex[iEdge].Z(), fun);
         }
      }

      //Draw the triangles that were found.  There can be up to five per cube
      for (Int_t iTriangle = 0; iTriangle < 5; iTriangle++) {
         if(gTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
            break;

         TGLTF3Painter::TriFace_t newTri;
         
         for (Int_t iCorner = 2; iCorner >= 0; --iCorner) {
            Int_t iVertex = gTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];

            newTri.fXYZ[iCorner].X() = asEdgeVertex[iVertex].X() * scaleX;
            newTri.fXYZ[iCorner].Y() = asEdgeVertex[iVertex].Y() * scaleY;
            newTri.fXYZ[iCorner].Z() = asEdgeVertex[iVertex].Z() * scaleZ;

            newTri.fNormals[iCorner] = asEdgeNorm[iVertex];
         }

         mesh.push_back(newTri);
      }
   }
   
   // For any edge, if one vertex is inside of the surface and the other is outside of the surface
   //  then the edge intersects the surface
   // For each of the 8 vertices of the cube can be two possible states : either inside or outside of the surface
   // For any cube the are 2^8=256 possible sets of vertex states
   // This table lists the edges intersected by the surface for all 256 possible vertex states
   // There are 12 edges.  For each entry in the table, if edge #n is intersected, then bit #n is set to 1

   Int_t gCubeEdgeFlags[256]=
   {
      0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 
      0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 
      0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 
      0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 
      0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 
      0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 
      0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 
      0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 
      0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 
      0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 
      0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 
      0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 
      0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 
      0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 
      0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 
      0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
   };

   //  For each of the possible vertex states listed in gCubeEdgeFlags there is a specific triangulation
   //  of the edge intersection points.  gTriangleConnectionTable lists all of them in the form of
   //  0-5 edge triples with the list terminated by the invalid value -1.
   //  For example: gTriangleConnectionTable[3] list the 2 triangles formed when corner[0] 
   //  and corner[1] are inside of the surface, but the rest of the cube is not.
   //
   //  I found this table in an example program someone wrote long ago.  It was probably generated by hand

   GLint gTriangleConnectionTable[256][16] =  
   {
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
      {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
      {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
      {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
      {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
      {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
      {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
      {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
      {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
      {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
      {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
      {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
      {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
      {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
      {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
      {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
      {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
      {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
      {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
      {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
      {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
      {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
      {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
      {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
      {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
      {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
      {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
      {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
      {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
      {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
      {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
      {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
      {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
      {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
      {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
      {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
      {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
      {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
      {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
      {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
      {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
      {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
      {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
      {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
      {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
      {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
      {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
      {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
      {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
      {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
      {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
      {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
      {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
      {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
      {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
      {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
      {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
      {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
      {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
      {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
      {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
      {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
      {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
      {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
      {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
      {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
      {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
      {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
      {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
      {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
      {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
      {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
      {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
      {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
      {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
      {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
      {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
      {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
      {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
      {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
      {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
      {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
      {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
      {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
      {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
      {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
      {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
      {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
      {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
      {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
      {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
      {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
      {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
      {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
      {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
      {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
      {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
   };
   
}
