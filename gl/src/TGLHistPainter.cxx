#include <algorithm>

#ifdef WIN32
#include "Windows4root.h"
#endif

#include "Riostream.h"

#include <GL/gl.h>
#include <GL/glu.h>

#include "THLimitsFinder.h"
#include "TVirtualGL.h"
#include "TVirtualPS.h"
#include "KeySymbols.h"
#include "TVirtualX.h"
#include "TGaxis.h"
#include "TString.h"
#include "TPoint.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAxis.h"
#include "TMath.h"
#include "TPad.h"
#include "TH2.h"
#include "TH1.h"
#include "TF3.h"

#include "TGLHistPainter.h"
#include "TGLOutput.h"
#include "gl2ps.h"

ClassImp(TGLHistPainter)

//______________________________________________________________________________
TGLHistPainter::TGLHistPainter(TH1 *hist)
                   : fDefaultPainter(0),
                     fHist(hist),
                     fF3(0),
                     fLastOption(kUnsupported),
                     fTF3Style(kDefault),
                     fAxisX(hist->GetXaxis()),
                     fAxisY(hist->GetYaxis()),
                     fAxisZ(hist->GetZaxis()),
                     fMinX(0.), fMaxX(0.), fScaleX(1.), 
                     fMinXScaled(0.), fMaxXScaled(0.),
                     fMinY(0.), fMaxY(0.), fScaleY(1.),
                     fMinYScaled(0.), fMaxYScaled(0.),
                     fMinZ(0.), fMaxZ(0.), fScaleZ(1.),
                     fMinZScaled(0.), fMaxZScaled(0.),
                     fFactor(1.),
                     fRotation(100, 100),
                     fFrustum(), fCenter(), fShift(0.), fViewport(),
                     fFirstBinX(0), fLastBinX(0),
                     fFirstBinY(0), fLastBinY(0),
                     fFirstBinZ(0), fLastBinZ(0),
                     fLogX(kFALSE), fLogY(kFALSE), fLogZ(kFALSE),
                     fGLDevice(-1),
                     f2DPass(kFALSE),
                     fTextureName(0),
                     fCurrentPainter(0),
                     fFrontPoint(2),
                     fZoom(1.)
{
   //Each TGLHistPainter has default painter as a member
   //to delegate unsupported calls
   fDefaultPainter = TVirtualHistPainter::HistPainter(fHist);
   SetTexture();
}

//______________________________________________________________________________
TGLHistPainter::~TGLHistPainter()
{
   //Destructor
   delete fDefaultPainter;
}

//______________________________________________________________________________
Int_t TGLHistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   //If fLastOption != kUnsupported, try to select hist or axis.
   //If not - gPad is selected (there are problems with TF2)
   if (fLastOption == kUnsupported)
      return fDefaultPainter ? fDefaultPainter->DistancetoPrimitive(px, py) : 1000;
   else {
      if ((fGLDevice = gPad->GetGLDevice()) == -1) {
         Error("DistancetoPrimitive", "Current pad (gPad) does not have gl device\n");

         return 1000; //TF2 will try to select something itself
      }

      if (!MakeCurrent())return 1000;

      const TGLHistPainter &cRef = *this;//To avoid dummy method call.

      if (!cRef.Select(px, py))
         gPad->SetSelected(gPad);// To avoid TF2 selection

      return 0;
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawPanel()
{
   //Interface to DrawPanel
   if (fLastOption == kUnsupported && fDefaultPainter)
      fDefaultPainter->DrawPanel();
}

//______________________________________________________________________________
void TGLHistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   //If fLastOption == kUnsupported, delegate call.
   //If not, try to process itself
   if (fLastOption == kUnsupported) {
      if(fDefaultPainter)
         fDefaultPainter->ExecuteEvent(event, px, py);
   } else {
      if (event != kKeyPress) {
         py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
         px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());
      }

      if ((fGLDevice = gPad->GetGLDevice()) == -1) {
         Error("ExecuteEvent", "current pad (gPad) does not have gl device\n");
         return;
      }

      if (!MakeCurrent())return;

      switch (event) {
      case kButton1Down :
         gGLManager->ExtractViewport(fGLDevice, fViewport);
         fRotation.SetBounds(fViewport[2], fViewport[3]);
         fRotation.Click(TPoint(px, py));
         gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
         break;
      case kButton1Motion :
         fRotation.Drag(TPoint(px, py));
         gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
         gGLManager->PaintSingleObject(this);
         break;
      case kButton1Up:
         gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
         break;
      case kMouseMotion:
         gPad->SetCursor(kRotate);//Does not work for TF2 (?)
         break;
      case kButton2Down:
         gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
         fCurrPos.fX = px;
         fCurrPos.fY = fViewport[3] - py;
         break;
      case kButton2Motion:
         gGLManager->PanObject(this, px, fViewport[3] - py);
         break;
      case kButton2Up:
         gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
         break;
      case kKeyPress:
      case 5:
      case 6:
         if (fLastOption == kTF3 && (py == kKey_s || py == kKey_S)) {
            if(fTF3Style < kMaple2)
               fTF3Style = EGLTF3Style(fTF3Style + 1);
            else
               fTF3Style = kDefault;

            gPad->Modified();
            gPad->Update();
         }
         if (event == 5 || py == kKey_J || py == kKey_j) {
            gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
            fZoom /= 1.2;
            gGLManager->PaintSingleObject(this);
            gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
         } else if (event == 6 || py == kKey_K || py == kKey_k) {
            gGLManager->MarkForDirectCopy(fGLDevice, kTRUE);
            fZoom *= 1.2;
            gGLManager->PaintSingleObject(this);
            gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
         }
         break;
      }
   }
}

//______________________________________________________________________________
void TGLHistPainter::FitPanel()
{
   //to be FIXed
   if (fLastOption == kUnsupported && fDefaultPainter)
      fDefaultPainter->FitPanel();
}

//______________________________________________________________________________
TList *TGLHistPainter::GetContourList(Double_t contour)const
{
   //to be FIXed
   return fDefaultPainter->GetContourList(contour);
}

//______________________________________________________________________________
char *TGLHistPainter::GetObjectInfo(Int_t px, Int_t py)const
{
   //to be FIXed
   return fDefaultPainter->GetObjectInfo(px, py);
}

//______________________________________________________________________________
TList *TGLHistPainter::GetStack()const
{
   //to be FIXed
   return fDefaultPainter->GetStack();
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Int_t x, Int_t y)
{
   //to be FIXed
   if (fLastOption == kUnsupported && fDefaultPainter)
      return fDefaultPainter->IsInside(x, y);
   
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Double_t x, Double_t y)
{
   //to be FIXed
   if (fLastOption == kUnsupported && fDefaultPainter)
      return fDefaultPainter->IsInside(x, y);

   return kFALSE;
}

//______________________________________________________________________________
void TGLHistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   //to be FIXed
   if (fDefaultPainter) 
      fDefaultPainter->PaintStat(dostat, fit);
}

//______________________________________________________________________________
void TGLHistPainter::ProcessMessage(const char *mess, const TObject *obj)
{
   //to be FIXed
   static const TString tf3String("SetF3");

   if (tf3String == mess) {
      fF3 = static_cast<TF3 *>((TObject *)obj);//in principle, we can use dynamic_cast and check result
   }// else if (fLastOption == kUnsupported && fDefaultPainter)
   
   fDefaultPainter->ProcessMessage(mess, obj);
}

//______________________________________________________________________________
void TGLHistPainter::SetHistogram(TH1 *hist)
{
   //to be FIXed
   fHist = hist;

   if(fDefaultPainter)
      fDefaultPainter->SetHistogram(hist);
}

//______________________________________________________________________________
void TGLHistPainter::SetStack(TList *stack)
{
   //to be FIXed
   if (fLastOption == kUnsupported && fDefaultPainter)
      fDefaultPainter->SetStack(stack);
}

//______________________________________________________________________________
Int_t TGLHistPainter::MakeCuts(char *cutsOpt)
{
   //to be FIXed
   return fDefaultPainter ? fDefaultPainter->MakeCuts(cutsOpt) : 0;
}

//______________________________________________________________________________
void TGLHistPainter::Paint(Option_t *o)
{
   //Final-overrider for TObject's Paint
   if (f2DPass) return;

   TString option(o);
   option.ToLower();

   if ((fLastOption = SetPaintFunction(option)) == kUnsupported) {
      gPad->SetCopyGLDevice(kFALSE);

      if (fDefaultPainter)
         fDefaultPainter->Paint(o);
   } else {
      
      fGLDevice = gPad->GetGLDevice();

      if (fGLDevice != -1 && SetVertices()) {
         if (!MakeCurrent()) return;
         gGLManager->PaintSingleObject(this);
      } else if (fDefaultPainter)
         fDefaultPainter->Paint(option.Data());
   }
}

//______________________________________________________________________________
void TGLHistPainter::Paint()
{
   //This function indirectly called via gGLManager->PaintSingleObject
   
   gPad->SetCopyGLDevice(kTRUE);
   //Calculate translation, frustum.
   CalculateTransformation();
   //Enable lighting etc.
   InitGL();
   //Save material/light properties in a stack
   glPushAttrib(GL_LIGHTING_BIT);
   //glViewport, glOrtho, glMatrixMode etc.
   SetCamera();
   ClearBuffers();
   //Main light
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);
   //Apply rotation + translation
   SetTransformation();
   //Define front point, draw frame and 'profiles'
   fFrontPoint = FrontPoint();
   if (gVirtualPS) PrintPlot();
   DrawFrame();
   //Surface/lego color
   SetPlotColor();
   //Call current painting function
   (this->*fCurrentPainter)();
   //Draw blue, semi-transparent plane
   DrawZeroPlane();
   //restore material properties from stack
   glPopAttrib();
   //now, gl drawing is finished, axes are drawn by TVirtualX
   glFlush();
   //Put content of GL buffer into pixmap/DIB
   gGLManager->ReadGLBuffer(fGLDevice);
   //Select this pixmap/DIB
   gGLManager->SelectOffScreenDevice(fGLDevice);
   //Draw axes into pixmap/DIB
   DrawAxes();
   gVirtualX->SelectWindow(gPad->GetPixmapID());
   //Flush pixmap during rotation
   gGLManager->Flush(fGLDevice);
}

//______________________________________________________________________________
void TGLHistPainter::PrintPlot()
{
   // Generate PS using gl2ps
   TGLOutput::StartEmbeddedPS();

   // Generate GL view
   FILE *output = fopen (gVirtualPS->GetName(), "a");
   Int_t gl2psFormat = GL2PS_EPS;
   Int_t gl2psSort = GL2PS_BSP_SORT;
   Int_t buffsize = 0, state = GL2PS_OVERFLOW;
                                                 
   while (state == GL2PS_OVERFLOW) {
      buffsize += 1024*1024;
      gl2psBeginPage ("ROOT Scene Graph", "ROOT", NULL,
                      gl2psFormat, gl2psSort, GL2PS_USE_CURRENT_VIEWPORT
                      | GL2PS_POLYGON_OFFSET_FILL | GL2PS_SILENT
                      | GL2PS_BEST_ROOT | GL2PS_OCCLUSION_CULL
                      | 0,
                      GL_RGBA, 0, NULL,0, 0, 0,
                      buffsize, output, NULL);
      DrawFrame();
      //Surface/lego color
      SetPlotColor();
      //Call current painting function
      (this->*fCurrentPainter)();
      state = gl2psEndPage();
   }
   
   fclose (output);

   TGLOutput::CloseEmbeddedPS();
   
   glFlush();
}

//______________________________________________________________________________
TGLHistPainter::EGLPaintOption TGLHistPainter::SetPaintFunction(TString &option)
{
   //Check, if Paint's option is supported
   const Ssiz_t glPos = option.Index("gl");
   if(glPos != kNPOS)
      option.Remove(glPos, 2);
   else
      return kUnsupported;
      
   if (fF3 && fLastOption == kTF3)
      return kTF3; //tf3 can be drawn only with tf3 ??
      
   const Ssiz_t len = option.Length();
   Ssiz_t pos = option.Index("lego");
   EGLPaintOption ret = kUnsupported;
   
   if (pos != kNPOS)
      if(len == 5) {
         switch (option[4]) {
         case '1' :
            ret = kLego;
            break;
         case '2' :
            ret = kLego2;
            break;
         }
      }else if (len == 4)
         ret = kLego;

   pos = option.Index("surf");

   if (pos != kNPOS)
      if (len == 5) {
         switch (option[4]) {
         case '1':
            ret = kSurface1;
            break;
         case '2':
            ret = kSurface2;
            break;
         case '4':
            ret = kSurface4;
            break;
         }         
      } else if (len == 4)
         ret = kSurface;

   if (ret < kSurface)
      fCurrentPainter = &TGLHistPainter::DrawLego;
   else if (ret < kTF3)
      fCurrentPainter = &TGLHistPainter::DrawSurface;
   else if (option == "tf3")
      ret = kTF3, fCurrentPainter = &TGLHistPainter::DrawTF3;

   return ret;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::SetVertices()
{
   //Set axes ranges, vertices, normals
   if (!SetAxes())
      return kFALSE;

   if (fLastOption < kSurface)
      SetTable();

   if (fLastOption == kSurface)
      SetMesh(), SetNormals();
   else if (fLastOption == kSurface4 || fLastOption == kSurface1 || fLastOption == kSurface2)
      SetMesh(), SetAverageNormals();
   else if (fLastOption == kTF3)
      SetTF3Mesh();

   return kTRUE;
}

namespace {

   Bool_t SetAxisRange(const TAxis *axis, Bool_t log, Int_t &first, Int_t &last,
                       Double_t &min, Double_t &max);

}

//______________________________________________________________________________
Bool_t TGLHistPainter::SetAxes()
{
   //Having TH1 pointer, setup min/max sizes and scales
   fLogX = gPad->GetLogx();
   if (!SetAxisRange(fAxisX, fLogX, fFirstBinX, fLastBinX, fMinX, fMaxX)) {
      Error("SetAxes", "cannot set X axis to log scale\n");

      return kFALSE;
   }

   fLogY = gPad->GetLogy();
   if (!SetAxisRange(fAxisY, fLogY, fFirstBinY, fLastBinY, fMinY, fMaxY)) {
      Error("SetAxes", "cannot set Y axis to log scale\n");

      return kFALSE;
   }

   if (fLastOption == kTF3) {
      fLogZ = gPad->GetLogz();
      if (!SetAxisRange(fAxisZ, fLogZ, fFirstBinZ, fLastBinZ, fMinZ, fMaxZ)) {
         Error("SetSizes", "cannot set Y axis to log scale\n");

         return kFALSE;
      }
   } else {
      fMaxZ = fHist->GetCellContent(fFirstBinX, fFirstBinY);
      fMinZ = fMaxZ;
      Double_t summ = 0.;
      Double_t positiveMin = -1.;

      for (Int_t i = fFirstBinX; i <= fLastBinX; ++i)
         for (Int_t j = fFirstBinY; j <= fLastBinY; ++j) {
            Double_t val = fHist->GetCellContent(i, j);
            if (positiveMin < 0. && val > 0.) 
               positiveMin = val;
            else if (val > 0.)
               positiveMin = TMath::Min(positiveMin, val);

            fMaxZ = TMath::Max(val, fMaxZ);
            fMinZ = TMath::Min(val, fMinZ);
            summ += val;
         }

      Bool_t maximum = fHist->GetMaximumStored() != -1111;
      Bool_t minimum = fHist->GetMinimumStored() != -1111;

      if (maximum) fMaxZ = fHist->GetMaximumStored();
      if (minimum) fMinZ = fHist->GetMinimumStored();
      if (fMinZ >= fMaxZ) fMinZ = 0.001 * fMaxZ;

      fLogZ = gPad->GetLogz();

      if (fLogZ && fMaxZ <= 0) {
         Error("SetSizes", "log scale is requested for Z, but maximum less or equal 0 (%f)", fMaxZ);
         return kFALSE;
      }

      fFactor = fHist->GetNormFactor() > 0 ? fHist->GetNormFactor() : summ;
      if (summ) fFactor /= summ;
      if (!fFactor) fFactor = 1.;

      fMaxZ *= fFactor;
      fMinZ *= fFactor;

      if (fLogZ) {
         if (fMinZ <= 0.) {
            fMinZ = TMath::Min(1., 0.001 * fMaxZ);
            fHist->SetMinimum(fMinZ);
         }

         fMinZ = TMath::Log10(fMinZ);
         fMaxZ = TMath::Log10(fMaxZ);

         if (positiveMin > 0.)
            fMinZ = TMath::Min(TMath::Log10(positiveMin), fMinZ);
      }
   }

   if (fLastOption != kTF3)
      SetZLevels();
      
   AdjustScales();

   return kTRUE;
}

//______________________________________________________________________________
void TGLHistPainter::AdjustScales()
{
   //Finds the maximum dimension and adjust scale coefficients
   Double_t xRange = fMaxX - fMinX;
   Double_t yRange = fMaxY - fMinY;
   Double_t zRange = fMinZ > 0. ? fMaxZ : fMaxZ - fMinZ;
   Double_t maxDim = TMath::Max(TMath::Max(xRange, yRange), zRange);
   
   fScaleX = maxDim / xRange;
   fScaleY = maxDim / yRange;
   fScaleZ = maxDim / zRange / 1.3;

   fMinXScaled = fMinX * fScaleX;
   fMaxXScaled = fMaxX * fScaleX;
   fMinYScaled = fMinY * fScaleY;
   fMaxYScaled = fMaxY * fScaleY;
   fMinZScaled = fMinZ * fScaleZ;
   fMaxZScaled = fMaxZ * fScaleZ;
}

//______________________________________________________________________________
void TGLHistPainter::SetTable()
{
   //Calculates table of X and Y for lego (Z is obtained during drawing) or
   //calculate mesh of triangles with vertices in the centres of bins
   const Int_t nX = fLastBinX - fFirstBinX + 1;
   const Int_t nY = fLastBinY - fFirstBinY + 1;

   fX.resize(nX + 1);
   fY.resize(nY + 1);

   if (fLogX)
      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         fX[i] = TMath::Log10(fAxisX->GetBinLowEdge(ir)) * fScaleX;
   else
      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         fX[i] = fAxisX->GetBinLowEdge(ir) * fScaleX;

   if (fLogY)
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr)
         fY[j] = TMath::Log10(fAxisY->GetBinLowEdge(jr)) * fScaleY;
   else
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr)
         fY[j] = fAxisY->GetBinLowEdge(jr) * fScaleY;

   const Double_t maxX = fAxisX->GetBinUpEdge(fLastBinX);
   fLogX ? fX[nX] = TMath::Log10(maxX) * fScaleX : fX[nX] = maxX * fScaleX;
   const Double_t maxY = fAxisY->GetBinUpEdge(fLastBinY);
   fLogY ? fY[nY] = TMath::Log10(maxY) * fScaleY : fY[nY] = maxY * fScaleY;
}

//______________________________________________________________________________
void TGLHistPainter::SetMesh()
{
   //Calculates table of X and Y for lego (Z is obtained during drawing) or
   //calculate mesh of triangles with vertices in the centres of bins
   const Int_t nX = fLastBinX - fFirstBinX + 1;
   const Int_t nY = fLastBinY - fFirstBinY + 1;

   fMesh.resize(nX * nY);
   fMesh.SetRowLen(nY);

   for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
         fLogX ? fMesh[i][j].X() = TMath::Log10(fAxisX->GetBinCenter(ir)) * fScaleX
               : fMesh[i][j].X() = fAxisX->GetBinCenter(ir) * fScaleX;
         fLogY ? fMesh[i][j].Y() = TMath::Log10(fAxisY->GetBinCenter(jr)) * fScaleY
               : fMesh[i][j].Y() = fAxisY->GetBinCenter(jr) * fScaleY;

         Double_t z = fHist->GetCellContent(ir, jr);
         
         if (fLogZ) {
            if (z <= 0)
               fMesh[i][j].Z() = fMinZ * fScaleZ;
            else
               fMesh[i][j].Z() = TMath::Log10(z) * fScaleZ;
         } else
            fMesh[i][j].Z() = z * fScaleZ;
      }
}

namespace {
   void MarchCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY, 
                  Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                  const TF3 *fun, std::vector<RootGL::TGLTriFace_t> &mesh);
}
//______________________________________________________________________________
void TGLHistPainter::SetTF3Mesh()
{
   //Build mesh for TF3 surface
   fF3Mesh.clear();

   const Int_t nX = fHist->GetNbinsX();
   const Int_t nY = fHist->GetNbinsY();
   const Int_t nZ = fHist->GetNbinsZ();

   const Double_t xMin = fAxisX->GetBinLowEdge(fAxisX->GetFirst());
   const Double_t xStep = (fAxisX->GetBinUpEdge(fAxisX->GetLast()) - xMin) / nX;
   const Double_t yMin = fAxisY->GetBinLowEdge(fAxisY->GetFirst());
   const Double_t yStep = (fAxisY->GetBinUpEdge(fAxisY->GetLast()) - yMin) / nY;
   const Double_t zMin = fAxisZ->GetBinLowEdge(fAxisZ->GetFirst());
   const Double_t zStep = (fAxisZ->GetBinUpEdge(fAxisZ->GetLast()) - zMin) / nZ;

   for (Int_t i = 0; i < nX; ++i)
      for (Int_t j= 0; j < nY; ++j)
         for (Int_t k = 0; k < nZ; ++k)
            MarchCube(xMin + i * xStep, yMin + j * yStep, zMin + k * zStep,
                      xStep, yStep, zStep, fScaleX, fScaleY, fScaleZ, fF3, fF3Mesh);
}

//______________________________________________________________________________
void TGLHistPainter::SetNormals()
{
   //Calculates normals for triangles in surface.
   //we have four points (cell contents of four neighbouring hist bins),
   //three points are in one plane, so build normals for 2 triangles
   const Int_t nX = fLastBinX - fFirstBinX;
   const Int_t nY = fLastBinY - fFirstBinY;
   
   fFaceNormals.resize(nX * nY);
   fFaceNormals.SetRowLen(nY);

   for (Int_t i = 0; i < nX; ++i)
      for (Int_t j = 0; j < nY; ++j) {
         //first "bottom-left" triangle
         TMath::Normal2Plane(fMesh[i][j + 1].CArr(), fMesh[i][j].CArr(), fMesh[i + 1][j].CArr(),
                             fFaceNormals[i][j].first.Arr());
         //second "top-right" triangle
         TMath::Normal2Plane(fMesh[i + 1][j].CArr(), fMesh[i + 1][j + 1].CArr(), fMesh[i][j + 1].CArr(),
                             fFaceNormals[i][j].second.Arr());
      }
}

//______________________________________________________________________________
void TGLHistPainter::SetAverageNormals()
{
   //One normal per vertex;
   //this normal is average of
   //neighbouring triangles normals
   const Int_t nX = fLastBinX - fFirstBinX + 1;
   const Int_t nY = fLastBinY - fFirstBinY + 1;
   
   fFaceNormals.resize((nX + 1) * (nY + 1));
   fFaceNormals.SetRowLen(nY + 1);
   //first, calculate normal for each triangle face
   for (Int_t i = 0; i < nX - 1; ++i)
      for (Int_t j = 0; j < nY - 1; ++j) {
         //first "bottom-left" triangle
         TMath::Normal2Plane(fMesh[i][j + 1].CArr(), fMesh[i][j].CArr(), fMesh[i + 1][j].CArr(),
                             fFaceNormals[i + 1][j + 1].first.Arr());
         //second "top-right" triangle
         TMath::Normal2Plane(fMesh[i + 1][j].CArr(), fMesh[i + 1][j + 1].CArr(), fMesh[i][j + 1].CArr(),
                             fFaceNormals[i + 1][j + 1].second.Arr());
      }
      
   fAverageNormals.resize(nX * nY);
   fAverageNormals.SetRowLen(nY);
   //second, calculate average normal for each vertex
   for (Int_t i = 0; i < nX; ++i)
      for (Int_t j = 0; j < nY; ++j) {
         TGLVector3 &norm = fAverageNormals[i][j];
         
         norm += fFaceNormals[i][j].second;
         norm += fFaceNormals[i][j + 1].first;
         norm += fFaceNormals[i][j + 1].second;
         norm += fFaceNormals[i + 1][j].first;
         norm += fFaceNormals[i + 1][j].second;
         norm += fFaceNormals[i + 1][j + 1].first;
         
         norm.Normalise();
      }
}

//______________________________________________________________________________
void TGLHistPainter::InitGL()const
{
   //gl initialization (Disable/Enable)
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);

   if (fLastOption <= kLego2)
      glEnable(GL_CULL_FACE), glCullFace(GL_BACK);
   else //for surface we cannot cull faces, because we can look at the surface "from bottom"
      glDisable(GL_CULL_FACE);

   if (fLastOption == kTF3) 
      glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
   else
      glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

//______________________________________________________________________________
Bool_t TGLHistPainter::MakeCurrent()const
{
   //Check gl context and make it current
   return fGLDevice != -1 && gGLManager->MakeCurrent(fGLDevice);
}

namespace {

   void DrawBoxFront(Double_t xMin, Double_t xMax, Double_t yMin, 
                     Double_t yMax, Double_t zMin, Double_t zMax,
                     Int_t frontPoint);
   void DrawBoxFrontTextured(Double_t x1, Double_t x2, Double_t y1, 
                             Double_t y2, Double_t z1, Double_t z2,
                             Double_t zMin, Double_t zRange, Int_t frontPoint);
}

//______________________________________________________________________________
void TGLHistPainter::DrawLego()const
{
   //Draws lego
   const Int_t nX = fLastBinX - fFirstBinX + 1;
   const Int_t nY = fLastBinY - fFirstBinY + 1;

   glEnable(GL_POLYGON_OFFSET_FILL);//[0
   glPolygonOffset(1.f, 1.f);
   //Cycle through table
   if (fLastOption != kLego2) {
      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
            Double_t zMax = fHist->GetCellContent(ir, jr);

            if (fLogZ)
               if (zMax <= 0.)
                  continue;
               else 
                  zMax = TMath::Log10(zMax) * fScaleZ;
            else zMax *= fScaleZ;

            DrawBoxFront(fX[i], fX[i + 1], fY[j], fY[j + 1], 0, zMax, fFrontPoint);
         }
   } else {
      EnableTexture();//[1

      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
            Double_t zMax = fHist->GetCellContent(ir, jr);

            if (fLogZ)
               if (zMax <= 0.)
                  continue;
               else 
                  zMax = TMath::Log10(zMax) * fScaleZ;
            else zMax *= fScaleZ;

            DrawBoxFrontTextured(fX[i], fX[i + 1], fY[j], fY[j + 1], 0., zMax,
                                 fMinZScaled, fMaxZScaled - fMinZScaled, fFrontPoint);
         }

      DisableTexture();//1]
   }

   glDisable(GL_POLYGON_OFFSET_FILL);//0]
   //Outlines cycle
   glDisable(GL_LIGHTING);//[2
   glColor3d(0., 0., 0.);
   glPolygonMode(GL_FRONT, GL_LINE);//[3

   for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
         Double_t zMax = fHist->GetCellContent(ir, jr);

         if (fLogZ)
            if (zMax <= 0.)
               continue;
            else 
               zMax = TMath::Log10(zMax) * fScaleZ;
         else zMax *= fScaleZ;

         DrawBoxFront(fX[i], fX[i + 1], fY[j], fY[j + 1], 0., zMax, fFrontPoint);
      }

   glPolygonMode(GL_FRONT, GL_FILL);//3]
   glEnable(GL_LIGHTING);//2]
}

namespace {

   void DrawFlatFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                     const TGLVector3 &norm);
   void DrawSmoothFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                       const TGLVector3 &norm1, const TGLVector3 &norm2,
                       const TGLVector3 &norm3);
   void DrawFaceTextured(const TGLVertex3 &v1, const TGLVertex3 &v2,
                        const TGLVertex3 &v3, const TGLVector3 &norm1,
                        const TGLVector3 &norm2, const TGLVector3 &norm3,
                        Double_t zMin, Double_t zMax);
   void DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                        const TGLVertex3 &v3, const TGLVertex3 &v4);
}

//______________________________________________________________________________
void TGLHistPainter::DrawSurface()const
{
   //Draw surf/surf1/surf2/surf4
   const Int_t nX = fLastBinX - fFirstBinX + 1;
   const Int_t nY = fLastBinY - fFirstBinY + 1;

   if (fLastOption == kSurface) {
      for (Int_t i = 0; i < nX - 1; ++i)
         for (Int_t j = 0; j < nY - 1; ++j) {
            DrawFlatFace(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1], fFaceNormals[i][j].first);
            DrawFlatFace(fMesh[i][j + 1], fMesh[i + 1][j + 1], fMesh[i + 1][j], fFaceNormals[i][j].second);
         }
   } else if (fLastOption == kSurface4) {
      for (Int_t i = 0; i < nX - 1; ++i)
         for (Int_t j = 0; j < nY - 1; ++j) {
            DrawSmoothFace(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1],
                     fAverageNormals[i + 1][j], fAverageNormals[i][j], fAverageNormals[i][j + 1]);
            DrawSmoothFace(fMesh[i][j + 1], fMesh[i + 1][j + 1], fMesh[i + 1][j], 
                     fAverageNormals[i][j + 1], fAverageNormals[i + 1][j + 1], fAverageNormals[i + 1][j]);
      }
   } else {
      //Paints surf1/surf2 options
      EnableTexture();//[0
      //Surface "surf2" is partially transparent
      if (fLastOption == kSurface2) {
         glDisable(GL_DEPTH_TEST);//[1
         glDepthMask(GL_FALSE);//[2
         glEnable(GL_BLEND);//[3
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      } else if (fLastOption == kSurface1) {
         glEnable(GL_POLYGON_OFFSET_FILL);//[4
         glPolygonOffset(1.f, 1.f);
      }
       
      //Blend : to get "correct transparency", I have to draw 
      //faces starting from the farthest. 
      Int_t i = 0, firstJ = 0;
      const Int_t addI = fFrontPoint == 2 || fFrontPoint == 1 ? i = 0, 1 : (i = nX - 2, -1);
      const Int_t addJ = fFrontPoint == 2 || fFrontPoint == 3 ? firstJ = 0, 1 : (firstJ = nY - 2, -1);

      for (; addI > 0 ? i < nX - 1 : i >= 0; i += addI) {
         for (Int_t j = firstJ; addJ > 0 ? j < nY - 1 : j >= 0; j += addJ) {
            DrawFaceTextured(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1],
                             fAverageNormals[i + 1][j], fAverageNormals[i][j], 
                             fAverageNormals[i][j + 1], fMinZScaled, fMaxZScaled);
            DrawFaceTextured(fMesh[i][j + 1], fMesh[i + 1][j + 1], fMesh[i + 1][j], 
                             fAverageNormals[i][j + 1], fAverageNormals[i + 1][j + 1], 
                             fAverageNormals[i + 1][j], fMinZScaled, fMaxZScaled);
         }
      }

      if (fLastOption == kSurface2) {
         glDisable(GL_BLEND);//3]
         glDepthMask(GL_TRUE);//2]
         glEnable(GL_DEPTH_TEST);//1]
      } else if (fLastOption == kSurface1) {
         //surf1 - textured, non-transparent, with outlines
         glDisable(GL_POLYGON_OFFSET_FILL);//4]
         //Outlines:
         glDisable(GL_LIGHTING);//[5
         glColor3d(0., 0., 0.);

         for (Int_t i = 0; i < nX - 1; ++i)
            for (Int_t j = 0; j < nY - 1; ++j)
               DrawQuadOutline(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1], fMesh[i + 1][j + 1]);

         glEnable(GL_LIGHTING);//5]
      }

      DisableTexture();//0]
   }
}

namespace {

   void GetColor(TGLVector3 &rfColor, const TGLVector3 &normal);
   
}

/*
DrawTF3 based on a small, nice and neat implementation of marc. cubes by Cory Bloyd (corysama@yahoo.com)
*/

//______________________________________________________________________________
void TGLHistPainter::DrawTF3()const
{
   //Draw TF3 surface
   if (fTF3Style > kDefault) {
      glDisable(GL_LIGHTING);//[0
   }

   if (fTF3Style == kMaple1) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[1
      glPolygonOffset(1.f, 1.f);
   } else if (fTF3Style == kMaple2)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[2

   glBegin(GL_TRIANGLES);

   TGLVector3 color;

   for (UInt_t i = 0, e = fF3Mesh.size(); i < e; ++i) {
      glNormal3dv(fF3Mesh[i].fNormals[0].CArr());
      GetColor(color, fF3Mesh[i].fNormals[0]);
      glColor3dv(color.CArr());
      glVertex3dv(fF3Mesh[i].fXYZ[0].CArr());
      glNormal3dv(fF3Mesh[i].fNormals[1].CArr());
      GetColor(color, fF3Mesh[i].fNormals[1]);
      glColor3dv(color.CArr());
      glVertex3dv(fF3Mesh[i].fXYZ[1].CArr());
      glNormal3dv(fF3Mesh[i].fNormals[2].CArr());
      GetColor(color, fF3Mesh[i].fNormals[2]);
      glColor3dv(color.CArr());
      glVertex3dv(fF3Mesh[i].fXYZ[2].CArr());
   }

   glEnd();

   if (fTF3Style == kMaple1) {
      glDisable(GL_POLYGON_OFFSET_FILL);//1]
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[3
      glColor3d(0., 0., 0.);

      glBegin(GL_TRIANGLES);

      for (UInt_t i = 0, e = fF3Mesh.size(); i < e; ++i) {
         glVertex3dv(fF3Mesh[i].fXYZ[0].CArr());
         glVertex3dv(fF3Mesh[i].fXYZ[1].CArr());
         glVertex3dv(fF3Mesh[i].fXYZ[2].CArr());
      }

      glEnd();

      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//3]
   } else if (fTF3Style == kMaple2)
      glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//2]

   if (fTF3Style > kDefault) {
      glEnable(GL_LIGHTING); //0]
   }
}

//______________________________________________________________________________
void TGLHistPainter::CalculateTransformation()
{
   //Sets viewport, bounds for arcball
   //Calculates arguments for glOrtho
   //Claculates center of scene and shift
   gGLManager->ExtractViewport(fGLDevice, fViewport);
   fRotation.SetBounds(fViewport[2], fViewport[3]);

   Double_t xRange = fMaxX - fMinX;
   Double_t yRange = fMaxY - fMinY;
   Double_t zRange = fMinZ > 0. ? fMaxZ : fMaxZ - fMinZ;

   Double_t maxDim = TMath::Max(TMath::Max(xRange, yRange), zRange);
   fCenter[0] = fMinX + xRange / 2;
   fCenter[1] = fMinY + yRange / 2;
   fCenter[2] = fMinZ > 0. ? zRange / 2 : fMinZ + zRange / 2;

   Double_t frx = 1., fry = 1.;

   if (fViewport[2] > fViewport[3])
      frx = fViewport[2] / double(fViewport[3]);
   else if (fViewport[2] < fViewport[3])
      fry = fViewport[3] / double(fViewport[2]);

   fFrustum[0] = maxDim;
   fFrustum[1] = maxDim;
   fFrustum[2] = -100 * maxDim;
   fFrustum[3] = 100 * maxDim;
   fShift = maxDim * 1.5;
}

/*
Two back faces cannot be seen, we know, which point is front, so we can
escape passing two polygons to gl.
For example : plane 0 - we draw it only if front point is 3 or 0
               | z
               |
               |       |
               |   0   |
               |      3|
            |1 0----|--3 ------>y
            | / 2   | /
            |/      |/
            1-------2
           / 
          /
           x
   */

namespace {

   const Int_t gBoxFrontQuads[][4] = {{0, 1, 2, 3}, {4, 0, 3, 5}, {4, 5, 6, 7}, {7, 6, 2, 1}};
   const Double_t gBoxFrontNormals[][3] = {{-1., 0., 0.}, {0., -1., 0.}, {1., 0., 0.}, {0., 1., 0.}};
   const Int_t gBoxFrontPlanes[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};

   //______________________________________________________________________________
   void DrawBoxFront(Double_t xMin, Double_t xMax, Double_t yMin, 
                     Double_t yMax, Double_t zMin, Double_t zMax, Int_t fp)
   {
      //Draws lego's bar as a 3d box
      if (zMax < zMin) 
         std::swap(zMax, zMin);
      //Top and bottom are always drawn (though I can skip one them)
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glVertex3d(xMax, yMin, zMax);
      glVertex3d(xMax, yMax, zMax);
      glVertex3d(xMin, yMax, zMax);
      glVertex3d(xMin, yMin, zMax);
      glEnd();

      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glVertex3d(xMax, yMin, zMin);
      glVertex3d(xMin, yMin, zMin);
      glVertex3d(xMin, yMax, zMin);
      glVertex3d(xMax, yMax, zMin);
      glEnd();
      
      const Double_t box[][3] = {{xMin, yMin, zMax}, {xMin, yMax, zMax}, {xMin, yMax, zMin}, {xMin, yMin, zMin},
                                 {xMax, yMin, zMax}, {xMax, yMin, zMin}, {xMax, yMax, zMin}, {xMax, yMax, zMax}};
      const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
      glVertex3dv(box[verts[0]]);
      glVertex3dv(box[verts[1]]);
      glVertex3dv(box[verts[2]]);
      glVertex3dv(box[verts[3]]);
      glEnd();
      
      verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
      glVertex3dv(box[verts[0]]);
      glVertex3dv(box[verts[1]]);
      glVertex3dv(box[verts[2]]);
      glVertex3dv(box[verts[3]]);
      glEnd();
   }

   //______________________________________________________________________________
   void DrawBoxFrontTextured(Double_t x1, Double_t x2, Double_t y1, 
                           Double_t y2, Double_t z1, Double_t z2,
                           Double_t zMin, Double_t zRange, Int_t fp)
   {
      //Draws lego's bar as a textured box
      if (z2 < z1) 
         std::swap(z2, z1);

      Double_t capZ = (z2 - zMin) / zRange;
      //Top and bottom are always drawn (though I can skip one them)
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glTexCoord1d(capZ); glVertex3d(x2, y1, z2);
      glTexCoord1d(capZ); glVertex3d(x2, y2, z2);
      glTexCoord1d(capZ); glVertex3d(x1, y2, z2);
      glTexCoord1d(capZ); glVertex3d(x1, y1, z2);
      glEnd();

      capZ = (z1 - zMin) / zRange;

      glBegin(GL_POLYGON);
      glNormal3d(0., 0., -1.);
      glTexCoord1d(capZ); glVertex3d(x2, y1, z1);
      glTexCoord1d(capZ); glVertex3d(x1, y1, z1);
      glTexCoord1d(capZ); glVertex3d(x1, y2, z1);
      glTexCoord1d(capZ); glVertex3d(x2, y2, z1);
      glEnd();

      const Double_t box[][3] = {{x1, y1, z2}, {x1, y2, z2}, {x1, y2, z1}, {x1, y1, z1},
                                 {x2, y1, z2}, {x2, y1, z1}, {x2, y2, z1}, {x2, y2, z2}};
      const Double_t z[] = {(z2 - zMin) / zRange, (z2 - zMin) / zRange,
                           (z1 - zMin) / zRange, (z1 - zMin) / zRange,
                           (z2 - zMin) / zRange, (z1 - zMin) / zRange,
                           (z1 - zMin) / zRange, (z2 - zMin) / zRange};
      const Int_t *verts = gBoxFrontQuads[gBoxFrontPlanes[fp][0]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][0]]);
      glTexCoord1d(z[verts[0]]), glVertex3dv(box[verts[0]]);
      glTexCoord1d(z[verts[1]]), glVertex3dv(box[verts[1]]);
      glTexCoord1d(z[verts[2]]), glVertex3dv(box[verts[2]]);
      glTexCoord1d(z[verts[3]]), glVertex3dv(box[verts[3]]);
      glEnd();
      
      verts = gBoxFrontQuads[gBoxFrontPlanes[fp][1]];

      glBegin(GL_POLYGON);
      glNormal3dv(gBoxFrontNormals[gBoxFrontPlanes[fp][1]]);
      glTexCoord1d(z[verts[0]]), glVertex3dv(box[verts[0]]);
      glTexCoord1d(z[verts[1]]), glVertex3dv(box[verts[1]]);
      glTexCoord1d(z[verts[2]]), glVertex3dv(box[verts[2]]);
      glTexCoord1d(z[verts[3]]), glVertex3dv(box[verts[3]]);
      glEnd();
   }

}

/*
      DrawFrame:
      Each front point has two opposite back planes 
      
              |       |
              |   0   |
              |      3|
           |1 0----|--3
           | / 2   | /
           |/      |/
           1-------2
      In backPlanes 2d array first subarray holds numbers of opposite planes.
      For example : point 0 has opposite planes 3 and 2, 1 - 0 and 3 etc.
*/

namespace {

   void DrawQuadFace(const TGLVertex3 &v0, const TGLVertex3 &v1, const TGLVertex3 &v2,
                 const TGLVertex3 &v3, const TGLVertex3 &normal);

}

//______________________________________________________________________________
void TGLHistPainter::DrawFrame()const
{
   //Draws frame box around hist or surface,
   //draws grids and 'profiles'
   //Planes are 85% opaque to make their color "softer"
   glEnable(GL_BLEND);//[0
   glDepthMask(GL_FALSE);//[1
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   Float_t backColor[] = {0.9f, 0.9f, 0.9f, 0.85f};

   if (gPad->GetFrameFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(gPad->GetFrameFillColor()))
         c->GetRGB(backColor[0], backColor[1], backColor[2]);

   glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);

   //First, bottom plane at the minimum;
   //draw it with offset to remove "artefacts" when have "near-zero" overlapping polygons 
   //(such polygons can be in lego, surface or it can be zero-plane itself)
   glEnable(GL_POLYGON_OFFSET_FILL);//[2
   glPolygonOffset(2.f, 2.f);//offset is 2., because lego uses offset 1

   Double_t zMin = fMinZ > 0. ? 0. : fMinZScaled;

   DrawQuadFace(TGLVertex3(fMinXScaled, fMinYScaled, zMin), TGLVertex3(fMaxXScaled, fMinYScaled, zMin), 
            TGLVertex3(fMaxXScaled, fMaxYScaled, zMin), TGLVertex3(fMinXScaled, fMaxYScaled, zMin),
            TGLVertex3(0., 0., 1.));

   glDisable(GL_POLYGON_OFFSET_FILL);//2]
   
   static const Int_t backPlanes[][2] = {{3, 2}, {0, 3}, {1, 0}, {2, 1}};
   //DrawBackPlane draws frame's part and corresponding profile
   DrawBackPlane(backPlanes[fFrontPoint][0]);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);
   DrawBackPlane(backPlanes[fFrontPoint][1]);

   glDepthMask(GL_TRUE);//1]
   glDisable(GL_BLEND);//0]
}

//______________________________________________________________________________
void TGLHistPainter::SetCamera()const
{
   //Viewport and projection
   glViewport(fViewport[0], fViewport[1], fViewport[2], fViewport[3]);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-fFrustum[0] * fZoom, fFrustum[0] * fZoom, - fFrustum[1] * fZoom, fFrustum[1] * fZoom, fFrustum[2], fFrustum[3]);

   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

//______________________________________________________________________________
void TGLHistPainter::SetTransformation()const
{
   //Applies rotations and translations before drawing
   glTranslated(0., 0., -fShift);
   glMultMatrixd(fRotation.GetRotMatrix());
   glRotated(45., 1., 0., 0.);
   glRotated(-45., 0., 1., 0.);
   glRotated(-90., 0., 1., 0.);
   glRotated(-90., 1., 0., 0.);
   glTranslated(-fPan[0], -fPan[1], -fPan[2]);
   glTranslated(-fCenter[0] * fScaleX, -fCenter[1] * fScaleY, -fCenter[2] * fScaleZ);
}

namespace {
   bool Compare(const TGLVertex3 &v1, const TGLVertex3 &v2)
   {
      return v1.Z() < v2.Z();
   }
}

/*
In minZ plane I have 4 points 0 {minX, minY} 1 {maxX, minY} 2 {maxX, maxY} 3 {minX, maxY}
                 0-----3
                /     /
               1-----2

*/

//______________________________________________________________________________
Int_t TGLHistPainter::FrontPoint()const
{
   //Converts 3d points into window coordinate system
   //and find the nearest
   Double_t mvMatrix[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
   Double_t prMatrix[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, prMatrix);
   
   Double_t zMin = fMinZScaled > 0. ? 0. : fMinZScaled;
   Double_t xy[][2] = {{fMinXScaled, fMinYScaled}, {fMaxXScaled, fMinYScaled},
                       {fMaxXScaled, fMaxYScaled}, {fMinXScaled, fMaxYScaled}};

   for (Int_t i = 0; i < 4; ++i) {
      gluProject(xy[i][0], xy[i][1], zMin, mvMatrix, prMatrix, fViewport,
                 &f2DAxes[i].X(), &f2DAxes[i].Y(), &f2DAxes[i].Z());
      gluProject(xy[i][0], xy[i][1], fMaxZScaled, mvMatrix, prMatrix, fViewport,
                 &f2DAxes[i + 4].X(), &f2DAxes[i + 4].Y(), &f2DAxes[i + 4].Z());
   }

   return std::min_element(f2DAxes, f2DAxes + 4, ::Compare) - f2DAxes;
}

//______________________________________________________________________________
void TGLHistPainter::DrawBackPlane(Int_t plane)const
{
   //Draw back plane with number 'plane'

   TGLVertex3 v0, v1, v2, v3, normal;
   Double_t zMin = fMinZScaled > 0. ? 0. : fMinZScaled;

   switch (plane) {
   case 0 :
      v0.Set(fMinXScaled, fMinYScaled, zMin);
      v1.Set(fMinXScaled, fMaxYScaled, zMin);
      v2.Set(fMinXScaled, fMaxYScaled, fMaxZScaled);
      v3.Set(fMinXScaled, fMinYScaled, fMaxZScaled);
      normal.Set(1., 0., 0.);

      break;
   case 1 :
      v0.Set(fMinXScaled, fMinYScaled, zMin);
      v1.Set(fMinXScaled, fMinYScaled, fMaxZScaled);
      v2.Set(fMaxXScaled, fMinYScaled, fMaxZScaled);
      v3.Set(fMaxXScaled, fMinYScaled, zMin);
      normal.Set(0., 1., 0.);

      break;
   case 2 :
      v0.Set(fMaxXScaled, fMinYScaled, zMin);
      v1.Set(fMaxXScaled, fMinYScaled, fMaxZScaled);
      v2.Set(fMaxXScaled, fMaxYScaled, fMaxZScaled);
      v3.Set(fMaxXScaled, fMaxYScaled, zMin);
      normal.Set(-1., 0., 0.);

      break;
   case 3 :
      v0.Set(fMaxXScaled, fMaxYScaled, zMin);
      v1.Set(fMaxXScaled, fMaxYScaled, fMaxZScaled);
      v2.Set(fMinXScaled, fMaxYScaled, fMaxZScaled);
      v3.Set(fMinXScaled, fMaxYScaled, zMin);
      normal.Set(0., -1., 0.);

      break;
   }

   DrawQuadFace(v0, v1, v2, v3, normal);
   //antialias back plane outline
   glEnable(GL_LINE_SMOOTH);//[0
   glEnable(GL_BLEND);//[1
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
   glDisable(GL_DEPTH_TEST);//[2
   glDisable(GL_LIGHTING);//[3
   glColor3d(0., 0., 0.);

   DrawQuadOutline(v0, v1, v2, v3);

   glEnable(GL_LIGHTING);//3]
   glEnable(GL_DEPTH_TEST);//2]
   glDisable(GL_BLEND);//1]
   glDisable(GL_LINE_SMOOTH);//0]
  
   DrawProfile(plane);
   DrawGrid(plane);
}

namespace {

   /*
   minZ plane, 4 points 0 {minX, minY} 1 {maxX, minY} 2 {maxX, maxY} 3 {minX, maxY}
                    0-----3
                   /     /
                  1-----2
   Each point has left and right neighbour:
        0     1     2     3
       / \   / \   / \   / \
      3   1 0   2 1   3 2   0
   */
   
   void Draw2DAxis(TAxis *axis, Double_t xMin, Double_t yMin, Double_t xMax, Double_t yMax,
                   Double_t min, Double_t max, Bool_t log, Bool_t z = kFALSE)
   {
      //Axes are drawn with help of TGaxis class
      std::string option;
      option.reserve(20);
      
      if (xMin > xMax || z) option += "SDH=+";
      else option += "SDH=-";
      
      if (log) option += 'G';
      
      Int_t nDiv = axis->GetNdivisions();
      
      if (nDiv < 0) {
         option += 'N';
         nDiv = -nDiv;
      }
      
      TGaxis axisPainter;
      axisPainter.SetLineWidth(1);
      
      static const Double_t zero = 0.001;
      
      if (TMath::Abs(xMax - xMin) >= zero || TMath::Abs(yMax - yMin) >= zero) {
         axisPainter.ImportAxisAttributes(axis);
         axisPainter.SetLabelOffset(axis->GetLabelOffset() + axis->GetTickLength());

         if (log) {
            min = TMath::Power(10, min);
            max = TMath::Power(10, max);
         }
         //Option time display is required ?
         if (axis->GetTimeDisplay()) {
            option += 't';

            if (!strlen(axis->GetTimeFormatOnly()))
               axisPainter.SetTimeFormat(axis->ChooseTimeFormat(max - min));
            else
               axisPainter.SetTimeFormat(axis->GetTimeFormat());
         }

         axisPainter.SetOption(option.c_str());
         axisPainter.PaintAxis(xMin, yMin, xMax, yMax, min, max, nDiv, option.c_str());
      }
   }
   
   const Int_t gFramePoints[][2] = {{3, 1}, {0, 2}, {1, 3}, {2, 0}};
   //Each point has two "neighbouring axes" (left and right). Axes types are 1 (ordinata) and 0 (abscissa)
   const Int_t gAxisType[][2] = {{1, 0}, {0, 1}, {1, 0}, {0, 1}};
}

//______________________________________________________________________________
void TGLHistPainter::DrawAxes()const
{
   //Using front point, find, where to draw axes and which labels to use for them
   //gVirtualX->SelectWindow(gGLManager->GetVirtualXInd(fGLDevice));
   gVirtualX->SetDrawMode(TVirtualX::kCopy);//TCanvas by default sets in kInverse

   const Int_t left = gFramePoints[fFrontPoint][0];
   const Int_t right = gFramePoints[fFrontPoint][1];

   const Double_t xLeft = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[left].X()));
   const Double_t yLeft = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[left].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));
   
   const Double_t xMid = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[fFrontPoint].X()));
   const Double_t yMid = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[fFrontPoint].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));

   const Double_t xRight = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[right].X()));
   const Double_t yRight = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[right].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));
   
   const Double_t points[][2] = {{fMinX, fMinY}, {fMaxX, fMinY}, {fMaxX, fMaxY}, {fMinX, fMaxY}};

   const Int_t leftType = gAxisType[fFrontPoint][0];
   const Int_t rightType = gAxisType[fFrontPoint][1];
   const Double_t leftLabel = points[left][leftType];
   const Double_t leftMidLabel = points[fFrontPoint][leftType];
   const Double_t rightMidLabel = points[fFrontPoint][rightType];
   const Double_t rightLabel = points[right][rightType];

   if (xLeft - xMid || yLeft - yMid) {//To supress error messages from TGaxis
      TAxis *axis = leftType ? fAxisY : fAxisX;

      if (leftLabel < leftMidLabel)
         Draw2DAxis(axis, xLeft, yLeft, xMid, yMid, leftLabel, leftMidLabel, leftType ? fLogY : fLogX);
      else
         Draw2DAxis(axis, xMid, yMid, xLeft, yLeft, leftMidLabel, leftLabel, leftType ? fLogY : fLogX);
   }

   if (xRight - xMid || yRight - yMid) {//To supress error messages from TGaxis
      TAxis *axis = rightType ? fAxisY : fAxisX;

      if (rightMidLabel < rightLabel)
         Draw2DAxis(axis, xMid, yMid, xRight, yRight, rightMidLabel, rightLabel, rightType ? fLogY : fLogX);
      else
         Draw2DAxis(axis, xRight, yRight, xMid, yMid, rightLabel, rightMidLabel, rightType ? fLogY : fLogX);
   }
    
   const Double_t xUp = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[left + 4].X()));
   const Double_t yUp = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[left + 4].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));

   Draw2DAxis(fAxisZ, xLeft, yLeft, xUp, yUp, fMinZ, fMaxZ, fLogZ, kTRUE);

/*   f2DPass = kTRUE;
   TObjOptLink *lnk = static_cast<TObjOptLink *>(gPad->GetListOfPrimitives()->FirstLink());
   
   while (lnk) {
      TObject *obj = lnk->GetObject();

      obj->Paint(lnk->GetOption());
      lnk = static_cast<TObjOptLink *>(lnk->Next());
   }

   f2DPass = kFALSE;*/
//FIXME   gGLManager->SelectGLPixmap(fGLDevice);
}

//______________________________________________________________________________
Bool_t TGLHistPainter::Select(Int_t x, Int_t y)const
{
   //Find hist "square" on screen
   SetCamera();
   SetTransformation();
   fFrontPoint = FrontPoint();
   
   Double_t xMin = f2DAxes[0].X();
   Double_t yMin = fViewport[3] - f2DAxes[0].Y();
   Double_t xMax = xMin;
   Double_t yMax = yMin;

   for (Int_t i = 1; i < 8; ++i) {
      xMin = TMath::Min(xMin, f2DAxes[i].X() + gPad->GetXlowNDC() * gPad->GetWw());
      xMax = TMath::Max(xMax, f2DAxes[i].X() + gPad->GetXlowNDC() * gPad->GetWw());
      yMin = TMath::Min(yMin, fViewport[3] - f2DAxes[i].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]);
      yMax = TMath::Max(yMax, fViewport[3] - f2DAxes[i].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]);
   }

   if (x >= xMin && x <= xMax && y >= yMin && y <= yMax) {
      //Select axes
      SelectAxes(fFrontPoint, x, y);

      return kTRUE;
   }
   
   return kFALSE;
}

namespace {

   Bool_t TestLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x, Double_t y)
   {
      if (x < TMath::Min(x1, x2) - 2 || x > TMath::Max(x1, x2) + 2 ||
          y < TMath::Min(y1, y2) - 2 || y > TMath::Max(y1, y2) + 2)
         return kFALSE;

      const Double_t a = y2 - y1;
      const Double_t b = x1 - x2;
      const Double_t c = -a * x1 -b * y1;
      const Double_t mu = c < 0. ? -1 / TMath::Sqrt(a * a + b * b) : 1 / TMath::Sqrt(a * a + b * b);

      return TMath::Abs(a * mu * x + b * mu * y + c * mu) < 3.;
   }

}

//______________________________________________________________________________
void TGLHistPainter::SelectAxes(Int_t front, Int_t x, Int_t y)const
{
   //Checks, if axis can be selected
   Int_t left = gFramePoints[front][0];
   Double_t xLeft = f2DAxes[left].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yLeft = fViewport[3] - f2DAxes[left].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];
   Double_t xMid = f2DAxes[front].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yMid = fViewport[3] - f2DAxes[front].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];

   if (TMath::Abs(xMid - xLeft) > 0.001 || TMath::Abs(yMid - yLeft) > 0.001) {
      if (TestLine(xLeft, yLeft, xMid, yMid, x, y)) {
         //Left was selected
         if (gAxisType[front][0])
            gPad->SetSelected(fAxisY);
         else
            gPad->SetSelected(fAxisX);

         return;
      }
   }

   Int_t right = gFramePoints[front][1];
   Double_t xRight = f2DAxes[right].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yRight = fViewport[3] - f2DAxes[right].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];

   if (TMath::Abs(xMid - xRight) > 0.001 || TMath::Abs(yMid - yRight) > 0.001) {
      if (TestLine(xMid, yMid, xRight, yRight, x, y)) {
         //Right was selected
         if (gAxisType[front][1])
            gPad->SetSelected(fAxisY);
         else
            gPad->SetSelected(fAxisX);

         return;
      }
   }

   Double_t xUp = f2DAxes[left + 4].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yUp = fViewport[3] - f2DAxes[left + 4].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];

   if (TMath::Abs(xLeft - xUp) > 0.001 || TMath::Abs(yLeft - yUp) > 0.001) {
      if (TestLine(xLeft, yLeft, xUp, yUp, x ,y))
         //Z was selected
         gPad->SetSelected(fAxisZ);
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawZeroPlane()const
{
   //Blue, semi-transparent plane at zero-level
   glEnable(GL_BLEND);//[0
   glDepthMask(GL_FALSE);//[1
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   const Float_t diffColor[] = {0.f, 0.3f, 0.8f, 0.15f};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT, GL_SHININESS, 70.f);

   if (fLastOption == kTF3) {
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffColor);
      glMaterialfv(GL_BACK, GL_SPECULAR, specColor);
      glMaterialf(GL_BACK, GL_SHININESS, 70.f);
   }

   //If abs(fMinZ) less than 1, I can get partially overlapping polys
   if (TMath::Abs(fMinZ) > 1.) {
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glVertex3d(fMinXScaled, fMinYScaled, 0.);
      glVertex3d(fMaxXScaled, fMinYScaled, 0.);
      glVertex3d(fMaxXScaled, fMaxYScaled, 0.);
      glVertex3d(fMinXScaled, fMaxYScaled, 0.);
      glEnd();
   }
      
   glDepthMask(GL_TRUE);//1]
   glDisable(GL_BLEND);//0]
}

//______________________________________________________________________________
void TGLHistPainter::DrawProfile(Int_t plane)const
{
   //Draw "shadows" for lego/surf
   if (fLastOption == kTF3) return;

   //Draws profiles on back planes
   //Profile's color is a mixture of back plane color and gray
   const Float_t color[] = {0.4, 0.4, 0.4, 0.25f};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   //To avoid different visual artefacts,
   //profile is drawn without depth test and
   //without writing into z-buffer
   glEnable(GL_BLEND);//[0
   glDepthMask(GL_FALSE);//[1
   glDisable(GL_DEPTH_TEST);//[2

   if (fLastOption == kLego || fLastOption == kLego2)
      glDisable(GL_CULL_FACE);//[3 To avoid point order checks during bin drawing

   if (!plane || plane == 2)
      fLastOption == kLego || fLastOption == kLego2 ? DrawLegoProfileY(plane) : DrawSurfaceProfileY(plane);
   else
      fLastOption == kLego || fLastOption == kLego2 ? DrawLegoProfileX(plane) : DrawSurfaceProfileX(plane);

   if (fLastOption == kLego || fLastOption == kLego2)
      glEnable(GL_CULL_FACE);//3]
      
   glEnable(GL_DEPTH_TEST);//2]
   glDepthMask(GL_TRUE);//1]
   glDisable(GL_BLEND);//0]
}

namespace {

   typedef std::pair<Double_t, Double_t> PD_t;

   //______________________________________________________________________________
   PD_t GetMaxContent(const TH1 *hist, Int_t firstBin, Int_t lastBin, Int_t colRow, Bool_t row)
   {
      //One column in 2d hist table, find minimum and maximum
      Double_t zMax = row ? hist->GetBinContent(colRow, firstBin) : 
                           hist->GetBinContent(firstBin, colRow);
      Double_t zMin = zMax;
      
      if (row)
         for (Int_t next = firstBin + 1; next <= lastBin; ++next) {
            zMax = TMath::Max(zMax, hist->GetBinContent(colRow, next));
            zMin = TMath::Min(zMin, hist->GetBinContent(colRow, next));
         }
      else
         for (Int_t next = firstBin + 1; next <= lastBin; ++next) {
            zMax = TMath::Max(zMax, hist->GetBinContent(next, colRow));
            zMin = TMath::Min(zMin, hist->GetBinContent(next, colRow));
         }      

      return PD_t(zMin, zMax);
   }

}

//______________________________________________________________________________
void TGLHistPainter::DrawLegoProfileX(Int_t plane)const
{
   //Draws X lego's profile on 'plane'
   //for each 'row' find min and max
   //and draw them as rectangle
   const Int_t nBins = fLastBinX - fFirstBinX + 1;
   const Double_t y = plane == 1 ? fMinYScaled : fMaxYScaled;
   const Double_t normal[] = {0., plane == 1 ? 1. : -1., 0.};

   for (Int_t i = 0, ir = fFirstBinX; i < nBins; ++i, ++ir) {
      PD_t z(GetMaxContent(fHist, fFirstBinY, fLastBinY, ir, kTRUE));

      if (fLogZ) {
         if (z.second <= 0.) continue;

         z.second = TMath::Log10(z.second);

         if (z.first > 0.)
            z.first = TMath::Log10(z.first);
         else
            z.first = 0.;
      }
      
      z.first *= fScaleZ;
      z.second *= fScaleZ;

      if (z.first > 0. && z.second > 0.)
         z.first = 0.;
      
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(fX[i], y, z.first);
      glVertex3d(fX[i], y, z.second);
      glVertex3d(fX[i + 1], y, z.second);
      glVertex3d(fX[i + 1], y, z.first);
      glEnd();
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawLegoProfileY(Int_t plane)const
{
   //Draws Y lego's profile on 'plane'
   //for each 'column' find min and max
   //and draw them as rectangle
   const Int_t nBins = fLastBinY - fFirstBinY + 1;
   const Double_t x = plane ? fMaxXScaled : fMinXScaled;
   const Double_t normal[] = {plane ? -1. : 1., 0., 0.};
   
   for (Int_t i = 0, ir = fFirstBinY; i < nBins; ++i, ++ir) {
      //PD_t z(GetMaxColumnContent(ir));
      PD_t z(GetMaxContent(fHist, fFirstBinX, fLastBinX, ir, kFALSE));
      
      if (fLogZ) {
         if (z.second <= 0.) continue;

         z.second = TMath::Log10(z.second);

         if (z.first > 0.)
            z.first = TMath::Log10(z.first);
         else
            z.first = 0.;
      }
      
      z.first *= fScaleZ;
      z.second *= fScaleZ;

      if (z.second > 0. && z.first > 0)
         z.first = 0.;
   
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(x, fY[i], z.first);
      glVertex3d(x, fY[i + 1], z.first);
      glVertex3d(x, fY[i + 1], z.second);
      glVertex3d(x, fY[i], z.second);
      glEnd();
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawSurfaceProfileX(Int_t plane)const
{
   //Draws X surface's profile on 'plane'
   const Int_t nBins = fLastBinX - fFirstBinX + 1;
   const Double_t y = plane == 1 ? fMinYScaled : fMaxYScaled;
   const Double_t normal[] = {0., plane == 1 ? 1. : -1., 0.};

   for (Int_t i = 0; i < nBins - 1; ++i) {
      Double_t xMin = fMesh[i][0].X();
      Double_t xMax = fMesh[i + 1][0].X();
      Double_t zMax1 = fMesh[i][0].Z();
      Double_t zMin1 = zMax1;
      Double_t zMax2 = fMesh[i + 1][0].Z();
      Double_t zMin2 = zMax2;
      
      for (Int_t j = 1; j < nBins; ++j) {
         zMax1 = TMath::Max(fMesh[i][j].Z(), zMax1);
         zMin1 = TMath::Min(fMesh[i][j].Z(), zMin1);
         zMax2 = TMath::Max(fMesh[i + 1][j].Z(), zMax2);
         zMin2 = TMath::Min(fMesh[i + 1][j].Z(), zMin2);
      }
     
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(xMin, y, zMin1);
      glVertex3d(xMin, y, zMax1);
      glVertex3d(xMax, y, zMax2);
      glVertex3d(xMax, y, zMin2);
      glEnd();
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawSurfaceProfileY(Int_t plane)const
{
   //Draws Y surface's profile on 'plane'
   const Int_t nBins = fLastBinY - fFirstBinY + 1;
   const Double_t x = plane ? fMaxXScaled : fMinXScaled;
   const Double_t normal[] = {plane ? -1. : 1., 0., 0.};

   for (Int_t i = 0; i < nBins - 1; ++i) {
      Double_t yMin = fMesh[0][i].Y();
      Double_t yMax = fMesh[0][i + 1].Y();
      Double_t zMax1 = fMesh[0][i].Z();
      Double_t zMin1 = zMax1;
      Double_t zMax2 = fMesh[0][i + 1].Z();
      Double_t zMin2 = zMax2;
      
      for (Int_t j = 1; j < nBins; ++j) {
         zMax1 = TMath::Max(fMesh[j][i].Z(), zMax1);
         zMin1 = TMath::Min(fMesh[j][i].Z(), zMin1);
         zMax2 = TMath::Max(fMesh[j][i + 1].Z(), zMax2);
         zMin2 = TMath::Min(fMesh[j][i + 1].Z(), zMin2);
      }
     
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(x, yMin, zMin1);
      glVertex3d(x, yMin, zMax1);
      glVertex3d(x, yMax, zMax2);
      glVertex3d(x, yMax, zMin2);
      glEnd();
   }
}

//______________________________________________________________________________
void TGLHistPainter::SetZLevels()
{
   //Define levels for grid
   Double_t zMin = fMinZ > 0. ? 0. : fMinZ;
   Double_t zMax = fMaxZ;//maxZ passes as non-const reference into Optimize, thus can be changed
   Int_t nDiv = fAxisZ->GetNdivisions() % 100;
   Int_t nBins = 0;
   Double_t binLow = 0., binHigh = 0., binWidth = 0.;
   
   THLimitsFinder::Optimize(zMin, zMax, nDiv, binLow, binHigh, nBins, binWidth, " ");
   fZLevels.resize(nBins + 1);
   
   for (Int_t i = 0; i < nBins + 1; ++i)
      fZLevels[i] = binLow + i * binWidth;
}

//______________________________________________________________________________
void TGLHistPainter::DrawGrid(Int_t plane)const
{
   //Grid at XOZ or YOZ back plane
   //Under win32 glPushAttrib does not help with GL_LINE_STIPPLE enable bit
   glPushAttrib(GL_LINE_BIT);//[0
   //Dot lines
   glEnable(GL_LINE_STIPPLE);//[1
   const UShort_t stipple = 0x5555;
   glLineStipple(1, stipple);
   //Do not need depth test
   glDisable(GL_DEPTH_TEST);//[2
   glDisable(GL_LIGHTING);//[3
   glColor3d(0., 0., 0.);
   //Antialias these lines
   glEnable(GL_LINE_SMOOTH);//[4
   glEnable(GL_BLEND);//[5
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      
   Double_t lineCaps[][4] = {
      {fMinXScaled, fMinXScaled, fMinYScaled, fMaxYScaled},
      {fMinXScaled, fMaxXScaled, fMinYScaled, fMinYScaled}, 
      {fMaxXScaled, fMaxXScaled, fMinYScaled, fMaxYScaled},
      {fMinXScaled, fMaxXScaled, fMaxYScaled, fMaxYScaled}
   };

   for (UInt_t i = 0; i < fZLevels.size(); ++i) {
      glBegin(GL_LINES);
      glVertex3d(lineCaps[plane][0], lineCaps[plane][2], fZLevels[i] * fScaleZ);
      glVertex3d(lineCaps[plane][1], lineCaps[plane][3], fZLevels[i] * fScaleZ);
      glEnd();
   }

   glDisable(GL_BLEND);//5]
   glDisable(GL_LINE_SMOOTH);//4]
   glEnable(GL_LIGHTING);//3]
   glEnable(GL_DEPTH_TEST);//2]
   //Under win32 push attrib does not help with GL_LINE_STIPPLE enable bit
   glDisable(GL_LINE_STIPPLE);//1]
   glPopAttrib();//0]
}

//______________________________________________________________________________
void TGLHistPainter::ClearBuffers()const
{
   //Clears gl buffers
   Color_t ci = gPad->GetFillColor();
   TColor *color = gROOT->GetColor(ci);
   Float_t sc[3] = {1.f, 1.f, 1.f};
   
   if (color)
      color->GetRGB(sc[0], sc[1], sc[2]);
   
   glClearColor(sc[0], sc[1], sc[2], 1.);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

namespace {
   const UChar_t gDefTexture1[] =
   {
      //R    G    B    A
      128, 0,   255, 200,
      169, 4,   240, 200,
      199, 73,  255, 200,
      222, 149, 253, 200,
      255, 147, 201, 200,
      255, 47,  151, 200,
      232, 0,   116, 200,
      253, 0,   0,   200,
      255, 62,  62,  200,
      217, 111, 15,  200,
      242, 151, 28,  200,
      245, 172, 73,  200,
      251, 205, 68,  200,
      255, 255, 21,  200,
      255, 255, 128, 200,
      255, 255, 185, 200
   };


   const UChar_t gDefTexture2[] =
   {
      //R    G    B    A
      230, 0,   115, 255,
      255, 62,  158, 255,
      255, 113, 113, 255,
      255, 98,  21,  255,
      255, 143, 89,  255,
      249, 158, 23,  255,
      252, 197, 114, 255,
      252, 228, 148, 255,
      66,  189, 121, 255,
      121, 208, 160, 255,
      89,  245, 37,  255,
      183, 251, 159, 255,
      0,   113, 225, 255,
      64,  159, 255, 255,
      145, 200, 255, 255,
      202, 228, 255, 255
   };
}

void TGLHistPainter::Pan(Int_t x, Int_t y)
{
   // Panning

   Double_t mvMatrix[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
   Double_t prMatrix[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, prMatrix);

   TGLVertex3 start, end;
   gluUnProject(fCurrPos.fX, fCurrPos.fY, 1., mvMatrix, prMatrix, fViewport, &start.X(), &start.Y(), &start.Z());
   gluUnProject(x, y, 1., mvMatrix, prMatrix, fViewport, &end.X(), &end.Y(), &end.Z());
   TGLVector3 delta = start - end;
   fPan = fPan + delta / 2.;
   fCurrPos.fX = x, fCurrPos.fY = y;
//   fFrustum[0] *= 1.2, fFrustum[1] *= 1.2;
   gGLManager->PaintSingleObject(this);
}


//______________________________________________________________________________
void TGLHistPainter::SetTexture()
{
   //Set default texture

   fTexture.resize(sizeof gDefTexture1);
   std::copy(gDefTexture1, gDefTexture1 + sizeof gDefTexture1, fTexture.begin());
}

void TGLHistPainter::SetPlotColor()const
{
   //Set color for lego/surface

   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 1.f};

   if (fLastOption != kLego2 && fLastOption != kSurface1 && fLastOption != kSurface2) {
      if (fHist->GetFillColor() != kWhite)
         if (TColor *c = gROOT->GetColor(fHist->GetFillColor()))
            c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   }

   if (fLastOption == kTF3 && fF3) {
      if (fF3->GetFillColor() != kWhite)
         if (TColor *c = gROOT->GetColor(fF3->GetFillColor()))
            c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   }
   
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT, GL_SHININESS, 70.f);

   if (fLastOption == kTF3) {
      glMaterialfv(GL_BACK, GL_DIFFUSE, diffColor);
      glMaterialfv(GL_BACK, GL_SPECULAR, specColor);
      glMaterialf(GL_BACK, GL_SHININESS, 70.f);
   }
}

void TGLHistPainter::EnableTexture()const
{
   //Enable 1D texture
   glEnable(GL_TEXTURE_1D);
   
   if (!glIsTexture(fTextureName)) {
      glGenTextures(1, &fTextureName);
   }

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glBindTexture(GL_TEXTURE_1D, fTextureName);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, fTexture.size() / 4, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, &fTexture[0]);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
}

void TGLHistPainter::DisableTexture()const
{
   //Disable 1D texture
   glDisable(GL_TEXTURE_1D);
   glDeleteTextures(1, &fTextureName);
}

namespace {

   //______________________________________________________________________________
   Bool_t SetAxisRange(const TAxis *axis, Bool_t log, Int_t &first, Int_t &last,
                       Double_t &min, Double_t &max)
   {
      //Sets-up parameters for X, Y or Z axis
      first = axis->GetFirst();
      last = axis->GetLast();
      min = axis->GetBinLowEdge(first);
      max = axis->GetBinLowEdge(last) + axis->GetBinWidth(last);

      if (log) {
         if (min <= 0.) 
            min = axis->GetBinUpEdge(axis->FindFixBin(0.01 * axis->GetBinWidth(first)));
         if (min <= 0. || max <= 0.)
            return kFALSE;
         Int_t bin = axis->FindFixBin(min);
         if (axis->GetBinLowEdge(bin) <= 0.) ++bin;//crashes under win32
         if (first < bin) first = bin;
         bin = axis->FindFixBin(max);
         if (last > bin) last = bin;
         min = TMath::Log10(min);
         max = TMath::Log10(max);
      }

      return kTRUE;
   }

   //______________________________________________________________________________
   void DrawFlatFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                     const TGLVector3 &norm)
   {
      //Draws triangle flat face, one normal per face
      glBegin(GL_POLYGON);
      glNormal3dv(norm.CArr());
      glVertex3dv(v1.CArr());
      glVertex3dv(v2.CArr());
      glVertex3dv(v3.CArr());
      glEnd();
   }

   //______________________________________________________________________________
   void DrawSmoothFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
                       const TGLVector3 &norm1, const TGLVector3 &norm2, const TGLVector3 &norm3)
   {
      //Draws triangle face, each vertex has its own averaged normal
      glBegin(GL_POLYGON);
      glNormal3dv(norm1.CArr());
      glVertex3dv(v1.CArr());
      glNormal3dv(norm2.CArr());
      glVertex3dv(v2.CArr());
      glNormal3dv(norm3.CArr());
      glVertex3dv(v3.CArr());
      glEnd();
   }

   //______________________________________________________________________________
   void DrawFaceTextured(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                         const TGLVertex3 &v3, const TGLVector3 &norm1,
                         const TGLVector3 &norm2, const TGLVector3 &norm3,
                         Double_t zMin, Double_t zMax)
   {
      //Draws texture triangle
      Double_t zRange = zMax - zMin;

      glBegin(GL_POLYGON);
      glNormal3dv(norm1.CArr());
      glTexCoord1d((v1.Z() - zMin) / zRange);
      glVertex3dv(v1.CArr());
      glNormal3dv(norm2.CArr());
      glTexCoord1d((v2.Z() - zMin) / zRange);
      glVertex3dv(v2.CArr());
      glNormal3dv(norm3.CArr());
      glTexCoord1d((v3.Z() - zMin) / zRange);
      glVertex3dv(v3.CArr());
      glEnd();   
   }

   //______________________________________________________________________________
   void DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                        const TGLVertex3 &v3, const TGLVertex3 &v4)
   {
      //Outline for surf1
      glBegin(GL_LINE_LOOP);
      glVertex3dv(v1.CArr());
      glVertex3dv(v2.CArr());
      glVertex3dv(v3.CArr());
      glVertex3dv(v4.CArr());
      glEnd();
   }

   //______________________________________________________________________________
   void DrawQuadFace(const TGLVertex3 &v0, const TGLVertex3 &v1, const TGLVertex3 &v2,
                     const TGLVertex3 &v3, const TGLVertex3 &normal)
   {
      glBegin(GL_POLYGON);
      glNormal3dv(normal.CArr());
      glVertex3dv(v0.CArr());
      glVertex3dv(v1.CArr());
      glVertex3dv(v2.CArr());
      glVertex3dv(v3.CArr());
      glEnd();
   }
   
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

   //MarchCube performs the Marching Cubes algorithm on a single cube
   void MarchCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY, 
                  Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                  const TF3 *fun, std::vector<RootGL::TGLTriFace_t> &mesh)
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

         using RootGL::TGLTriFace_t;
         TGLTriFace_t newTri;
         
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
