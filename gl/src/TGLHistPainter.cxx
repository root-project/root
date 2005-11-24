#include <algorithm>
#include <iostream>
#include <string>

#ifdef WIN32
#include "Windows4root.h"
#endif

#include <GL/gl.h>
#include <GL/glu.h>

#include "THLimitsFinder.h"
#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGaxis.h"
#include "TString.h"
#include "TPoint.h"
#include "TColor.h"
#include "TROOT.h"
#include "TAxis.h"
#include "TMath.h"
#include "TPad.h"
#include "TH1.h"

#include "TGLHistPainter.h"

ClassImp(TGLHistPainter)

//______________________________________________________________________________
TGLHistPainter::TGLHistPainter(TH1 *hist)
                   : fDefaultPainter(0),
                     fHist(hist),
                     fLastOption(kUnsupported),
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
                     fLogX(kFALSE), fLogY(kFALSE), fLogZ(kFALSE),
                     fGLDevice(-1),
                     f2DPass(kFALSE)
{
   //Each TGLHistPainter has default painter as a member
   //to delegate unsupported calls

   InitDefaultPainter();
}

//______________________________________________________________________________
TGLHistPainter::~TGLHistPainter()
{
   //

   delete fDefaultPainter;
}

//______________________________________________________________________________
Int_t TGLHistPainter::DistancetoPrimitive(Int_t px, Int_t py)
{
   //If fLastOption != kUnsupported, try to select hist or axis.
   //if not - gPad is selected (there are problems with TF2)

   if (fLastOption == kUnsupported)
      return fDefaultPainter->DistancetoPrimitive(px, py);
   else {
      if ((fGLDevice = gPad->GetGLDevice()) == -1) {
         Error("DistancetoPrimitive", "Current pad (gPad) does not have gl device\n");
         return 9999; //TF2 will try to select something itself
      }

      gGLManager->MakeCurrent(fGLDevice);

      if (!Select(px, py))
         gPad->SetSelected(gPad);// To avoid TF2 selection

      return 0;
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawPanel()
{
   //FIX
   
   if (fLastOption == kUnsupported)
      fDefaultPainter->DrawPanel();
}

//______________________________________________________________________________
void TGLHistPainter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   //If FLastOption == kUnsupported, delegate call.
   //If not, try to process itself

   if (fLastOption == kUnsupported)
      fDefaultPainter->ExecuteEvent(event, px, py);
   else {
      //
      py -= Int_t((1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh());
      px -= Int_t(gPad->GetXlowNDC() * gPad->GetWw());

      if ((fGLDevice = gPad->GetGLDevice()) == -1) {
         Error("ExecuteEvent", "current pad (gPad) does not have gl device\n");
         return;
      }

      gGLManager->MakeCurrent(fGLDevice);

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
         gGLManager->Flush(fGLDevice);
         break;
      case kButton1Up:
         gGLManager->MarkForDirectCopy(fGLDevice, kFALSE);
         break;
      case kMouseMotion:
         gPad->SetCursor(kRotate);//Does not work for TF2 (?)
         break;
      }
   }
}

//______________________________________________________________________________
void TGLHistPainter::FitPanel()
{
   //FIX
   
   if (fLastOption == kUnsupported)
      fDefaultPainter->FitPanel();
}

//______________________________________________________________________________
TList *TGLHistPainter::GetContourList(Double_t contour)const
{
   //FIX

   return fDefaultPainter->GetContourList(contour);
}

//______________________________________________________________________________
char *TGLHistPainter::GetObjectInfo(Int_t px, Int_t py)const
{
   //FIX

   return fDefaultPainter->GetObjectInfo(px, py);
}

//______________________________________________________________________________
TList *TGLHistPainter::GetStack()const
{
   //FIX

   return fDefaultPainter->GetStack();
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Int_t x, Int_t y)
{
   //FIX

   if (fLastOption == kUnsupported)
      return fDefaultPainter->IsInside(x, y);
   
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::IsInside(Double_t x, Double_t y)
{
   //FIX

   if (fLastOption == kUnsupported)
      return fDefaultPainter->IsInside(x, y);

   return kFALSE;
}

//______________________________________________________________________________
void TGLHistPainter::PaintStat(Int_t dostat, TF1 *fit)
{
   //FIX

   fDefaultPainter->PaintStat(dostat, fit);
}

//______________________________________________________________________________
void TGLHistPainter::ProcessMessage(const char *mess, const TObject *obj)
{
   //FIX

   if (fLastOption == kUnsupported)
      fDefaultPainter->ProcessMessage(mess, obj);
}

//______________________________________________________________________________
void TGLHistPainter::SetHistogram(TH1 *hist)
{
   //FIX

   fHist = hist;
   fDefaultPainter->SetHistogram(hist);
}

//______________________________________________________________________________
void TGLHistPainter::SetStack(TList *stack)
{
   //FIX
   
   if (fLastOption == kUnsupported)
      fDefaultPainter->SetStack(stack);
}

//______________________________________________________________________________
Int_t TGLHistPainter::MakeCuts(char *cutsOpt)
{
   //FIX

   return fDefaultPainter->MakeCuts(cutsOpt);
}

//______________________________________________________________________________
void TGLHistPainter::InitDefaultPainter()
{
   //FIX
   
   fDefaultPainter = TVirtualHistPainter::HistPainter(fHist);
}

//______________________________________________________________________________
void TGLHistPainter::Paint(Option_t *o)
{
   //Final-overrider for TOvject's Paint, checks, if 
   //painter can draw itself or should pass to default painter

   if (f2DPass) return;

   TString option(o);
   option.ToLower();

   if ((fLastOption = GetPaintOption(option)) == kUnsupported) {
      gPad->SetCopyGLDevice(kFALSE);
      fDefaultPainter->Paint(o);
   } else {
      
      fGLDevice = gPad->GetGLDevice();

      if (fGLDevice != -1) {
         if (!InitPainter()) return;

         gGLManager->SelectGLPixmap(fGLDevice);
         gGLManager->MakeCurrent(fGLDevice);
         gGLManager->PaintSingleObject(this);
         //to deselect pixmap from Win32 DC
         //gGLManager->GetVirtualXInd(fGLDevice);
         gVirtualX->SelectWindow(gPad->GetPixmapID());
      }
      else
         fDefaultPainter->Paint(o);
   }
}

//______________________________________________________________________________
void TGLHistPainter::Paint()
{
   //Paint method, which is indirectly called by gGLManager
   //at this moment, MakeCurrent must be done already for fGLDevice

   gGLManager->SelectGLPixmap(fGLDevice);
   gPad->SetCopyGLDevice(kTRUE);

   InitGL();
   SetGLParameters();
   ClearBuffer();

   switch (fLastOption) {
   case kLego :
      PaintLego();
      break;
   case kSurface:
      PaintSurface();
      break;
   case kSurface4:
      PaintSurface4();
      break;
   default:;//to shut up g++
   }
}

//______________________________________________________________________________
TGLHistPainter::EGLPaintOption TGLHistPainter::GetPaintOption(const TString &o)
{
   //Check, if Paint's option is supported

   std::string option(o.Data());
   std::string::size_type start = option.find("lego");

   if (start != std::string::npos)
      if (option.length() == 5 && option[4] == '1' || option.length() == 4)
         return kLego;
   
   start = option.find("surf");
   
   if (start != std::string::npos)
      if (option.length() == 5 && option[4] == '4')
         return kSurface4;
      else if(option.length() == 4)
         return kSurface;
         
   return kUnsupported;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::InitPainter()
{
   //It's clear :)

   if (!SetSizes())
      return kFALSE;

   FillVertices();

   if (fLastOption == kSurface)
      SetNormals();
   else if (fLastOption == kSurface4)
      SetAverageNormals();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::SetSizes()
{
   //Having TH1 pointer, setup min/max sizes and scales

   fLogX = gPad->GetLogx();
   if (!SetAxisRange(fAxisX, fLogX, fFirstBinX, fLastBinX, fMinX, fMaxX)) {
      Error("SetSizes", "cannot set X axis to log scale\n");
      return kFALSE;
   }

   fLogY = gPad->GetLogy();
   if (!SetAxisRange(fAxisY, fLogY, fFirstBinY, fLastBinY, fMinY, fMaxY)) {
      Error("SetSizes", "cannot set Y axis to log scale\n");
      return kFALSE;
   }

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
   if (fMinZ >= fMaxZ) fMinZ = 0.001 * fMaxZ;//NEW

   fLogZ = gPad->GetLogz();

   if (fLogZ && fMaxZ <= 0) {
      Error("SetSizes", "log scale is requested for Z, but maximum less or equal 0 (%f)", fMaxZ);
      return kFALSE;
   }
/*//NEW COMMENTED
   if (fMinZ >= fMaxZ && fLogZ)
      if (fMaxZ > 0.) 
         fMinZ = 0.001 * fMaxZ;
      else {
         Error("SetSizes", "log scale is requested for Z, but maximum less or equal 0 (%f)", fMaxZ);
         return kFALSE;
   }
*/
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
 //     if (minimum) fMinZ += TMath::Log10(0.5); // ???
      fMaxZ = TMath::Log10(fMaxZ);
  //    if (maximum) fMaxZ += TMath::Log10(2 * (0.9 / 0.95)); //???
      
      if (positiveMin > 0.)
         fMinZ = TMath::Min(TMath::Log10(positiveMin), fMinZ);
   }

   SetZLevels();
   AdjustScales();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLHistPainter::SetAxisRange(const TAxis *axis, Bool_t log, Int_t &first, Int_t &last,
                                          Double_t &min, Double_t &max)
{
   //Sets-up parameters for X or Y axis

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
void TGLHistPainter::AdjustScales()
{
   //Finds the maximum dimension and adjust scale coefficients

   Double_t xRange = fMaxX - fMinX;
   Double_t yRange = fMaxY - fMinY;
   Double_t zRange = fMinZ > 0. ? fMaxZ : fMaxZ - fMinZ;

   Double_t maxDim = TMath::Max(TMath::Max(xRange, yRange), zRange);
   
   fScaleX = maxDim / xRange;
   fScaleY = maxDim / yRange;
   fScaleZ = maxDim / zRange / 1.5;
   fMinXScaled = fMinX * fScaleX;
   fMaxXScaled = fMaxX * fScaleX;
   fMinYScaled = fMinY * fScaleY;
   fMaxYScaled = fMaxY * fScaleY;
   fMinZScaled = fMinZ * fScaleZ;
   fMaxZScaled = fMaxZ * fScaleZ;
}

//______________________________________________________________________________
void TGLHistPainter::FillVertices()
{
   //Calculates table of X and Y for lego (Z is obtained during drawing) or
   //calculate mesh of triangles with vertices in the centres of bins

   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;

   if (fLastOption == kLego) {
      fTable.resize((nX + 1) * (nY + 1));
      fTable.SetRowLen(nY + 1);

      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
            if (fLogX) fTable[i][j].first = TMath::Log10(fAxisX->GetBinLowEdge(ir)) * fScaleX;
            else fTable[i][j].first = fAxisX->GetBinLowEdge(ir) * fScaleX;
            if (fLogY) fTable[i][j].second = TMath::Log10(fAxisY->GetBinLowEdge(jr)) * fScaleY;
            else fTable[i][j].second = fAxisY->GetBinLowEdge(jr) * fScaleY;
         }

      Double_t maxX = fAxisX->GetBinLowEdge(fLastBinX) + fAxisX->GetBinWidth(fLastBinX);

      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
         if (fLogX) fTable[nX][j].first = TMath::Log10(maxX) * fScaleX;
         else fTable[nX][j].first = maxX * fScaleX;
         if (fLogY) fTable[nX][j].second = TMath::Log10(fAxisY->GetBinLowEdge(jr)) * fScaleY;
         else fTable[nX][j].second = fAxisY->GetBinLowEdge(jr) * fScaleY;
      }

      Double_t maxY = fAxisY->GetBinLowEdge(fLastBinY) + fAxisY->GetBinWidth(fLastBinY);

      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir) {
         if (fLogX) fTable[i][nY].first = TMath::Log10(fAxisX->GetBinLowEdge(ir)) * fScaleX;
         else fTable[i][nY].first = fAxisX->GetBinLowEdge(ir) * fScaleX;
         if (fLogY) fTable[i][nY].second = TMath::Log10(maxY) * fScaleY;
         else fTable[i][nY].second = maxY * fScaleY;
      }

      if (fLogX) fTable[nX][nY].first = TMath::Log10(maxX) * fScaleX;
      else fTable[nX][nY].first = maxX * fScaleX;
      if (fLogY) fTable[nX][nY].second = TMath::Log10(maxY) * fScaleY;
      else fTable[nX][nY].second = maxY * fScaleY;
   } else {
      fMesh.resize(nX * nY);
      fMesh.SetRowLen(nY);

      for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
         for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
            if (fLogX) fMesh[i][j].X() = TMath::Log10(fAxisX->GetBinCenter(ir)) * fScaleX;
            else fMesh[i][j].X() = fAxisX->GetBinCenter(ir) * fScaleX;
            if (fLogY) fMesh[i][j].Y() = TMath::Log10(fAxisY->GetBinCenter(jr)) * fScaleY;
            else fMesh[i][j].Y() = fAxisY->GetBinCenter(jr) * fScaleY;

            Double_t z = fHist->GetCellContent(ir, jr);
            
            if (fLogZ) {
            //bizarre
               if (z <= 0)
                  fMesh[i][j].Z() = fMinZ * fScaleZ;
               else
                  fMesh[i][j].Z() = TMath::Log10(z) * fScaleZ;
            } else
               fMesh[i][j].Z() = z * fScaleZ;
         }
   }
}

//______________________________________________________________________________
void TGLHistPainter::SetNormals()
{
   //Calculates normals for triangles in surface.
   //"flat" normals == 1 normal per triangle
   //we have : four points (cell contents of four neighbouring hist bins)
   //but only three points are shurely in one plane, so build 2 triangles and their normals
   
   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;
   
   fFaceNormals.resize((nX - 1) * (nY - 1));
   fFaceNormals.SetRowLen(nY - 1);

   for (Int_t i = 0; i < nX - 1; ++i)
      for (Int_t j = 0; j < nY - 1; ++j) {
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
   //Calculate averaged normals.
   //"averaged" normals == normal per vertex
   //This normal is average of
   //neighbouring triangles normals
   
   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;
   
   fFaceNormals.resize((nX + 1) * (nY + 1));
   fFaceNormals.SetRowLen(nY + 1);
   
   //first, calculate normal for each triangle face
   for (Int_t i = 0; i < nX - 1; ++i)
      for (Int_t j = 0; j < nY - 1; ++j) {
         //second "top-right" triangle
         TMath::Normal2Plane(fMesh[i][j + 1].CArr(), fMesh[i][j].CArr(), fMesh[i + 1][j].CArr(),
                             fFaceNormals[i + 1][j + 1].first.Arr());
         //second "top-right" triangle
         TMath::Normal2Plane(fMesh[i + 1][j].CArr(), fMesh[i + 1][j + 1].CArr(), fMesh[i][j + 1].CArr(),
                             fFaceNormals[i + 1][j + 1].second.Arr());
      }
      
   fAverageNormals.resize(nX * nY);
   fAverageNormals.SetRowLen(nY);
      
   //second, lets calculate average normal for each vertex
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
   //Simple gl initialization
   
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   if (fLastOption == kLego)//with lego we should avoid back faces (or lego is really sloooow)
      glEnable(GL_CULL_FACE), glCullFace(GL_BACK);
   else //for surface we cannot cull faces, because we can look at the surface "from bottom"
      glDisable(GL_CULL_FACE);
}

//______________________________________________________________________________
void TGLHistPainter::PaintLego()const
{
   //Draws lego and "profiles" on the back planes

   SetCamera();
   //main light
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);

   SetTransformation();
   
   Int_t fp = FrontPoint();
   DrawFrame(fp);

   Float_t difColor[] = {0.8f, 0.8f, 0.8f, 1.f};

   if (fHist->GetFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(difColor[0], difColor[1], difColor[2]);
   
   //save material properties in stack
   glPushAttrib(GL_LIGHTING_BIT);
   
   glMaterialfv(GL_FRONT, GL_DIFFUSE, difColor);
   const Float_t specColor[] = {0.5f, 0.f, 0.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT, GL_SHININESS, 70.f);
   //
   //cycle through table
   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;

   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.f, 1.f);

   for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
         Double_t xMin = fTable[i][j].first;
         Double_t xMax = fTable[i + 1][j].first;
         Double_t yMin = fTable[i][j].second;
         Double_t yMax = fTable[i][j + 1].second;

         Double_t zMin = 0.;//fMinZ * fScaleZ;
         Double_t zMax = fHist->GetCellContent(ir, jr);

         if (fLogZ)
            if (zMax <= 0.)
               continue;//I simply ignore this bizzare situation
            else 
               zMax = TMath::Log10(zMax) * fScaleZ;
         else zMax *= fScaleZ;


         DrawBox(xMin, xMax, yMin, yMax, zMin, zMax, fp);
      }

   glDisable(GL_POLYGON_OFFSET_FILL);

   //outlines cycle
   glDisable(GL_LIGHTING);
   glColor3d(0., 0., 0.);
   glPolygonMode(GL_FRONT, GL_LINE);

   for (Int_t i = 0, ir = fFirstBinX; i < nX; ++i, ++ir)
      for (Int_t j = 0, jr = fFirstBinY; j < nY; ++j, ++jr) {
         Double_t xMin = fTable[i][j].first;
         Double_t xMax = fTable[i + 1][j].first;
         Double_t yMin = fTable[i][j].second;
         Double_t yMax = fTable[i][j + 1].second;

         Double_t zMin = 0.;//fMinZ * fScaleZ;
         Double_t zMax = fHist->GetCellContent(ir, jr);

         if (fLogZ)
            if (zMax <= 0.)
               continue;//I simply ignore this bizzare situation
            else 
               zMax = TMath::Log10(zMax) * fScaleZ;
         else zMax *= fScaleZ;

         
         DrawBox(xMin, xMax, yMin, yMax, zMin, zMax, fp);
      }

   glPolygonMode(GL_FRONT, GL_FILL);
   glEnable(GL_LIGHTING);

   //restore material properties from stack
   glPopAttrib();

   DrawZeroPlane();

   glFlush();
   //now, gl drawing is finished, axes are drawn by TVirtualX
   DrawAxes(fp);
}

//______________________________________________________________________________
void TGLHistPainter::PaintSurface()const
{
   //Draws surface as a set of triangles, each triangle has one normal

   SetCamera();
   //main light
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);

   SetTransformation();

   Int_t fp = FrontPoint();
   DrawFrame(fp);

   Float_t color[] = {0.8f, 0.8f, 0.8f, 1.f};

   if (fHist->GetFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(color[0], color[1], color[2]);

   //save material properties in stack
   glPushAttrib(GL_LIGHTING_BIT);
   
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color);
   const Float_t spec[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
   glMaterialf(GL_FRONT, GL_SHININESS, 80.);

   //cycle through table
   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;

   for (Int_t i = 0; i < nX - 1; ++i)
      for (Int_t j = 0; j < nY - 1; ++j) {
         DrawFlatFace(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1], fFaceNormals[i][j].first);
         DrawFlatFace(fMesh[i][j + 1], fMesh[i + 1][j + 1], fMesh[i + 1][j], fFaceNormals[i][j].second);
      }

   //restore material properties from stack
   glPopAttrib();
   
   DrawZeroPlane();

   glFlush();
   DrawAxes(fp);
}

//______________________________________________________________________________
void TGLHistPainter::PaintSurface4()const
{
   //Draws surface with "averaged" normals
   //If you have nearly smooth surface, it will be smooth :)
   
   SetCamera();
   //main light
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);

   SetTransformation();

   Int_t fp = FrontPoint();
   DrawFrame(fp);

   Float_t color[] = {0.8f, 0.8f, 0.8f, 1.f};

   if (fHist->GetFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(color[0], color[1], color[2]);

   //save material properties in stack
   glPushAttrib(GL_LIGHTING_BIT);
   
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color);
   const Float_t spec[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
   glMaterialf(GL_FRONT, GL_SHININESS, 80.);
   //cycle through table
   Int_t nX = fLastBinX - fFirstBinX + 1;
   Int_t nY = fLastBinY - fFirstBinY + 1;

   for (Int_t i = 0; i < nX - 1; ++i)
      for (Int_t j = 0; j < nY - 1; ++j) {
         DrawFace(fMesh[i + 1][j], fMesh[i][j], fMesh[i][j + 1],
                  fAverageNormals[i + 1][j], fAverageNormals[i][j], fAverageNormals[i][j + 1]);
         DrawFace(fMesh[i][j + 1], fMesh[i + 1][j + 1], fMesh[i + 1][j], 
                  fAverageNormals[i][j + 1], fAverageNormals[i + 1][j + 1], fAverageNormals[i + 1][j]);
      }

   //restore material properties from stack
   glPopAttrib();
   
   DrawZeroPlane();

   glFlush();
   DrawAxes(fp);
}

//______________________________________________________________________________
void TGLHistPainter::SetGLParameters()
{
   //Sets viewport, bounds for arcball
   //Calculates arguments for glOrtho
   //Claculates center of scene and shift

   gGLManager->ExtractViewport(fGLDevice, fViewport);
   fRotation.SetBounds(fViewport[2], fViewport[3]);
   glViewport(fViewport[0], fViewport[1], fViewport[2], fViewport[3]);

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

   fFrustum[0] = maxDim / 1.15 * frx;
   fFrustum[1] = maxDim / 1.15 * fry;
   fFrustum[2] = maxDim * 0.707;
   fFrustum[3] = 3 * maxDim;
   fShift = maxDim * 1.7;
}

//______________________________________________________________________________
void TGLHistPainter::DrawBox(Double_t xmin, Double_t xmax, Double_t ymin, 
                             Double_t ymax, Double_t zmin, Double_t zmax, Int_t fp)
{
   //Draws lego's bar as 3d box
   
   if (zmax < zmin) 
      std::swap(zmax, zmin);

   //top and bottom are always drawn (though I can skip one them)
   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glVertex3d(xmax, ymin, zmax);
   glVertex3d(xmax, ymax, zmax);
   glVertex3d(xmin, ymax, zmax);
   glVertex3d(xmin, ymin, zmax);
   glEnd();

   glBegin(GL_POLYGON);
   glNormal3d(0., 0., -1.);
   glVertex3d(xmax, ymin, zmin);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymax, zmin);
   glVertex3d(xmax, ymax, zmin);
   glEnd();

   //two back faces cannot be seen, I know, which point is front, so I can
   //escape passing to polygons to gl.
   //For example : plane 0, I draw it only if front point is 3 or 0
   /*
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

   switch (fp) {
   case 0://0 is the front point, draw planes 0 and 1
      glBegin(GL_POLYGON);
      glNormal3d(-1., 0., 0.);//plane 0
      glVertex3d(xmin, ymin, zmax);
      glVertex3d(xmin, ymax, zmax);
      glVertex3d(xmin, ymax, zmin);
      glVertex3d(xmin, ymin, zmin);
      glEnd();
      glBegin(GL_POLYGON);
      glNormal3d(0., -1., 0.);//plane 1
      glVertex3d(xmax, ymin, zmax);
      glVertex3d(xmin, ymin, zmax);
      glVertex3d(xmin, ymin, zmin);
      glVertex3d(xmax, ymin, zmin);
      glEnd();
      break;
   case 1://1 is the front point, draw planes 1 and 2
      glBegin(GL_POLYGON);
      glNormal3d(0., -1., 0.);//plane 1
      glVertex3d(xmax, ymin, zmax);
      glVertex3d(xmin, ymin, zmax);
      glVertex3d(xmin, ymin, zmin);
      glVertex3d(xmax, ymin, zmin);
      glEnd();
      glBegin(GL_POLYGON);
      glNormal3d(1., 0., 0.);//plane 2
      glVertex3d(xmax, ymin, zmax);
      glVertex3d(xmax, ymin, zmin);
      glVertex3d(xmax, ymax, zmin);
      glVertex3d(xmax, ymax, zmax);
      glEnd();
      break;
   case 2://2 is the front point, draw planes 2 and 3
      glBegin(GL_POLYGON);
      glNormal3d(1., 0., 0.);//plane 2
      glVertex3d(xmax, ymin, zmax);
      glVertex3d(xmax, ymin, zmin);
      glVertex3d(xmax, ymax, zmin);
      glVertex3d(xmax, ymax, zmax);
      glEnd();
      glBegin(GL_POLYGON);
      glNormal3d(0., 1., 0.);//plane 3
      glVertex3d(xmax, ymax, zmax);
      glVertex3d(xmax, ymax, zmin);
      glVertex3d(xmin, ymax, zmin);
      glVertex3d(xmin, ymax, zmax);
      glEnd();
      break;
   case 3://3 is the front point, draw planes 3 and 0
      glBegin(GL_POLYGON);
      glNormal3d(-1., 0., 0.);//plane 0
      glVertex3d(xmin, ymin, zmax);
      glVertex3d(xmin, ymax, zmax);
      glVertex3d(xmin, ymax, zmin);
      glVertex3d(xmin, ymin, zmin);
      glEnd();
      glBegin(GL_POLYGON);
      glNormal3d(0., 1., 0.);//plane 3
      glVertex3d(xmax, ymax, zmax);
      glVertex3d(xmax, ymax, zmin);
      glVertex3d(xmin, ymax, zmin);
      glVertex3d(xmin, ymax, zmax);
      glEnd();
      break;
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawFlatFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
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
void TGLHistPainter::DrawFace(const TGLVertex3 &v1, const TGLVertex3 &v2, const TGLVertex3 &v3,
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
void TGLHistPainter::DrawFrame(Int_t frontPoint)const
{
   //Draws frame box around histogramm, surface or surface4,
   //draws grids and "profiles" for lego

   //save material properties in stack
   glPushAttrib(GL_LIGHTING_BIT);
   
   //back planes and bottom are always drawn first, in fact, I do not need depth test.
   //Planes are 85% opaque to make their color "softer"
   glEnable(GL_BLEND);
   glDepthMask(GL_FALSE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   Float_t backColor[] = {0.9f, 0.9f, 0.9f, 0.85f};

   if (gPad->GetFrameFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(gPad->GetFrameFillColor()))
         c->GetRGB(backColor[0], backColor[1], backColor[2]);

   glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);

   //First, bottom plane at the minimum;
   //draw it with offset to remove "artefacts" when have "near-zero" overlapping polygons 
   //(such polygons can be in lego, surface or it can be zero-plane itself)
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(2.f, 2.f);//offset is 2., because lego uses offset 1

   Double_t zMin = fMinZ > 0. ? 0. : fMinZScaled;

   glBegin(GL_POLYGON);
   glNormal3d(0., 0., 1.);
   glVertex3d(fMinXScaled, fMinYScaled, zMin);
   glVertex3d(fMaxXScaled, fMinYScaled, zMin);
   glVertex3d(fMaxXScaled, fMaxYScaled, zMin);
   glVertex3d(fMinXScaled, fMaxYScaled, zMin);
   glEnd();

   glDisable(GL_POLYGON_OFFSET_FILL);
   
/*
      Each front point has two opposite back planes 
      
              |       |
              |   0   |
              |      3|
           |1 0----|--3
           | / 2   | /
           |/      |/
           1-------2
      For example in backPlanes 2d array first subarray holds numbers of opposite planes:
      
      Point 0 has 3 and 2, 1 - 0 and 3 etc.
*/
   Int_t backPlanes[][2] = {{3, 2}, {0, 3}, {1, 0}, {2, 1}};
   //DrawBackPlane should draw frame's part and corresponding profile for lego
   DrawBackPlane(backPlanes[frontPoint][0]);
   glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);
   DrawBackPlane(backPlanes[frontPoint][1]);

   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);
   
   //restore material properties
   glPopAttrib();
}

//______________________________________________________________________________
void TGLHistPainter::SetCamera()const
{
   //Clears gl buffer, sets projection
/*   
   Color_t ci = gPad->GetFillColor();
   TColor *color = gROOT->GetColor(ci);
   Float_t sc[3] = {1.f, 1.f, 1.f};
   
   if (color)
      color->GetRGB(sc[0], sc[1], sc[2]);
   
   glClearColor(sc[0], sc[1], sc[2], 1.);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
*/
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(-fFrustum[0], fFrustum[0], -fFrustum[1], fFrustum[1], fFrustum[2], fFrustum[3]);

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

   TGLVertex3 v0, v1, v2, v3;
   TGLVector3 normal;
   
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

   glBegin(GL_POLYGON);
   glNormal3dv(normal.CArr());
   glVertex3dv(v0.CArr());
   glVertex3dv(v1.CArr());
   glVertex3dv(v2.CArr());
   glVertex3dv(v3.CArr());
   glEnd();

   //try to antialias back plane outline
   glEnable(GL_LINE_SMOOTH);
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);


   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glColor3d(0., 0., 0.);
   glBegin(GL_LINE_LOOP);
   glVertex3dv(v0.CArr());
   glVertex3dv(v1.CArr());
   glVertex3dv(v2.CArr());
   glVertex3dv(v3.CArr());
   glEnd();
   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);

   glDisable(GL_BLEND);
   glDisable(GL_LINE_SMOOTH);
   
//   if (fLastOption == kLego)
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
   
   //each point has two "neighbouring axes" (left and right). Axes types are 1 (ordinata) and 0 (abscissa)
   const Int_t gAxisType[][2] = {{1, 0}, {0, 1}, {1, 0}, {0, 1}};
}

//______________________________________________________________________________
void TGLHistPainter::DrawAxes(Int_t frontPoint)const
{
   //Using front point, find, where to draw axes and which labels to use for them

   Int_t pixmap = gGLManager->GetVirtualXInd(fGLDevice);
   gVirtualX->SelectWindow(pixmap);
   gVirtualX->SetDrawMode(TVirtualX::kCopy);//TCanvas by default sets in kInverse

   Int_t left = gFramePoints[frontPoint][0];
   Int_t right = gFramePoints[frontPoint][1];
   
   //Now, only 2D coors to draw axes

   Double_t xLeft = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[left].X()));
   Double_t yLeft = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[left].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));
   
   Double_t xMid = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[frontPoint].X()));
   Double_t yMid = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[frontPoint].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));

   Double_t xRight = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[right].X()));
   Double_t yRight = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[right].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));
   
   const Double_t points[][2] = {{fMinX, fMinY}, {fMaxX, fMinY}, {fMaxX, fMaxY}, {fMinX, fMaxY}};
   Int_t leftType = gAxisType[frontPoint][0];
   Int_t rightType = gAxisType[frontPoint][1];
   Double_t leftLabel = points[left][leftType];
   Double_t leftMidLabel = points[frontPoint][leftType];
   Double_t rightMidLabel = points[frontPoint][rightType];
   Double_t rightLabel = points[right][rightType];

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
    
   Double_t xUp = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() + f2DAxes[left + 4].X()));
   Double_t yUp = gPad->AbsPixeltoY(Int_t(fViewport[3] - f2DAxes[left + 4].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1]));

   Draw2DAxis(fAxisZ, xLeft, yLeft, xUp, yUp, fMinZ, fMaxZ, fLogZ, kTRUE);

   f2DPass = kTRUE;
/*
   TObjOptLink *lnk = static_cast<TObjOptLink *>(gPad->GetListOfPrimitives()->FirstLink());
   
   while (lnk) {
      TObject *obj = lnk->GetObject();

      obj->Paint(lnk->GetOption());
      lnk = static_cast<TObjOptLink *>(lnk->Next());
   }*/
   
   f2DPass = kFALSE;

   gGLManager->SelectGLPixmap(fGLDevice);
}

//______________________________________________________________________________
Bool_t TGLHistPainter::Select(Int_t x, Int_t y)const
{
   //find hist "square" on screen
   SetCamera();
   SetTransformation();
   Int_t frontPoint = FrontPoint();
   
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
      //select axes
      SelectAxes(frontPoint, x, y);
      return kTRUE;
   }
   
   return kFALSE;
}

namespace {
   Bool_t TestLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t x, Double_t y)
   {
      Double_t xMin = TMath::Min(x1, x2);
      Double_t xMax = TMath::Max(x1, x2);
      Double_t yMin = TMath::Min(y1, y2);
      Double_t yMax = TMath::Max(y1, y2);

      if (x < xMin - 2 || x > xMax + 2 || y < yMin - 2 || y > yMax + 2)
         return kFALSE;

      Double_t a = y2 - y1;
      Double_t b = x1 - x2;
      Double_t c = -a * x1 -b * y1;
      Double_t mu = 1 / TMath::Sqrt(a * a + b * b);

      if (c < 0.) mu *= -1.;
      
      Double_t r = TMath::Abs(a * mu * x + b * mu * y + c * mu);

      return r < 3.;
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
      //left was selected
         if (gAxisType[front][0]) gPad->SetSelected(fAxisY);
         else gPad->SetSelected(fAxisX);
         return;
      }
   }

   Int_t right = gFramePoints[front][1];
   Double_t xRight = f2DAxes[right].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yRight = fViewport[3] - f2DAxes[right].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];

   if (TMath::Abs(xMid - xRight) > 0.001 || TMath::Abs(yMid - yRight) > 0.001) {
      if (TestLine(xMid, yMid, xRight, yRight, x, y)) {
      //right was selected
         if (gAxisType[front][1]) gPad->SetSelected(fAxisY);
         else gPad->SetSelected(fAxisX);
         return;
      }
   }

   Double_t xUp = f2DAxes[left + 4].X() + gPad->GetXlowNDC() * gPad->GetWw();
   Double_t yUp = fViewport[3] - f2DAxes[left + 4].Y() + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() + fViewport[1];

   if (TMath::Abs(xLeft - xUp) > 0.001 || TMath::Abs(yLeft - yUp) > 0.001) {
      if (TestLine(xLeft, yLeft, xUp, yUp, x ,y))
      //left was selected
         gPad->SetSelected(fAxisZ);
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawZeroPlane()const
{
   //Blue, semi-transparent plane at zero-level
   
   glPushAttrib(GL_LIGHTING_BIT);

   glEnable(GL_BLEND);
   glDepthMask(GL_FALSE);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   const Float_t color[] = {0.f, 0.3f, 0.8f, 0.15f};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color);

   //if abs(fMinZ) less than 1, I can get partially overlapping polys
   if (TMath::Abs(fMinZ) > 1.) {
      glBegin(GL_POLYGON);
      glNormal3d(0., 0., 1.);
      glVertex3d(fMinXScaled, fMinYScaled, 0.);
      glVertex3d(fMaxXScaled, fMinYScaled, 0.);
      glVertex3d(fMaxXScaled, fMaxYScaled, 0.);
      glVertex3d(fMinXScaled, fMaxYScaled, 0.);
      glEnd();
   }
      
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);
   
   glPopAttrib();
}

//______________________________________________________________________________
void TGLHistPainter::DrawProfile(Int_t plane)const
{
   //Draws profiles on back planes

   //profile's color is a mixture of back plane color and gray
   const Float_t color[] = {0.4, 0.4, 0.4, 0.25f};
   glMaterialfv(GL_FRONT, GL_DIFFUSE, color);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   
   //to avoid different visual artefacts,
   //profile is drawn without depth test and
   //without z-buffer
   glEnable(GL_BLEND);
   glDepthMask(GL_FALSE);
   glDisable(GL_DEPTH_TEST);
   if (fLastOption == kLego)
      glDisable(GL_CULL_FACE);//to avoid point order checks during bin drawing

   if (!plane || plane == 2)
      fLastOption == kLego ? DrawLegoProfileY(plane) : DrawSurfaceProfileY(plane);
   else
      fLastOption == kLego ? DrawLegoProfileX(plane) : DrawSurfaceProfileX(plane);

   if (fLastOption == kLego)
      glEnable(GL_CULL_FACE);
   glEnable(GL_DEPTH_TEST);
   glDepthMask(GL_TRUE);
   glDisable(GL_BLEND);
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
      Double_t xMin = fTable[i][0].first;
      Double_t xMax = fTable[i + 1][0].first;
      PD_t z = GetMaxRowContent(ir);
      //MAXROW      
      if (fLogZ) {
         if (z.second <= 0.) continue;
         z.second = TMath::Log10(z.second);
         if (z.first > 0.)
            z.first = TMath::Log10(z.first);
         else z.first = 0.;
      }
      
      z.first *= fScaleZ;
      z.second *= fScaleZ;

      if (z.first > 0. && z.second > 0.) z.first = 0.;
      
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(xMin, y, z.first);
      glVertex3d(xMin, y, z.second);
      glVertex3d(xMax, y, z.second);
      glVertex3d(xMax, y, z.first);
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
      Double_t yMin = fTable[0][i].second;
      Double_t yMax = fTable[0][i + 1].second;
      PD_t z = GetMaxColumnContent(ir);
      //MAXCOL
      if (fLogZ) {
         if (z.second <= 0.) continue;
         z.second = TMath::Log10(z.second);
         if (z.first > 0.)
            z.first = TMath::Log10(z.first);
         else z.first = 0.;
      }
      
      z.first *= fScaleZ;
      z.second *= fScaleZ;

      if (z.second > 0. && z.first > 0) z.first = 0.;
   
      glBegin(GL_POLYGON);
      glNormal3dv(normal);
      glVertex3d(x, yMin, z.first);
      glVertex3d(x, yMax, z.first);
      glVertex3d(x, yMax, z.second);
      glVertex3d(x, yMin, z.second);
      glEnd();
   }
}

//______________________________________________________________________________
void TGLHistPainter::DrawSurfaceProfileX(Int_t plane)const
{
   //Draws X surface's profile on 'plane'
   //for each 'row' find min and max
   //and draw them as rectangle

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
   //for each 'row' find min and max
   //and draw them as rectangle

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
TGLHistPainter::PD_t TGLHistPainter::GetMaxRowContent(Int_t row)const
{
   //Having one row in 2d hist, find minimum and maximum

   Double_t zMax = fHist->GetBinContent(row, fFirstBinY);
   Double_t zMin = zMax;      

   for (Int_t next = fFirstBinY + 1; next <= fLastBinY; ++next) {
      zMax = TMath::Max(zMax, fHist->GetBinContent(row, next));
      zMin = TMath::Min(zMin, fHist->GetBinContent(row, next));
   }
      
   return std::make_pair(zMin, zMax);
}

//______________________________________________________________________________
TGLHistPainter::PD_t TGLHistPainter::GetMaxColumnContent(Int_t col)const
{
   //Having one column in 2d hist, find minimum and maximum

   Double_t zMax = fHist->GetBinContent(fFirstBinX, col);
   Double_t zMin = zMax;
   
   for (Int_t next = fFirstBinX + 1; next <= fLastBinX; ++next) {
      zMax = TMath::Max(zMax, fHist->GetBinContent(next, col));
      zMin = TMath::Min(zMin, fHist->GetBinContent(next, col));
   }
      
   return std::make_pair(zMin, zMax);
}

//______________________________________________________________________________
void TGLHistPainter::SetZLevels()
{
   //Levels for grid

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

   //First, save current line stipple state
   //to avoid problems somewhere
   glPushAttrib(GL_LINE_BIT);

   glEnable(GL_LINE_STIPPLE);
   const UShort_t stipple = 0x5555;
   glLineStipple(1, stipple);

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glColor3d(0., 0., 0.);
   
   //try to anti-alias these lines
   glEnable(GL_LINE_SMOOTH);
   glEnable(GL_BLEND);
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

   //disable AA
   glDisable(GL_BLEND);
   glDisable(GL_LINE_SMOOTH);   

   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   
   glPopAttrib();
}

void TGLHistPainter::ClearBuffer()const
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
