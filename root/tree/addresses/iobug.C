#include "TFile.h"
#include "TTree.h"
#include "TClonesArray.h"

#include "TH1F.h"

class Graph : public TNamed, public TAttLine, public TAttFill, public TAttMarker {

protected:

   Int_t              fMaxSize;   //!Current dimension of arrays fX and fY
   Int_t              fNpoints;   //Number of points <= fMaxSize
   Double_t          *fX;         //[fNpoints] array of X points
   Double_t          *fY;         //[fNpoints] array of Y points
   TList             *fFunctions; //Pointer to list of functions (fits and user)
   TH1F              *fHistogram; //Pointer to histogram used for drawing axis
   Double_t           fMinimum;   //Minimum value for plotting along y
   Double_t           fMaximum;   //Maximum value for plotting along y

   Double_t** AllocateArrays(Int_t Narrays, Int_t arraySize)
   {
      // Allocate arrays.

      if (arraySize < 0) {
         arraySize = 0;
      }
      Double_t **newarrays = new Double_t*[Narrays];
      if (!arraySize) {
         for (Int_t i = 0; i < Narrays; ++i)
            newarrays[i] = 0;
      } else {
         for (Int_t i = 0; i < Narrays; ++i)
            newarrays[i] = new Double_t[arraySize];
      }
      fMaxSize = arraySize;
      return newarrays;
   }

   virtual Double_t **Allocate(Int_t newsize) {
      return AllocateArrays(2, newsize);
   }

   virtual void CopyAndRelease(Double_t **newarrays, Int_t ibegin, Int_t iend,
                       Int_t obegin)
   {
      // Copy points from fX and fY to arrays[0] and arrays[1]
      // or to fX and fY if arrays == 0 and ibegin != iend.
      // If newarrays is non null, replace fX, fY with pointers from newarrays[0,1].
      // Delete newarrays, old fX and fY

      CopyPoints(newarrays, ibegin, iend, obegin);
      if (newarrays) {
         delete[] fX;
         fX = newarrays[0];
         delete[] fY;
         fY = newarrays[1];
         delete[] newarrays;
      }
   }

   virtual Bool_t CopyPoints(Double_t **arrays, Int_t ibegin, Int_t iend,
                     Int_t obegin)
   {
      // Copy points from fX and fY to arrays[0] and arrays[1]
      // or to fX and fY if arrays == 0 and ibegin != iend.

      if (ibegin < 0 || iend <= ibegin || obegin < 0) { // Error;
         return kFALSE;
      }
      if (!arrays && ibegin == obegin) { // No copying is needed
         return kFALSE;
      }
      Int_t n = (iend - ibegin) * sizeof(Double_t);
      if (arrays) {
         memmove(&arrays[0][obegin], &fX[ibegin], n);
         memmove(&arrays[1][obegin], &fY[ibegin], n);
      } else {
         memmove(&fX[obegin], &fX[ibegin], n);
         memmove(&fY[obegin], &fY[ibegin], n);
      }
      return kTRUE;
   }

   Double_t **ExpandAndCopy(Int_t size, Int_t iend)
   {
      // if size > fMaxSize allocate new arrays of 2*size points
      //  and copy oend first points.
      // Return pointer to new arrays.

      if (size <= fMaxSize) {
         return 0;
      }
      Double_t **newarrays = Allocate(2 * size);
      CopyPoints(newarrays, 0, iend, 0);
      return newarrays;
   }

   virtual void FillZero(Int_t begin, Int_t end)
   {
      // Set zero values for point arrays in the range [begin, end)
      // Should be redefined in descendant classes

      memset(fX + begin, 0, (end - begin)*sizeof(Double_t));
      memset(fY + begin, 0, (end - begin)*sizeof(Double_t));
   }

public:

   Graph() : fMaxSize(0),fNpoints(0), fX(0),fY(0),fFunctions(0),fHistogram(0),fMinimum(0),fMaximum(0) {}
   Graph(Int_t n) : TNamed("Graph", "Graph"), TAttLine(), TAttFill(1, 1001), TAttMarker(),
                    fMaxSize(n),fNpoints(n), fX(new Double_t[n]),fY(new Double_t[n]),fFunctions(0),fHistogram(0),fMinimum(0),fMaximum(0)
   {
      FillZero(0, n);
   }
   ~Graph() override {
      delete [] fX;
      delete [] fY;
   }

   void SetPoint(Int_t i, Double_t x, Double_t y)
   {
      // Set x and y values for point number i.

      if (i < 0) return;
      if (fHistogram) {
         delete fHistogram;
         fHistogram = 0;
      }
      if (i >= fMaxSize) {
         Double_t **ps = ExpandAndCopy(i + 1, fNpoints);
         CopyAndRelease(ps, 0, 0, 0);
      }
      if (i >= fNpoints) {
         // points above i can be not initialized
         // set zero up to i-th point to avoid redefenition
         // of this method in descendant classes
         FillZero(fNpoints, i + 1);
         fNpoints = i + 1;
      }
      fX[i] = x;
      fY[i] = y;
   }

   ClassDefOverride(Graph,4)  //Graph graphics class
};

class GraphErrors : public Graph {

protected:
   Double_t    *fEX;        //[fNpoints] array of X errors
   Double_t    *fEY;        //[fNpoints] array of Y errors

public:
   GraphErrors() : Graph(0), fEX(0),fEY(0) {}
   GraphErrors(Int_t n) : Graph(n),fEX(new Double_t[n]),fEY(new Double_t[n])
   {
      FillZero(0, n);
   }
   ~GraphErrors() override {
      delete [] fEX;
      delete [] fEY;
   }

   Double_t **Allocate(Int_t newsize) override {
      return AllocateArrays(4, newsize);
   }

   void CopyAndRelease(Double_t **newarrays, Int_t ibegin, Int_t iend,
                       Int_t obegin) override
   {
      Double_t *newex, *newey;
      if (newarrays) {
         // newarrays will be deleted by Graph::CopyAndRelease.
         newex = newarrays[2];
         newey = newarrays[3];
      }
      Graph::CopyAndRelease(newarrays, ibegin, iend, obegin);
      if (newarrays) {
         delete[] fEX;
         fEX = newex;
         delete[] fEY;
         fEY = newey;
      }
   }

   Bool_t CopyPoints(Double_t **arrays, Int_t ibegin, Int_t iend,
                     Int_t obegin) override
   {
      // Copy points from fEX and fEY to arrays[0] and arrays[1]
      // or to fEX and fEY if arrays == 0 and ibegin != iend.

      if (Graph::CopyPoints(arrays, ibegin, iend, obegin)) {
         Int_t n = (iend - ibegin) * sizeof(Double_t);
         if (arrays) {
            memmove(&arrays[2][obegin], &fEX[ibegin], n);
            memmove(&arrays[3][obegin], &fEY[ibegin], n);
         } else {
            memmove(&fEX[obegin], &fEX[ibegin], n);
            memmove(&fEY[obegin], &fEY[ibegin], n);
         }
         return kTRUE;
      } else
         return kFALSE;

   }

   void FillZero(Int_t begin, Int_t end) override
   {
      // Set zero values for point arrays in the range [begin, end)
      // Should be redefined in descendant classes

      Graph::FillZero(begin, end);
      memset(fEX + begin, 0, (end - begin)*sizeof(Double_t));
      memset(fEY + begin, 0, (end - begin)*sizeof(Double_t));
   }

   ClassDefOverride(GraphErrors,3)  //A graph with error bars
};

void iobug(int split = 0, int classtype = 0, int clonesmode = 0, int show = 0, int dumpmode = 0)
{
   // root -b -q iobug.C(0,0)  OK
   // root -b -q iobug.C(1,0)  crash
   // root -b -q iobug.C(2,0)  OK
   // root -b -q iobug.C(0,1)  OK
   // root -b -q iobug.C(1,1)  Bad numerical expressions
   // root -b -q iobug.C(2,1)  wrong result
   Graph* g = 0;
   Graph* g2 = 0;
   Graph* g3 = 0;
   if (clonesmode == 0) {
      clonesmode = 1;
   }
   TClonesArray* clones = 0;
   if (classtype == 0) {
      g = new Graph(2);
      clones = new TClonesArray("Graph");
      new((*clones)[0]) Graph(2);
      g2 = (Graph*) (*clones)[0];
      new((*clones)[1]) Graph(2);
      g3 = (Graph*) (*clones)[1];
   } else {
      g = new GraphErrors(2);
      clones = new TClonesArray("GraphErrors");
      new((*clones)[0]) GraphErrors(2);
      g2 = (GraphErrors*) (*clones)[0];
      new((*clones)[1]) GraphErrors(2);
      g3 = (GraphErrors*) (*clones)[1];
   }
   g->SetPoint(0, 1, 2);
   g->SetPoint(1, 3, 4);
   g->SetMarkerColor(2);
   g->SetMarkerSize(1.2);
   g->SetMarkerStyle(21);

   g2->SetPoint(0, 1, 2);
   g2->SetPoint(1, 3, 4);
   g2->SetMarkerColor(2);
   g2->SetMarkerSize(1.4);
   g2->SetMarkerStyle(27);

   g3->SetPoint(7, 8, 9);
   g3->SetPoint(10, 13, 14);
   g3->SetMarkerColor(3);
   g3->SetMarkerSize(1.5);
   g3->SetMarkerStyle(30);

   delete gFile; // This will set gFile to zero.

   TFile* f =  new TFile("problem.root", "RECREATE");
   TTree* t = new TTree("graphs", "problematic graphs");
   if (!f || !t) return;

   if (clonesmode & 0x1) {
      // Remember "g" is local, so we must break the
      // connection between "g" and this branch before
      // leaving the routine.
      t->Branch("graph", g->ClassName(), &g, 32000, split);
   }

   if (clonesmode & 0x2) {
      // Remember "clones" is local, so we must break the
      // connection between "clones" and this branch before
      // leaving the routine.
      t->Branch("graphCl", &clones, 32000, split);
   }

   t->Fill();

   g->SetMarkerColor(3);
   g->SetMarkerSize(1.3);
   g->SetMarkerStyle(24);

   g2->SetMarkerColor(4);
   g2->SetMarkerSize(1.6);
   g2->SetMarkerStyle(33);

   g3->SetMarkerColor(5);
   g3->SetMarkerSize(1.7);
   g3->SetMarkerStyle(36);

   t->Fill();

   if (show) {
      t->Show(0);
   }

   t->Write();

   if (dumpmode & 0x1) {
      g->Dump();
   }

   if (clonesmode & 0x1) {
      delete g;
      g = 0;
      // Remember "g" is local, so we must break the
      // connection between "g" and this branch before
      // leaving the routine.
      //
      // Note: Because we set g to zero, an object will
      //       be allocated by this call.
      t->SetBranchAddress("graph", &g);
   }

   if (clonesmode & 0x2) {
      delete clones;
      clones = 0;
      // Remember "clones" is local, so we must break the
      // connection between "clones" and this branch before
      // leaving the routine.
      //
      // Note: Because we set clones to zero, an object will
      //       be allocated by this call.
      t->SetBranchAddress("graphCl", &clones);
   }

   t->GetEntry(0);

   if (dumpmode & 0x2) {
      g->Dump();
   }

   //f.Write();

   if (dumpmode & 0x4) {
      t->Print();
   }

   if (clonesmode & 0x1) {
      t->Scan("fMarkerColor:fMarkerSize:graph.fMarkerStyle", "", "colsize=20 precision=6");
   }

   if (clonesmode & 0x2) {
      t->Scan("fMarkerColor:graphCl.fMarkerSize:graphCl.fMarkerStyle", "", "colsize=20 precision=6");
   }

   //
   // Break the connections between the tree
   // and local variables before returning.
   //

   if (clonesmode & 0x1) {
      t->SetBranchAddress("graph", 0);
   }

   if (clonesmode & 0x2) {
      t->SetBranchAddress("graphCl", 0);
   }

   // return f;
}

