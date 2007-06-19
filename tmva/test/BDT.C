#include "tmvaglob.C"

// this macro displays a decision tree read in from the weight file


// input: - No. of tree
//        - the weight file from which the tree is read
void BDT(Int_t itree=1, TString fname= "weights/MVAnalysis_BDT.weights.txt", Bool_t useTMVAStyle = kTRUE) 
{
   // set style and remove existing canvas'
   TMVAGlob::Initialize( useTMVAStyle );
 
   draw_tree(itree,fname);
}

//_______________________________________________________________________
void draw_tree(Int_t itree, TString fname= "weights/MVAnalysis_BDT.weights.txt"){
   // returns poiter to 2D histogram with tree drawing in it.

   TString *vars;   
   TMVA::DecisionTree *d = read_tree(vars,itree,fname);
   if (d == 0) return;
   
   Int_t  depth = d->GetDepth();
   Double_t xmax= 2*depth + 0.5;
   Double_t xmin= -xmax;
   Double_t ystep = 1./(depth+1);

   char buffer[100];
   sprintf (buffer, "Decision Tree No.: %d",itree);
   TCanvas *c1=new TCanvas("c1",buffer,200,0,1000,600);
   c1->Draw();

   draw_node( (TMVA::DecisionTreeNode*)d->GetRoot(), 0.5, 1.-0.5*ystep, 0.25, ystep ,vars);
  
   // make the legend
   Double_t yup=0.99;
   Double_t ydown=yup-ystep/2.5;
   Double_t dy= ystep/2.5 * 0.2;

 
   TPaveText *whichTree = new TPaveText(0.85,ydown,0.98,yup, "NDC");
   whichTree->SetBorderSize(1);
   whichTree->SetFillStyle(1);
   whichTree->SetFillColor(5);
   whichTree->AddText(buffer);
   whichTree->Draw();

   TPaveText *intermediate = new TPaveText(0.02,ydown,0.15,yup, "NDC");
   intermediate->SetBorderSize(1);
   intermediate->SetFillStyle(1);
   intermediate->SetFillColor(3);
   intermediate->AddText("Intermediate Nodes");
   intermediate->Draw();



   ydown = ydown - ystep/2.5 -dy;
   yup   = yup - ystep/2.5 -dy;
   TPaveText *signalleaf = new TPaveText(0.02,ydown ,0.15,yup, "NDC");
   signalleaf->SetBorderSize(1);
   signalleaf->SetFillStyle(1);
   signalleaf->SetFillColor(4);
   signalleaf->SetTextColor(10);
   signalleaf->AddText("Signal Leaf Nodes");
   signalleaf->Draw();

   ydown = ydown - ystep/2.5 -dy;
   yup   = yup - ystep/2.5 -dy;
   TPaveText *backgroundleaf = new TPaveText(0.02,ydown,0.15,yup, "NDC");
   backgroundleaf->SetBorderSize(1);
   backgroundleaf->SetFillStyle(1);
   backgroundleaf->SetFillColor(2);
   backgroundleaf->AddText("Background Leaf Nodes");
   backgroundleaf->Draw();

   c1->Update();
   TString fname = Form("plots/BDT_%i", itree );
   TMVAGlob::imgconv( c1, fname );   
}   
      
//_______________________________________________________________________
void draw_node(  TMVA::DecisionTreeNode *n, 
                 Double_t x, Double_t y, 
                 Double_t xscale,  Double_t yscale, TString * vars) {
   // recursively puts an entries in the histogram for the node and its daughters
   //

   if (n->GetLeft() != NULL){
      TLine *a1 = new TLine(x-xscale/2,y,x-xscale,y-yscale/2);
      a1->SetLineWidth(2);
      a1->Draw();
      draw_node((TMVA::DecisionTreeNode*) n->GetLeft(), x-xscale, y-yscale, xscale/2, yscale, vars);
   }
   if (n->GetRight() != NULL){
      TLine *a1 = new TLine(x+xscale/2,y,x+xscale,y-yscale/2);
      a1->SetLineWidth(2);
      a1->Draw();
      draw_node((TMVA::DecisionTreeNode*) n->GetRight(), x+xscale, y-yscale, xscale/2, yscale, vars  );
   }


   TPaveText *t = new TPaveText(x-xscale/2,y-yscale/2,x+xscale/2,y+yscale/2, "NDC");

   t->SetBorderSize(1);

   t->SetFillStyle(1);
   if (n->GetNodeType() == 1) { t->SetFillColor(4); t->SetTextColor(10); }
   else if (n->GetNodeType() == -1) t->SetFillColor(2);
   else if (n->GetNodeType() == 0) t->SetFillColor(3);

   char buffer[25];
   sprintf(buffer,"N=%d",n->GetNEvents());
   t->AddText(buffer);
   sprintf(buffer,"S/(S+B)=%4.3f",n->GetPurity());
   t->AddText(buffer);

   if (n->GetNodeType() == 0){
      if (n->GetCutType()){
         t->AddText(TString(vars[n->GetSelector()]+">"+=::Form("%5.3g",n->GetCutValue())));
      }else{
         t->AddText(TString(vars[n->GetSelector()]+"<"+=::Form("%5.3g",n->GetCutValue())));
      }
   }

//    sprintf(buffer,"seq=%d",n->GetSequence());
//    t->AddText(buffer);
//    sprintf(buffer,"depth=%d",n->GetDepth());
//    t->AddText(buffer);
   sprintf(buffer,"type=%d",n->GetNodeType());
   t->AddText(buffer);


   if (n->GetNodeType() == 1) t->SetFillColor(4);
   else if (n->GetNodeType() == -1) t->SetFillColor(2);
   else if (n->GetNodeType() == 0) t->SetFillColor(3);
   


   t->Draw();

   return;
}

TMVA::DecisionTree* read_tree(TString * &vars, Int_t itree=1, TString fname= "weights/MVAnalysis_BDT.weights.txt")
{
   cout << "reading Tree " << itree << " from weight file: " << fname << endl;
   ifstream fin( fname );
   if (!fin.good( )) { // file not found --> Error
      cout << "Weight file: " << fname << " does not exist" << endl;
      return;
   }
   
   Int_t   idummy;
   Float_t fdummy;
   TString dummy = "";
   
   // file header with name
   while (!dummy.Contains("#VAR")) fin >> dummy;
   fin >> dummy >> dummy >> dummy; // the rest of header line
   
   // number of variables
   Int_t nVars;
   fin >> dummy >> nVars;
   // at this point, we should have idummy == nVars
   // cout << "rread nVars = " << nVars <<endl;


   // variable mins and maxes
   vars = new TString[nVars];
   for (Int_t i = 0; i < nVars; i++) fin >> vars[i] >> dummy >> dummy >> dummy;

   char buffer[20];
   char line[256];
   sprintf(buffer,"Tree %d",itree);

   while (!dummy.Contains(buffer)) {
      fin.getline(line,256);
      dummy = TString(line);
   }
   
   TMVA::DecisionTreeNode *n = new TMVA::DecisionTreeNode();
   char pos="s";
   UInt_t depth =0;
   n->ReadRec(fin,pos,depth);
   TMVA::DecisionTree *d = new TMVA::DecisionTree(n);
   
   // d->Print(cout);


   fin.close();
   
   return d;
}

