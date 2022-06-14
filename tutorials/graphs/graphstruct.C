/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// Draw a simple graph structure.
/// The graph layout is made using graphviz. This macro creates some
/// nodes and edges and change a few graphical attributes on some of them.
///
/// \macro_image
/// \macro_code
/// \note For this to work, ROOT has to be compiled with gviz ON
/// \author Olivier Couet

TCanvas* graphstruct()
{
   #if __has_include("TGraphStruct.h") // handy check on whether gviz was installed
   TGraphStruct *gs = new TGraphStruct();

   // create some nodes and put them in the graph in one go ...
   TGraphNode *n0 = gs->AddNode("n0","Node 0");
   TGraphNode *n1 = gs->AddNode("n1","First node");
   TGraphNode *n2 = gs->AddNode("n2","Second node");
   TGraphNode *n3 = gs->AddNode("n3","Third node");
   TGraphNode *n4 = gs->AddNode("n4","Fourth node");
   TGraphNode *n5 = gs->AddNode("n5","5th node");
   TGraphNode *n6 = gs->AddNode("n6","Node number six");
   TGraphNode *n7 = gs->AddNode("n7","Node 7");
   TGraphNode *n8 = gs->AddNode("n8","Node 8");
   TGraphNode *n9 = gs->AddNode("n9","Node 9");

   n4->SetTextSize(0.03);
   n6->SetTextSize(0.03);
   n2->SetTextSize(0.04);

   n3->SetTextFont(132);

   n0->SetTextColor(kRed);

   n9->SetFillColor(kRed-10);
   n0->SetFillColor(kYellow-9);
   n7->SetFillColor(kViolet-9);

   // some edges ...
   gs->AddEdge(n0,n1)->SetLineColor(kRed);
   TGraphEdge *e06 = gs->AddEdge(n0,n6);
   e06->SetLineColor(kRed-3);
   e06->SetLineWidth(4);
   gs->AddEdge(n1,n7);
   gs->AddEdge(n4,n6);
   gs->AddEdge(n3,n9);
   gs->AddEdge(n6,n8);
   gs->AddEdge(n7,n2);
   gs->AddEdge(n8,n3);
   gs->AddEdge(n2,n3);
   gs->AddEdge(n9,n0);
   gs->AddEdge(n1,n4);
   gs->AddEdge(n1,n6);
   gs->AddEdge(n2,n5);
   gs->AddEdge(n3,n6);
   gs->AddEdge(n4,n5);

   TCanvas *c = new TCanvas("c","c",800,600);
   c->SetFillColor(38);
   gs->Draw();
   return c;
   #else
   return new TCanvas("c","c",800,600);
   #endif
}
