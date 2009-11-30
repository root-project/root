iterplugin *plugin = 0;

void runplugin()
{
//+ Creates and runs a simple iterator plugin connected to TGeoPainter iterator.
// It demonstrates the possibility to dynamically change the color of drawn
// volumes acording some arbitrary criteria *WITHOUT* changing the color of the 
// same volume drawn on branches that do not match the criteria.
//
// To run:
// root[0]   .L iterplugin.cxx+
// root[1]   .x runplugin.C
// root[2]   select(2,kMagenta);
// root[3]   select(3,kBlue)
// ...

   gROOT->ProcessLine(".x $ROOTSYS/tutorials/geom/rootgeom.C");
   plugin = new iterplugin();
   gGeoManager->GetGeomPainter()->SetIteratorPlugin(plugin);
}

void select(Int_t replica=1, Int_t color=kGreen)
{
// Change current color. Replica range: 1-4
   plugin->Select(replica, color);
   gGeoManager->GetGeomPainter()->ModifiedPad();
}
