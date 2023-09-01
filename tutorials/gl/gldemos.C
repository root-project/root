/// \file
/// \ingroup tutorial_gl
/// Menu for running GL demos.
///
/// \macro_code
///
/// \author Timur Pocheptsov

void gldemos()
{
   TControlBar *bar = new TControlBar("vertical", "GL painter demo",20,20);
   bar->AddButton("Help on demos", "help()", "Description");
   bar->AddButton("glsurfaces", ".x $ROOTSYS/tutorials/gl/glsurfaces.C", "Surface painter example");
   bar->AddButton("glrose", ".x $ROOTSYS/tutorials/gl/glrose.C", "Surface in polar system");
   bar->AddButton("gltf3", ".x $ROOTSYS/tutorials/gl/gltf3.C", "TF3 painter");
   bar->AddButton("glbox", ".x $ROOTSYS/tutorials/gl/glbox.C", "BOX painter");
   bar->AddButton("glparametric", ".x $ROOTSYS/tutorials/gl/glparametric.C", "Parametric surface");
   bar->Show();
}

void help()
{
   new TCanvas("chelp","Help on gldemos",200,10,700,600);

   TPaveLabel *title = new TPaveLabel(0.04, 0.86, 0.96, 0.98, "These demos show different gl painters.");
   title->SetFillColor(32);
   title->Draw();

   TPaveText *hdemo = new TPaveText(0.04, 0.04, 0.96, 0.8);
   hdemo->SetTextAlign(12);
   hdemo->SetTextFont(52);
   hdemo->SetTextColor(kBlue);
   hdemo->AddText("1. Glsurfaces demo shows glsurf4, glsurf1, glsurf3, glsurf1cyl, glsurfpol, gltf3 options.");
   hdemo->AddText("2. Glrose demontrates \"glsurf2pol\" drawing option and user-defined palette.");
   hdemo->AddText("3. Gltf3 demo shows \"gltf3\" option.");
   hdemo->AddText("4. Glbox demo shows \"glbox\" and \"glbox1\" options for TH3.");
   hdemo->AddText("5. Glparametric demo shows how to define and display parametric surfaces.");
   hdemo->AddText("You can zoom any plot: press 'J', 'K', 'j', 'k' keys, or use mouse wheel.");
   hdemo->AddText("Rotate any plot:");
   hdemo->AddText("  ---select plot with mouse cursor,");
   hdemo->AddText("  ---move mouse cursor, pressing and holding left mouse button ");
   hdemo->AddText("Pan plot:");
   hdemo->AddText("  ---select with mouse cursor a part of a plot, other than back box planes ,");
   hdemo->AddText("  ---move mouse cursor, pressing and holding middle mouse button ");
   hdemo->AddText("Selected part of a plot is higlighted (TF3 higlighting is not implemented yet.)");
   hdemo->AddText("You can select one of back box planes, press middle mouse button and move cursor-");
   hdemo->AddText("this will create \"slice\" (TF does not support yet).");
   hdemo->AddText("After the slice was created, you can project it on a back box");
   hdemo->AddText("  ---press key 'p' (now implemented only for surf options ).");
   hdemo->AddText("Left double click removes all slices/projections.");

   hdemo->Draw();

}
