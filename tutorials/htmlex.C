{
   //
   // All ROOT tutorials as well as the class descriptions have been
   // generated automatically by ROOT itself via the services of
   // the THtml class. Please read this class description and
   // begin_html<a href=../../Conventions.html>Coding Conventions</a></b>end_html.
   // The following macro illustrates how to generate the html code
   // for one class using the Make function.
   // This example also shows how to convert to html a macro, including
   // the generation of a "gif" file produced by the macro.

   //
   // How to generate HTML files for a single class
   // (in this example class name is TBRIK), ...
   THtml htmlex;
   htmlex.MakeClass("TBRIK");

   //
   // and how to generate html code for all classes, including an index.
   //
   //gHtml.MakeAll();

   // execute a macro
   //exec something.mac

   // Invoke the TSystem class to execute a shell script.
   // Here we call the "xpick" program to capture the graphics window
   // into a gif file.

   //gSystem.Exec("xpick html/gif/shapes.gif");


   // Convert this macro into html
   htmlex.Convert("htmlex.C","Automatic HTML document generation");

   // The following line is an example of comment showing how
   // to include HTML instructions in a comment line.

   // here is the begin_html<a href="gif/shapes.gif">output</a> end_html
}
