\defgroup PS Graphics file output
\ingroup Graphics2D
\brief Interfaces to various file output formats

These classes are the backends allowing to generate PS, PDF, LaTeX, SVG and all kinds
of binary files. They are used when the methods `TPad::SaveAS` or `TPad::Print`
are invoked. This methods are also accessible interactively from the `File` menu
of a `TCanvas` window.

  - psview.C is an example showing how to display PS, EPS, PDF files in canvas.

