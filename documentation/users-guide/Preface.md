# Preface {.unnumbered}

In late 1994, we decided to learn
and investigate Object Oriented programming and C++ to better judge
the suitability of these relatively new techniques for scientific
programming. We knew that there is no better way to learn a new
programming environment than to use it to write a program that can
solve a real problem. After a few weeks, we had our first
histogramming package in C++. A few weeks later we had a rewrite of
the same package using the, at that time, very new template features
of C++. Again, a few weeks later we had another rewrite of the package
without templates since we could only compile the version with
templates on one single platform using a specific compiler. Finally,
after about four months we had a histogramming package that was faster
and more efficient than the well-known FORTRAN based HBOOK
histogramming package. This gave us enough confidence in the new
technologies to decide to continue the development. Thus was born
ROOT. Since its first public release at the end of 1995, ROOT has
enjoyed an ever-increasing popularity. Currently it is being used in
all major High Energy and Nuclear Physics laboratories around the
world to monitor, to store and to analyse data. In the other sciences
as well as the medical and financial industries, many people are using
ROOT. We estimate the current user base to be around several thousand
people. In 1997, Eric Raymond analysed in his paper "The Cathedral and
the Bazaar" the development method that makes Linux such a success.
The essence of that method is: "release early, release often and
listen to your customers". This is precisely how ROOT is being
developed. Over the last five years, many of our "customers" became
co-developers. Here we would like to thank our main co-developers and
contributors:

**Masaharu Goto** wrote the C++ interpreter CINT that was an
essential part of ROOT before ROOT 6. Despite being 8 time zones ahead
of us, we have the feeling he has been sitting in the room next door
since 1995.

**Andrei** and **Mihaela Gheata** (Alice collaboration) are co-authors
of the ROOT geometry classes and Virtual Monte-Carlo. They have been
working with the ROOT team since 2000.

**Olivier Couet**, who after a successful development and maintenance
of PAW, has joined the ROOT team in 2000 and has been working on the
graphics sub-system.

**Ilka Antcheva** has been working on the Graphical User Interface
classes. She is also responsible for this latest edition of the Users
Guide with a better style, improved index and several new chapters
(since 2002).

**Bertrand Bellenot** has been developing and maintaining the Win32GDK
version of ROOT. Bertrand has also many other contributions like the
nice RootShower example (since 2001).

**Valeriy Onoutchin** has been working on several ROOT packages, in
particular the graphics sub-system for Windows and the GUI Builder
(since 2000).

**Gerri Ganis** has been working on the authentication procedures to
be used by the root daemons and the PROOF system (since 2002).

**Maarten Ballintijn** (MIT) is one of the main developers of the
PROOF sub-system (since 1995).

**Valeri Fine** (now at BNL) ported ROOT to Windows and contributed
largely to the 3-D graphics. He is currently working on the Qt layer
of ROOT (since 1995).

**Victor Perevoztchikov** (BNL) worked on key elements of the I/O
system, in particular the improved support for STL collections
(1997-2001).

**Nenad Buncic** developed the HTML documentation generation system
and integrated the X3D viewer inside ROOT (1995-1997).

**Suzanne Panacek** was the author of the first version of this User's
Guide and very active in preparing tutorials and giving lectures about
ROOT (1999-2002).

**Axel Naumann** has been developing further the HTML Reference Guide
and helps in porting ROOT under Windows (cygwin/gcc implementation)
(since 2000).

**Anna Kreshuk** has developed the Linear Fitter and Robust Fitter
classes as well as many functions in TMath, TF1, TGraph (since 2005).

**Richard Maunder** has contributed to the GL viewer classes (since
2004).

**Timur Pocheptsov** has contributed to the GL viewer classes and GL
in pad classes (since 2004).

**Sergei Linev** has developed the XML driver and the TSQLFile classes
(since 2003).

**Stefan Roiser** has been contributing to the reflex and cintex
packages (since 2005).

**Lorenzo Moneta** has been contributing the MathCore, MathMore,
Smatrix & Minuit2 packages (since 2005).

**Wim Lavrijsen** is the author of the PyRoot package (since 2004).

Further we would like to thank all the people mentioned in the
`$ROOTSYS/README/CREDITS` file for their contributions, and finally,
everybody who gave comments, reported bugs and provided fixes.

Happy ROOTing!

Rene Brun & Fons Rademakers

Geneva, July 2007

