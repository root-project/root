\page Minuit2Page Minuit2


The **Minuit2** library is a new object-oriented implementation, written in C++,
of the popular MINUIT minimization package. These new version provides basically
all the functionality present in the old Fortran version, with almost equivalent
numerical accuracy and computational performances. Furthermore, it contains new
functionality, like the possibility to set single side parameter limits or the
FUMILI algorithm, which is an optimized method for least square and log likelihood
minimizations. The package has been originally developed by M. Winkler and F. James.
More information on the new C++ version can be found on the
[MINUIT Web Site](http://www.cern.ch/minuit).

Minuit2, originally developed in the SEAL project, is now distributed within %ROOT.
The API has been then changed in this new version to follow the %ROOT coding convention
(function names starting with capital letters) and the classes have been moved inside
the namespace _ROOT::Minuit2_. In addition, the %ROOT distribution contains classes
needed to integrate Minuit2 in the %ROOT framework.

A new class has been introduced, ROOT::Minuit2::Minuit2Minimizer, which implements
the interface ROOT::Math::Minimizer. Within %ROOT, it can be instantiates also using
the %ROOT plug-in manager. This class provides a convenient entry point for using Minuit2\.
An example of using this interface is the %ROOT tutorial _tutorials/fit/NumericalMinimization.C_
or the Minuit2 test program
[<tt>testMinimize.cxx</tt>](https://github.com/cxx-hep/root-cern/blob/master/math/minuit2/test/testMinimize.cxx).

A standalone version of Minuit2 (independent of %ROOT) can be easily built and installed using `CMake`. See this [`README`](https://github.com/root-project/root/blob/master/math/minuit2/README.md) for the instructions on how to get the sources, building and installing a stand-alone Minuit2.

The [Minuit2 User Guide](https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html)
provides all the information needed for using directly (without add-on packages like %ROOT) Minuit2.

## References

1.  F. James, _Fortran MINUIT Reference Manual_ ([html](https://cern-tex.web.cern.ch/cern-tex/minuit/minmain.html));
2.  F. James and M. Winkler, _C++ MINUIT User's Guide_ ([html](https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html) and [pdf](https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.pdf));
3.  F. James, _Minuit Tutorial on Function Minimization_ ([pdf](http://seal.cern.ch/documents/minuit/mntutorial.pdf));
4.  F. James, _The Interpretation of Errors in Minuit_ ([pdf](http://seal.cern.ch/documents/minuit/mnerror.pdf));
