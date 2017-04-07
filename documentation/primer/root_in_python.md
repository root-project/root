# ROOT in Python #

ROOT offers the possibility to interface to Python via a set of bindings called 
PyROOT.
Python is used in a wide variety of application areas and one of the most used 
scripting languages today. 
With the help of PyROOT it becomes possible to combine the power of a scripting
language with ROOT tools. Introductory material to Python is available from many
sources on the web, see e. g. http://docs.python.org. 

## PyROOT ##

The access to ROOT classes and their methods in PyROOT is almost identical to C++
macros, except for the special language features of Python, most importantly dynamic
type declaration at the time of assignment. Coming back to our first example, simply
plotting a function in ROOT, the following C++ code:

``` {.cpp}
TF1 *f1 = new TF1("f2","[0]*sin([1]*x)/x",0.,10.);
f1->SetParameter(0,1);
f1->SetParameter(1,1);
f1->Draw();
```

in Python becomes:

``` {.python}
import ROOT
f1 = ROOT.TF1("f2","[0]*sin([1]*x)/x",0.,10.)
f1.SetParameter(0,1);
f1.SetParameter(1,1);
f1.Draw();
```

A slightly more advanced example hands over data defined in the macro to the ROOT
class `TGraphErrors`. Note that a Python array can be used to pass data between
Python and ROOT. The first line in the Python script allows it to be executed
directly from the operating system, without the need to start the script from
python or the highly recommended powerful interactive shell ipython. The last line
in the python script is there to allow you to have a look at the graphical output
in the ROOT canvas before it disappears upon termination of the script.

Here is the C++ version:

``` {.cpp}
@ROOT_INCLUDE_FILE macros/TGraphFit.C
```

In Python it looks like this:

``` {.python}
@ROOT_INCLUDE_FILE macros/TGraphFit.py
```

Comparing the C++ and Python versions in these two examples, it now should be
clear how easy it is to convert any ROOT Macro in C++ to a Python version.

As another example, let us revisit macro3 from Chapter 4. A straight-forward
Python version relying on the ROOT class `TMath`:

``` {.python}
@ROOT_INCLUDE_FILE macros/macro3.py
```

### More Python- less C++ ###

You may have noticed already that there are some Python modules providing
functionality similar to ROOT classes, which fit more seamlessly into your
Python code.

A more “pythonic” version of the above macro3 would use a replacement of the
ROOT class TMath for the provisoining of data to TGraphPolar. With the math
package, the part of the code becomes

``` {.cpp}
import math
from array import array
from ROOT import TCanvas , TGraphPolar
...
ipt=range(0,npoints)
r=array('d',map(lambda x: x*(rmax-rmin)/(npoints-1.)+rmin,ipt))
theta=array('d',map(math.sin,r))
e=array('d',npoints*[0.])
...

```

#### Customised Binning ####
This example combines comfortable handling of arrays in Python to define
variable bin sizes of a ROOT histogram. All we need to know is the interface
of the relevant ROOT class and its methods (from the ROOT documentation):

``` {.cpp}
TH1F(const char* name , const char* title , Int_t nbinsx , const Double_t* xbins)
```

Here is the Python code:

``` {.python}
import ROOT
from array import array
arrBins = array('d' ,(1 ,4 ,9 ,16) ) # array of bin edges
histo = ROOT.TH1F("hist", "hist", len(arrBins)-1, arrBins)
# fill it with equally spaced numbers
for i in range (1 ,16) :
   histo.Fill(i)
histo.Draw ()
```

## Custom code: from C++ to Python ##
The ROOT interpreter and type sytem offer interesting possibilities when it comes
to JITting of C++ code.
Take for example this header file, containing a class and a function.

```{.cpp}
// file cpp2pythonExample.h
#include "stdio.h"

class A{
public:
 A(int i):m_i(i){}
 int getI() const {return m_i;}
private:
 int m_i=0;
};

void printA(const A& a ){
  printf ("The value of A instance is %i.\n",a.getI());
}
```

```{ .python }
>>> import ROOT
>>> ROOT.gInterpreter.ProcessLine('#include "cpp2pythonExample.h"')
>>> a = ROOT.A(123)
>>> ROOT.printA(a)
The value of A instance is 123.
```

This example might seem trivial, but it shows a powerful ROOT feature. 
C++ code can be JITted within PyROOT and the entities defined in C++ can be 
transparently used in Python!