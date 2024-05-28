\addtogroup tutorial_FOAM

@{

### What is FOAM ?

FOAM is simplified version of multi-dimensional general
purpose Monte Carlo event generator (integrator) with hyper-cubical
"foam of cells". Certain features of full version of FOAM are omitted.
mFOAM is intended  as an easy to use tool for MC
simulation/integration in few dimensions. It relies heavily on ROOT package,
borrowing persistency of classes from ROOT. mFOAM can be easily used from
the ROOT shell. For more difficult problems the full FOAM may be better.

### How to run application programs ?

The application program can be run in two modes: it can be simply
interpreted by CLING or compiled. The first method is simpler but
results in slower execution. The second method employs ACLiC -
The Automatic Compiler of Libraries, which automatizes the
process of compilation and linking.

In $(ROOTSYS)/tutorials there are 3 demonstration programs, each
one with its corresponding python scripts:

#### foam_kanwa.C
#### foam_kanwa.py
is a simple example of how to run FOAM in interactive
mode. To run this macro issue either of the following simple commands from the
Linux shell:

```
  root foam_kanwa.C
  ipython3 foam_kanwa.py
```

or from CLING:

```
  root [0] .x foam_kanwa.C
```
or from ipython3:
```
  IP[1] %run foam_demo.py
```

The Simulation will start and a graphical canvas with a plot
of the distribution function will appear. In this example
we defined the distribution function as a global
function Camel2. You can edit it for your own requirements.

#### foam_demo.C or foam_demo.py
show usage of FOAM in compiled mode, which is
the preferred method.  The integrand-function is defined
now as a Density method from class TFDISTR inherited from the
abstract class  TFoamIntegrand. Users can modify its interface,the 
integrand-function, according to their needs but they should
always be remembered of define the Density-method which provides the
density distribution.
Enter into the CLING interpreter(root[]) and type:

```
  root [0] gSystem->Load("libFoam.so")
  root [1] .x foam_demo.C+
```
or ipython3 interpreter :
```
 IP[1]: import foam_demo
 IP[2]: foam_demo()
```

to load FOAM library, and then compile and execute the macro
foam_demo.C(or foam_demo.py).
A shared object foam_demo_C.so is created in the current
directory. At the end of its exploration phase, a FOAM object 
called 'FoamX' will be written into disk, along with its distribution function.

#### foam_demopers.C or foam_demopers.py
demonstrates persistency of FOAM classes.
To run this macro type:

```
  root [0] .x foam_demopers.C
```
or
```
  IP[1] %run foam_demopers.py
```

This Program(.C or .py) reads the FOAM object stored in disk, checks its
consistency, and prints geometry of cells. Next it starts the
the generation of events(with TFDISTR or TFDISTR_Py). Which can be interpreted directly by CLING
because the TFDISTR class was already compiled and is available in
`foam_demo_C.so` library. In python, TFDISTR_Py is available in memory because we already 
loaded with 'import foam_demo'. Either way, TFDISTR or TFDISTR_Py has to be defined previously.

If you have any problems at interpreting or at running those macros. 
Do not hesitate to contact us: the ROOT-Team. Or if you found 
that root crashes let us know, please. We area always founding ways to improve.
The ROOT-Team.
@}
