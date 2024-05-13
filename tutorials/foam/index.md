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

In $(ROOTSYS)/tutorials there are 3 demonstration programs:

#### foam_kanwa.C
is a simple example how to run FOAM in interactive
mode. To run this macro issue the  following simple command from the
Linux shell:

```
  root foam_kanwa.C
```

or from CLING:

```
  root [0] .x foam_kanwa.C
```

Simulation will start and graphical canvas with plot
of the distribution function appear. In this example
we defined the distribution function simply as a global
function function Camel2.

#### foam_demo.C
shows usage of FOAM in compiled mode, which is
the preferred method.  The integrand function is defined
now as a Density method from class TFDISTR inheriting from
abstract class  TFoamIntegrand. User can modify interface to
integrand function according to their needs but they should
always remember to define Density  method which provides the
density distribution.
Enter CLING interpreter and type:

```
  root [0] gSystem->Load("libFoam.so")
  root [1] .x foam_demo.C+
```

to load FOAM library, compile and execute macro foam_demo.C.
A shared object foam_demo_C.so is created in the current
directory. At the end of exploration phase FOAM object
including distribution function will be written to disk.

#### foam_demopers.C
demonstrates persistency of FOAM classes.
To run this macro type:

```
  root [0] .x foam_demopers.C
```

Program reads the FOAM object from disk, checks its
consistency and prints geometry of cells. Next starts the
the generation. It can be interpreted directly by CLING
because compiled TFDISTR class is already available in
`foam_demo_C.so` library.

@}