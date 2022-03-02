\defgroup Materials_classes Materials
\ingroup Geometry

  - [Materials and Tracking Media](\ref GM00)
    - [Elements, Materials and Mixtures](\ref GM00a)
    - [Radionuclides](\ref GM00b)
    - [Tracking Media](\ref GM00c)
    - [User Interface for Handling Materials and Media](\ref GM00d)


\anchor GM00
## Materials and Tracking Media

We have mentioned that volumes are the building blocks for geometry, but
they describe real objects having well defined properties. In fact,
there are just two of them: the material they are made from and their
geometrical `shape`. These have to be created before creating the volume
itself, so we will describe the bits and pieces needed for making the
geometry before moving to an architectural point of view.

As far as materials are concerned, they represent the physical
properties of the solid from which a volume is made. Materials are just
a support for the data that has to be provided to the tracking engine
that uses this geometry package. Due to this fact, the
TGeoMaterial class is more like a thin data structure needed for
building the corresponding native materials of the Monte-Carlo tracking
code that uses TGeo.

\anchor GM00a
### Elements, Materials and Mixtures

In order to make easier material and mixture creation, one can use the
pre-built table of elements owned by TGeoManager class:

~~~{.cpp}
TGeoElementTable *table = gGeoManager->GetElementTable();
TGeoElement *element1 = table->GetElement(Int_t Z);
TGeoElement *element2 = table->FindElement("Copper");
~~~

Materials made of single elements can be defined by their atomic mass
(`A`), charge (`Z`) and density (`rho`). One can also create a material
by specifying the element that it is made of. Optionally the radiation
and absorption lengths can be also provided; otherwise they can be
computed on-demand [`G3`]. The class representing them is
TGeoMaterial:

~~~{.cpp}
TGeoMaterial(const char *name,Double_t a,Double_t z,
    Double_t density, Double_t radlen=0,Double_t intlen=0);
TGeoMaterial(const char *name, TGeoElement *elem,
    Double_t density);
TGeoMaterial(const char* name, Double_t a, Double_t z,
    Double_t rho,
    TGeoMaterial::EGeoMaterialState state,
    Double_t temperature = STP_temperature,
    Double_t pressure = STP_pressure)
~~~

Any material or derived class is automatically indexed after creation.
The assigned index is corresponding to the last entry in the list of
materials owned by TGeoManager class. This can be changed using
the `TGeoMaterial::SetIndex()` method, however it is not
recommended while using the geometry package interfaced with a transport
MC. Radiation and absorption lengths can be set using:

~~~{.cpp}
TGeoMaterial::SetRadLen(Double_t radlen, Double_t intlen);
~~~

-   `radlen:` radiation length. If `radlen<=0` the value is computed
    using GSMATE algorithm in GEANT3
-   `intlen:` absorption length

Material state, temperature and pressure can be changed via setters.
Another material property is transparency. It can be defined and used
while viewing the geometry with OpenGL.

~~~{.cpp}
void SetTransparency (Char_t transparency = 0)
~~~

-   `transparency:` between 0 (opaque default) to 100 (fully
    transparent)

One can attach to a material a user-defined object storing Cerenkov
properties. Another hook for material shading properties is currently
not in use. Mixtures are materials made of several elements. They are
represented by the class TGeoMixture, deriving from
TGeoMaterial and defined by their number of components and the
density:

~~~{.cpp}
TGeoMixture(const char *name,Int_t nel,Double_t rho);
~~~

Elements have to be further defined one by one:

~~~{.cpp}
void TGeoMixture::DefineElement(Int_t iel,Double_t a,Double_t z,
    Double_t weigth);
void TGeoMixture::DefineElement(Int_t iel, TGeoElement *elem,
    Double_t weight);
void TGeoMixture::DefineElement(Int_t iel, Int_t z, Int_t natoms);
~~~

or:

~~~{.cpp}
void AddElement(TGeoMaterial* mat, Double_t weight);
void AddElement(TGeoElement* elem, Double_t weight);
void AddElement(TGeoElement* elem, Int_t natoms);
void AddElement(Double_t a, Double_t z, Double_t weight)
~~~

-   `iel:` index of the element` [0,nel-1]`
-   `a` and `z:` the atomic mass and charge
-   `weight:` proportion by mass of the elements
-   `natoms`: number of atoms of the element in the molecule making the
    mixture

The radiation length is automatically computed when all elements are
defined. Since tracking MC provide several other ways to create
materials/mixtures, the materials classes are likely to evolve as the
interfaces to these engines are being developed. Generally in the
process of tracking material properties are not enough and more specific
media properties have to be defined. These highly depend on the MC
performing tracking and sometimes allow the definition of different
media properties (e.g. energy or range cuts) for the same material.

\anchor GM00b
### Radionuclides

A new class TGeoElementRN was introduced in this version to
provide support for radioactive nuclides and their decays. A database of
3162 radionuclides can be loaded on demand via the table of elements
(TGeoElementTable class). One can make then materials/mixtures
based on these radionuclides and use them in a geometry

~~~{.cpp}
root[] TGeoManager *geom = new TGeoManager("geom","radionuclides");
root[] TGeoElementTable *table = geom->GetElementTable();
root[] TGeoElementRN *c14 = table->GetElementRN(14,6);  // A,Z
root[] c14->Print();
6-C-014 ENDF=60140; A=14; Z=6; Iso=0; Level=0[MeV]; Dmass=3.0199[MeV];
Hlife=1.81e+11[s]  J/P=0+; Abund=0; Htox=5.8e-10; Itox=5.8e-10; Stat=0
Decay modes:
BetaMinus            Diso:   0 BR:   100.000% Qval: 0.1565
~~~

One can make materials or mixtures from radionuclides:

~~~{.cpp}
root[] TGeoMaterial *mat = new TGeoMaterial("C14", c14, 2.0);
~~~

The following properties of radionuclides can be currently accessed via
getters in the TGeoElementRN class:

Atomic number and charge (from the base class TGeoElement)

-   Isomeric number (`ISO`)
-   ENDF code - following the convention: `ENDF=10000*Z+100*A+ISO`
-   Isomeric energy level [`MeV`]
-   Mass excess [`MeV`]
-   Half life [`s`]
-   Spin/Parity - can be retrieved with: `TGeoElementRN::GetTitle()`
-   Hynalation and ingestion toxicities
-   List of decays - `TGeoElementRN::GetDecays()`

The radioactive decays of a radionuclide are represented by the class
TGeoDecayChannel and they are stored in a TObjArray. Decay
provides:

-   Decay mode
-   Variation of isomeric number
-   `Q` value for the decay [`GeV`]
-   Parent element
-   Daughter element

Radionuclides are linked one to each other via their decays, until the
last element in the decay chain which must be stable. One can iterate
decay chains using the iterator TGeoElemIter:

~~~{.cpp}
root[] TGeoElemIter next(c14);
root[] TGeoElementRN *elem;
root[] while ((elem=next())) next.Print();
6-C-014 (100% BetaMinus) T1/2=1.81e+11
7-N-014 stable
~~~

To create a radioactive material based on a radionuclide, one should
use the constructor:

~~~{.cpp}
TGeoMaterial(const char *name, TGeoElement *elem, Double_t density)
~~~

To create a radioactive mixture, one can use radionuclides as well as
stable elements:

~~~{.cpp}
TGeoMixture(const char *name, Int_t nelements, Double_t density);
TGeoMixture::AddElement(TGeoElement *elem,
    Double_t weight_fraction);
~~~

Once defined, one can retrieve the time evolution for the radioactive
materials/mixtures by using one of the next two methods:

#### Method 1

~~~{.cpp}
TGeoMaterial::FillMaterialEvolution(TObjArray *population, Double_t precision=0.001)
~~~

To use this method, one has to provide an empty TObjArray object
that will be filled with all elements coming from the decay chain of the
initial radionuclides contained by the material/mixture. The precision
represent the cumulative branching ratio for which decay products are
still considered.

\image html geometry003.png width=600px


The population list may contain stable elements as well as
radionuclides, depending on the initial elements. To test if an element
is a radionuclide:

~~~{.cpp}
Bool_t TGeoElement::IsRadioNuclide() const
~~~

All radionuclides in the output population list have attached objects
that represent the time evolution of their fraction of nuclei with
respect to the top radionuclide in the decay chain. These objects
(Bateman solutions) can be retrieved and drawn:

~~~{.cpp}
TGeoBatemanSol *TGeoElementRN::Ratio();
void TGeoBatemanSol::Draw();
~~~

#### Method 2

Another method allows to create the evolution of a given radioactive
material/mixture at a given moment in time:

~~~{.cpp}
TGeoMaterial::DecayMaterial(Double_t time, Double_t precision=0.001)
~~~

The method will create the mixture that result from the decay of a
initial material/mixture at time, while all resulting elements having a
fractional weight less than precision are excluded.

A demo macro for radioactive material features is
`$ROOTSYS/tutorials/geom/RadioNuclides.C` It demonstrates also the decay
of a mixture made of radionuclides.

\image html geometry004.png width=600px

\anchor GM00c
### Tracking Media

The class TGeoMedium describes tracking media properties. This has
a pointer to a material and the additional data members representing the
properties related to tracking.

~~~{.cpp}
TGeoMedium(const char *name,Int_t numed,TGeoMaterial *mat,
           Double_t *params=0);
~~~

-   `name:` name assigned to the medium
-   `mat:` pointer to a material
-   `params:` array of additional parameters

Another constructor allows effectively defining tracking parameters in
GEANT3 style:

~~~{.cpp}
TGeoMedium(const char *name,Int_t numed,Int_t imat,Int_t ifield,
Double_t fieldm,Double_t tmaxfd,Double_t stemax,
Double_t deemax,Double_t epsil,Double_t stmin);
~~~

This constructor is reserved for creating tracking media from the VMC
interface [...]:

-   `numed:` user-defined medium index
-   `imat:` unique ID of the material
-   `others:` see G3 documentation

Looking at our simple world example, one can see that for creating
volumes one needs to create tracking media before. The way to proceed
for those not interested in performing tracking with external MC's is to
define and use only one `dummy tracking medium` as in the example (or a
`NULL` pointer).

\anchor GM00d
### User Interface for Handling Materials and Media

The TGeoManager class contains the API for accessing and handling
defined materials:

~~~{.cpp}
TGeoManager::GetMaterial(name);
~~~

