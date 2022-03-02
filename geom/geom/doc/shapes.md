\defgroup Shapes_classes Shapes
\ingroup Geometry
\brief Shapes are geometrical objects that provide the basic modeling functionality.

  - [Primitive Shapes](\ref SHAPES01)
  - [Navigation Methods Performed By Shapes](\ref SHAPES02)
  - [Creating Shapes](\ref SHAPES03)
  - [Dividing Shapes](\ref SHAPES04)
  - [Parametric Shapes](\ref SHAPES05)

The "shapes" provide the definition of the `local` coordinate
system of the volume. Any volume must have a shape. Any shape recognized
by the modeller has to derive from the base **`TGeoShape`** class,
providing methods for:

-   Finding out if a point defined in their local frame is contained or
    not by the shape;
-   Computing the distance to enter/exit the shape from a local point,
    given a known direction;
-   Computing the maximum distance in any direction from a local point
    that does NOT result in a boundary crossing of the shape (safe
    distance);
-   Computing the cosines of the normal vector to the crossed shape
    surface, given a starting local point and an ongoing direction.

All the features above are globally managed by the modeller in order to
provide navigation functionality. In addition to those, shapes have also
to implement additional specific abstract methods:

-   Computation of the minimal box bounding the shape, given that this
    box have to be aligned with the local coordinates;
-   Algorithms for dividing the shape along a given axis.

The modeller currently provides a set of 20 basic shapes, which we will
call `primitives`. It also provides a special class allowing the
creation of shapes as a result of Boolean operations between primitives.
These are called `composite shapes` and the composition operation can be
recursive (combined composites). This allows the creation of a quite
large number of different shape topologies and combinations. You can
have a look and run the tutorial: geodemo.C

\image html geom_primitive_shapes.png Primitive Shapes - the general inheritance scheme

Shapes are named objects and all primitives have constructors like:

~~~ {.cpp}
TGeoXXX(const char *name,<type> param1,<type> param2, ...);
TGeoXXX(<type> param1,<type> param2, ...);
~~~

Naming shape primitive is mandatory only for the primitives used in
Boolean composites (see "Composite Shapes"). For the sake of simplicity,
we will describe only the constructors in the second form.

\anchor SHAPES01
### Primitive Shapes

  - Boxes: TGeoBBox class
  - Parallelepiped: TGeoPara class
  - Trapezoids: TGeoTrd1, TGeoTrd2 classes
  - General Trapezoid: TGeoTrap class
  - Twisted Trapezoid: TGeoGtra class
  - Arbitrary 8 vertices shapes: TGeoArb8 class
  - Tubes: TGeoTube class
  - Tube Segments: TGeoTubeSeg class
  - Cut Tubes: TGeoCtub class
  - Elliptical Tubes: TGeoEltu class
  - Hyperboloids: TGeoHype class
  - Cones: TGeoCone class
  - Cone Segments: TGeoConeSeg class
  - Sphere: TGeoSphere class
  - Torus: TGeoTorus class
  - Paraboloid: TGeoParaboloid class
  - Polycone: TGeoPcon class
  - Polygon: TGeoPgon class
  - Polygonal extrusion: TGeoXtru class
  - Half Spaces: TGeoHalfSpace class
  - Composite Shapes: TGeoCompositeShape class

\anchor SHAPES02
### Navigation Methods Performed By Shapes

Shapes are named objects and register themselves to the `manager class`
at creation time. This is responsible for their final deletion. Shapes
can be created without name if their retrieval by name is no needed.
Generally shapes are objects that are useful only at geometry creation
stage. The pointer to a shape is in fact needed only when referring to a
given volume and it is always accessible at that level. Several volumes
may reference a single shape; therefore its deletion is not possible
once volumes were defined based on it.

The navigation features related for instance to tracking particles are
performed in the following way: Each shape implement its specific
algorithms for all required tasks in its local reference system. Note
that the manager class handles global queries related to geometry.
However, shape-related queries might be sometimes useful:

~~~ {.cpp}
Bool_t TGeoShape::Contains(Double_t *point[3]);
~~~

The method above returns `kTRUE` if the point \*point is actually inside
the shape. The point has to be defined in the local shape reference. For
instance, for a box having `DX,DY` and `DZ `half-lengths a point will be
considered inside if:

`-DX <= point[0] <= DX`

`-DY <= point[1] <= DY`

`-DZ <= point[2] <= DZ`

~~~ {.cpp}
Double_t TGeoShape::DistFromInside(Double_t *point[3],
Double_t *dir[3], Int_t iact,Double_t step,Double_t *safe);
~~~

The method computes the distance to exiting a shape from a given point
`inside`, along a given direction. This direction is given by its
director cosines with respect to the local shape coordinate system. This
method provides additional information according the value of `iact`
input parameter:

-   `iact = 0`computes only safe distance and fill it at the location
    given by SAFE;
-   `iact = 1`a proposed STEP is supplied. The safe distance is computed
    first. If this is bigger than STEP than the proposed step is
    approved and returned by the method since it does not cross the
    shape boundaries. Otherwise, the distance to exiting the shape is
    computed and returned;
-   `iact = 2`computes both safe distance and distance to exiting,
    ignoring the proposed step;
-   `iact > 2`computes only the distance to exiting, ignoring anything
    else

~~~ {.cpp}
Double_t TGeoShape::DistFromOutside(Double_t *point[3],
Double_t *dir[3],Int_t iact,Double_t step,Double_t *safe);
~~~

This method computes the distance to entering a shape from a given point
`outside`. It acts in the same way as the previous method.

~~~ {.cpp}
Double_t TGeoShape::Safety(Double_t *point[3],Bool_t inside);
~~~

This computes the maximum shift of a point in any direction that does
not change its `inside/outside `state (does not cross shape boundaries).
The state of the point has to be properly supplied.

~~~ {.cpp}
Double_t *TGeoShape::ComputeNormal(Double_t *point[3],
Double_t *dir[3],Double_t *norm[3]);
~~~

The method above computes the director cosines of normal to the crossed
shape surface from a given point towards direction. This is filled into
the `norm` array, supplied by the user. The normal vector is always
chosen such that its dot product with the direction is positive defined.

\anchor SHAPES03
### Creating Shapes

Shape objects embeds only the minimum set of parameters that are fully
describing a valid physical shape. For instance, the half-length, the
minimum and maximum radius represent a tube. Shapes are used together
with media in order to create volumes, which in their turn are the main
components of the geometrical tree. A specific shape can be created
stand-alone:

~~~ {.cpp}
TGeoBBox *box = new TGeoBBox("s_box",halfX,halfY,halfZ); // named
TGeoTube *tub = new TGeoTube(rmin,rmax,halfZ); // no name
//...  (See all specific shape constructors)
~~~

Sometimes it is much easier to create a volume having a given shape in
one step, since shapes are not directly linked in the geometrical tree
but volumes are:

~~~ {.cpp}
TGeoVolume *vol_box = gGeoManager->MakeBox("BOX_VOL",pmed,halfX,
halfY,halfZ);
TGeoVolume *vol_tub = gGeoManager->MakeTube("TUB_VOL",pmed,rmin,
rmax,halfZ);
// ...(See MakeXXX() utilities in TGeoManager class)
~~~

\anchor SHAPES04
### Dividing Shapes

Shapes can generally be divided along a given axis. Supported axes are:
`X`, `Y`, `Z`, `Rxy`, `Phi`, `Rxyz`. A given shape cannot be divided
however on any axis. The general rule is that that divisions are
possible on whatever axis that produces still known shapes as slices.
The division of shapes are performed by the call `TGeoShape::Divide()`,
but this operation can be done only via `TGeoVolume::Divide()` method.
In other words, the algorithm for dividing a specific shape is known by
the shape object, but is always invoked in a generic way from the volume
level. Details on how to do that can be found in the paragraph ‘Dividing
volumes'. One can see how all division options are interpreted and which
their result inside specific shape classes is.

\anchor SHAPES05
### Parametric Shapes

Shapes generally have a set of parameters that is well defined at build
time. In fact, when the final geometrical hierarchy is assembled and the
geometry is closed, all constituent shapes `MUST`**have well defined and
valid parameters. In order to ease-up geometry creation, some
parameterizations are however allowed.

For instance let's suppose that we need to define several volumes having
exactly the same properties but different sizes. A way to do this would
be to create as many different volumes and shapes. The modeller allows
however the definition of a single volume having undefined shape
parameters.

~~~ {.cpp}
TGeoManager::Volume(const char *name,const char *shape,Int_t nmed);
~~~

-   `name:` the name of the newly created volume;
-   `shape:`the type of the associated shape. This has to contain the
    case-insensitive first 4 letters of the corresponding class name
    (e.g. "`tubs`" will match **`TGeoTubeSeg`**, "`bbox`" will match
    **`TGeoBBox`**)
-   `nmed:` the medium number.

This will create a special volume that will not be directly used in the
geometry, but whenever positioned will require a list of actual
parameters for the current shape that will be created in this process.
Such volumes having shape parameters known only when used have to be
positioned only with **`TGeoManager::Node()` method (see ‘Creating and
Positioning Volumes').**

Other case when shape parameterizations are quite useful is scaling
geometry structures. Imagine that we would like to enlarge/shrink a
detector structure on one or more axes. This happens quite often in real
life and is handled by "fitting mother" parameters. This is accomplished
by defining shapes with one or more invalid (negative) parameters. For
instance, defining a box having `dx=10.`, `dy=10.`, and `dz=-1` will not
generate an error but will be interpreted in a different way: A special
volume **`TGeoVolumeMulti`** will be created. Whenever positioned inside
a mother volume, this will create a normal **`TGeoVolume`** object
having as shape a box with `dz` fitting the corresponding `dz `of the
mother shape. Generally, this type of parameterization is used when
positioning volumes in containers having a matching shape, but it works
also for most reasonable combinations.

\defgroup Tubes Tubes
\ingroup Shapes_classes
Tubes have Z as their symmetry axis.

\defgroup Cones Cones
\ingroup Shapes_classes
Conical tube classes.

\defgroup Trapezoids Trapezoids
\ingroup Shapes_classes
In general, we will call trapezoidal shapes having 8 vertices and up to
6 trapezoid faces. Besides that, two of the opposite faces are parallel
to XY plane and are positioned at ` dZ`. Since general trapezoids are
seldom used in detector geometry descriptions, there are several
primitives implemented in the modeller for particular cases.


