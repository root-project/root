\defgroup GenVector Physics Vectors
\ingroup Math
\brief Generic 2D, 3D and 4D vector classes and their transformations (rotations).

These classes represent vectors and their operations and transformations, such as rotations and Lorentz transformations, in two, three and four dimensions.
The 4D space-time is used for physics vectors representing relativistic particles in [Minkowski-space](https://en.wikipedia.org/wiki/Minkowski_space).
These vectors are different from Linear Algebra vectors or `std::vector` which describe generic N-dimensional vectors.

## Points and Vector concept

Mathematically vectors and points are two distinct concepts. They have different
transformations, like vectors only rotate while points rotate and translate. You can
add two vectors but not two points and the difference between two points is a vector.
We then distinguish for the 2 and 3 dimensional case, between points and vectors,
modeling them with different classes:
*   ROOT::Math::DisplacementVector2D and ROOT::Math::DisplacementVector3D describe
    2 and 3 component direction and magnitude vectors, not rooted at any particular point;
*   ROOT::Math::PositionVector2D and ROOT::Math::PositionVector3D model points in
    2 and 3 dimensions.

ROOT::Math::LorentzVector models 4D space-time vectors. There is no class for a 4D point.


\anchor GenVectorCoordinateSystems
## Coordinate Systems

### Generic Coordinate System

The vector classes are based on a generic type of coordinate system, expressed as a
template parameter of the class. Various classes exist to describe the various
coordinates systems:
*   **2D coordinate system** classes:
    *   ROOT::Math::Cartesian2D, based on <em>(x,y)</em> ;
    *   ROOT::Math::Polar2D, based on <em>(r, phi)</em> ;
*   **3D coordinate system** classes:
    *   ROOT::Math::Cartesian3D, based on <em>(x,y,z)</em>;
    *   ROOT::Math::Polar3D, based on <em>(r, theta, phi)</em>;
    *   ROOT::Math::Cylindrical3D, based on <em>(rho, z, phi)</em>
    *   ROOT::Math::CylindricalEta3D, based on <em>(rho, eta, phi)</em>, where eta is the pseudo-rapidity;
*   **4D coordinate system** classes:
    *   ROOT::Math::PxPyPzE4D, based on based on <em>(px,py,pz,E)</em>;
    *   ROOT::Math::PxPyPzM4D, based on based on <em>(px,py,pz,M)</em>;
    *   ROOT::Math::PtEtaPhiE4D, based on based on <em>(pt,eta,phi,E)</em>;
    *   ROOT::Math::PtEtaPhiM4D, based on based on <em>(pt,eta,phi,M)</em>;

The angle _theta_ is defined between [0,\f$\pi\f$] and _phi_ between [-\f$\pi\f$,\f$\pi\f$].
The angles are expressed in radians.

Users can define the Vectors according to the coordinate type which
is most efficient for their use. Transformations between the various coordinate
systems are available through copy constructors or the assignment `operator =`.
For maximum flexibility and minimize in some use case memory allocation, the
coordinate system classes are templated on the scalar type.

### Coordinate System Tag

The 2D and 3D points and vector classes can be associated to a tag defining the
coordinate system. This can be used to distinguish between vectors of different
coordinate systems like global or local vectors. The coordinate system tag is a
template parameter of the ROOT::Math::DisplacementVector3D
(and ROOT::Math::DisplacementVector2D) and ROOT::Math::PositionVector3D
(and ROOT::Math::PositionVector2D) classes. A default tag,
ROOT::Math::DefaultCoordinateSystemTag, exists for users who don't need this
functionality.


\anchor GenVectorTypedefs
## Concrete Vector typedefs

To avoid exposing templated parameters to the users, typedefs are defined for all types of vectors based an `double`s and `float`s.

### Point2D

Type definitions for points in two dimensions, based on ROOT::Math::PositionVector2D, are defined by `Math/Point2D.h`:

*   ROOT::Math::XYPoint vector based on x,y coordinates (cartesian) in double precision
*   ROOT::Math::XYPointF vector based on x,y coordinates (cartesian) in float precision
*   ROOT::Math::Polar2DPoint vector based on r,phi coordinates (polar) in double precision
*   ROOT::Math::Polar2DPointF vector based on r,phi coordinates (polar) in float precision

### Vector2D

Type definitions for vectors in two dimensions, based on ROOT::Math::DisplacementVector2D, are defined by `Math/Vector2D.h`:

*   ROOT::Math::XYVector vector based on x,y coordinates (cartesian) in double precision
*   ROOT::Math::XYVectorF vector based on x,y coordinates (cartesian) in float precision
*   ROOT::Math::Polar2DVector vector based on r,phi coordinates (polar) in double precision
*   ROOT::Math::Polar2DVectorF vector based on r,phi coordinates (polar) in float precision

### Point3D

Type definitions for points in three dimensions, based on ROOT::Math::PositionVector3D, are defined by `Math/Point3D.h`:

*   ROOT::Math::XYZPoint point based on x,y,z coordinates (cartesian) in double precision
*   ROOT::Math::XYZPointF point based on x,y,z coordinates (cartesian) in float precision
*   ROOT::Math::Polar3DPoint point based on r,theta,phi coordinates (polar) in double precision
*   ROOT::Math::Polar3DPointF point based on r,theta,phi coordinates (polar) in float precision
*   ROOT::Math::RhoZPhiPoint point based on rho,z,phi coordinates (cylindrical using z) in double precision
*   ROOT::Math::RhoZPhiPointF point based on rho,z,phi coordinates (cylindrical using z) in float precision
*   ROOT::Math::RhoEtaPhiPoint point based on rho,eta,phi coordinates (cylindrical using eta instead of z) in double precision
*   ROOT::Math::RhoEtaPhiPointF point based on rho,eta,phi coordinates (cylindrical using eta instead of z) in float precision

### Vector3D

Type definitions for vectors in three dimensions, based on ROOT::Math::DisplacementVector3D, are defined by `Math/Vector3D.h`:

*   ROOT::Math::XYZVector vector based on x,y,z coordinates (cartesian) in double precision
*   ROOT::Math::XYZVectorF vector based on x,y,z coordinates (cartesian) in float precision
*   ROOT::Math::Polar3DVector vector based on r,theta,phi coordinates (polar) in double precision
*   ROOT::Math::Polar3DVectorF vector based on r,theta,phi coordinates (polar) in float precision
*   ROOT::Math::RhoZPhiVector vector based on rho, z,phi coordinates (cylindrical) in double precision
*   ROOT::Math::RhoZPhiVectorF vector based on rho, z,phi coordinates (cylindrical) in float precision
*   ROOT::Math::RhoEtaPhiVector vector based on rho,eta,phi coordinates (cylindrical using eta instead of z) in double precision
*   ROOT::Math::RhoEtaPhiVectorF vector based on rho,eta,phi coordinates (cylindrical using eta instead of z) in float precision

### LorentzVector

Type definitions for Lorentz vectors in four dimensions, based on ROOT::Math::LorentzVector, are defined by `Math/Vector4D.h`:

*   ROOT::Math::XYZTVector vector based on x,y,z,t coordinates (cartesian) in double precision
*   ROOT::Math::XYZTVectorF vector based on x,y,z,t coordinates (cartesian) in float precision
*   ROOT::Math::PtEtaPhiEVector vector based on pt (rho),eta,phi and E (t) coordinates in double precision
*   ROOT::Math::PtEtaPhiMVector vector based on pt (rho),eta,phi and M (t) coordinates in double precision
*   ROOT::Math::PxPyPzMVector vector based on px,py,pz and M (mass) coordinates in double precision
*   ROOT::Math::PxPyPzEVector vector based on px,py,pz and E (energy) coordinates in double precision

The metric used for all the LorentzVector's is (-,-,-,+)


\anchor GenVectorOperations
## Operations

### Constructors and Assignment

Vectors can be constructed by passing its coordinate representation.
In addition the vector classes can be constructed by any vector which implements the
accessors x(), y() and z(). This can be another vector based on a different coordinate
system type or even any vector of a different package, like the CLHEP `Hep3Vector` that
implements the required signatures.

```
XYZVector v1(1,2,3);
RhoEtaPhiVector r2(v1);
CLHEP::Hep3Vector q(1,2,3);
XYZVector v3(q)
```

### Arithmetic Operations

The following operations are possible between vector classes, even of different
coordinate system types:

```
v1 += v2;
v1 -= v2;
v1 = - v2;
v1 *= a;
v1 /= a;
v2 = a * v1;
v2 = v1 / a;
v2 = v1 * a;
v3 = v1 + v2;
v3 = v1 - v2;
```

Note that the multiplication between two vectors using the `operator *` is not supported
because it is ambiguous.

### Other Methods

The vector classes support methods for:

- computation of the dot product via Dot(),
- comptuation of the cross product via Cross(),
- construction of a unit vector via Unit().


\anchor GenVectorTransformations
## Transformations

The transformations are modeled using simple (non-template) classes, using `double` as
the scalar type to avoid too large numerical errors. The transformations are grouped
in Rotations (in 3 dimensions), Lorentz transformations. Each group has several members which may
model physically equivalent trasformations but with different internal representations.
Transformation classes can operate on all type of vectors using the `operator()`
or the `operator *` and the transformations can also be combined via the `operator *`.
In more detail the transformations available are:

### 3D Rotations

*   ROOT::Math::Rotation3D, rotation described by a 3x3 matrix of doubles
*   ROOT::Math::EulerAngles rotation described by the three Euler angles (phi, theta and psi) following the GoldStein [definition](http://mathworld.wolfram.com/EulerAngles.html).
*   ROOT::Math::RotationZYX rotation described by three angles defining a rotation first along the Z axis, then along the rotated Y' axis and then along the rotated X'' axis.
*   ROOT::Math::AxisAngle, rotation described by a vector (axis) and an angle
*   ROOT::Math::Quaternion, rotation described by a quaternion (4 numbers)
*   ROOT::Math::RotationX, specialized rotation along the X axis
*   ROOT::Math::RotationY, specialized rotation along the Y axis
*   ROOT::Math::RotationZ, specialized rotation along the Z axis

### 3D Transformation

*   ROOT::Math::Translation3D, (only translation) described by a 3D vector
*   ROOT::Math::Transform3D, (rotations and then translation) described by a 3x4 matrix (12 numbers)

### Lorentz Rotation

*   ROOT::Math::LorentzRotation , 4D rotation (3D rotation plus a boost) described by a 4x4 matrix
*   ROOT::Math::Boost, a Lorentz boost in an arbitrary direction and described by a 4x4 symmetric matrix (10 numbers)
*   ROOT::Math::BoostX, a boost in the X axis direction
*   ROOT::Math::BoostY, a boost in the Y axis direction
*   ROOT::Math::BoostZ, a boost in the Z axis direction

## Compatibility with CLHEP Vector classes

For compatibility with CLHEP, the vector classes can be constructed easily
from a CLHEP `Hep3Vector` or `HepLorentzVector`, by using a template constructor, which
requires only that the classes implement the accessors `x()`, `y()` and `z()` (and `t()`
for `HepLorentzVector`).
The vector classes also provide member function with the same naming convention
as CLHEP for the most used functions like `x()`, `y()` and `z()`.

## Additional Documentation

A more detailed description of all the GenVector classes is available in this [document](https://root.cern/topical/GenVector.pdf).
