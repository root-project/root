\defgroup GenVectorX Physics Vectors
\ingroup Math
\brief Vector classes (2D, 3D and 4D / Lorentz vector) and their transformations to be used in SYCL or CUDA kernels for heterogeneous computing.

These classes represent vectors and their operations and transformations, such as rotations and Lorentz transformations, in two, three and four dimensions.
The 4D space-time is used for physics vectors representing relativistic particles in [Minkowski-space](https://en.wikipedia.org/wiki/Minkowski_space).
These vectors are different from Linear Algebra vectors or `std::vector` which describe generic N-dimensional vectors.

Hint: the most commonly used Lorentz vector class is ROOT::MathSYCL::PtEtaPhiMVector or ROOT::MathCUDA::PtEtaPhiMVector, respectively a typedef to ROOT::MathSYCL::LorentzVector < ROOT::MathSYCL::PtEtaPhiM4D < double > > and ROOT::MathCUDA::LorentzVector < ROOT::MathCUDA::PtEtaPhiM4D < double > >.


## Additional Documentation

A more detailed description of the GenVectorX package is available in this [document](https://arxiv.org/pdf/2312.02756).
