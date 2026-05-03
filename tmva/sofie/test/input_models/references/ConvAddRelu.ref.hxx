namespace ConvAddRelu_ExpectedOutput {
// Conv (3x3 all-ones kernel, no padding) + Add (bias 0.5) + Relu, applied to
// a 4x4 ramp input starting at -7 so that the Relu clips part of the output
float outputs[] = {0., 0., 18.5, 27.5};
} // namespace ConvAddRelu_ExpectedOutput
