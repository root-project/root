

## 3D Graphics Libraries

### Gl in Pad

- Transparency is now implemented for "GL in Pad" (`gStyle->SetCanvasPreferGL(1)`).
- Introduce the flag `CanvasPreferGL` in `rootrc.in`. So OpenGL can be use by
  default. The default value for this flag is 0 (no OpenGL).
- Fix size issues with the FTGL text.
- Make `TMathText` work with FTGL
- Linear and radial color gradients are implemented for "GL in Pad"
  (only a simple radial color gradient),
  see also the notes about TLinearGradient and TRadialGradient classes.
- "GL in Pad" and gl hist painters were updated to support Retina displays
  (OS X + Cocoa).
