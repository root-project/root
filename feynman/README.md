# TFeynman

## This is a class that makes it easier to create good-looking Feynman diagrams using ROOT components. 

### Usage:

``` c++

- Quark(
    Double_t x1, // x-Coordinate of starting point
    Double_t y1, // y-coordinate of starting point
    Double_t x2, // x-coordinate of second point
    Double_t y2, // y-coordinate of second point
    Double_t labelPositionX, // x-coordinate of the particle-label
    Double_t labelPositionY, y-coordinate of the particle label 
    const char * quarkName, // name of the quark (either u, d, c, s, t, b or q)
    bool isMatter // is the particle matter (if false then antimatter)
    )
- QuarkAntiQuark(
    Double_t x1, // 
    Double_t y1, // coordinates of center of the circle
    Double_t rad, // radius of the circle
    Double_t labelPositionX, // 
    Double_t labelPositionY, // coordinates of the label (just define the coordinates of the first label)
    const char * quarkName // name of the first quark (u, d, c, s, t, b, q), antiparticle is automatically generated
    )
- Lepton(
    Double_t x1, //
    Double_t y1, // starting coordinates
    Double_t x2, 
    Double_t y2, // stopping coordinates
    Double_t labelPositionX, //
    Double_t labelPositionY, // label coordinates
    const char * whichLepton, // name of the lepton (e, en, m, mn, t, tn) -> Electron, Electron-Neutrino etc.
    bool isMatter // true or false if it is matter or antimatter
    )
- LeptonAntiLepton(
    Double_t x1, //
    Double_t y1, // coordinates of center of the circle
    Double_t rad, // radius of the circle
    Double_t labelPositionX, //
    Double_t labelPositionY, // coordinates of the circle
    const char * whichLepton, // name of the first lepton
    const char * whichAntiLepton // name of the antilepton
    )
- Photon(
    Double_t x1, 
    Double_t y1, // starting coordinates
    Double_t x2, 
    Double_t y2, // stopping coordinates
    Double_t labelPositionX, 
    Double_t labelPositionY // label coordinates
    )
- CurvedPhoton( // for when a Photon is emitted and reabsorbed
    Double_t x1, 
    Double_t y1, // coordinates of center of arc
    Double_t rad, // radius of arc
    Double_t phimin, // minimum angle (see TArc)
    Double_t phimax, // maximum angle (See TAcr)
    Double_t labelPositionX, 
    Double_t labelPositionY // position of label
    ) 
- Gluon(
    Double_t x1, 
    Double_t y1, // starting position
    Double_t x2, 
    Double_t y2, // stopping position
    Double_t labelPositionX, 
    Double_t labelPositionY // label position
    )
- CurvedGluon(
    Double_t x1, 
    Double_t y1, // position of center of Arc
    Double_t rad, // radius of arc
    Double_t phimin, // maximum angle (see TArc)
    Double_t phimax, //minimum angle (see TArc)
    Double_t labelPositionX, 
    Double_t labelPositionY // position of label
    )
- WeakBoson(
    Double_t x1, 
    Double_t y1, // starting position
    Double_t x2, 
    Double_t y2, // stopping position
    Double_t labelPositionX, 
    Double_t labelPositionY, // label position
    const char *whichWeakBoson // name of weak force boson in Latex (Z_{0}, W^{+}, W^{-})
    )
- CurvedWeakBoson(
    Double_t x1, 
    Double_t y1, // position of center of arc
    Double_t rad, // radius of arc
    Double_t phimin, // maximum angle (See TArc)
    Double_t phimax, // minimum angle (see TArc)
    Double_t labelPositionX, 
    Double_t labelPositionY,// position of the label 
    const char* whichWeakBoson // name of the weak boson in Latex (Z_{0}, W^{+}, W^{-})
    )
- Higgs(
    Double_t x1, 
    Double_t y1, // starting position
    Double_t x2, 
    Double_t y2, // stopping position
    Double_t labelPositionX, 
    Double_t labelPositionY // label position
)
- CurvedHiggs(
    Double_t x1, 
    Double_t y1, // position of center of Arc
    Double_t rad, // radius of arc
    Double_t phimin, //maximum angle (see TArc)
    Double_t phimax, //minimum angle (see TArc)
    Double_t labelPositionX, 
    Double_t labelPositionY // position of label
)

```