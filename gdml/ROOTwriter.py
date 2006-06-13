# @(#)root/gdml:$Name:$:$Id:$
# Author: Witold Pokorski   05/06/2006

from math import *
from units import *

# This class provides ROOT binding for the 'writer' class. It implements specific
# methods for all the supported TGeo classes which call the appropriate 'add-element'
# methods from the 'writer' class.

# The list of presently supported classes is the following:

# Materials:
# TGeoElement
# TGeoMaterial
# GeoMixture

# Solids:
# TGeoBBox
# TGeoSphere
# TGeoConeSeg
# TGeoCone
# TGeoPara
# TGeoTrap
# TGeoTrd2
# TGeoTubeSeg
# TGeoPcon
# TGeoTorus
# TGeoPgon
# TGeoEltu
# TGeoHype
# TGeoUnion
# TGeoIntersection
# TGeoSubtraction

# Geometry:
# TGeoVolume

# In addition the class contains three methods 'dumpMaterials', 'dumpSolids' and 'examineVol'
# which retrieve from the memory the materials, the solids and the geometry tree
# respectively. The user should instanciate this class passing and instance of 'writer'
# class as argument. In order to export the geometry in the form of a GDML file,
# the three methods (dumpMaterials, dumpSolids and examineVol) should be called.
# The argument of 'dumpMaterials' method should be the list of materials,
# the argument of the 'dumpSolids' method should be the list of solids and
# the argument of the 'examineVol' method should be the top volume of
# the geometry tree.

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class ROOTwriter(object):

    def __init__(self, writer):
        self.writer = writer
        self.vols = []
        self.elements = []
        pass

    def rotXYZ(self, r):
        cosb = ( r[0]**2 + r[1]**2 )**0.5
        if cosb > 0 :
            a = atan2( r[5], r[8] ) * rad
            b = atan2( -r[2], cosb ) * rad
            c = atan2( r[1], r[0] ) * rad
        else:
            a = atan2( -r[7], r[4] ) * rad
            b = atan2( -r[2], cosb ) * rad
            c = 0.
        return (a, b, c)

    def TGeoBBox(self, solid):
        self.writer.addBox(solid.GetName(), 2*solid.GetDX(), 2*solid.GetDY(), 2*solid.GetDZ())

    def TGeoSphere(self, solid):
        self.writer.addSphere(solid.GetName(), solid.GetRmin(), solid.GetRmax(),
                             solid.GetPhi1(), solid.GetPhi2() - solid.GetPhi1(),
                             solid.GetTheta1(), solid.GetTheta2() - solid.GetTheta1())

    def TGeoConeSeg(self, solid):
        self.writer.addCone(solid.GetName(), 2*solid.GetDz(), solid.GetRmin1(), solid.GetRmin2(),
                           solid.GetRmax1(), solid.GetRmax2(),
                           solid.GetPhi1(), solid.GetPhi2() - solid.GetPhi1())

    def TGeoCone(self, solid):
        self.writer.addCone(solid.GetName(), 2*solid.GetDz(), solid.GetRmax1(), solid.GetRmax2(),
                           solid.GetRmin1(), solid.GetRmin2(), 0, 360)

    def TGeoPara(self, solid):
        self.writer.addPara(solid.GetName(), solid.GetX(), solid.GetY(), solid.GetZ(),
                           solid.GetAlpha(), solid.GetTheta(), solid.GetPhi())

    def TGeoTrap(self, solid):
        self.writer.addTrap(solid.GetName(), 2*solid.GetDz(), solid.GetTheta(), solid.GetPhi(),
                           2*solid.GetH1(), 2*solid.GetBl1(), 2*solid.GetTl1(), solid.GetAlpha1(),
                           2*solid.GetH2(), 2*solid.GetBl2(), 2*solid.GetTl2(), solid.GetAlpha2())

    def TGeoTrd2(self, solid):
        self.writer.addTrd(solid.GetName(), 2*solid.GetDx1(), 2*solid.GetDx2(), 2*solid.GetDy1(),
                          2*solid.GetDy2(), 2*solid.GetDz())

    def TGeoTubeSeg(self, solid):
        self.writer.addTube(solid.GetName(), solid.GetRmin(), solid.GetRmax(),
                           2*solid.GetDz(), solid.GetPhi1(), solid.GetPhi2()-solid.GetPhi1())

    def TGeoPcon(self, solid):
        zplanes = []
        for i in range(solid.GetNz()):
            zplanes.append( (solid.GetZ(i), solid.GetRmin(i), solid.GetRmax(i)) )
        self.writer.addPolycone(solid.GetName(), solid.GetPhi1(), solid.GetDphi(), zplanes)

    def TGeoTorus(self, solid):
        self.writer.addTorus(solid.GetName(), solid.GetR(), solid.GetRmin(), solid.GetRmax(),
                            solid.GetPhi1(), solid.GetDphi())

    def TGeoPgon(self, solid):
        zplanes = []
        for i in range(solid.GetNz()):
            zplanes.append( (solid.GetZ(i), solid.GetRmin(i), solid.GetRmax(i)) )
        self.writer.addPolyhedra(solid.GetName(), solid.GetPhi1(), solid.GetPhi1() + solid.GetDphi(),
                                solid.GetNedges(), zplanes)

    def TGeoEltu(self, solid):
        self.writer.addEltube(solid.GetName(), solid.GetA(), solid.GetB(), solid.GetDz())

    def TGeoHype(self, solid):
        self.writer.addHype(solid.GetName(), solid.GetRmin(), solid.GetRmax(),
                           solid.GetStIn(), solid.GetStOut(), 2*solid.GetDz())

    def TGeoUnion(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())

        self.writer.addUnion(solid.GetName(),
                            solid.GetBoolNode().GetLeftShape().GetName(),
                            solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                            lrot,
                            solid.GetBoolNode().GetRightShape().GetName(),
                            solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                            rrot)

    def TGeoIntersection(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())

        self.writer.addIntersection(solid.GetName(),
                                   solid.GetBoolNode().GetLeftShape().GetName(),
                                   solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                                   lrot,
                                   solid.GetBoolNode().GetRightShape().GetName(),
                                   solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                                   rrot)

    def TGeoSubtraction(self, solid):
        lrot = self.rotXYZ(solid.GetBoolNode().GetLeftMatrix().Inverse().GetRotationMatrix())
        rrot = self.rotXYZ(solid.GetBoolNode().GetRightMatrix().Inverse().GetRotationMatrix())

        self.writer.addSubtraction(solid.GetName(),
                                  solid.GetBoolNode().GetLeftShape().GetName(),
                                  solid.GetBoolNode().GetLeftMatrix().GetTranslation(),
                                  lrot,
                                  solid.GetBoolNode().GetRightShape().GetName(),
                                  solid.GetBoolNode().GetRightMatrix().GetTranslation(),
                                  rrot)

    def TGeoCompositeShape(self, shape):
        eval('self.'+shape.GetBoolNode().__class__.__name__)(shape)

    def dumpMaterials(self, matlist):
        print 'Found ', matlist.GetSize(),' materials'
        for mat in matlist:
            if not mat.IsMixture():
                self.writer.addMaterial(mat.GetName(), mat.GetA(), mat.GetZ(), mat.GetDensity())
            else:
                elems = {}
                for index in range(mat.GetNelements()):
                    elems[mat.GetElement(index).GetTitle()] = mat.GetWmixt()[index]
                    el = mat.GetElement(index)
                    if el not in self.elements:
                        self.elements.append(el)
                        self.writer.addElement(mat.GetElement(index).GetName(), mat.GetElement(index).GetTitle(),
                                              mat.GetElement(index).Z(), mat.GetElement(index).A())

                    self.writer.addMixture(mat.GetName(), mat.GetDensity(), elems)

    def dumpSolids(self, shapelist):
        print 'Found ', shapelist.GetEntries(), ' shapes'
        for shape in shapelist:
            eval('self.'+shape.__class__.__name__)(shape)

    def examineVol(self, volume):
        daughters = []
        if volume.GetNodes():
            for node in volume.GetNodes():
                subvol = node.GetVolume()
                if subvol not in self.vols:
                    self.vols.append(subvol)
                    self.examineVol(subvol)

                name = node.GetName()+'in'+volume.GetName()
                pos = node.GetMatrix().GetTranslation()
                self.writer.addPosition(name+'pos', pos[0], pos[1], pos[2])
                r = self.rotXYZ(node.GetMatrix().GetRotationMatrix())
                rotname = ''
                if r[0]!=0.0 or r[1]!=0.0 or r[2]!=0.0:
                    self.writer.addRotation(name+'rot', r[0], r[1], r[2])
                    rotname = name+'rot'
                daughters.append( (node.GetVolume().GetName(),
                                   name+'pos', rotname) )

        self.writer.addVolume(volume.GetName(), volume.GetShape().GetName(),
                             volume.GetMaterial().GetName(), daughters)

