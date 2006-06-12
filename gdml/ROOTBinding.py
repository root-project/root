#!/usr/bin/env python2.3
# -*- Mode: Python -*-

import ROOT
import re
import array

# This module contains the implementation of two classes 'solid_reflection' and
# 'ROOTBinding'. The first one is just a small helper class used internally by the
# 'reflection' method (for reflected solids).

# The 'ROOTBinding' class is a collection of methods which are actually called in
# order to instanciate different TGeo objects. It provides the 'binding' between
# different GDML elements and ROOT TGeo classes.
# An instance of this class should be used as the argument of the 'processes' class
# constructor (see processes.py and GDMLContentHandler.py modules).

# The presently supported list of TGeo classes is the following:

# Materials:
# TGeoElement
# TGeoMaterial
# TGeoMixture

# Solids:
# TGeoBBox
# TGeoTubeSeg
# TGeoConeSeg
# TGeoPcon
# TGeoTrap
# TGeoTrd2
# TGeoSphere
# TGeoPara
# TGeoTorus
# TGeoPgon
# TGeoEltu
# TGeoCompositeShape (subtraction, union, intersection)

# Geometry:
# TGeoVolume
# TGeoVolumeAssembly
# divisions

# For any new TGeo class (new solids, etc), one needs to add the appropriate method
# in ROOTBinding class (to be called by the appropriate 'process' for processes.py
# module). 

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class solid_reflection(object):
    def __init__(self, name, solid, refl):
        self.name = name
        self.solid = solid
        self.matrix = refl

class ROOTBinding(object):

    def __init__(self):
	self.medid = 0
	self.volid = 0
        self.reflections = {}
	
    def position(self, x, y, z):
	return ROOT.TGeoTranslation(x,y,z)

    def rotation(self, x, y, z):
	rot = ROOT.TGeoRotation()
	rot.RotateZ(-z)
	rot.RotateY(-y)
	rot.RotateX(-x)
 	return rot

    def element(self, name, formula, z, a):
        # Z can be in general 'float' !!!!!!!!!!!!!
        # we have to convert it to 'int' for ROOT !!!!!!!!!!!!
	return ROOT.TGeoElement(re.sub('0x........','',name), formula, int(z), a)

    def isotope(self, name, z, n, a, d):
	print 'Isotopes not supported by the GDML->TGeo converter, sorry!'
	return 0

    def material(self, name, a, z, rho):
	return ROOT.TGeoMaterial(re.sub('0x........','',name), a, z, rho)

    def mixmat(self, name, ncompo, density):
	return ROOT.TGeoMixture(re.sub('0x........','',name), ncompo, density)

    def mix_addele(self, mixture, element, id, fraction):
	mixture.DefineElement(id, element, fraction)
	return

    def mixele(self, name, ncompo, density):
	return ROOT.TGeoMixture(re.sub('0x........','',name), ncompo, density)

    def mix_addiso(self, element, isotope, id, fraction):
	return 0

    def medium(self, name, material):
	self.medid = self.medid + 1
	return ROOT.TGeoMedium(re.sub('0x........','',name), self.medid, material)

    def logvolume(self, name, solid, medium, reflex):
        if reflex == 0:
            vol = ROOT.TGeoVolume(re.sub('0x........','',name), solid, medium)
        else:
            vol = ROOT.TGeoVolume(re.sub('0x........','',name), reflex, medium)            
        return vol

    def assembly(self, name):
	return ROOT.TGeoVolumeAssembly(name)

    def physvolume(self, name, lv, motherlv, rot, pos, reflected_vol):
	self.volid = self.volid + 1
        # we need to check whether reflected_vol is not 0 and if yes get the reflection matrix
	matr = ROOT.TGeoCombiTrans(pos, rot)
#        if reflected_vol != 0:
#            refl_mtx =  self.reflections[reflected_vol].matrix
#            matr = refl_mtx*matr
	motherlv.AddNode(lv, self.volid, matr)
	return

    def divisionvol(self, name, lv, motherlv, axis, number, width, offset):
        naxis = 0
	self.volid = self.volid + 1
	matr = ROOT.TGeoCombiTrans()

        if axis == 'kXAxis':
            naxis = 1
        elif axis == 'kYAxis':
            naxis = 2
        elif axis == 'kZAxis':
            naxis = 3
        elif axis == 'kRho':
            naxis = 1
        elif axis == 'kPhi':
            naxis = 2

        motherlv.Divide(name, naxis, number, offset, width)
 
    def box(self, name, x, y, z):
	return ROOT.TGeoBBox(re.sub('0x........','',name), x, y, z)

    def tube(self, name, rmin, rmax, z, startphi, deltaphi):
	return ROOT.TGeoTubeSeg(re.sub('0x........','',name), rmin, rmax, z, startphi, 
				startphi+deltaphi)

    def cone(self, name, rmin1, rmax1, rmin2, rmax2, z, 
	     startphi, deltaphi):
	return ROOT.TGeoConeSeg(re.sub('0x........','',name), z, rmin1, rmax1, rmin2, rmax2, 
				startphi, startphi+deltaphi)

    def polycone(self, name, startphi, deltaphi, zrs):
	poly = ROOT.TGeoPcon(re.sub('0x........','',name), startphi, deltaphi, zrs.__len__())

	nb = 0
	for zpl in zrs:
	    poly.DefineSection(nb, zpl[0], zpl[1], zpl[2])
	    nb = nb + 1

	return poly

    def trap(self, name, x1, x2, x3, x4, y1, y2, z, 
	     alpha1, alpha2, phi, theta):
	return ROOT.TGeoTrap(re.sub('0x........','',name), z, theta, phi, y1, x1, x2, 
			     alpha1, y2, x3, x4, alpha2)

    def trd(self, name, x1, x2, y1, y2, z):
	return ROOT.TGeoTrd2(name, x1, x2, y1, y2, z)

    def sphere(self, name, rmin, rmax, 
	       startphi, deltaphi, 
	       starttheta, deltatheta):
	return ROOT.TGeoSphere(re.sub('0x........','',name), rmin, rmax, 
			       starttheta, starttheta+deltatheta,
			       startphi, startphi+deltaphi)

    def orb(self, name, r):
	return ROOT.TGeoSphere(re.sub('0x........','',name), 0, r, 0, 180, 0, 360)

    def para(self, name, x, y, z, alpha, theta, phi):
	return ROOT.TGeoPara(re.sub('0x........','',name), x, y, z, alpha, theta, phi)

    def torus(self, name, rmin, rmax, rtor, startphi, deltaphi):
	return ROOT.TGeoTorus(re.sub('0x........','',name), rtor, rmin, rmax, startphi, deltaphi) 

    def polyhedra(self, name, startphi, deltaphi, numsides, zrs):
	polyh = ROOT.TGeoPgon(re.sub('0x........','',name), startphi, deltaphi, 
			      numsides, zrs.__len__())

	nb = 0
	for zpl in zrs:
	    polyh.DefineSection(nb, zpl[0], zpl[1], zpl[2])
	    nb = nb + 1

	return polyh

    def eltube(self, name, dx, dy, dz):
	return ROOT.TGeoEltu(re.sub('0x........','',name), dx, dy, dz)

    def subtraction(self, name, first, second, pos, rot):
	matr = ROOT.TGeoCombiTrans(pos, ROOT.TGeoRotation(rot.Inverse()))
	sub = ROOT.TGeoSubtraction(first, second, ROOT.TGeoCombiTrans(), matr)
	return ROOT.TGeoCompositeShape(re.sub('0x........','',name), sub)


    def union(self, name, first, second, pos, rot):
	matr = ROOT.TGeoCombiTrans(pos, ROOT.TGeoRotation(rot.Inverse()))
	un = ROOT.TGeoUnion(first, second, ROOT.TGeoCombiTrans(), matr)
	return ROOT.TGeoCompositeShape(re.sub('0x........','',name), un)

    def intersection(self, name, first, second, pos, rot):
	matr = ROOT.TGeoCombiTrans(ROOT.TGeoTranslation(pos), ROOT.TGeoRotation(rot.Inverse()))
	int = ROOT.TGeoIntersection(first, second, ROOT.TGeoCombiTrans(), matr)
	return ROOT.TGeoCompositeShape(re.sub('0x........','',name), int)

    def reflection(self, name, solid,
                   sx, sy, sz,
                   rx, ry, rz,
                   dx, dy, dz):

        rot = ROOT.TGeoRotation()
	rot.RotateZ(-rz)
	rot.RotateY(-ry)
	rot.RotateX(-rx)

        refl = ROOT.TGeoRotation()
        matrix = array.array('d',[sx,0,0,0,sy,0,0,0,sz])
        refl.SetMatrix(matrix)

        rot = rot*refl

#        rot.Print()
#        refl_matx = ROOT.TGeoGenTrans(dx, dy, dz, sx, sy, sz, rot)
        newrot = ROOT.TGeoRotation(rot)
#        newrot.Print()
        
        refl_matx = ROOT.TGeoCombiTrans(dx, dy, dz, newrot)
        rs = solid_reflection(name, solid, refl_matx)
        self.reflections[name] = rs

        return rs
    
