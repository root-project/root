#!/usr/bin/env python2.3
# -*- Mode: Python -*-
# @(#)root/gdml:$Name:  $:$Id: processes.py,v 1.3 2006/06/13 20:46:53 rdm Exp $
# Author: Witold Pokorski   05/06/2006
#
from units import *
from math import *

# This class contains the implementation of the SAX parser 'processes' associated to
# all the supported GDML elements.
# User should never need to explicitely instaciate this class. It is internally
# used by the GDMLContentHandler.

# Whenever a new element is added to GDML schema, this class needs to be extended.
# The appropriate method (process) needs to be implemented, as well as the new
# element needs to be added to the 'gdmlel_dict' dictionary (at the end of the present file)
# where the mapping between the GDML element name and the name of the appropriate process is
# specified.

# The constructor of this class requires the 'binding' as argument.
# The 'binding' is an application-specific mapping of GDML elements (materials,
# solids, etc) to specific objects which should be instanciated by the converted.
# In the present case (ROOT) the binding is implemented in the ROOTBinding module.

# For any question or remarks concerning this code, please send an email to
# Witold.Pokorski@cern.ch.

class processes(object):

    def __init__(self, binding):
        self.bind = binding

        self.define_dict = {}
        self.volumes_dict = {}
        self.reflsolids_dict = {}
        self.solids_dict = {}
        self.elements_dict = {}
	self.isotopes_dict = {}
        self.materials_dict = {}
	self.mediums_dict = {}

        self.reflections = {} # map (reflected_solid_name, solid_name)
        self.reflectedvols = {} # map (name of volume using reflected solid, solid)
        self.auxmap = {}

        self.world = 0

    def gdml_proc(self, elem):
	pass

    def setup_proc(self, elem):
        for el in elem[2]:
            if el[0]=='world':
                self.world = self.volumes_dict[el[1]['ref']]

    def const_proc(self, elem):
	globals()[elem[1]['name']] = eval(elem[1]['value'])

    def position_proc(self, elem):
        lun = '*'+elem[1].get('lunit','mm')

	x = elem[1].get('x','0')
	y = elem[1].get('y','0')
	z = elem[1].get('z','0')

	pos = self.bind.position(eval(x+lun), eval(y+lun), eval(z+lun))
        self.define_dict[elem[1]['name']] = pos

    def rotation_proc(self,elem):
        aun = '*'+elem[1].get('aunit','deg')

	dx = elem[1].get('x','0')
	dy = elem[1].get('y','0')
	dz = elem[1].get('z','0')

	rot = self.bind.rotation(eval(dx+aun), eval(dy+aun), eval(dz+aun))
        self.define_dict[elem[1]['name']] = rot

    def element_proc(self,elem):
        ncompo = 0
        fractions = {}
	density = 0

	if elem[1].has_key('Z'):
	    atom = 0
	    for subele in elem[2]:
		if subele[0] == 'atom':
		    atom = eval(subele[1]['value'])

	    ele = self.bind.element(elem[1]['name'],
				    elem[1]['formula'],
				    eval(elem[1]['Z']),
				    atom)

	    self.elements_dict[elem[1]['name']] =  ele
	else:
	    for subele in elem[2]:
		if subele[0] == 'fraction':
		    ncompo = ncompo + 1
		    fractions[subele[1]['ref']] = eval(subele[1]['n'])
		elif subele[0] == 'composite':
		    ncompo = ncompo + 1
		    fractions[subele[1]['ref']] = int(eval(subele[1]['n']))
		elif subele[0] == 'D':
		    density = eval(subele[1]['value'])

	    mele = self.bind.mixele(elem[1]['name'], ncompo, density)

	    i = 0
	    for frac_name in fractions.keys():
		self.bind.mix_addiso(mele,
				     self.elements_dict[frac_name],
				     i, fractions[frac_name])
		i = i + 1

    def isotope_proc(self,elem):
	a = 0
	d = 0
	for subele in elem[2]:
	    if subele[0]=='atom':
		a = eval(subele[1]['value'])
	    elif subele[0]=='D':
		d = eval(subele[1]['value'])

	iso = self.bind.isotope(elem[1]['name'],
				int(eval(elem[1]['Z'])),
				int(eval(elem[1]['N'])),
				a,d)

	self.isotopes_dict[elem[1]['name']] = iso

    def material_proc(self,elem):
        ncompo = 0
        fractions = {}

	if elem[1].has_key('Z'):
	    a = 0
	    d = 0
	    for subele in elem[2]:
		if subele[0]=='atom':
		    a = eval(subele[1]['value'])
		elif subele[0]=='D':
		    d = eval(subele[1]['value'])
	    mat = self.bind.material(elem[1]['name'], a, eval(elem[1]['Z']), d)

	    # is this OK??????????
	    # probaly not OK for G4
	    mat_ele = self.bind.element(elem[1]['name'],'',
					eval(elem[1]['Z']), a)
	    self.elements_dict[elem[1]['name']] = mat_ele
	else:
	    for subele in elem[2]:
		if subele[0] == 'fraction':
		    ncompo = ncompo + 1
		    fractions[subele[1]['ref']] = eval(subele[1]['n'])
		elif subele[0] == 'composite':
		    ncompo = ncompo + 1
		    fractions[subele[1]['ref']] = int(eval(subele[1]['n']))
		elif subele[0] == 'D':
		    density = eval(subele[1]['value'])
	    mat = self.bind.mixmat(elem[1]['name'], ncompo, density)

            i = 0
            for frac_name in fractions.keys():
                if self.elements_dict.has_key(frac_name):
                    self.bind.mix_addele(mat,
                                         self.elements_dict[frac_name],
                                         i, fractions[frac_name])
                else:
                    self.bind.mix_addele(mat,
                                         self.materials_dict[frac_name],
                                         i, fractions[frac_name])
		i = i + 1

	self.materials_dict[elem[1]['name']] = mat
	med = self.bind.medium(elem[1]['name'], mat)
	self.mediums_dict[elem[1]['name']] = med

    def volume_proc(self,elem):
        auxpairs = []
        reflex = 0
        solidname = 0
	for subele in elem[2]:
            if subele[0] == 'solidref':
		if self.solids_dict.has_key(subele[1]['ref']):
		    solid = self.solids_dict[subele[1]['ref']]
                elif self.reflsolids_dict.has_key(subele[1]['ref']):
                    solid = self.reflsolids_dict[subele[1]['ref']]
                    solidname = subele[1]['ref']
                    reflex = self.solids_dict[self.reflections[subele[1]['ref']]]
                else:
		    print 'Solid ',subele[1]['ref'],' not defined yet!'
            elif subele[0] == 'materialref':
		if self.mediums_dict.has_key(subele[1]['ref']):
		    medium = self.mediums_dict[subele[1]['ref']]
		else:
		    print 'Medium ',subele[1]['ref'],' not defined yet!'
            elif subele[0] == 'auxiliary':
                auxpairs.append((subele[1]['auxtype'], subele[1]['auxvalue']))

        lvol = self.bind.logvolume(elem[1]['name'], solid, medium, reflex)
        self.volumes_dict[elem[1]['name']] = lvol
        if reflex != 0:
            self.reflectedvols[elem[1]['name']] = solidname

        if auxpairs != []:
            self.auxmap[lvol] = auxpairs

        for subele in elem[2]:
            if subele[0] == 'physvol':
                # we need this variable to tell physvol that we are dealing with reflection
                reflected_vol = 0
                pos = self.bind.position(0,0,0)
                rot = self.bind.rotation(0,0,0)
                for subsubel in subele[2]:
                    if subsubel[0] == 'volumeref':
                        lv = self.volumes_dict[subsubel[1]['ref']]
                        if self.reflectedvols.has_key(subsubel[1]['ref']):
                            reflected_vol = self.reflectedvols[subsubel[1]['ref']]
                    elif subsubel[0] == 'positionref':
                        pos = self.define_dict[subsubel[1]['ref']]
                    elif subsubel[0] == 'rotationref':
                        rot = self.define_dict[subsubel[1]['ref']]

		self.bind.physvolume(elem[1]['name'],
				       lv, lvol, rot, pos, reflected_vol)
            elif subele[0] == 'divisionvol':
                lun = '*'+elem[1].get('lunit','mm')
                # we deal here with divisions
                for subsubel in subele[2]:
                    if subsubel[0] == 'volumeref':
                        lv = self.volumes_dict[subsubel[1]['ref']]
                self.bind.divisionvol(elem[1]['name'],
                                      lv, lvol,
                                      subele[1]['axis'],
                                      eval(subele[1]['number']),
                                      eval(subele[1]['width']+lun),
                                      eval(subele[1]['offset']+lun))

    def assembly_proc(self,elem):
	assem = self.bind.assembly(elem[1]['name'])
        self.volumes_dict[elem[1]['name']] = assem

        for subele in elem[2]:
            if subele[0] == 'physvol':
	        reflected_vol = 0
                pos = self.bind.position(0,0,0)
                rot = self.bind.rotation(0,0,0)
                for subsubel in subele[2]:
                    if subsubel[0] == 'volumeref':
                        lv = self.volumes_dict[subsubel[1]['ref']]
                    elif subsubel[0] == 'positionref':
                        pos = self.define_dict[subsubel[1]['ref']]
                    elif subsubel[0] == 'rotationref':
                        rot = self.define_dict[subsubel[1]['ref']]

		self.bind.physvolume(elem[1]['name'],
				       lv, assem, rot, pos, reflected_vol)

    def box_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')

	box = self.bind.box(elem[1]['name'],
			    eval(elem[1]['x']+lun)/2,
			    eval(elem[1]['y']+lun)/2,
			    eval(elem[1]['z']+lun)/2)

        self.solids_dict[elem[1]['name']] = box

    def paraboloid_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	paraboloid = self.bind.paraboloid(elem[1]['name'],
			                  eval(elem[1]['rlo']+lun),
			                  eval(elem[1]['rhi']+lun),
			                  eval(elem[1]['dz']+lun))

        self.solids_dict[elem[1]['name']] = paraboloid
	
    def arb8_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')

	arb8 = self.bind.arb8(elem[1]['name'],
			      eval(elem[1]['v1x']+lun),
			      eval(elem[1]['v1y']+lun),
			      eval(elem[1]['v2x']+lun),
			      eval(elem[1]['v2y']+lun),
			      eval(elem[1]['v3x']+lun),
			      eval(elem[1]['v3y']+lun),
			      eval(elem[1]['v4x']+lun),
			      eval(elem[1]['v4y']+lun),
			      eval(elem[1]['v5x']+lun),
			      eval(elem[1]['v5y']+lun),
			      eval(elem[1]['v6x']+lun),
			      eval(elem[1]['v6y']+lun),
			      eval(elem[1]['v7x']+lun),
			      eval(elem[1]['v7y']+lun),
			      eval(elem[1]['v8x']+lun),
			      eval(elem[1]['v8y']+lun),
			      eval(elem[1]['dz']+lun))

        self.solids_dict[elem[1]['name']] = arb8

    def tube_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	tube = self.bind.tube(elem[1]['name'],
			      eval(elem[1].get('rmin','0')+lun),
			      eval(elem[1]['rmax']+lun),
			      eval(elem[1]['z']+lun)/2,
			      eval(elem[1].get('startphi','0')+aun),
			      eval(elem[1]['deltaphi']+aun))

        self.solids_dict[elem[1]['name']] = tube
	
    def cutTube_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	cutTube = self.bind.cutTube(elem[1]['name'],
			            eval(elem[1].get('rmin','0')+lun),
			            eval(elem[1]['rmax']+lun),
			            eval(elem[1]['z']+lun)/2,
			            eval(elem[1].get('startphi','0')+aun),
			            eval(elem[1]['deltaphi']+aun),
				    eval(elem[1]['lowX']+lun),
				    eval(elem[1]['lowY']+lun),
				    eval(elem[1]['lowZ']+lun),
				    eval(elem[1]['highX']+lun),
				    eval(elem[1]['highY']+lun),
				    eval(elem[1]['highZ']+lun))

        self.solids_dict[elem[1]['name']] = cutTube

    def cone_proc(self,elem):
 	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')


	cone = self.bind.cone(elem[1]['name'],
			      eval(elem[1].get('rmin1','0')+lun),
			      eval(elem[1]['rmax1']+lun),
			      eval(elem[1].get('rmin2','0')+lun),
			      eval(elem[1]['rmax2']+lun),
			      eval(elem[1]['z']+lun)/2,
			      eval(elem[1].get('startphi','0')+aun),
			      eval(elem[1]['deltaphi']+aun))

        self.solids_dict[elem[1]['name']] = cone

    def polycone_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	zrs=[]
	for subele in elem[2]:
	    zrs.append((eval(subele[1]['z']+lun),
			eval(subele[1]['rmin']+lun),
			eval(subele[1]['rmax']+lun)))

	polycone = self.bind.polycone(elem[1]['name'],
				      eval(elem[1].get('startphi','0')+aun),
				      eval(elem[1]['deltaphi']+aun),
				      zrs)

	self.solids_dict[elem[1]['name']] = polycone

    def trap_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	trap = self.bind.trap(elem[1]['name'],
			      eval(elem[1]['x1']+lun)/2,
			      eval(elem[1]['x2']+lun)/2,
			      eval(elem[1]['x3']+lun)/2,
			      eval(elem[1]['x4']+lun)/2,
			      eval(elem[1]['y1']+lun)/2,
			      eval(elem[1]['y2']+lun)/2,
			      eval(elem[1]['z']+lun)/2,
			      eval(elem[1]['alpha1']+aun),
			      eval(elem[1]['alpha2']+aun),
			      eval(elem[1]['phi']+aun),
			      eval(elem[1]['theta']+aun))

 	self.solids_dict[elem[1]['name']] = trap
	
    def twisttrap_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	twisttrap = self.bind.twisttrap(elem[1]['name'],
			                eval(elem[1]['x1']+lun)/2,
			                eval(elem[1]['x2']+lun)/2,
			                eval(elem[1]['x3']+lun)/2,
			                eval(elem[1]['x4']+lun)/2,
			                eval(elem[1]['y1']+lun)/2,
			                eval(elem[1]['y2']+lun)/2,
			                eval(elem[1]['z']+lun)/2,
			                eval(elem[1]['alpha1']+aun),
			                eval(elem[1]['alpha2']+aun),
			                eval(elem[1]['phi']+aun),
			                eval(elem[1]['theta']+aun),
					eval(elem[1]['twist']+aun))

 	self.solids_dict[elem[1]['name']] = twisttrap

    def trd_proc(self,elem):
	lun = '*'+elem[1].get('lunit','mm')

	trd = self.bind.trd(elem[1]['name'],
			    eval(elem[1]['x1']+lun)/2,
			    eval(elem[1]['x2']+lun)/2,
			    eval(elem[1]['y1']+lun)/2,
			    eval(elem[1]['y2']+lun)/2,
			    eval(elem[1]['z']+lun)/2)

	self.solids_dict[elem[1]['name']] = trd

    def sphere_proc(self, elem):
	 lun = '*'+elem[1].get('lunit','mm')
	 aun = '*'+elem[1].get('aunit','deg')

	 sphere = self.bind.sphere(elem[1]['name'],
				   eval(elem[1].get('rmin','0')+lun),
				   eval(elem[1]['rmax']+lun),
				   eval(elem[1].get('startphi','0')+aun),
				   eval(elem[1]['deltaphi']+aun),
				   eval(elem[1].get('starttheta','0')+aun),
				   eval(elem[1]['deltatheta']+aun))

	 self.solids_dict[elem[1]['name']] = sphere

    def orb_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')

	orb = self.bind.orb(elem[1]['name'],
			    eval(elem[1]['r']))

	self.solids_dict[elem[1]['name']] = orb

    def para_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	para = self.bind.para(elem[1]['name'],
			      eval(elem[1]['x']+lun),
			      eval(elem[1]['y']+lun),
			      eval(elem[1]['z']+lun),
			      eval(elem[1]['alpha']+aun),
			      eval(elem[1]['theta']+aun),
			      eval(elem[1]['phi']+aun))

	self.solids_dict[elem[1]['name']] = para

    def torus_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	torus = self.bind.torus(elem[1]['name'],
				eval(elem[1]['rmin']+lun),
				eval(elem[1]['rmax']+lun),
				eval(elem[1]['rtor']+lun),
				eval(elem[1]['startphi']+aun),
				eval(elem[1]['deltaphi']+aun))

	self.solids_dict[elem[1]['name']] = torus

    def hype_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	hype = self.bind.hype(elem[1]['name'],
				eval(elem[1]['rmin']+lun),
				eval(elem[1]['rmax']+lun),
				eval(elem[1]['inst']+aun),
				eval(elem[1]['outst']+aun),
				eval(elem[1]['z']+lun)/2)

	self.solids_dict[elem[1]['name']] = hype

    def polyhedra_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	zrs=[]
	for subele in elem[2]:
	    zrs.append((eval(subele[1]['z']+lun),
			eval(subele[1]['rmin']+lun),
			eval(subele[1]['rmax']+lun)))

	polyh = self.bind.polyhedra(elem[1]['name'],
				    eval(elem[1]['startphi']+aun),
				    eval(elem[1]['deltaphi']+aun),
				    int(eval(elem[1]['numsides'])),
				    zrs)

	self.solids_dict[elem[1]['name']] = polyh

    def xtru_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	aun = '*'+elem[1].get('aunit','deg')

	vertices=[]
	sections=[]
	for subele in elem[2]:
	    if subele[0]=='twoDimVertex':
	        vertices.append((eval(subele[1]['x']+lun), eval(subele[1]['y']+lun)))
	    elif subele[0]=='section':
	        sections.append((eval(subele[1]['zOrder']+lun), eval(subele[1]['zPosition']+lun), eval(subele[1]['xOffset']+lun), eval(subele[1]['yOffset']+lun), eval(subele[1]['scalingFactor']+lun)))

	xtru = self.bind.xtrusion(elem[1]['name'], vertices, sections)

	self.solids_dict[elem[1]['name']] = xtru
    
    def eltube_proc(self, elem):
	lun = '*'+elem[1].get('lunit','mm')
	eltube = self.bind.eltube(elem[1]['name'],
				  eval(elem[1]['dx']+lun),
				  eval(elem[1]['dy']+lun),
				  eval(elem[1]['dz']+lun))

	self.solids_dict[elem[1]['name']] = eltube

    def subtraction_proc(self, elem):
	pos = self.bind.position(0,0,0)
	rot = self.bind.rotation(0,0,0)

        for subele in elem[2]:
            if subele[0] == 'first':
		first = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'second':
		second = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'position':
		self.position_proc(subele)
		pos = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'positionref':
		pos = self.define_dict[subele[1]['ref']]
	    elif subele[0] == 'rotation':
		self.rotation_proc(subele)
		rot = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'rotationref':
		rot = self.define_dict[subele[1]['ref']]

	subt = self.bind.subtraction(elem[1]['name'],
				     first,
				     second,
				     pos,
				     rot)

	self.solids_dict[elem[1]['name']] = subt

    def union_proc(self, elem):
	pos = self.bind.position(0,0,0)
	rot = self.bind.rotation(0,0,0)

        for subele in elem[2]:
            if subele[0] == 'first':
		first = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'second':
		second = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'position':
		self.position_proc(subele)
		pos = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'positionref':
		pos = self.define_dict[subele[1]['ref']]
	    elif subele[0] == 'rotation':
		self.rotation_proc(subele)
		rot = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'rotationref':
		rot = self.define_dict[subele[1]['ref']]

	uni = self.bind.union(elem[1]['name'],
				     first,
				     second,
				     pos,
				     rot)

	self.solids_dict[elem[1]['name']] = uni

    def intersection_proc(self, elem):
	pos = self.bind.position(0,0,0)
	rot = self.bind.rotation(0,0,0)

        for subele in elem[2]:
            if subele[0] == 'first':
		first = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'second':
		second = self.solids_dict[subele[1]['ref']]
	    elif subele[0] == 'position':
		self.position_proc(subele)
		pos = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'positionref':
		pos = self.define_dict[subele[1]['ref']]
	    elif subele[0] == 'rotation':
		self.rotation_proc(subele)
		rot = self.define_dict[subele[1]['name']]
	    elif subele[0] == 'rotationref':
		rot = self.define_dict[subele[1]['ref']]

	inte = self.bind.intersection(elem[1]['name'],
				     first,
				     second,
				     pos,
				     rot)

	self.solids_dict[elem[1]['name']] = inte

    def reflection_proc(self, elem):
        refl = self.bind.reflection(elem[1]['name'], elem[1]['solid'],
                                    eval(elem[1]['sx']), eval(elem[1]['sy']), eval(elem[1]['sz']),
                                    eval(elem[1]['rx']), eval(elem[1]['ry']), eval(elem[1]['rz']),
                                    eval(elem[1]['dx']), eval(elem[1]['dy']), eval(elem[1]['dz']))
        self.reflsolids_dict[elem[1]['name']] = refl
        self.reflections[elem[1]['name']] = elem[1]['solid']


### dictionary mapping element name to 'process method'

    gdmlel_dict = { 'gdml':gdml_proc, 'setup':setup_proc, 'constant':const_proc,
                    'position':position_proc, 'rotation':rotation_proc,
                    'element':element_proc, 'isotope':isotope_proc,
		    'material':material_proc, 'twistTrap':twisttrap_proc,
                    'volume':volume_proc, 'assembly':assembly_proc, 'cutTube':cutTube_proc,
		    'box':box_proc, 'xtru': xtru_proc, 'arb8':arb8_proc, 'tube':tube_proc,
		    'cone':cone_proc, 'polycone':polycone_proc, 'hype':hype_proc,
		    'trap':trap_proc, 'trd':trd_proc, 'sphere':sphere_proc,
		    'orb':orb_proc, 'para':para_proc, 'torus':torus_proc,
		    'polyhedra':polyhedra_proc, 'eltube':eltube_proc,
		    'subtraction':subtraction_proc, 'union':union_proc, 'paraboloid':paraboloid_proc,
		    'intersection':intersection_proc,
                    'reflectedSolid':reflection_proc}

