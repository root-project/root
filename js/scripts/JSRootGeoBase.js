/** @file JSRootGeoBase.js */
/// Basic functions for work with TGeo classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( [ 'JSRootCore', 'threejs', 'ThreeCSG' ], factory );
   } else if (typeof exports === 'object' && typeof module !== 'undefined') {
      factory(require("./JSRootCore.js"), require("three"), require("./ThreeCSG.js"));
   } else {
      if (typeof JSROOT == 'undefined')
         throw new Error('JSROOT is not defined', 'JSRootGeoBase.js');
      if (typeof THREE == 'undefined')
         throw new Error('THREE is not defined', 'JSRootGeoBase.js');
      if (typeof ThreeBSP == 'undefined')
         throw new Error('ThreeBSP is not defined', 'JSRootGeoBase.js');
      factory(JSROOT, THREE, ThreeBSP);
   }
} (function( JSROOT, THREE, ThreeBSP ) {

   "use strict";

   /** @namespace JSROOT.GEO */
   /// Holder of all TGeo-related functions and classes
   JSROOT.GEO = {
         GradPerSegm: 6,     // grad per segment in cylinder/spherical symmetry shapes
         CompressComp: true,  // use faces compression in composite shapes
         CompLimit: 20        // maximal number of components in composite shape
    };

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.BITS = {
         kVisOverride     : JSROOT.BIT(0),           // volume's vis. attributes are overwritten
         kVisNone         : JSROOT.BIT(1),           // the volume/node is invisible, as well as daughters
         kVisThis         : JSROOT.BIT(2),           // this volume/node is visible
         kVisDaughters    : JSROOT.BIT(3),           // all leaves are visible
         kVisOneLevel     : JSROOT.BIT(4),           // first level daughters are visible (not used)
         kVisStreamed     : JSROOT.BIT(5),           // true if attributes have been streamed
         kVisTouched      : JSROOT.BIT(6),           // true if attributes are changed after closing geom
         kVisOnScreen     : JSROOT.BIT(7),           // true if volume is visible on screen
         kVisContainers   : JSROOT.BIT(12),          // all containers visible
         kVisOnly         : JSROOT.BIT(13),          // just this visible
         kVisBranch       : JSROOT.BIT(14),          // only a given branch visible
         kVisRaytrace     : JSROOT.BIT(15)           // raytracing flag
      };

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.TestBit = function(volume, f) {
      var att = volume.fGeoAtt;
      return att === undefined ? false : ((att & f) !== 0);
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.SetBit = function(volume, f, value) {
      if (volume.fGeoAtt === undefined) return;
      volume.fGeoAtt = value ? (volume.fGeoAtt | f) : (volume.fGeoAtt & ~f);
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.ToggleBit = function(volume, f) {
      if (volume.fGeoAtt !== undefined)
         volume.fGeoAtt = volume.fGeoAtt ^ (f & 0xffffff);
   }

   /** @memberOf JSROOT.GEO
    * implementation of TGeoVolume::InvisibleAll */
   JSROOT.GEO.InvisibleAll = function(flag) {
      if (flag===undefined) flag = true;

      JSROOT.GEO.SetBit(this, JSROOT.GEO.BITS.kVisThis, !flag);
      // JSROOT.GEO.SetBit(this, JSROOT.GEO.BITS.kVisDaughters, !flag);

      if (this.fNodes)
         for (var n=0;n<this.fNodes.arr.length;++n) {
            var sub = this.fNodes.arr[n].fVolume;
            JSROOT.GEO.SetBit(sub, JSROOT.GEO.BITS.kVisThis, !flag);
            // JSROOT.GEO.SetBit(sub, JSROOT.GEO.BITS.kVisDaughters, !flag);
         }
   }

   /** method used to avoid duplication of warnings
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.warn = function(msg) {
      if (JSROOT.GEO._warn_msgs === undefined) JSROOT.GEO._warn_msgs = {};
      if (JSROOT.GEO._warn_msgs[msg] !== undefined) return;
      JSROOT.GEO._warn_msgs[msg] = true;
      console.warn(msg);
   }

   /** @brief Returns kind of the geo node
    *  @desc  0 - TGeoNode
    *         1 - TEveGeoNode
    *        -1 - unsupported
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.NodeKind = function(obj) {
      if ((obj === undefined) || (obj === null) || (typeof obj !== 'object')) return -1;
      return ('fShape' in obj) && ('fTrans' in obj) ? 1 : 0;
   }

   /** @brief Returns number of shapes
    *  @desc Used to count total shapes number in composites
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.CountNumShapes = function(shape) {
      if (!shape) return 0;
      // if (shape._typename=="TGeoHalfSpace") JSROOT.GEO.HalfSpace = true;
      if (shape._typename!=='TGeoCompositeShape') return 1;
      return JSROOT.GEO.CountNumShapes(shape.fNode.fLeft) + JSROOT.GEO.CountNumShapes(shape.fNode.fRight);
   }

   // ==========================================================================

   JSROOT.GEO.GeometryCreator = function(numfaces) {
      this.nfaces = numfaces;
      this.indx = 0;
      this.pos = new Float32Array(numfaces*9);
      this.norm = new Float32Array(numfaces*9);

      return this;
   }

   JSROOT.GEO.GeometryCreator.prototype.AddFace3 = function(x1,y1,z1,
                                                            x2,y2,z2,
                                                            x3,y3,z3) {
      var indx = this.indx, pos = this.pos;
      pos[indx] = x1;
      pos[indx+1] = y1;
      pos[indx+2] = z1;
      pos[indx+3] = x2;
      pos[indx+4] = y2;
      pos[indx+5] = z2;
      pos[indx+6] = x3;
      pos[indx+7] = y3;
      pos[indx+8] = z3;
      this.last4 = false;
      this.indx = indx + 9;
   }

   JSROOT.GEO.GeometryCreator.prototype.StartPolygon = function() {}
   JSROOT.GEO.GeometryCreator.prototype.StopPolygon = function() {}

   JSROOT.GEO.GeometryCreator.prototype.AddFace4 = function(x1,y1,z1,
                                                            x2,y2,z2,
                                                            x3,y3,z3,
                                                            x4,y4,z4,
                                                            reduce) {
      // from four vertices one normally creates two faces (1,2,3) and (1,3,4)
      // if (reduce==1), first face is reduced
      // if (reduce==2), second face is reduced

      var indx = this.indx, pos = this.pos;

      if (reduce!==1) {
         pos[indx] = x1;
         pos[indx+1] = y1;
         pos[indx+2] = z1;
         pos[indx+3] = x2;
         pos[indx+4] = y2;
         pos[indx+5] = z2;
         pos[indx+6] = x3;
         pos[indx+7] = y3;
         pos[indx+8] = z3;
         indx+=9;
      }

      if (reduce!==2) {
         pos[indx] = x1;
         pos[indx+1] = y1;
         pos[indx+2] = z1;
         pos[indx+3] = x3;
         pos[indx+4] = y3;
         pos[indx+5] = z3;
         pos[indx+6] = x4;
         pos[indx+7] = y4;
         pos[indx+8] = z4;
         indx+=9;
      }

      this.last4 = (indx !== this.indx + 9);
      this.indx = indx;
   }

   JSROOT.GEO.GeometryCreator.prototype.SetNormal4 = function(nx1,ny1,nz1,
                                                              nx2,ny2,nz2,
                                                              nx3,ny3,nz3,
                                                              nx4,ny4,nz4,
                                                              reduce) {
     // same as AddFace4, assign normals for each individual vertex
     // reduce has same meaning and should be the same

      if (this.last4 && reduce)
         return console.error('missmatch between AddFace4 and SetNormal4 calls');

      var indx = this.indx - (this.last4 ? 18 : 9), norm = this.norm;

      if (reduce!==1) {
         norm[indx] = nx1;
         norm[indx+1] = ny1;
         norm[indx+2] = nz1;
         norm[indx+3] = nx2;
         norm[indx+4] = ny2;
         norm[indx+5] = nz2;
         norm[indx+6] = nx3;
         norm[indx+7] = ny3;
         norm[indx+8] = nz3;
         indx+=9;
      }

      if (reduce!==2) {
         norm[indx] = nx1;
         norm[indx+1] = ny1;
         norm[indx+2] = nz1;
         norm[indx+3] = nx3;
         norm[indx+4] = ny3;
         norm[indx+5] = nz3;
         norm[indx+6] = nx4;
         norm[indx+7] = ny4;
         norm[indx+8] = nz4;
      }
   }

   JSROOT.GEO.GeometryCreator.prototype.RecalcZ = function(func) {
      var pos = this.pos,
          last = this.indx,
          indx = last - (this.last4 ? 18 : 9);

      while (indx < last) {
         pos[indx+2] = func(pos[indx], pos[indx+1], pos[indx+2]);
         indx+=3;
      }
   }

   JSROOT.GEO.GetNormal = function(x1,y1,z1,x2,y2,z2,x3,y3,z3) {

      var pA = new THREE.Vector3(x1,y1,z1),
          pB = new THREE.Vector3(x2,y2,z2),
          pC = new THREE.Vector3(x3,y3,z3),
          cb = new THREE.Vector3(),
          ab = new THREE.Vector3();

      cb.subVectors( pC, pB );
      ab.subVectors( pA, pB );
      cb.cross(ab );

      return cb;
   }

   JSROOT.GEO.GeometryCreator.prototype.CalcNormal = function() {
      var indx = this.indx, norm = this.norm;

      if (!this.cb) {
         this.pA = new THREE.Vector3();
         this.pB = new THREE.Vector3();
         this.pC = new THREE.Vector3();
         this.cb = new THREE.Vector3();
         this.ab = new THREE.Vector3();
      }

      this.pA.fromArray( this.pos, this.indx - 9 );
      this.pB.fromArray( this.pos, this.indx - 6 );
      this.pC.fromArray( this.pos, this.indx - 3 );

      this.cb.subVectors( this.pC, this.pB );
      this.ab.subVectors( this.pA, this.pB );
      this.cb.cross( this.ab );

      this.SetNormal(this.cb.x, this.cb.y, this.cb.z);
   }

   JSROOT.GEO.GeometryCreator.prototype.SetNormal = function(nx,ny,nz) {
      var indx = this.indx - 9, norm = this.norm;

      norm[indx]   = norm[indx+3] = norm[indx+6] = nx;
      norm[indx+1] = norm[indx+4] = norm[indx+7] = ny;
      norm[indx+2] = norm[indx+5] = norm[indx+8] = nz;

      if (this.last4) {
         indx -= 9;
         norm[indx]   = norm[indx+3] = norm[indx+6] = nx;
         norm[indx+1] = norm[indx+4] = norm[indx+7] = ny;
         norm[indx+2] = norm[indx+5] = norm[indx+8] = nz;
      }
   }

   JSROOT.GEO.GeometryCreator.prototype.SetNormal_12_34 = function(nx12,ny12,nz12,nx34,ny34,nz34,reduce) {
      // special shortcut, when same normals can be applied for 1-2 point and 3-4 point
      if (reduce===undefined) reduce = 0;

      var indx = this.indx - ((reduce>0) ? 9 : 18), norm = this.norm;

      if (reduce!==1) {
         norm[indx]   = nx12;
         norm[indx+1] = ny12;
         norm[indx+2] = nz12;
         norm[indx+3] = nx12;
         norm[indx+4] = ny12;
         norm[indx+5] = nz12;
         norm[indx+6] = nx34;
         norm[indx+7] = ny34;
         norm[indx+8] = nz34;
         indx+=9;
      }

      if (reduce!==2) {
         norm[indx]   = nx12;
         norm[indx+1] = ny12;
         norm[indx+2] = nz12;
         norm[indx+3] = nx34;
         norm[indx+4] = ny34;
         norm[indx+5] = nz34;
         norm[indx+6] = nx34;
         norm[indx+7] = ny34;
         norm[indx+8] = nz34;
         indx+=9;
      }
   }

   JSROOT.GEO.GeometryCreator.prototype.Create = function() {
      if (this.nfaces !== this.indx/9)
         console.error('Mismatch with created ' + this.nfaces + ' and filled ' + this.indx/9 + ' number of faces');

      var geometry = new THREE.BufferGeometry();
      geometry.addAttribute( 'position', new THREE.BufferAttribute( this.pos, 3 ) );
      geometry.addAttribute( 'normal', new THREE.BufferAttribute( this.norm, 3 ) );
      return geometry;
   }

   // ================================================================================

   // same methods as GeometryCreator, but with different implementation

   JSROOT.GEO.PolygonsCreator = function() {
      this.polygons = [];
   }

   JSROOT.GEO.PolygonsCreator.prototype.StartPolygon = function(normal) {
      this.multi = 1;
      this.mnormal = normal;
   }

   JSROOT.GEO.PolygonsCreator.prototype.StopPolygon = function() {
      if (!this.multi) return;
      this.multi = 0;
      console.error('Polygon should be already closed at this moment');
   }

   JSROOT.GEO.PolygonsCreator.prototype.AddFace3 = function(x1,y1,z1,
                                                            x2,y2,z2,
                                                            x3,y3,z3) {
      this.AddFace4(x1,y1,z1,x2,y2,z2,x3,y3,z3,x3,y3,z3,2);
   }


   JSROOT.GEO.PolygonsCreator.prototype.AddFace4 = function(x1,y1,z1,
                                                            x2,y2,z2,
                                                            x3,y3,z3,
                                                            x4,y4,z4,
                                                            reduce) {
      // from four vertices one normally creates two faces (1,2,3) and (1,3,4)
      // if (reduce==1), first face is reduced
      //  if (reduce==2), second face is reduced

      if (reduce === undefined) reduce = 0;

      this.v1 = new ThreeBSP.Vertex( x1, y1, z1, 0, 0, 0 );
      this.v2 = (reduce===1) ? null : new ThreeBSP.Vertex( x2, y2, z2, 0, 0, 0 );
      this.v3 = new ThreeBSP.Vertex( x3, y3, z3, 0, 0, 0 );
      this.v4 = (reduce===2) ? null : new ThreeBSP.Vertex( x4, y4, z4, 0, 0, 0 );

      this.reduce = reduce;

      if (this.multi) {

         if (reduce!==2) console.error('polygon not supported for not-reduced faces');

         var polygon;

         if (this.multi++ === 1) {
            polygon = new ThreeBSP.Polygon;

            polygon.vertices.push(this.mnormal ? this.v2 : this.v3);
            this.polygons.push(polygon);
         } else {
            polygon = this.polygons[this.polygons.length-1];
            // check that last vertice equals to v2
            var last = this.mnormal ? polygon.vertices[polygon.vertices.length-1] : polygon.vertices[0],
                comp = this.mnormal ? this.v2 : this.v3;

            if (comp.diff(last) > 1e-12)
               console.error('vertex missmatch when building polygon');
         }

         var first = this.mnormal ? polygon.vertices[0] : polygon.vertices[polygon.vertices.length-1],
             next = this.mnormal ? this.v3 : this.v2;

         if (next.diff(first) < 1e-12) {
            //console.log('polygon closed!!!', polygon.vertices.length);
            this.multi = 0;
         } else
         if (this.mnormal) {
            polygon.vertices.push(this.v3);
         } else {
            polygon.vertices.unshift(this.v2);
         }

         return;

      }

      var polygon = new ThreeBSP.Polygon;

      switch (reduce) {
         case 0: polygon.vertices.push(this.v1, this.v2, this.v3, this.v4); break;
         case 1: polygon.vertices.push(this.v1, this.v3, this.v4); break;
         case 2: polygon.vertices.push(this.v1, this.v2, this.v3); break;
      }

      this.polygons.push(polygon);
   }

   JSROOT.GEO.PolygonsCreator.prototype.SetNormal4 = function(nx1,ny1,nz1,
                                                              nx2,ny2,nz2,
                                                              nx3,ny3,nz3,
                                                              nx4,ny4,nz4,
                                                              reduce) {
      this.v1.setnormal(nx1,ny1,nz1);
      if (this.v2) this.v2.setnormal(nx2,ny2,nz2);
      this.v3.setnormal(nx3,ny3,nz3);
      if (this.v4) this.v4.setnormal(nx4,ny4,nz4);
   }

   JSROOT.GEO.PolygonsCreator.prototype.SetNormal_12_34 = function(nx12,ny12,nz12,nx34,ny34,nz34,reduce) {
      // special shortcut, when same normals can be applied for 1-2 point and 3-4 point
      this.v1.setnormal(nx12,ny12,nz12);
      if (this.v2) this.v2.setnormal(nx12,ny12,nz12);
      this.v3.setnormal(nx34,ny34,nz34);
      if (this.v4) this.v4.setnormal(nx34,ny34,nz34);
   }

   JSROOT.GEO.PolygonsCreator.prototype.CalcNormal = function() {

      if (!this.cb) {
         this.pA = new THREE.Vector3();
         this.pB = new THREE.Vector3();
         this.pC = new THREE.Vector3();
         this.cb = new THREE.Vector3();
         this.ab = new THREE.Vector3();
      }

      this.pA.set( this.v1.x, this.v1.y, this.v1.z);

      if (this.reduce!==1) {
         this.pB.set( this.v2.x, this.v2.y, this.v2.z);
         this.pC.set( this.v3.x, this.v3.y, this.v3.z);
      } else {
         this.pB.set( this.v3.x, this.v3.y, this.v3.z);
         this.pC.set( this.v4.x, this.v4.y, this.v4.z);
      }

      this.cb.subVectors( this.pC, this.pB );
      this.ab.subVectors( this.pA, this.pB );
      this.cb.cross( this.ab );

      this.SetNormal(this.cb.x, this.cb.y, this.cb.z);
   }


   JSROOT.GEO.PolygonsCreator.prototype.SetNormal = function(nx,ny,nz) {
      this.v1.setnormal(nx,ny,nz);
      if (this.v2) this.v2.setnormal(nx,ny,nz);
      this.v3.setnormal(nx,ny,nz);
      if (this.v4) this.v4.setnormal(nx,ny,nz);
   }

   JSROOT.GEO.PolygonsCreator.prototype.RecalcZ = function(func) {
      this.v1.z = func(this.v1.x, this.v1.y, this.v1.z);
      if (this.v2) this.v2.z = func(this.v2.x, this.v2.y, this.v2.z);
      this.v3.z = func(this.v3.x, this.v3.y, this.v3.z);
      if (this.v4) this.v4.z = func(this.v4.x, this.v4.y, this.v4.z);
   }

   JSROOT.GEO.PolygonsCreator.prototype.Create = function() {
      return { polygons: this.polygons };
   }

   // ================= all functions to create geometry ===================================

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createCubeBuffer = function(shape, faces_limit) {

      if (faces_limit < 0) return 12;

      var dx = shape.fDX, dy = shape.fDY, dz = shape.fDZ;

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(12);

      creator.AddFace4(dx,dy,dz, dx,-dy,dz, dx,-dy,-dz, dx,dy,-dz); creator.SetNormal(1,0,0);

      creator.AddFace4(-dx,dy,-dz, -dx,-dy,-dz, -dx,-dy,dz, -dx,dy,dz); creator.SetNormal(-1,0,0);

      creator.AddFace4(-dx,dy,-dz, -dx,dy,dz, dx,dy,dz, dx,dy,-dz); creator.SetNormal(0,1,0);

      creator.AddFace4(-dx,-dy,dz, -dx,-dy,-dz, dx,-dy,-dz, dx,-dy,dz); creator.SetNormal(0,-1,0);

      creator.AddFace4(-dx,dy,dz, -dx,-dy,dz, dx,-dy,dz, dx,dy,dz); creator.SetNormal(0,0,1);

      creator.AddFace4(dx,dy,-dz, dx,-dy,-dz, -dx,-dy,-dz, -dx,dy,-dz); creator.SetNormal(0,0,-1);

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.create8edgesBuffer = function( v, faces_limit ) {

      var indicies = [ 4,7,6,5,  0,3,7,4,  4,5,1,0,  6,2,1,5,  7,3,2,6,  1,2,3,0 ];

      var creator = (faces_limit > 0) ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(12);

      for (var n=0;n<indicies.length;n+=4) {
         var i1 = indicies[n]*3,
             i2 = indicies[n+1]*3,
             i3 = indicies[n+2]*3,
             i4 = indicies[n+3]*3;
         creator.AddFace4(v[i1], v[i1+1], v[i1+2], v[i2], v[i2+1], v[i2+2],
                          v[i3], v[i3+1], v[i3+2], v[i4], v[i4+1], v[i4+2]);
         if (n===0) creator.SetNormal(0,0,1); else
         if (n===20) creator.SetNormal(0,0,-1); else creator.CalcNormal();
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createParaBuffer = function( shape, faces_limit ) {

      if (faces_limit < 0) return 12;

      var txy = shape.fTxy, txz = shape.fTxz, tyz = shape.fTyz;

      var v = [
          -shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
           shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ ];

      return JSROOT.GEO.create8edgesBuffer(v, faces_limit );
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createTrapezoidBuffer = function( shape, faces_limit ) {

      if (faces_limit < 0) return 12;

      var y1, y2;
      if (shape._typename == "TGeoTrd1") {
         y1 = y2 = shape.fDY;
      } else {
         y1 = shape.fDy1; y2 = shape.fDy2;
      }

      var v = [
            -shape.fDx1,  y1, -shape.fDZ,
             shape.fDx1,  y1, -shape.fDZ,
             shape.fDx1, -y1, -shape.fDZ,
            -shape.fDx1, -y1, -shape.fDZ,
            -shape.fDx2,  y2,  shape.fDZ,
             shape.fDx2,  y2,  shape.fDZ,
             shape.fDx2, -y2,  shape.fDZ,
            -shape.fDx2, -y2,  shape.fDZ
         ];

      return JSROOT.GEO.create8edgesBuffer(v, faces_limit );
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createArb8Buffer = function( shape, faces_limit ) {

      if (faces_limit < 0) return 12;

      var vertices = [
            shape.fXY[0][0], shape.fXY[0][1], -shape.fDZ,
            shape.fXY[1][0], shape.fXY[1][1], -shape.fDZ,
            shape.fXY[2][0], shape.fXY[2][1], -shape.fDZ,
            shape.fXY[3][0], shape.fXY[3][1], -shape.fDZ,
            shape.fXY[4][0], shape.fXY[4][1],  shape.fDZ,
            shape.fXY[5][0], shape.fXY[5][1],  shape.fDZ,
            shape.fXY[6][0], shape.fXY[6][1],  shape.fDZ,
            shape.fXY[7][0], shape.fXY[7][1],  shape.fDZ
         ],
         indicies = [
            4,7,6,   6,5,4,   0,3,4,   7,4,3,
            4,5,0,   1,0,5,   6,2,1,   1,5,6,
            7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      // detect same vertices on both Z-layers
      for (var side=0;side<vertices.length;side += vertices.length/2)
         for (var n1 = side; n1 < side + vertices.length/2 - 3 ; n1+=3)
            for (var n2 = n1+3; n2 < side + vertices.length/2 ; n2+=3)
               if ((vertices[n1] === vertices[n2]) &&
                   (vertices[n1+1] === vertices[n2+1]) &&
                   (vertices[n1+2] === vertices[n2+2])) {
                      for (var k=0;k<indicies.length;++k)
                        if (indicies[k] === n2/3) indicies[k] = n1/3;
                  }


      var map = [], // list of existing faces (with all rotations)
          numfaces = 0;

      for (var k=0;k<indicies.length;k+=3) {
         var id1 = indicies[k]*100   + indicies[k+1]*10 + indicies[k+2],
             id2 = indicies[k+1]*100 + indicies[k+2]*10 + indicies[k],
             id3 = indicies[k+2]*100 + indicies[k]*10   + indicies[k+1];

         if ((indicies[k] == indicies[k+1]) || (indicies[k] == indicies[k+2]) || (indicies[k+1] == indicies[k+2]) ||
             (map.indexOf(id1)>=0) || (map.indexOf(id2)>=0) || (map.indexOf(id3)>=0)) {
            indicies[k] = indicies[k+1] = indicies[k+2] = -1;
         } else {
            map.push(id1,id2,id3);
            numfaces++;
         }
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // var creator = new JSROOT.GEO.GeometryCreator(numfaces);

      for (var n=0; n < indicies.length; n+=6) {
         var i1 = indicies[n]   * 3,
             i2 = indicies[n+1] * 3,
             i3 = indicies[n+2] * 3,
             i4 = indicies[n+3] * 3,
             i5 = indicies[n+4] * 3,
             i6 = indicies[n+5] * 3,
             norm = null;

         if ((i1>=0) && (i4>=0) && faces_limit) {
            // try to identify two faces with same normal - very useful if one can create face4
            if (n===0) norm = new THREE.Vector3(0,0,1); else
            if (n===30) norm = new THREE.Vector3(0,0,-1); else {
               var norm1 = JSROOT.GEO.GetNormal(vertices[i1], vertices[i1+1], vertices[i1+2],
                                                vertices[i2], vertices[i2+1], vertices[i2+2],
                                                vertices[i3], vertices[i3+1], vertices[i3+2]);

               norm1.normalize();

               var norm2 = JSROOT.GEO.GetNormal(vertices[i4], vertices[i4+1], vertices[i4+2],
                                                vertices[i5], vertices[i5+1], vertices[i5+2],
                                                vertices[i6], vertices[i6+1], vertices[i6+2]);

               norm2.normalize();

               if (norm1.distanceToSquared(norm2) < 1e-12) norm = norm1;
            }
         }

         if (norm !== null) {
            creator.AddFace4(vertices[i1], vertices[i1+1], vertices[i1+2],
                             vertices[i2], vertices[i2+1], vertices[i2+2],
                             vertices[i3], vertices[i3+1], vertices[i3+2],
                             vertices[i5], vertices[i5+1], vertices[i5+2]);
            creator.SetNormal(norm.x, norm.y, norm.z);
         }  else {
            if (i1>=0) {
               creator.AddFace3(vertices[i1], vertices[i1+1], vertices[i1+2],
                                vertices[i2], vertices[i2+1], vertices[i2+2],
                                vertices[i3], vertices[i3+1], vertices[i3+2]);
               creator.CalcNormal();
            }
            if (i4>=0) {
               creator.AddFace3(vertices[i4], vertices[i4+1], vertices[i4+2],
                                vertices[i5], vertices[i5+1], vertices[i5+2],
                                vertices[i6], vertices[i6+1], vertices[i6+2]);
               creator.CalcNormal();
            }
         }
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createSphereBuffer = function( shape, faces_limit ) {
      var radius = [shape.fRmax, shape.fRmin],
          phiStart = shape.fPhi1,
          phiLength = shape.fPhi2 - shape.fPhi1,
          thetaStart = shape.fTheta1,
          thetaLength = shape.fTheta2 - shape.fTheta1,
          widthSegments = shape.fNseg,
          heightSegments = shape.fNz,
          noInside = (radius[1] <= 0);

      // widthSegments = 20; heightSegments = 10;
      // phiStart = 0; phiLength = 360; thetaStart = 0;  thetaLength = 180;

      if (faces_limit > 0) {
         var fact = (noInside ? 2 : 4) * widthSegments * heightSegments / faces_limit;

         if (fact > 1.) {
            widthSegments = Math.max(4, Math.floor(widthSegments/Math.sqrt(fact)));
            heightSegments = Math.max(4, Math.floor(heightSegments/Math.sqrt(fact)));
         }
      }

      var numoutside = widthSegments * heightSegments * 2,
          numtop = widthSegments * 2,
          numbottom = widthSegments * 2,
          numcut = phiLength === 360 ? 0 : heightSegments * (noInside ? 2 : 4),
          epsilon = 1e-10;

      if (noInside) numbottom = numtop = widthSegments;

      if (faces_limit < 0) return numoutside * (noInside ? 1 : 2) + numtop + numbottom + numcut;

      var _sinp = new Float32Array(widthSegments+1),
          _cosp = new Float32Array(widthSegments+1),
          _sint = new Float32Array(heightSegments+1),
          _cost = new Float32Array(heightSegments+1);

      for (var n=0;n<=heightSegments;++n) {
         var theta = (thetaStart + thetaLength/heightSegments*n)*Math.PI/180;
         _sint[n] = Math.sin(theta);
         _cost[n] = Math.cos(theta);
      }

      for (var n=0;n<=widthSegments;++n) {
         var phi = (phiStart + phiLength/widthSegments*n)*Math.PI/180;
         _sinp[n] = Math.sin(phi);
         _cosp[n] = Math.cos(phi);
      }

      if (Math.abs(_sint[0]) <= epsilon) { numoutside -= widthSegments; numtop = 0; }
      if (Math.abs(_sint[heightSegments]) <= epsilon) { numoutside -= widthSegments; numbottom = 0; }

      var numfaces = numoutside * (noInside ? 1 : 2) + numtop + numbottom + numcut;

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // var creator = new JSROOT.GEO.GeometryCreator(numfaces);

      for (var side=0;side<2;++side) {
         if ((side===1) && noInside) break;

         var r = radius[side],
             s = (side===0) ? 1 : -1,
             d1 = 1 - side, d2 = 1 - d1;

         // use direct algorithm for the sphere - here normals and position can be calculated directly
         for (var k=0;k<heightSegments;++k) {

            var k1 = k + d1, k2 = k + d2;

            var skip = 0;
            if (Math.abs(_sint[k1]) <= epsilon) skip = 1; else
            if (Math.abs(_sint[k2]) <= epsilon) skip = 2;

            for (var n=0;n<widthSegments;++n) {
               creator.AddFace4(
                     r*_sint[k1]*_cosp[n],   r*_sint[k1] *_sinp[n],   r*_cost[k1],
                     r*_sint[k1]*_cosp[n+1], r*_sint[k1] *_sinp[n+1], r*_cost[k1],
                     r*_sint[k2]*_cosp[n+1], r*_sint[k2] *_sinp[n+1], r*_cost[k2],
                     r*_sint[k2]*_cosp[n],   r*_sint[k2] *_sinp[n],   r*_cost[k2],
                     skip);
               creator.SetNormal4(
                     s*_sint[k1]*_cosp[n],   s*_sint[k1] *_sinp[n],   s*_cost[k1],
                     s*_sint[k1]*_cosp[n+1], s*_sint[k1] *_sinp[n+1], s*_cost[k1],
                     s*_sint[k2]*_cosp[n+1], s*_sint[k2] *_sinp[n+1], s*_cost[k2],
                     s*_sint[k2]*_cosp[n],   s*_sint[k2] *_sinp[n],   s*_cost[k2],
                     skip);
            }
         }
      }

      // top/bottom
      for (var side=0; side<=heightSegments; side+=heightSegments)
         if (Math.abs(_sint[side]) >= epsilon) {
            var ss = _sint[side], cc = _cost[side],
                d1 = (side===0) ? 0 : 1, d2 = 1 - d1;
            for (var n=0;n<widthSegments;++n) {
               creator.AddFace4(
                     radius[1] * ss * _cosp[n+d1], radius[1] * ss * _sinp[n+d1], radius[1] * cc,
                     radius[0] * ss * _cosp[n+d1], radius[0] * ss * _sinp[n+d1], radius[0] * cc,
                     radius[0] * ss * _cosp[n+d2], radius[0] * ss * _sinp[n+d2], radius[0] * cc,
                     radius[1] * ss * _cosp[n+d2], radius[1] * ss * _sinp[n+d2], radius[1] * cc,
                     noInside ? 2 : 0);
               creator.CalcNormal();
            }
         }

      // cut left/right sides
      if (phiLength < 360) {
         for (var side=0;side<=widthSegments;side+=widthSegments) {
            var ss = _sinp[side], cc = _cosp[side],
                d1 = (side === 0) ? 1 : 0, d2 = 1 - d1;

            for (var k=0;k<heightSegments;++k) {
               creator.AddFace4(
                     radius[1] * _sint[k+d1] * cc, radius[1] * _sint[k+d1] * ss, radius[1] * _cost[k+d1],
                     radius[0] * _sint[k+d1] * cc, radius[0] * _sint[k+d1] * ss, radius[0] * _cost[k+d1],
                     radius[0] * _sint[k+d2] * cc, radius[0] * _sint[k+d2] * ss, radius[0] * _cost[k+d2],
                     radius[1] * _sint[k+d2] * cc, radius[1] * _sint[k+d2] * ss, radius[1] * _cost[k+d2],
                     noInside ? 2 : 0);
               creator.CalcNormal();
            }
         }
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createTubeBuffer = function( shape, faces_limit) {
      var outerR, innerR; // inner/outer tube radius
      if ((shape._typename == "TGeoCone") || (shape._typename == "TGeoConeSeg")) {
         outerR = [ shape.fRmax2, shape.fRmax1 ];
         innerR = [ shape.fRmin2, shape.fRmin1 ];
      } else {
         outerR = [ shape.fRmax, shape.fRmax ];
         innerR = [ shape.fRmin, shape.fRmin ];
      }

      var hasrmin = (innerR[0] > 0) || (innerR[1] > 0),
          thetaStart = 0, thetaLength = 360;

      if ((shape._typename == "TGeoConeSeg") || (shape._typename == "TGeoTubeSeg") || (shape._typename == "TGeoCtub")) {
         thetaStart = shape.fPhi1;
         thetaLength = shape.fPhi2 - shape.fPhi1;
      }

      var radiusSegments = Math.max(4, Math.round(thetaLength/JSROOT.GEO.GradPerSegm));

      // external surface
      var numfaces = radiusSegments * (((outerR[0] <= 0) || (outerR[1] <= 0)) ? 1 : 2);

      // internal surface
      if (hasrmin)
         numfaces += radiusSegments * (((innerR[0] <= 0) || (innerR[1] <= 0)) ? 1 : 2);

      // upper cap
      if (outerR[0] > 0) numfaces += radiusSegments * ((innerR[0]>0) ? 2 : 1);
      // bottom cup
      if (outerR[1] > 0) numfaces += radiusSegments * ((innerR[1]>0) ? 2 : 1);

      if (thetaLength < 360)
         numfaces += ((outerR[0] > innerR[0]) ? 2 : 0) + ((outerR[1] > innerR[1]) ? 2 : 0);

      if (faces_limit < 0) return numfaces;

      var phi0 = thetaStart*Math.PI/180,
          dphi = thetaLength/radiusSegments*Math.PI/180,
          _sin = new Float32Array(radiusSegments+1),
          _cos = new Float32Array(radiusSegments+1);

      for (var seg=0; seg<=radiusSegments; ++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // var creator = new JSROOT.GEO.GeometryCreator(numfaces);

      var calcZ;

      if (shape._typename == "TGeoCtub")
         calcZ = function(x,y,z) {
            var arr = (z<0) ? shape.fNlow : shape.fNhigh;
            return ((z<0) ? -shape.fDz : shape.fDz) - (x*arr[0] + y*arr[1]) / arr[2];
         }

      // create outer/inner tube
      for (var side = 0; side<2; ++side) {
         if ((side === 1) && !hasrmin) break;

         var R = (side === 0) ? outerR : innerR,
             d1 = side, d2 = 1 - side, nxy = 1., nz = 0;

         if (R[0] !== R[1]) {
            var angle = Math.atan2((R[1]-R[0]), 2*shape.fDZ);
            nxy = Math.cos(angle);
            nz = Math.sin(angle);
         }

         if (side === 1) { nxy *= -1; nz *= -1; };

         var reduce = 0;
         if (R[0] <= 0) reduce = 2; else
         if (R[1] <= 0) reduce = 1;

         for (var seg=0;seg<radiusSegments;++seg) {
            creator.AddFace4(
                  R[0] * _cos[seg+d1], R[0] * _sin[seg+d1],  shape.fDZ,
                  R[1] * _cos[seg+d1], R[1] * _sin[seg+d1], -shape.fDZ,
                  R[1] * _cos[seg+d2], R[1] * _sin[seg+d2], -shape.fDZ,
                  R[0] * _cos[seg+d2], R[0] * _sin[seg+d2],  shape.fDZ,
                  reduce );

            if (calcZ) creator.RecalcZ(calcZ);

            creator.SetNormal_12_34(nxy*_cos[seg+d1], nxy*_sin[seg+d1], nz,
                                    nxy*_cos[seg+d2], nxy*_sin[seg+d2], nz,
                                    reduce);
         }
      }

      // create upper/bottom part
      for (var side = 0; side<2; ++side) {
         if (outerR[side] <= 0) continue;

         var d1 = side, d2 = 1- side,
             sign = (side == 0) ? 1 : -1,
             reduce = (innerR[side] <= 0) ? 2 : 0;
         if ((reduce==2) && (thetaLength === 360) && !calcZ ) creator.StartPolygon(side===0);
         for (var seg=0;seg<radiusSegments;++seg) {
            creator.AddFace4(
                  innerR[side] * _cos[seg+d1], innerR[side] * _sin[seg+d1], sign*shape.fDZ,
                  outerR[side] * _cos[seg+d1], outerR[side] * _sin[seg+d1], sign*shape.fDZ,
                  outerR[side] * _cos[seg+d2], outerR[side] * _sin[seg+d2], sign*shape.fDZ,
                  innerR[side] * _cos[seg+d2], innerR[side] * _sin[seg+d2], sign*shape.fDZ,
                  reduce);
            if (calcZ) {
               creator.RecalcZ(calcZ);
               creator.CalcNormal();
            } else {
               creator.SetNormal(0,0,sign);
            }
         }

         creator.StopPolygon();
      }

      // create cut surfaces
      if (thetaLength < 360) {
         creator.AddFace4(innerR[1] * _cos[0], innerR[1] * _sin[0], -shape.fDZ,
                          outerR[1] * _cos[0], outerR[1] * _sin[0], -shape.fDZ,
                          outerR[0] * _cos[0], outerR[0] * _sin[0],  shape.fDZ,
                          innerR[0] * _cos[0], innerR[0] * _sin[0],  shape.fDZ,
                          (outerR[0] === innerR[0]) ? 2 : ((innerR[1]===outerR[1]) ? 1 : 0) );
         if (calcZ) creator.RecalcZ(calcZ);
         creator.CalcNormal();

         creator.AddFace4(innerR[0] * _cos[radiusSegments], innerR[0] * _sin[radiusSegments],  shape.fDZ,
                          outerR[0] * _cos[radiusSegments], outerR[0] * _sin[radiusSegments],  shape.fDZ,
                          outerR[1] * _cos[radiusSegments], outerR[1] * _sin[radiusSegments], -shape.fDZ,
                          innerR[1] * _cos[radiusSegments], innerR[1] * _sin[radiusSegments], -shape.fDZ,
                          (outerR[0] === innerR[0]) ? 1 : ((innerR[1]===outerR[1]) ? 2 : 0));

         if (calcZ) creator.RecalcZ(calcZ);
         creator.CalcNormal();
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createEltuBuffer = function( shape , faces_limit ) {
      var radiusSegments = Math.max(4, Math.round(360/JSROOT.GEO.GradPerSegm));

      if (faces_limit < 0) return radiusSegments*4;

      // calculate all sin/cos tables in advance
      var x = new Float32Array(radiusSegments+1),
          y = new Float32Array(radiusSegments+1);
      for (var seg=0; seg<=radiusSegments; ++seg) {
          var phi = seg/radiusSegments*2*Math.PI;
          x[seg] = shape.fRmin*Math.cos(phi);
          y[seg] = shape.fRmax*Math.sin(phi);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(radiusSegments*4),
          nx1 = 1, ny1 = 0, nx2 = 1, ny2 = 0;

      // create tube faces
      for (var seg=0; seg<radiusSegments; ++seg) {
         creator.AddFace4(x[seg],   y[seg],   +shape.fDZ,
                          x[seg],   y[seg],   -shape.fDZ,
                          x[seg+1], y[seg+1], -shape.fDZ,
                          x[seg+1], y[seg+1],  shape.fDZ);

         // calculate normals ourself
         nx1 = nx2; ny1 = ny2;
         nx2 = x[seg+1] * shape.fRmax / shape.fRmin;
         ny2 = y[seg+1] * shape.fRmin / shape.fRmax;
         var dist = Math.sqrt(nx2*nx2 + ny2*ny2);
         nx2 = nx2 / dist; ny2 = ny2/dist;

         creator.SetNormal_12_34(nx1,ny1,0,nx2,ny2,0);
      }

      // create top/bottom sides
      for (var side=0;side<2;++side) {
         var sign = (side===0) ? 1 : -1, d1 = side, d2 = 1 - side;
         for (var seg=0; seg<radiusSegments; ++seg) {
            creator.AddFace3(0,          0,          sign*shape.fDZ,
                             x[seg+d1],  y[seg+d1],  sign*shape.fDZ,
                             x[seg+d2],  y[seg+d2],  sign*shape.fDZ);
            creator.SetNormal(0, 0, sign);
         }
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createTorusBuffer = function( shape, faces_limit ) {
      var radius = shape.fR,
          radialSegments = Math.max(6, Math.round(360/JSROOT.GEO.GradPerSegm)),
          tubularSegments = Math.max(8, Math.round(shape.fDphi/JSROOT.GEO.GradPerSegm));

      var numfaces = (shape.fRmin > 0 ? 4 : 2) * radialSegments * (tubularSegments + (shape.fDphi !== 360 ? 1 : 0));

      if (faces_limit < 0) return numfaces;

      if ((faces_limit > 0) && (numfaces > faces_limit)) {
         radialSegments = Math.floor(radialSegments/Math.sqrt(numfaces / faces_limit));
         tubularSegments = Math.floor(tubularSegments/Math.sqrt(numfaces / faces_limit));
         numfaces = (shape.fRmin > 0 ? 4 : 2) * radialSegments * (tubularSegments + (shape.fDphi !== 360 ? 1 : 0));
      }

      var _sinr = new Float32Array(radialSegments+1),
          _cosr = new Float32Array(radialSegments+1),
          _sint = new Float32Array(tubularSegments+1),
          _cost = new Float32Array(tubularSegments+1);

      for (var n=0;n<=radialSegments;++n) {
         _sinr[n] = Math.sin(n/radialSegments*2*Math.PI);
         _cosr[n] = Math.cos(n/radialSegments*2*Math.PI);
      }

      for (var t=0;t<=tubularSegments;++t) {
         var angle = (shape.fPhi1 + shape.fDphi*t/tubularSegments)/180*Math.PI;
         _sint[t] = Math.sin(angle);
         _cost[t] = Math.cos(angle);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // use vectors for normals calculation
      var p1 = new THREE.Vector3(), p2 = new THREE.Vector3(), p3 = new THREE.Vector3(), p4 = new THREE.Vector3(),
          n1 = new THREE.Vector3(), n2 = new THREE.Vector3(), n3 = new THREE.Vector3(), n4 = new THREE.Vector3(),
          center1 = new THREE.Vector3(), center2 = new THREE.Vector3();

      for (var side=0;side<2;++side) {
         if ((side > 0) && (shape.fRmin <= 0)) break;
         var tube = (side > 0) ? shape.fRmin : shape.fRmax,
             d1 = 1 - side, d2 = 1 - d1, ns = side>0 ? -1 : 1;

         for (var t=0;t<tubularSegments;++t) {
            var t1 = t + d1, t2 = t + d2;
            center1.x = radius * _cost[t1]; center1.y = radius * _sint[t1];
            center2.x = radius * _cost[t2]; center2.y = radius * _sint[t2];

            for (var n=0;n<radialSegments;++n) {
               p1.x = (radius + tube * _cosr[n])   * _cost[t1]; p1.y = (radius + tube * _cosr[n])   * _sint[t1]; p1.z = tube*_sinr[n];
               p2.x = (radius + tube * _cosr[n+1]) * _cost[t1]; p2.y = (radius + tube * _cosr[n+1]) * _sint[t1]; p2.z = tube*_sinr[n+1];
               p3.x = (radius + tube * _cosr[n+1]) * _cost[t2]; p3.y = (radius + tube * _cosr[n+1]) * _sint[t2]; p3.z = tube*_sinr[n+1];
               p4.x = (radius + tube * _cosr[n])   * _cost[t2]; p4.y = (radius + tube * _cosr[n])   * _sint[t2]; p4.z = tube*_sinr[n];

               creator.AddFace4(p1.x, p1.y, p1.z,
                                p2.x, p2.y, p2.z,
                                p3.x, p3.y, p3.z,
                                p4.x, p4.y, p4.z);

               n1.subVectors( p1, center1 ).normalize();
               n2.subVectors( p2, center1 ).normalize();
               n3.subVectors( p3, center2 ).normalize();
               n4.subVectors( p4, center2 ).normalize();

               creator.SetNormal4(ns*n1.x, ns*n1.y, ns*n1.z,
                                  ns*n2.x, ns*n2.y, ns*n2.z,
                                  ns*n3.x, ns*n3.y, ns*n3.z,
                                  ns*n4.x, ns*n4.y, ns*n4.z);
            }
         }
      }

      if (shape.fDphi !== 360)
         for (var t=0;t<=tubularSegments;t+=tubularSegments) {
            var tube1 = shape.fRmax, tube2 = shape.fRmin,
                d1 = (t>0) ? 0 : 1, d2 = 1 - d1,
                skip = (shape.fRmin) > 0 ?  0 : 1,
                nsign = t>0 ? 1 : -1;
            for (var n=0;n<radialSegments;++n) {
               creator.AddFace4((radius + tube1 * _cosr[n+d1]) * _cost[t], (radius + tube1 * _cosr[n+d1]) * _sint[t], tube1*_sinr[n+d1],
                                (radius + tube2 * _cosr[n+d1]) * _cost[t], (radius + tube2 * _cosr[n+d1]) * _sint[t], tube2*_sinr[n+d1],
                                (radius + tube2 * _cosr[n+d2]) * _cost[t], (radius + tube2 * _cosr[n+d2]) * _sint[t], tube2*_sinr[n+d2],
                                (radius + tube1 * _cosr[n+d2]) * _cost[t], (radius + tube1 * _cosr[n+d2]) * _sint[t], tube1*_sinr[n+d2], skip);
               creator.SetNormal(-nsign* _sint[t], nsign * _cost[t], 0);
            }
         }

      return creator.Create();
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createPolygonBuffer = function( shape, faces_limit ) {
      var thetaStart = shape.fPhi1,
          thetaLength = shape.fDphi,
          radiusSegments = 60, factor = 1;

      if (shape._typename == "TGeoPgon") {
         radiusSegments = shape.fNedges;
         factor = 1. / Math.cos(Math.PI/180 * thetaLength / radiusSegments / 2);
      } else {
         radiusSegments = Math.max(5, Math.round(thetaLength/JSROOT.GEO.GradPerSegm));
      }

      var usage = new Int16Array(2*shape.fNz), numusedlayers = 0, hasrmin = false;

      for (var layer=0; layer < shape.fNz; ++layer)
         if (shape.fRmin[layer] > 0) hasrmin = true;

      // return very rough estimation, number of faces may be much less
      if (faces_limit < 0) return (hasrmin ? 4 : 2) * radiusSegments * (shape.fNz-1);

      // coordinate of point on cut edge (x,z)
      var pnts = (thetaLength === 360) ? null : [];

      // first analyse levels - if we need to create all of them
      for (var side = 0; side < 2; ++side) {
         var rside = (side === 0) ? 'fRmax' : 'fRmin';

         for (var layer=0; layer < shape.fNz; ++layer) {

            // first create points for the layer
            var layerz = shape.fZ[layer], rad = shape[rside][layer];

            usage[layer*2+side] = 0;

            if ((layer > 0) && (layer < shape.fNz-1))
               if (((shape.fZ[layer-1] === layerz) && (shape[rside][layer-1] === rad)) ||
                   ((shape[rside][layer+1] === rad) && (shape[rside][layer-1] === rad))) {

                  // same Z and R as before - ignore
                  // or same R before and after

                  continue;
               }

            if ((layer>0) && ((side === 0) || hasrmin)) {
               usage[layer*2+side] = 1;
               numusedlayers++;
            }

            if (pnts !== null) {
               if (side === 0) {
                  pnts.push(new THREE.Vector2(factor*rad, layerz));
               } else if (rad < shape.fRmax[layer]) {
                  pnts.unshift(new THREE.Vector2(factor*rad, layerz));
               }
            }
         }
      }

      var numfaces = numusedlayers*radiusSegments*2;
      if (shape.fRmin[0] !== shape.fRmax[0]) numfaces += radiusSegments * (hasrmin ? 2 : 1);
      if (shape.fRmin[shape.fNz-1] !== shape.fRmax[shape.fNz-1]) numfaces += radiusSegments * (hasrmin ? 2 : 1);

      var cut_faces = null;

      if (pnts!==null) {
         if (pnts.length === shape.fNz * 2) {
            // special case - all layers are there, create faces ourself
            cut_faces = [];
            for (var layer = shape.fNz-1; layer>0; --layer) {
               if (shape.fZ[layer] === shape.fZ[layer-1]) continue;
               var right = 2*shape.fNz - 1 - layer;
               cut_faces.push([right, layer - 1, layer]);
               cut_faces.push([right, right + 1, layer-1]);
            }

         } else {
            // let three.js calculate our faces
            // console.log('triangulate polygon ' + shape.fShapeId);
            cut_faces = THREE.ShapeUtils.triangulateShape(pnts, []);
         }
         numfaces += cut_faces.length*2;
      }

      var phi0 = thetaStart*Math.PI/180, dphi = thetaLength/radiusSegments*Math.PI/180;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments+1),
          _cos = new Float32Array(radiusSegments+1);
      for (var seg=0;seg<=radiusSegments;++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // add sides
      for (var side = 0; side < 2; ++side) {
         var rside = (side === 0) ? 'fRmax' : 'fRmin',
             z1 = shape.fZ[0], r1 = factor*shape[rside][0],
             d1 = 1 - side, d2 = side;

         for (var layer=0; layer < shape.fNz; ++layer) {

            if (usage[layer*2+side] === 0) continue;

            var z2 = shape.fZ[layer], r2 = factor*shape[rside][layer],
                nxy = 1, nz = 0;

            if ((r2 !== r1)) {
               var angle = Math.atan2((r2-r1), (z2-z1));
               nxy = Math.cos(angle);
               nz = Math.sin(angle);
            }

            if (side>0) { nxy*=-1; nz*=-1; }

            for (var seg=0;seg < radiusSegments;++seg) {
               creator.AddFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z1,
                                r2 * _cos[seg+d1], r2 * _sin[seg+d1], z2,
                                r2 * _cos[seg+d2], r2 * _sin[seg+d2], z2,
                                r1 * _cos[seg+d2], r1 * _sin[seg+d2], z1);
               creator.SetNormal_12_34(nxy*_cos[seg+d1], nxy*_sin[seg+d1], nz, nxy*_cos[seg+d2], nxy*_sin[seg+d2], nz);
            }

            z1 = z2; r1 = r2;
         }
      }

      // add top/bottom
      for (var layer=0; layer < shape.fNz; layer += (shape.fNz-1)) {

         var rmin = factor*shape.fRmin[layer], rmax = factor*shape.fRmax[layer];

         if (rmin === rmax) continue;

         var layerz = shape.fZ[layer],
             d1 = (layer===0) ? 1 : 0, d2 = 1 - d1,
             normalz = (layer===0) ? -1: 1;

         if (!hasrmin && !cut_faces) creator.StartPolygon(layer>0);

         for (var seg=0;seg < radiusSegments;++seg) {
            creator.AddFace4(rmin * _cos[seg+d1], rmin * _sin[seg+d1], layerz,
                             rmax * _cos[seg+d1], rmax * _sin[seg+d1], layerz,
                             rmax * _cos[seg+d2], rmax * _sin[seg+d2], layerz,
                             rmin * _cos[seg+d2], rmin * _sin[seg+d2], layerz,
                             hasrmin ? 0 : 2);
            creator.SetNormal(0, 0, normalz);
         }

         creator.StopPolygon();
      }

      if (cut_faces)
         for (var seg = 0; seg <= radiusSegments; seg += radiusSegments) {
            var d1 = (seg === 0) ? 1 : 2, d2 = 3 - d1;
            for (var n=0;n<cut_faces.length;++n) {
               var a = pnts[cut_faces[n][0]],
                   b = pnts[cut_faces[n][d1]],
                   c = pnts[cut_faces[n][d2]];

               creator.AddFace3(a.x * _cos[seg], a.x * _sin[seg], a.y,
                                b.x * _cos[seg], b.x * _sin[seg], b.y,
                                c.x * _cos[seg], c.x * _sin[seg], c.y);

               creator.CalcNormal();
            }
         }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createXtruBuffer = function( shape, faces_limit ) {
      var nfaces = (shape.fNz-1) * shape.fNvert * 2;

      if (faces_limit < 0) return nfaces + shape.fNvert*3;

      // create points
      var pnts = [];
      for (var vert = 0; vert < shape.fNvert; ++vert)
         pnts.push(new THREE.Vector2(shape.fX[vert], shape.fY[vert]));

      // console.log('triangulate Xtru ' + shape.fShapeId);
      var faces = THREE.ShapeUtils.triangulateShape(pnts , []);
      if (faces.length < pnts.length-2) {
         JSROOT.GEO.warn('Problem with XTRU shape ' +shape.fName + ' with ' + pnts.length + ' vertices');
         faces = [];
      } else {
         nfaces += faces.length * 2;
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(nfaces);

      for (var layer = 0; layer < shape.fNz-1; ++layer) {
         var z1 = shape.fZ[layer], scale1 = shape.fScale[layer],
             z2 = shape.fZ[layer+1], scale2 = shape.fScale[layer+1],
             x01 = shape.fX0[layer], x02 = shape.fX0[layer+1],
             y01 = shape.fY0[layer], y02 = shape.fY0[layer+1];

         for (var vert1 = 0; vert1 < shape.fNvert; ++vert1) {
            var vert2 = (vert1+1) % shape.fNvert;
            creator.AddFace4(scale1 * shape.fX[vert1] + x01, scale1 * shape.fY[vert1] + y01, z1,
                             scale2 * shape.fX[vert1] + x02, scale2 * shape.fY[vert1] + y02, z2,
                             scale2 * shape.fX[vert2] + x02, scale2 * shape.fY[vert2] + y02, z2,
                             scale1 * shape.fX[vert2] + x01, scale1 * shape.fY[vert2] + y01, z1);
            creator.CalcNormal();
         }
      }

      for (layer = 0; layer <= shape.fNz-1; layer+=(shape.fNz-1)) {
         var z = shape.fZ[layer], scale = shape.fScale[layer],
             x0 = shape.fX0[layer], y0 = shape.fY0[layer];

         for (var n=0;n<faces.length;++n) {
            var face = faces[n],
                pnt1 = pnts[face[0]],
                pnt2 = pnts[face[(layer===0) ? 2 : 1]],
                pnt3 = pnts[face[(layer===0) ? 1 : 2]];

            creator.AddFace3(scale * pnt1.x + x0, scale * pnt1.y + y0, z,
                             scale * pnt2.x + x0, scale * pnt2.y + y0, z,
                             scale * pnt3.x + x0, scale * pnt3.y + y0, z);
            creator.SetNormal(0,0,layer===0 ? -1 : 1);
         }
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createParaboloidBuffer = function( shape, faces_limit ) {

      var radiusSegments = Math.max(4, Math.round(360/JSROOT.GEO.GradPerSegm)),
          heightSegments = 30;

      if (faces_limit > 0) {
         var fact = 2*radiusSegments*(heightSegments+1) / faces_limit;
         if (fact > 1.) {
            radiusSegments = Math.max(5, Math.floor(radiusSegments/Math.sqrt(fact)));
            heightSegments = Math.max(5, Math.floor(heightSegments/Math.sqrt(fact)));
         }
      }

      var zmin = -shape.fDZ, zmax = shape.fDZ, rmin = shape.fRlo, rmax = shape.fRhi;

      // if no radius at -z, find intersection
      if (shape.fA >= 0) {
         if (shape.fB > zmin) zmin = shape.fB;
      } else {
         if (shape.fB < zmax) zmax = shape.fB;
      }

      var ttmin = Math.atan2(zmin, rmin), ttmax = Math.atan2(zmax, rmax);

      var numfaces = (heightSegments+1)*radiusSegments*2;
      if (rmin===0) numfaces -= radiusSegments*2; // complete layer
      if (rmax===0) numfaces -= radiusSegments*2; // complete layer

      if (faces_limit < 0) return numfaces;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments+1),
          _cos = new Float32Array(radiusSegments+1);
      for (var seg=0;seg<=radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      var lastz = zmin, lastr = 0, lastnxy = 0, lastnz = -1;

      for (var layer = 0; layer <= heightSegments + 1; ++layer) {

         var layerz = 0, radius = 0, nxy = 0, nz = -1;

         if ((layer === 0) && (rmin===0)) continue;

         if ((layer === heightSegments + 1) && (lastr === 0)) break;

         switch (layer) {
            case 0: layerz = zmin; radius = rmin; break;
            case heightSegments: layerz = zmax; radius = rmax; break;
            case heightSegments + 1: layerz = zmax; radius = 0; break;
            default: {
               var tt = Math.tan(ttmin + (ttmax-ttmin) * layer / heightSegments);
               var delta = tt*tt - 4*shape.fA*shape.fB; // should be always positive (a*b<0)
               radius = 0.5*(tt+Math.sqrt(delta))/shape.fA;
               if (radius < 1e-6) radius = 0;
               layerz = radius*tt;
            }
         }

         nxy = shape.fA * radius;
         nz = (shape.fA > 0) ? -1 : 1;

         var skip = 0;
         if (lastr === 0) skip = 1; else
         if (radius === 0) skip = 2;

         for (var seg=0; seg<radiusSegments; ++seg) {
            creator.AddFace4(radius*_cos[seg],   radius*_sin[seg], layerz,
                             lastr*_cos[seg],    lastr*_sin[seg], lastz,
                             lastr*_cos[seg+1],  lastr*_sin[seg+1], lastz,
                             radius*_cos[seg+1], radius*_sin[seg+1], layerz, skip);

            // use analytic normal values when open/closing paraboloid around 0
            // cut faces (top or bottom) set with simple normal
            if ((skip===0) || ((layer===1) && (rmin===0)) || ((layer===heightSegments+1) && (rmax===0)))
               creator.SetNormal4(nxy*_cos[seg],       nxy*_sin[seg],       nz,
                                  lastnxy*_cos[seg],   lastnxy*_sin[seg],   lastnz,
                                  lastnxy*_cos[seg+1], lastnxy*_sin[seg+1], lastnz,
                                  nxy*_cos[seg+1],     nxy*_sin[seg+1],     nz, skip);
            else
               creator.SetNormal(0, 0, (layer < heightSegments) ? -1 : 1);
         }

         lastz = layerz; lastr = radius;
         lastnxy = nxy; lastnz = nz;
      }

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createHypeBuffer = function( shape, faces_limit ) {

      if ((shape.fTin===0) && (shape.fTout===0))
         return JSROOT.GEO.createTubeBuffer(shape, faces_limit);

      var radiusSegments = Math.max(4, Math.round(360/JSROOT.GEO.GradPerSegm)),
          heightSegments = 30;

      var numfaces = radiusSegments * (heightSegments + 1) * ((shape.fRmin > 0) ? 4 : 2);

      if (faces_limit < 0) return numfaces;

      if ((faces_limit > 0) && (faces_limit > numfaces)) {
         radiusSegments = Math.max(4, Math.floor(radiusSegments/Math.sqrt(numfaces/faces_limit)));
         heightSegments = Math.max(4, Math.floor(heightSegments/Math.sqrt(numfaces/faces_limit)));
         numfaces = radiusSegments * (heightSegments + 1) * ((shape.fRmin > 0) ? 4 : 2);
      }

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments+1), _cos = new Float32Array(radiusSegments+1);
      for (var seg=0;seg<=radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(numfaces);

      // in-out side
      for (var side=0;side<2;++side) {
         if ((side > 0) && (shape.fRmin <= 0)) break;

         var r0 = (side > 0) ? shape.fRmin : shape.fRmax,
             tsq = (side > 0) ? shape.fTinsq : shape.fToutsq,
             d1 = 1- side, d2 = 1 - d1;

         // vertical layers
         for (var layer=0;layer<heightSegments;++layer) {
            var z1 = -shape.fDz + layer/heightSegments*2*shape.fDz,
                z2 = -shape.fDz + (layer+1)/heightSegments*2*shape.fDz,
                r1 = Math.sqrt(r0*r0+tsq*z1*z1),
                r2 = Math.sqrt(r0*r0+tsq*z2*z2);

            for (var seg=0; seg<radiusSegments; ++seg) {
               creator.AddFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z1,
                                r2 * _cos[seg+d1], r2 * _sin[seg+d1], z2,
                                r2 * _cos[seg+d2], r2 * _sin[seg+d2], z2,
                                r1 * _cos[seg+d2], r1 * _sin[seg+d2], z1);
               creator.CalcNormal();
            }
         }
      }

      // add caps
      for(var layer=0; layer<2; ++layer) {
         var z = (layer === 0) ? shape.fDz : -shape.fDz,
             r1 = Math.sqrt(shape.fRmax*shape.fRmax + shape.fToutsq*z*z),
             r2 = (shape.fRmin > 0) ? Math.sqrt(shape.fRmin*shape.fRmin + shape.fTinsq*z*z) : 0,
             skip = (shape.fRmin > 0) ? 0 : 1,
             d1 = 1 - layer, d2 = 1 - d1;
          for (var seg=0; seg<radiusSegments; ++seg) {
             creator.AddFace4(r1 * _cos[seg+d1], r1 * _sin[seg+d1], z,
                              r2 * _cos[seg+d1], r2 * _sin[seg+d1], z,
                              r2 * _cos[seg+d2], r2 * _sin[seg+d2], z,
                              r1 * _cos[seg+d2], r1 * _sin[seg+d2], z, skip);
             creator.SetNormal(0,0, (layer===0) ? 1 : -1)
          }

      }

      return creator.Create();
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createMatrix = function(matrix) {

      if (!matrix) return null;

      var translation = null, rotation = null, scale = null;

      switch (matrix._typename) {
         case 'TGeoTranslation': translation = matrix.fTranslation; break;
         case 'TGeoRotation': rotation = matrix.fRotationMatrix; break;
         case 'TGeoScale': scale = matrix.fScale; break;
         case 'TGeoGenTrans':
            scale = matrix.fScale; // no break, translation and rotation follows
         case 'TGeoCombiTrans':
            translation = matrix.fTranslation;
            if (matrix.fRotation) rotation = matrix.fRotation.fRotationMatrix;
            break;
         case 'TGeoHMatrix':
            translation = matrix.fTranslation;
            rotation = matrix.fRotationMatrix;
            scale = matrix.fScale;
            break;
         case 'TGeoIdentity':
            break;
         default:
            console.warn('unsupported matrix ' + matrix._typename);
      }

      if (!translation && !rotation && !scale) return null;

      var res = new THREE.Matrix4();

      if (rotation)
         res.set(rotation[0], rotation[1], rotation[2],  0,
                 rotation[3], rotation[4], rotation[5],  0,
                 rotation[6], rotation[7], rotation[8],  0,
                           0,           0,           0,  1);

      if (translation)
         res.setPosition(new THREE.Vector3(translation[0], translation[1], translation[2]));

      if (scale)
         res.scale(new THREE.Vector3(scale[0], scale[1], scale[2]));

      return res;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.getNodeMatrix = function(kind, node) {
      // returns transformation matrix for the node
      // created after node visibility flag is checked and volume cut is performed

      var matrix = null;

      if (kind === 1) {
         // special handling for EVE nodes

         matrix = new THREE.Matrix4();

         if (node.fTrans) {
            matrix.set(node.fTrans[0],  node.fTrans[4],  node.fTrans[8],  0,
                       node.fTrans[1],  node.fTrans[5],  node.fTrans[9],  0,
                       node.fTrans[2],  node.fTrans[6],  node.fTrans[10], 0,
                                    0,               0,               0,  1);
            // second - set position with proper sign
            matrix.setPosition({ x: node.fTrans[12], y: node.fTrans[13], z: node.fTrans[14] });
         }
      } else if (node.fMatrix) {
         matrix = JSROOT.GEO.createMatrix(node.fMatrix);
      } else if ((node._typename == "TGeoNodeOffset") && node.fFinder) {
         var kPatternReflected = JSROOT.BIT(14);
         if ((node.fFinder.fBits & kPatternReflected) !== 0)
            JSROOT.GEO.warn('Unsupported reflected pattern ' + node.fFinder._typename);

         // if (node.fFinder._typename === 'TGeoPatternCylR') { }
         // if (node.fFinder._typename === 'TGeoPatternSphR') { }
         // if (node.fFinder._typename === 'TGeoPatternSphTheta') { }
         // if (node.fFinder._typename === 'TGeoPatternSphPhi') { }
         // if (node.fFinder._typename === 'TGeoPatternHoneycomb') { }
         switch(node.fFinder._typename) {
           case 'TGeoPatternX':
           case 'TGeoPatternY':
           case 'TGeoPatternZ':
           case 'TGeoPatternParaX':
           case 'TGeoPatternParaY':
           case 'TGeoPatternParaZ':
              var _shift = node.fFinder.fStart + (node.fIndex + 0.5) * node.fFinder.fStep;

              matrix = new THREE.Matrix4();

              switch (node.fFinder._typename[node.fFinder._typename.length-1]) {
                 case 'X': matrix.setPosition(new THREE.Vector3(_shift, 0, 0)); break;
                 case 'Y': matrix.setPosition(new THREE.Vector3(0, _shift, 0)); break;
                 case 'Z': matrix.setPosition(new THREE.Vector3(0, 0, _shift)); break;
              }
              break;

           case 'TGeoPatternCylPhi':
              var phi = (Math.PI/180)*(node.fFinder.fStart+(node.fIndex+0.5)*node.fFinder.fStep),
                  _cos = Math.cos(phi), _sin = Math.sin(phi);

              matrix = new THREE.Matrix4();

              matrix.set(_cos, -_sin,  0,  0,
                         _sin,  _cos,  0,  0,
                            0,     0,  1,  0,
                            0,     0,  0,  1);
              break;

           case 'TGeoPatternCylR':
               // seems to be, require no transformation
               matrix = new THREE.Matrix4();
               break;

           case 'TGeoPatternTrapZ':
              var dz = node.fFinder.fStart + (node.fIndex+0.5)*node.fFinder.fStep;
              matrix = new THREE.Matrix4();
              matrix.setPosition(new THREE.Vector3(node.fFinder.fTxz*dz, node.fFinder.fTyz*dz, dz));
              break;

           default:
              JSROOT.GEO.warn('Unsupported pattern type ' + node.fFinder._typename);
              break;
         }
      }

      return matrix;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createComposite = function ( shape, faces_limit ) {
      /*
      if ((faces_limit === -1) || (faces_limit === 0))  {
         var cnt = JSROOT.GEO.CountNumShapes(shape);

         if (cnt > JSROOT.GEO.CompLimit) {
            JSROOT.GEO.warn("composite shape " + shape.fShapeId + " has " + cnt + " components, replace by most left");
            var matrix = new THREE.Matrix4();
            while (shape.fNode && shape.fNode.fLeft) {
               var m1 = JSROOT.GEO.createMatrix(shape.fNode.fLeftMat);
               if (m1) matrix.multiply(m1);
               shape = shape.fNode.fLeft;
            }
            var res = JSROOT.GEO.createGeometry(shape, faces_limit);
            if (res && (faces_limit===0)) res.applyMatrix(matrix);
            return res;
         }
      }
      */

      if (faces_limit < 0)
         return JSROOT.GEO.createGeometry(shape.fNode.fLeft, -10) +
                JSROOT.GEO.createGeometry(shape.fNode.fRight, -10);

      var geom1, geom2, bsp1, bsp2, return_bsp = false,
          matrix1 = JSROOT.GEO.createMatrix(shape.fNode.fLeftMat),
          matrix2 = JSROOT.GEO.createMatrix(shape.fNode.fRightMat);

      // seems to be, IE has smaller stack for functions calls and ThreeCSG fails with larger shapes
      if (faces_limit === 0) faces_limit = (JSROOT.browser && JSROOT.browser.isIE) ? 2000 : 4000;
                        else return_bsp = true;

      if (matrix1 && (matrix1.determinant() < -0.9))
         JSROOT.GEO.warn('Axis reflection in left composite shape - not supported');

      if (matrix2 && (matrix2.determinant() < -0.9))
         JSROOT.GEO.warn('Axis reflections in right composite shape - not supported');

      if (shape.fNode.fLeft._typename == "TGeoHalfSpace") {
         geom1 = JSROOT.GEO.createHalfSpace(shape.fNode.fLeft);
      } else {
         geom1 = JSROOT.GEO.createGeometry(shape.fNode.fLeft, faces_limit);
      }

      if (!geom1) return null;

      var n1 = JSROOT.GEO.numGeometryFaces(geom1), n2 = 0;
      if (geom1._exceed_limit) n1 += faces_limit;

      if (n1 < faces_limit) {

         if (shape.fNode.fRight._typename == "TGeoHalfSpace") {
            geom2 = JSROOT.GEO.createHalfSpace(shape.fNode.fRight, geom1);
         } else {
            geom2 = JSROOT.GEO.createGeometry(shape.fNode.fRight, faces_limit);
         }

         n2 = JSROOT.GEO.numGeometryFaces(geom2);
      }

      if ((n1 + n2 >= faces_limit) || !geom2) {
         if (geom1.polygons) {
            geom1 = ThreeBSP.CreateBufferGeometry(geom1.polygons);
            n1 = JSROOT.GEO.numGeometryFaces(geom1);
         }
         if (matrix1) geom1.applyMatrix(matrix1);
         // if (!geom1._exceed_limit) console.log('reach faces limit', faces_limit, 'got', n1, n2);
         geom1._exceed_limit = true;
         return geom1;
      }

      bsp1 = new ThreeBSP.Geometry(geom1, matrix1, JSROOT.GEO.CompressComp ? 0 : undefined);

      bsp2 = new ThreeBSP.Geometry(geom2, matrix2, bsp1.maxid);

      // take over maxid from both geometries
      bsp1.maxid = bsp2.maxid;

      switch(shape.fNode._typename) {
         case 'TGeoIntersection': bsp1.direct_intersect(bsp2);  break; // "*"
         case 'TGeoUnion': bsp1.direct_union(bsp2); break;   // "+"
         case 'TGeoSubtraction': bsp1.direct_subtract(bsp2); break; // "/"
         default:
            JSROOT.GEO.warn('unsupported bool operation ' + shape.fNode._typename + ', use first geom');
      }

      if (JSROOT.GEO.numGeometryFaces(bsp1) === 0) {
         JSROOT.GEO.warn('Zero faces in comp shape'
               + ' left: ' + shape.fNode.fLeft._typename +  ' ' + JSROOT.GEO.numGeometryFaces(geom1) + ' faces'
               + ' right: ' + shape.fNode.fRight._typename + ' ' + JSROOT.GEO.numGeometryFaces(geom2) + ' faces'
               + '  use first');
         bsp1 = new ThreeBSP.Geometry(geom1, matrix1);
      }

      return return_bsp ? { polygons: bsp1.toPolygons() } : bsp1.toBufferGeometry();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.projectGeometry = function(geom, matrix, projection, position, flippedMesh) {

      if (!geom.boundingBox) geom.computeBoundingBox();

      var box = geom.boundingBox.clone();

      box.applyMatrix4(matrix);

      if (!position) position = 0;

      if (((box.min[projection]>=position) && (box.max[projection]>=position)) ||
          ((box.min[projection]<=position) && (box.max[projection]<=position))) {
         return null; // not interesting
      }

      var bsp1 = new ThreeBSP.Geometry(geom, matrix, 0, flippedMesh),
          sizex = 2*Math.max(Math.abs(box.min.x), Math.abs(box.max.x)),
          sizey = 2*Math.max(Math.abs(box.min.y), Math.abs(box.max.y)),
          sizez = 2*Math.max(Math.abs(box.min.z), Math.abs(box.max.z)),
          size = 10000;

      switch (projection) {
         case "x": size = Math.max(sizey,sizez); break;
         case "y": size = Math.max(sizex,sizez); break;
         case "z": size = Math.max(sizex,sizey); break;
      }

      var bsp2 = ThreeBSP.CreateNormal(projection, position, size);

      bsp1.cut_from_plane(bsp2);

      return bsp2.toBufferGeometry();
   }

   /** Creates geometry model for the provided shape
    * @memberOf JSROOT.GEO
    *
    * If @par limit === 0 (or undefined) returns THREE.BufferGeometry
    * If @par limit < 0 just returns estimated number of faces
    * If @par limit > 0 return list of ThreeBSP polygons (used only for composite shapes)
    * */
   JSROOT.GEO.createGeometry = function( shape, limit ) {
      if (limit === undefined) limit = 0;

      try {
         switch (shape._typename) {
            case "TGeoBBox": return JSROOT.GEO.createCubeBuffer( shape, limit );
            case "TGeoPara": return JSROOT.GEO.createParaBuffer( shape, limit );
            case "TGeoTrd1":
            case "TGeoTrd2": return JSROOT.GEO.createTrapezoidBuffer( shape, limit );
            case "TGeoArb8":
            case "TGeoTrap":
            case "TGeoGtra": return JSROOT.GEO.createArb8Buffer( shape, limit );
            case "TGeoSphere": return JSROOT.GEO.createSphereBuffer( shape , limit );
            case "TGeoCone":
            case "TGeoConeSeg":
            case "TGeoTube":
            case "TGeoTubeSeg":
            case "TGeoCtub": return JSROOT.GEO.createTubeBuffer( shape, limit );
            case "TGeoEltu": return JSROOT.GEO.createEltuBuffer( shape, limit );
            case "TGeoTorus": return JSROOT.GEO.createTorusBuffer( shape, limit );
            case "TGeoPcon":
            case "TGeoPgon": return JSROOT.GEO.createPolygonBuffer( shape, limit );
            case "TGeoXtru": return JSROOT.GEO.createXtruBuffer( shape, limit );
            case "TGeoParaboloid": return JSROOT.GEO.createParaboloidBuffer( shape, limit );
            case "TGeoHype": return JSROOT.GEO.createHypeBuffer( shape, limit );
            case "TGeoCompositeShape": return JSROOT.GEO.createComposite( shape, limit );
            case "TGeoShapeAssembly": break;
            case "TGeoScaledShape": {
               var res = JSROOT.GEO.createGeometry(shape.fShape, limit);
               if (shape.fScale && (limit>=0) && (typeof res === 'object') && (typeof res.scale === 'function'))
                  res.scale(shape.fScale.fScale[0],shape.fScale.fScale[1],shape.fScale.fScale[2]);
               return res;
            }
            case "TGeoHalfSpace":
               if (limit < 0) return 1; // half space if just plane used in composite
               // no break here - warning should appear
            default: JSROOT.GEO.warn('unsupported shape type ' + shape._typename);
         }
      } catch(e) {
         var place = "";
         if (e.stack !== undefined) {
            place = e.stack.split("\n")[0];
            if (place.indexOf(e.message) >= 0) place = e.stack.split("\n")[1];
                                          else place = " at: " + place;
         }
         JSROOT.GEO.warn(shape._typename + " err: " + e.message + place);
      }

      return limit < 0 ? 0 : null;
   }

   /** Creates half-space geometry for given shape
    * Just big-enough triangle to make BSP calculations
    * @memberOf JSROOT.GEO */

   JSROOT.GEO.createHalfSpace = function(shape, geom) {
      if (!shape || !shape.fN || !shape.fP) return null;

      // shape.fP = [0,0,15]; shape.fN = [0,1,1];

      var vertex = new THREE.Vector3(shape.fP[0], shape.fP[1], shape.fP[2]);

      var normal = new THREE.Vector3(shape.fN[0], shape.fN[1], shape.fN[2]);
      normal.normalize();

      var sz = 1e10;
      if (geom) {
         // using real size of other geometry, we probably improve precision
         var box = JSROOT.GEO.geomBoundingBox(geom);
         if (box) sz = box.getSize(new THREE.Vector3()).length() * 1000;
      }

      // console.log('normal', normal, 'vertex', vertex, 'size', sz);

      var v1 = new THREE.Vector3(-sz, -sz/2, 0),
          v2 = new THREE.Vector3(0, sz, 0),
          v3 = new THREE.Vector3(sz, -sz/2, 0),
          v4 = new THREE.Vector3(0, 0, -sz);

      var geometry = new THREE.Geometry();

      geometry.vertices.push(v1, v2, v3, v4);

      geometry.faces.push( new THREE.Face3( 0, 2, 1 ) );
      geometry.faces.push( new THREE.Face3( 0, 1, 3 ) );
      geometry.faces.push( new THREE.Face3( 1, 2, 3 ) );
      geometry.faces.push( new THREE.Face3( 2, 0, 3 ) );

      geometry.lookAt(normal);
      geometry.computeFaceNormals();

      v1.add(vertex);
      v2.add(vertex);
      v3.add(vertex);
      v4.add(vertex);

      /*
      // it suppose to be top corner of tetrahedron
      var v0 = vertex.clone().addScaledVector(normal, sz);

      // plane to verify our calculations
      var plane = new THREE.Plane(normal);

      // translate all vertices and plane
      plane.translate(vertex);

      console.log('Distance plane to fP', plane.distanceToPoint(vertex), "expect 0");
      console.log('Distance plane to v0', plane.distanceToPoint(v0), "expect", sz);
      console.log('Distance plane to v1', plane.distanceToPoint(v1), "expect 0");
      console.log('Distance plane to v2', plane.distanceToPoint(v2), "expect 0");
      console.log('Distance plane to v3', plane.distanceToPoint(v3), "expect 0");
      console.log('Distance plane to v4', plane.distanceToPoint(v4), "expect", sz);
      console.log('Distoance v0 to v4', v0.distanceTo(v4), "expect 0");
      */

      // return null;
      return geometry;
   }

   /** Provides info about geo object, used for tooltip info */
   JSROOT.GEO.provideInfo = function(obj) {
      var info = [], shape = null;

      if (obj.fVolume !== undefined) shape = obj.fVolume.fShape; else
      if (obj.fShape !== undefined) shape = obj.fShape; else
      if ((obj.fShapeBits !== undefined) && (obj.fShapeId !== undefined)) shape = obj;

      if (!shape) {
         info.push(obj._typename);
         return info;
      }

      var sz = Math.max(shape.fDX, shape.fDY, shape.fDZ);
      var useexp = (sz>1e7) || (sz<1e-7);

      function conv(v) {
         if (v===undefined) return "???";
         if ((v==Math.round(v) && v<1e7)) return Math.round(v);
         return useexp ? v.toExponential(4) : v.toPrecision(7);
      }

      info.push(shape._typename);

      info.push("DX="+conv(shape.fDX) + " DY="+conv(shape.fDY) + " DZ="+conv(shape.fDZ));

      switch (shape._typename) {
         case "TGeoBBox": break;
         case "TGeoPara": info.push("Alpha=" + shape.fAlpha + " Phi=" + shape.fPhi + " Theta=" + shape.fTheta); break;
         case "TGeoTrd2": info.push("Dy1=" + conv(shape.fDy1) + " Dy2=" + conv(shape.fDy1));
         case "TGeoTrd1": info.push("Dx1=" + conv(shape.fDx1) + " Dx2=" + conv(shape.fDx1)); break;
         case "TGeoArb8": break;
         case "TGeoTrap": break;
         case "TGeoGtra": break;
         case "TGeoSphere":
            info.push("Rmin=" + conv(shape.fRmin) + " Rmax=" + conv(shape.fRmax));
            info.push("Phi1=" + shape.fPhi1 + " Phi2=" + shape.fPhi2);
            info.push("Theta1=" + shape.fTheta1 + " Theta2=" + shape.fTheta2);
            break;
         case "TGeoConeSeg":
            info.push("Phi1=" + shape.fPhi1 + " Phi2=" + shape.fPhi2);
         case "TGeoCone":
            info.push("Rmin1=" + conv(shape.fRmin1) + " Rmax1=" + conv(shape.fRmax1));
            info.push("Rmin2=" + conv(shape.fRmin2) + " Rmax2=" + conv(shape.fRmax2));
            break;
         case "TGeoCtub":
         case "TGeoTubeSeg":
            info.push("Phi1=" + shape.fPhi1 + " Phi2=" + shape.fPhi2);
         case "TGeoEltu":
         case "TGeoTube":
            info.push("Rmin=" + conv(shape.fRmin) + " Rmax=" + conv(shape.fRmax));
            break;
         case "TGeoTorus":
            info.push("Rmin=" + conv(shape.fRmin) + " Rmax=" + conv(shape.fRmax));
            info.push("Phi1=" + shape.fPhi1 + " Dphi=" + shape.fDphi);
            break;
         case "TGeoPcon":
         case "TGeoPgon": break;
         case "TGeoXtru": break;
         case "TGeoParaboloid":
            info.push("Rlo=" + conv(shape.fRlo) + " Rhi=" + conv(shape.fRhi));
            info.push("A=" + conv(shape.fA) + " B=" + conv(shape.fB));
            break;
         case "TGeoHype":
            info.push("Rmin=" + conv(shape.fRmin) + " Rmax=" + conv(shape.fRmax));
            info.push("StIn=" + conv(shape.fStIn) + " StOut=" + conv(shape.fStOut));
            break;
         case "TGeoCompositeShape": break;
         case "TGeoShapeAssembly": break;
         case "TGeoScaledShape":
            info = JSROOT.GEO.provideInfo(shape.fShape);
            if (shape.fScale)
               info.unshift('Scale X=' + shape.fScale.fScale[0] + " Y=" + shape.fScale.fScale[1] + " Z=" + shape.fScale.fScale[2]);
            break;
      }

      return info;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.CreateProjectionMatrix = function(camera) {
      var cameraProjectionMatrix = new THREE.Matrix4();

      camera.updateMatrixWorld();
      camera.matrixWorldInverse.getInverse( camera.matrixWorld );
      cameraProjectionMatrix.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse);

      return cameraProjectionMatrix;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.CreateFrustum = function(source) {
      if (!source) return null;

      if (source instanceof THREE.PerspectiveCamera)
         source = JSROOT.GEO.CreateProjectionMatrix(source);

      var frustum = new THREE.Frustum();
      frustum.setFromMatrix(source);

      frustum.corners = new Float32Array([
          1,  1,  1,
          1,  1, -1,
          1, -1,  1,
          1, -1, -1,
         -1,  1,  1,
         -1,  1, -1,
         -1, -1,  1,
         -1, -1, -1,
          0,  0,  0 // also check center of the shape
      ]);

      frustum.test = new THREE.Vector3(0,0,0);

      frustum.CheckShape = function(matrix, shape) {
         var pnt = this.test, len = this.corners.length, corners = this.corners, i;

         for (i = 0; i < len; i+=3) {
            pnt.x = corners[i] * shape.fDX;
            pnt.y = corners[i+1] * shape.fDY;
            pnt.z = corners[i+2] * shape.fDZ;
            if (this.containsPoint(pnt.applyMatrix4(matrix))) return true;
        }

        return false;
      }

      frustum.CheckBox = function(box) {
         var pnt = this.test, cnt = 0;
         pnt.set(box.min.x, box.min.y, box.min.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.min.x, box.min.y, box.max.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.min.x, box.max.y, box.min.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.min.x, box.max.y, box.max.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.max.x, box.max.y, box.max.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.max.x, box.min.y, box.max.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.max.x, box.max.y, box.min.z);
         if (this.containsPoint(pnt)) cnt++;
         pnt.set(box.max.x, box.max.y, box.max.z);
         if (this.containsPoint(pnt)) cnt++;
         return cnt>5; // only if 6 edges and more are seen, we think that box is fully visible
      }

      return frustum;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.VisibleByCamera = function(camera, matrix, shape) {
      var frustum = new THREE.Frustum();
      var cameraProjectionMatrix = new THREE.Matrix4();

      camera.updateMatrixWorld();
      camera.matrixWorldInverse.getInverse( camera.matrixWorld );
      cameraProjectionMatrix.multiplyMatrices( camera.projectionMatrix, camera.matrixWorldInverse);
      frustum.setFromMatrix( cameraProjectionMatrix );

      var corners = [
         new THREE.Vector3(  shape.fDX/2.0,  shape.fDY/2.0,   shape.fDZ/2.0 ),
         new THREE.Vector3(  shape.fDX/2.0,  shape.fDY/2.0,  -shape.fDZ/2.0 ),
         new THREE.Vector3(  shape.fDX/2.0, -shape.fDY/2.0,   shape.fDZ/2.0 ),
         new THREE.Vector3(  shape.fDX/2.0, -shape.fDY/2.0,  -shape.fDZ/2.0 ),
         new THREE.Vector3( -shape.fDX/2.0,  shape.fDY/2.0,   shape.fDZ/2.0 ),
         new THREE.Vector3( -shape.fDX/2.0,  shape.fDY/2.0,  -shape.fDZ/2.0 ),
         new THREE.Vector3( -shape.fDX/2.0, -shape.fDY/2.0,   shape.fDZ/2.0 ),
         new THREE.Vector3( -shape.fDX/2.0, -shape.fDY/2.0,  -shape.fDZ/2.0 )
               ];
      for (var i = 0; i < corners.length; i++) {
         if (frustum.containsPoint(corners[i].applyMatrix4(matrix))) return true;
      }

      return false;
   }

   /** Returns number of faces for provided geometry
    * @param geom  - can be THREE.Geometry, THREE.BufferGeometry, ThreeBSP.Geometry or interim array of polygons
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.numGeometryFaces = function(geom) {
      if (!geom) return 0;

      if (geom instanceof ThreeBSP.Geometry)
         return geom.tree.numPolygons();

      if (geom.type == 'BufferGeometry') {
         var attr = geom.getAttribute('position');
         return attr ? attr.count / 3 : 0;
      }

      // special array of polygons
      if (geom.polygons)
         return geom.polygons.length;

      return geom.faces.length;
   }

   /** Returns number of faces for provided geometry
    * @param geom  - can be THREE.Geometry, THREE.BufferGeometry, ThreeBSP.Geometry or interim array of polygons
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.numGeometryVertices = function(geom) {
      if (!geom) return 0;

      if (geom instanceof ThreeBSP.Geometry)
         return geom.tree.numPolygons() * 3;

      if (geom.type == 'BufferGeometry') {
         var attr = geom.getAttribute('position');
         return attr ? attr.count : 0;
      }

      if (geom.polygons)
         return geom.polygons.length * 4;

      return geom.vertices.length;
   }

   /** Returns bounding box
    * @memberOf JSROOT.GEO */
   JSROOT.GEO.geomBoundingBox = function(geom) {
      if (!geom) return null;

      var polygons = null;

      if (geom instanceof ThreeBSP.Geometry)
         polygons = geom.tree.collectPolygons([]);
      else if (geom.polygons)
         polygons = geom.polygons;

      if (polygons!==null) {
         var box = new THREE.Box3();
         for (var n=0;n<polygons.length;++n) {
            var polygon = polygons[n], nvert = polygon.vertices.length;
            for (var k=0;k<nvert;++k)
               box.expandByPoint(polygon.vertices[k]);
         }
         return box;
      }

      if (!geom.boundingBox) geom.computeBoundingBox();

      return geom.boundingBox.clone();
   }

   /** Compares two stacks. Returns length where stacks are the same
    * @memberOf JSROOT.GEO
    * @private */
   JSROOT.GEO.CompareStacks = function(stack1, stack2) {
      if (!stack1 || !stack2) return 0;
      if (stack1 === stack2) return stack1.length;
      var len = Math.min(stack1.length, stack2.length);
      for (var k=0;k<len;++k)
         if (stack1[k] !== stack2[k]) return k;
      return len;
   }

   /** Checks if two stack arrays are identical
    * @memberOf JSROOT.GEO
    * @private */
   JSROOT.GEO.IsSameStack = function(stack1, stack2) {
      if (!stack1 || !stack2) return false;
      if (stack1 === stack2) return true;
      if (stack1.length !== stack2.length) return false;
      for (var k=0;k<stack1.length;++k)
         if (stack1[k] !== stack2[k]) return false;
      return true;
   }


   // ====================================================================

   // class for working with cloned nodes

   JSROOT.GEO.ClonedNodes = function(obj, clones) {
      this.toplevel = true; // indicate if object creates top-level structure with Nodes and Volumes folder
      this.name_prefix = ""; // name prefix used for nodes names
      this.maxdepth = 1;  // maximal hierarchy depth, required for transparency
      this.fVisLevel = 4;  // maximal depth of nodes visibility aka gGeoManager->SetVisLevel, same default
      this.fMaxVisNodes = 10000; // maximal number of visisble nodes aka gGeoManager->fMaxVisNodes

      if (obj) {
         if (obj.$geoh) this.toplevel = false;
         this.CreateClones(obj);
      } else if (clones) {
         this.nodes = clones;
      }
   }

   /** Set maximal depth for nodes visibility */
   JSROOT.GEO.ClonedNodes.prototype.SetVisLevel = function(lvl) {
      this.fVisLevel = lvl && !isNaN(lvl) ? lvl : 4;
   }

   JSROOT.GEO.ClonedNodes.prototype.GetVisLevel = function() {
      return this.fVisLevel;
   }

   /** Set maximal depth for nodes visibility */
   JSROOT.GEO.ClonedNodes.prototype.SetMaxVisNodes = function(v) {
      this.fMaxVisNodes = !isNaN(v) ? v : 10000;
   }

   JSROOT.GEO.ClonedNodes.prototype.GetMaxVisNodes = function() {
      return this.fMaxVisNodes;
   }

   /** Insert node into existing array */
   JSROOT.GEO.ClonedNodes.prototype.updateNode = function(node) {
      if (node && !isNaN(node.id) && (node.id < this.nodes.length))
         this.nodes[node.id] = node;
   }

   /** Returns TGeoShape for element with given indx */
   JSROOT.GEO.ClonedNodes.prototype.GetNodeShape = function(indx) {
      if (!this.origin || !this.nodes) return null;
      var obj = this.origin[indx], clone = this.nodes[indx];
      if (!obj || !clone) return null;
      if (clone.kind === 0) {
         if (obj.fVolume) return obj.fVolume.fShape;
      } else {
         return obj.fShape;
      }
      return null;
   }

   JSROOT.GEO.ClonedNodes.prototype.Cleanup = function(drawnodes, drawshapes) {
      // function to cleanup as much as possible structures
      // drawnodes and drawshapes are arrays created during building of geometry

      if (drawnodes) {
         for (var n=0;n<drawnodes.length;++n) {
            delete drawnodes[n].stack;
            drawnodes[n] = undefined;
         }
      }

      if (drawshapes) {
         for (var n=0;n<drawshapes.length;++n) {
            delete drawshapes[n].geom;
            drawshapes[n] = undefined;
         }
      }

      if (this.nodes) {
         for (var n=0;n<this.nodes.length;++n) {
            if (this.nodes[n])
               delete this.nodes[n].chlds;
         }
      }

      delete this.nodes;
      delete this.origin;

      delete this.sortmap;

   }

   /** Create complete description for provided Geo object */
   JSROOT.GEO.ClonedNodes.prototype.CreateClones = function(obj, sublevel, kind) {
       if (!sublevel) {

          if (obj && obj._typename == "$$Shape$$")
             return this.CreateClonesForShape(obj);

          this.origin = [];
          sublevel = 1;
          kind = JSROOT.GEO.NodeKind(obj);
       }

       if ((kind < 0) || !obj || ('_refid' in obj)) return;

       obj._refid = this.origin.length;
       this.origin.push(obj);
       if (sublevel>this.maxdepth) this.maxdepth = sublevel;

       var chlds = null;
       if (kind===0)
          chlds = (obj.fVolume && obj.fVolume.fNodes) ? obj.fVolume.fNodes.arr : null;
       else
          chlds = obj.fElements ? obj.fElements.arr : null;

       if (chlds !== null) {
          JSROOT.GEO.CheckDuplicates(obj, chlds);
          for (var i = 0; i < chlds.length; ++i)
             this.CreateClones(chlds[i], sublevel+1, kind);
       }

       if (sublevel > 1) return;

       this.nodes = [];

       var sortarr = [];

       // first create nodes objects
       for (var n=0; n<this.origin.length; ++n) {
          var obj = this.origin[n];
          var node = { id: n, kind: kind, vol: 0, nfaces: 0 };
          this.nodes.push(node);
          sortarr.push(node); // array use to produce sortmap
       }

       // than fill children lists
       for (var n=0;n<this.origin.length;++n) {
          var obj = this.origin[n], clone = this.nodes[n];

          var chlds = null, shape = null;

          if (kind===1) {
             shape = obj.fShape;
             if (obj.fElements) chlds = obj.fElements.arr;
          } else if (obj.fVolume) {
             shape = obj.fVolume.fShape;
             if (obj.fVolume.fNodes) chlds = obj.fVolume.fNodes.arr;
          }

          var matrix = JSROOT.GEO.getNodeMatrix(kind, obj);
          if (matrix) {
             clone.matrix = matrix.elements; // take only matrix elements, matrix will be constructed in worker
             if (clone.matrix[0] === 1) {
                var issimple = true;
                for (var k=1;(k<clone.matrix.length) && issimple;++k)
                   issimple = (clone.matrix[k] === ((k===5) || (k===10) || (k===15) ? 1 : 0));
                if (issimple) delete clone.matrix;
             }
          }
          if (shape) {
             clone.fDX = shape.fDX;
             clone.fDY = shape.fDY;
             clone.fDZ = shape.fDZ;
             clone.vol = shape.fDX*shape.fDY*shape.fDZ;
             if (shape.$nfaces === undefined)
                shape.$nfaces = JSROOT.GEO.createGeometry(shape, -1);
             clone.nfaces = shape.$nfaces;
             if (clone.nfaces <= 0) clone.vol = 0;
          }

          if (!chlds) continue;

          // in cloned object children is only list of ids
          clone.chlds = new Array(chlds.length);
          for (var k=0;k<chlds.length;++k)
             clone.chlds[k] = chlds[k]._refid;
       }

       // remove _refid identifiers from original objects
       for (var n=0;n<this.origin.length;++n)
          delete this.origin[n]._refid;

       // do sorting once
       sortarr.sort(function(a,b) { return b.vol - a.vol; });

       // remember sort map and also sortid
       this.sortmap = new Array(this.nodes.length);
       for (var n=0;n<this.nodes.length;++n) {
          this.sortmap[n] = sortarr[n].id;
          sortarr[n].sortid = n;
       }
   }

   /** Create elementary item with single already existing shape,
    * used by details view of geometry shape */
   JSROOT.GEO.ClonedNodes.prototype.CreateClonesForShape = function(obj) {
      this.origin = [];

      // indicate that just plain shape is used
      this.plain_shape = obj;

      var node = {
            id: 0, sortid: 0, kind: 2,
            name: "Shape",
            nfaces: obj.nfaces,
            fDX: 1, fDY: 1, fDZ: 1, vol: 1,
            vis: true
         };

      this.nodes = [ node ];
   }

   /** Count all visisble nodes */
   JSROOT.GEO.ClonedNodes.prototype.CountVisibles = function() {
      var cnt = 0;
      if (this.nodes)
         for (var k=0;k<this.nodes.length;++k)
            if (this.nodes[k].vis)
               cnt++;
      return cnt;
   }

   /** Mark visisble nodes. Set only basic flags, actual visibility depends from hierarchy  */
   JSROOT.GEO.ClonedNodes.prototype.MarkVisibles = function(on_screen, copy_bits, cloning, hide_top_volume) {
      if (!this.nodes) return 0;
      if (this.plain_shape) return 1;

      var res = 0, simple_copy = cloning && (cloning.length === this.nodes.length);

      if (!simple_copy && !this.origin) return 0;

      for (var n=0;n<this.nodes.length;++n) {
         var clone = this.nodes[n];

         clone.vis = 0; // 1 - only with last level
         delete clone.nochlds;

         if (simple_copy) {
            clone.vis = cloning[n].vis;
            clone.nochlds = cloning[n].nochlds;
            if (clone.vis) res++;
            continue;
         }

         var obj = this.origin[n];

         if (clone.kind === 0) {
            if (obj.fVolume) {
               if (on_screen) {
                  // on screen bits used always, childs always checked
                  clone.vis = JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisOnScreen) ? 99 : 0;

                  if (copy_bits) {
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisNone, false);
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisThis, (clone.vis > 0));
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisDaughters, true);
                  }
               } else {
                  clone.vis = !JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisNone) &&
                               JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisThis) ? 99 : 0;

                  if (!JSROOT.GEO.TestBit(obj, JSROOT.GEO.BITS.kVisDaughters) ||
                      !JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisDaughters)) clone.nochlds = true;

                  // node with childs only shown in case if it is last level in hierarchy
                  if ((clone.vis > 0) && clone.chlds && !clone.nochlds) clone.vis = 1;

                  // special handling for top node
                  if (n==0) {
                     if (hide_top_volume) clone.vis = 0;
                     delete clone.nochlds;
                  }
               }
            }
         } else {
            clone.vis = obj.fRnrSelf ? 99 : 0;

            // when the only node is selected, draw it
            if ((n===0) && (this.nodes.length===1)) clone.vis = 99;

            this.vislevel = 9999; // automatically take all volumes
         }

         // shape with zero volume or without faces will not be observed
         if ((clone.vol <= 0) || (clone.nfaces <= 0)) clone.vis = 0;

         if (clone.vis) res++;
      }

      return res;
   }

   /** After visibility flags is set, produce idshift for all nodes as it would be maximum level @private */
   JSROOT.GEO.ClonedNodes.prototype.ProduceIdShits = function() {
      for (var k=0;k<this.nodes.length;++k)
         this.nodes[k].idshift = -1;

      function scan_func(nodes, node) {
         if (node.idshift < 0) {
            node.idshift = 0;
            if (node.chlds)
               for(var k = 0; k<node.chlds.length; ++k)
                  node.idshift += scan_func(nodes, nodes[node.chlds[k]]);
         }

         return node.idshift + 1;
      }

      scan_func(this.nodes, this.nodes[0]);
   }

   /** Extract only visibility flags, used to transfer them to the worker @private */
   JSROOT.GEO.ClonedNodes.prototype.GetVisibleFlags = function() {
      var res = new Array(this.nodes.length);
      for (var n=0;n<this.nodes.length;++n)
         res[n] = { vis: this.nodes[n].vis, nochlds: this.nodes[n].nochlds };
      return res;
   }

   /** Scan visible nodes in hierarchy, starting from nodeid
     * Each entry in hierarchy get its unique id, which is not changed with visibility flags
     * @private */
   JSROOT.GEO.ClonedNodes.prototype.ScanVisible = function(arg, vislvl) {

      if (!this.nodes) return 0;

      if (vislvl === undefined) {
         if (!arg) arg = {};

         vislvl = arg.vislvl || this.vislevel || 4; // default 3 in ROOT
         if (vislvl > 88) vislvl = 88;

         arg.stack = new Array(100); // current stack
         arg.nodeid = 0;
         arg.counter = 0; // sequence ID of the node, used to identify it later
         arg.last = 0;
         arg.CopyStack = function(factor) {
            var entry = { nodeid: this.nodeid, seqid: this.counter, stack: new Array(this.last) };
            if (factor) entry.factor = factor; // factor used to indicate importance of entry, will be build as first
            for (var n=0;n<this.last;++n) entry.stack[n] = this.stack[n+1]; // copy stack
            return entry;
         }

         if (arg.domatrix) {
            arg.matrices = [];
            arg.mpool = [ new THREE.Matrix4() ]; // pool of Matrix objects to avoid permanent creation
            arg.getmatrix = function() { return this.matrices[this.last]; }
         }
      }

      var res = 0, node = this.nodes[arg.nodeid];

      if (arg.domatrix) {
         if (!arg.mpool[arg.last+1])
            arg.mpool[arg.last+1] = new THREE.Matrix4();

         var prnt = (arg.last > 0) ? arg.matrices[arg.last-1] : new THREE.Matrix4();
         if (node.matrix) {
            arg.matrices[arg.last] = arg.mpool[arg.last].fromArray(prnt.elements);
            arg.matrices[arg.last].multiply(arg.mpool[arg.last+1].fromArray(node.matrix));
         } else {
            arg.matrices[arg.last] = prnt;
         }
      }

      if (node.nochlds) vislvl = 0;

      if (node.vis > vislvl) {
         if (!arg.func || arg.func(node)) res++;
      }

      arg.counter++;

      if ((vislvl > 0) && node.chlds) {
         arg.last++;
         for (var i = 0; i < node.chlds.length; ++i) {
            arg.nodeid = node.chlds[i];
            arg.stack[arg.last] = i; // in the stack one store index of child, it is path in the hierarchy
            res += this.ScanVisible(arg, vislvl-1);
         }
         arg.last--;
      } else {
         arg.counter += (node.idshift || 0);
      }

      if (arg.last === 0) {
         delete arg.last;
         delete arg.stack;
         delete arg.CopyStack;
         delete arg.counter;
         delete arg.matrices;
         delete arg.mpool;
         delete arg.getmatrix;
      }

      return res;
   }

   /** @brief Return node name with given id.
    * @desc Either original object or description is used
    * @private */
   JSROOT.GEO.ClonedNodes.prototype.GetNodeName = function(nodeid) {
      if (this.origin) {
         var obj = this.origin[nodeid];
         return obj ? JSROOT.GEO.ObjectName(obj) : "";
      }
      var node = this.nodes[nodeid];
      return node ? node.name : "";
   }

   /** @brief Returns description for provide stack
    * @private */
   JSROOT.GEO.ClonedNodes.prototype.ResolveStack = function(stack, withmatrix) {

      var res = { id: 0, obj: null, node: this.nodes[0], name: this.name_prefix };

      // if (!this.toplevel || (this.nodes.length === 1) || (res.node.kind === 1)) res.name = "";

      if (withmatrix) {
         res.matrix = new THREE.Matrix4();
         if (res.node.matrix) res.matrix.fromArray(res.node.matrix);
      }

      if (this.origin)
         res.obj = this.origin[0];

      //if (!res.name)
      //   res.name = this.GetNodeName(0);

      if (stack)
         for(var lvl=0;lvl<stack.length;++lvl) {
            res.id = res.node.chlds[stack[lvl]];
            res.node = this.nodes[res.id];

            if (this.origin)
               res.obj = this.origin[res.id];

            var subname = this.GetNodeName(res.id);
            if (subname) {
               if (res.name) res.name+="/";
               res.name += subname;
            }

            if (withmatrix && res.node.matrix)
               res.matrix.multiply(new THREE.Matrix4().fromArray(res.node.matrix));
         }

      return res;
   }

   /** Create stack array based on nodes ids array.
    * Ids list should correspond to existing nodes hierarchy */
   JSROOT.GEO.ClonedNodes.prototype.MakeStackByIds = function(ids) {
      if (!ids) return null;

      if (ids[0] !== 0) {
         console.error('wrong ids - first should be 0');
         return null;
      }

      var node = this.nodes[0], stack = [];

      for (var k=1;k<ids.length;++k) {
         var nodeid = ids[k];
         if (!node) return null;
         var chindx = node.chlds.indexOf(nodeid);
         if (chindx < 0) {
            console.error('wrong nodes ids ' + ids[k] + ' is not child of ' + ids[k-1]);
            return null;
         }

         stack.push(chindx);
         node = this.nodes[nodeid];
      }

      return stack;
   }

   /** Retuns ids array which correspond to the stack */
   JSROOT.GEO.ClonedNodes.prototype.MakeIdsByStack = function(stack) {
      if (!stack) return null;
      var node = this.nodes[0], ids = [0];
      for (var k=0;k<stack.length;++k) {
         var id = node.chlds[stack[k]];
         ids.push(id);
         node = this.nodes[id];
      }
      return ids;
   }

   /** Returns true if stack includes at any place provided nodeid */
   JSROOT.GEO.ClonedNodes.prototype.IsNodeInStack = function(nodeid, stack) {

      if (!nodeid) return true;

      var node = this.nodes[0], id = 0;

      for(var lvl = 0; lvl < stack.length; ++lvl) {
         id = node.chlds[stack[lvl]];
         if (id == nodeid) return true;
         node = this.nodes[id];
      }

      return false;
   }

   /** find stack by name which include names of all parents */
   JSROOT.GEO.ClonedNodes.prototype.FindStackByName = function(fullname) {

      var names = fullname.split('/'), currid = 0, stack = [];

      if (this.GetNodeName(currid) !== names[0]) return null;

      for (var n=1;n<names.length;++n) {
         var node = this.nodes[currid];
         if (!node.chlds) return null;

         for (var k=0;k<node.chlds.length;++k) {
            var chldid = node.chlds[k];
            if (this.GetNodeName(chldid) === names[n]) { stack.push(k); currid = chldid; break; }
         }

         // no new entry - not found stack
         if (stack.length === n - 1) return null;
      }

      return stack;
   }

   /** @brief Provide different properties of draw entry nodeid
    * @desc Only if node visible, material will be created*/
   JSROOT.GEO.ClonedNodes.prototype.getDrawEntryProperties = function(entry) {

      var clone = this.nodes[entry.nodeid];
      var visible = true;

      if (clone.kind === 2) {
         var prop = { name: clone.name, nname: clone.name, shape: null, material: null, chlds: null };
         var _opacity = entry.opacity || 1;
         prop.fillcolor = new THREE.Color( entry.color ? "rgb(" + entry.color + ")" : "blue" );
         prop.material = new THREE.MeshLambertMaterial( { transparent: _opacity < 1,
                          opacity: _opacity, wireframe: false, color: prop.fillcolor,
                          side: THREE.FrontSide /* THREE.DoubleSide*/, vertexColors: THREE.NoColors /*THREE.VertexColors */,
                          depthWrite: _opacity == 1 } );
         prop.material.inherentOpacity = _opacity;

         return prop;
      }

      if (!this.origin) {
         console.error('origin not there - kind', clone.kind, entry.nodeid, clone);
         return null;
      }

      var node = this.origin[entry.nodeid];

      if (clone.kind === 1) {
         // special handling for EVE nodes

         var prop = { name: JSROOT.GEO.ObjectName(node), nname: JSROOT.GEO.ObjectName(node), shape: node.fShape, material: null, chlds: null };

         if (node.fElements !== null) prop.chlds = node.fElements.arr;

         if (visible) {
            var _opacity = Math.min(1, node.fRGBA[3]);
            prop.fillcolor = new THREE.Color( node.fRGBA[0], node.fRGBA[1], node.fRGBA[2] );
            prop.material = new THREE.MeshLambertMaterial( { transparent: _opacity < 1,
                             opacity: _opacity, wireframe: false, color: prop.fillcolor,
                             side: THREE.FrontSide /* THREE.DoubleSide*/, vertexColors: THREE.NoColors /*THREE.VertexColors */,
                             depthWrite: _opacity == 1 } );
            prop.material.inherentOpacity = _opacity;
         }

         return prop;
      }

      var volume = node.fVolume;

      var prop = { name: JSROOT.GEO.ObjectName(volume), nname: JSROOT.GEO.ObjectName(node), volume: node.fVolume, shape: volume.fShape, material: null, chlds: null };

      if (node.fVolume.fNodes !== null) prop.chlds = node.fVolume.fNodes.arr;

      if (volume) prop.linewidth = volume.fLineWidth;

      if (visible) {
         var _opacity = 1.0;
         if (entry.custom_color)
            prop.fillcolor = entry.custom_color;
         else if ((volume.fFillColor > 1) && (volume.fLineColor == 1))
            prop.fillcolor = JSROOT.Painter.root_colors[volume.fFillColor];
         else if (volume.fLineColor >= 0)
            prop.fillcolor = JSROOT.Painter.root_colors[volume.fLineColor];

         if (volume.fMedium && volume.fMedium.fMaterial) {
            var fillstyle = volume.fMedium.fMaterial.fFillStyle,
                transparency = (fillstyle < 3000 || fillstyle > 3100) ? 0 : fillstyle - 3000;
            if (transparency > 0)
               _opacity = (100.0 - transparency) / 100.0;
            if (prop.fillcolor === undefined)
               prop.fillcolor = JSROOT.Painter.root_colors[volume.fMedium.fMaterial.fFillColor];
         }
         if (prop.fillcolor === undefined)
            prop.fillcolor = "lightgrey";

         prop.material = new THREE.MeshLambertMaterial( { transparent: _opacity < 1,
                              opacity: _opacity, wireframe: false, color: prop.fillcolor,
                              side: THREE.FrontSide /* THREE.DoubleSide */, vertexColors: THREE.NoColors /*THREE.VertexColors*/,
                              depthWrite: _opacity == 1 } );
         prop.material.inherentOpacity = _opacity;

      }

      return prop;
   }

   JSROOT.GEO.ClonedNodes.prototype.CreateObject3D = function(stack, toplevel, options) {
      // create hierarchy of Object3D for given stack entry
      // such hierarchy repeats hierarchy of TGeoNodes and set matrix for the objects drawing
      // also set renderOrder, required to handle transparency

      var node = this.nodes[0], three_prnt = toplevel, draw_depth = 0,
          force = (typeof options == 'object') || (options==='force');

      for(var lvl=0; lvl<=stack.length; ++lvl) {
         var nchld = (lvl > 0) ? stack[lvl-1] : 0;
         // extract current node
         if (lvl>0)  node = this.nodes[node.chlds[nchld]];

         var obj3d = undefined;

         if (three_prnt.children)
            for (var i=0;i<three_prnt.children.length;++i) {
               if (three_prnt.children[i].nchld === nchld) {
                  obj3d = three_prnt.children[i];
                  break;
               }
            }

         if (obj3d) {
            three_prnt = obj3d;
            if (obj3d.$jsroot_drawable) draw_depth++;
            continue;
         }

         if (!force) return null;

         obj3d = new THREE.Object3D();

         if (node.matrix) {
            // console.log(stack.toString(), lvl, 'matrix ', node.matrix.toString());
            obj3d.matrix.fromArray(node.matrix);
            obj3d.matrix.decompose( obj3d.position, obj3d.quaternion, obj3d.scale );
         }

         // this.accountNodes(obj3d);
         obj3d.nchld = nchld; // mark index to find it again later

         // add the mesh to the scene
         three_prnt.add(obj3d);

         // this is only for debugging - test inversion of whole geometry
         if ((lvl==0) && (typeof options == 'object') && options.scale) {
            if ((options.scale.x<0) || (options.scale.y<0) || (options.scale.z<0)) {
               obj3d.scale.copy(options.scale);
               obj3d.updateMatrix();
            }
         }

         obj3d.updateMatrixWorld();

         three_prnt = obj3d;
      }

      if ((options === 'mesh') || (options === 'delete_mesh')) {
         var mesh = null;
         if (three_prnt)
            for (var n=0; (n<three_prnt.children.length) && !mesh;++n) {
               var chld = three_prnt.children[n];
               if ((chld.type === 'Mesh') && (chld.nchld === undefined)) mesh = chld;
            }

         if ((options === 'mesh') || !mesh) return mesh;

         var res = three_prnt;
         while (mesh && (mesh !== toplevel)) {
            three_prnt = mesh.parent;
            three_prnt.remove(mesh);
            mesh = (three_prnt.children.length == 0) ? three_prnt : null;
         }

         return res;
      }

      if (three_prnt) {
         three_prnt.$jsroot_drawable = true;
         three_prnt.$jsroot_depth = draw_depth;
      }

      return three_prnt;
   }

   JSROOT.GEO.ClonedNodes.prototype.GetVolumeBoundary = function(viscnt, facelimit, nodeslimit) {

      var result = { min: 0, max: 1, sortidcut: 0 };

      if (!this.sortmap) {
         console.error('sorting map do not exist');
         return result;
      }

      var maxNode, currNode, cnt=0, facecnt=0;

      for (var n = 0; (n < this.sortmap.length) && (cnt < nodeslimit) && (facecnt < facelimit); ++n) {
         var id = this.sortmap[n];
         if (viscnt[id] === 0) continue;
         currNode = this.nodes[id];
         if (!maxNode) maxNode = currNode;
         cnt += viscnt[id];
         facecnt += viscnt[id] * currNode.nfaces;
      }

      if (!currNode) {
         console.error('no volumes selected');
         return result;
      }

      // console.log('Volume boundary ' + currNode.vol + '  cnt ' + cnt + '  faces ' + facecnt);
      result.max = maxNode.vol;
      result.min = currNode.vol;
      result.sortidcut = currNode.sortid; // latest node is not included
      return result;
   }

   /** @brief Collects visible nodes, using maxlimit
     * @desc One can use map to define cut based on the volume or serious of cuts */
   JSROOT.GEO.ClonedNodes.prototype.CollectVisibles = function(maxnumfaces, frustum) {

      // in simple case shape as it is
      if (this.plain_shape)
         return { lst: [ { nodeid: 0, seqid: 0, stack: [], factor: 1, shapeid: 0, server_shape: this.plain_shape } ], complete: true };

      var total = 0, arg = {
         facecnt: 0,
         viscnt: new Array(this.nodes.length), // counter for each node
         reset: function() {
            this.total = 0;
            this.facecnt = 0;
            for (var n=0;n<this.viscnt.length;++n) this.viscnt[n] = 0;
         },
         // nodes: this.nodes,
         func: function(node) {
            this.total++;
            this.facecnt += node.nfaces;
            this.viscnt[node.id]++;
            return true;
         }
      };

      arg.vislvl = this.GetVisLevel();

      arg.reset();
      total = this.ScanVisible(arg);

      var maxnumnodes = this.GetMaxVisNodes();

      if (maxnumnodes > 0) {
         while ((total > maxnumnodes) && (arg.vislvl > 1)) {
            arg.vislvl--;
            arg.reset();
            total = this.ScanVisible(arg);
         }
      }

      this.actual_level = arg.vislvl; // not used, can be shown somewhere in the gui

      var minVol = 0, maxVol = 0, camVol = -1, camFact = 10, sortidcut = this.nodes.length + 1;

      console.log('Total visible nodes ' + total + ' numfaces ' + arg.facecnt);

      if (arg.facecnt > maxnumfaces) {

         var bignumfaces = maxnumfaces * (frustum ? 0.8 : 1.0),
             bignumnodes = maxnumnodes * (frustum ? 0.8 : 1.0);

         // define minimal volume, which always to shown
         var boundary = this.GetVolumeBoundary(arg.viscnt, bignumfaces, bignumnodes);

         minVol = boundary.min;
         maxVol = boundary.max;
         sortidcut = boundary.sortidcut;

         if (frustum) {
             arg.domatrix = true;
             arg.frustum = frustum;
             arg.totalcam = 0;
             arg.func = function(node) {
                if (node.vol <= minVol) // only small volumes are interesting
                   if (this.frustum.CheckShape(this.getmatrix(), node)) {
                      this.viscnt[node.id]++;
                      this.totalcam += node.nfaces;
                   }

                return true;
             }

             for (var n=0;n<arg.viscnt.length;++n) arg.viscnt[n] = 0;

             this.ScanVisible(arg);

             if (arg.totalcam > maxnumfaces*0.2)
                camVol = this.GetVolumeBoundary(arg.viscnt, maxnumfaces*0.2, maxnumnodes*0.2).min;
             else
                camVol = 0;

             camFact = maxVol / ((camVol>0) ? (camVol>0) : minVol);

             // console.log('Limit for camera ' + camVol + '  faces in camera view ' + arg.totalcam);
         }
      }

      arg.items = [];

      arg.func = function(node) {
         if (node.sortid < sortidcut) {
            this.items.push(this.CopyStack());
         } else if ((camVol >= 0) && (node.vol > camVol)) {
            if (this.frustum.CheckShape(this.getmatrix(), node))
               this.items.push(this.CopyStack(camFact));
         }
         return true;
      }

      this.ScanVisible(arg);

      return { lst: arg.items, complete: minVol === 0 };
   }

   /** @brief merge list of drawn objects
     * @desc in current list we should mark if object already exists
     * from previous list we should collect objects which are not there */
   JSROOT.GEO.ClonedNodes.prototype.MergeVisibles = function(current, prev) {

      var indx2 = 0, del = [];
      for (var indx1=0; (indx1<current.length) && (indx2<prev.length); ++indx1) {

         while ((indx2 < prev.length) && (prev[indx2].seqid < current[indx1].seqid)) {
            del.push(prev[indx2++]); // this entry should be removed
         }

         if ((indx2 < prev.length) && (prev[indx2].seqid === current[indx1].seqid)) {
            if (prev[indx2].done) current[indx1].done = true; // copy ready flag
            indx2++;
         }
      }

      // remove rest
      while (indx2<prev.length)
         del.push(prev[indx2++]);

      return del; //
   }

   /** @brief Collect all uniques shapes which should be build
    *  @desc Check if same shape used many time for drawing */
   JSROOT.GEO.ClonedNodes.prototype.CollectShapes = function(lst) {

      // nothing else - just that single shape
      if (this.plain_shape)
         return [ this.plain_shape ];

      var shapes = [];

      for (var i=0;i<lst.length;++i) {
         var entry = lst[i];
         var shape = this.GetNodeShape(entry.nodeid);

         if (!shape) continue; // strange, but avoid misleading

         if (shape._id === undefined) {
            shape._id = shapes.length;

            shapes.push({ id: shape._id, shape: shape, vol: this.nodes[entry.nodeid].vol, refcnt: 1, factor: 1, ready: false });

            // shapes.push( { obj: shape, vol: this.nodes[entry.nodeid].vol });
         } else {
            shapes[shape._id].refcnt++;
         }

         entry.shape = shapes[shape._id]; // remember shape used

         // use maximal importance factor to push element to the front
         if (entry.factor && (entry.factor>entry.shape.factor))
            entry.shape.factor = entry.factor;
      }

      // now sort shapes in volume decrease order
      shapes.sort(function(a,b) { return b.vol*b.factor - a.vol*a.factor; })

      // now set new shape ids according to the sorted order and delete temporary field
      for (var n=0;n<shapes.length;++n) {
         var item = shapes[n];
         item.id = n; // set new ID
         delete item.shape._id; // remove temporary field
      }

      // as last action set current shape id to each entry
      for (var i=0;i<lst.length;++i) {
         var entry = lst[i];
         if (entry.shape) {
            entry.shapeid = entry.shape.id; // keep only id for the entry
            delete entry.shape; // remove direct references
         }
      }

      return shapes;
   }

   JSROOT.GEO.ClonedNodes.prototype.MergeShapesLists = function(oldlst, newlst) {

      if (!oldlst) return newlst;

      // set geometry to shape object itself
      for (var n=0;n<oldlst.length;++n) {
         var item = oldlst[n];

         item.shape._geom = item.geom;
         delete item.geom;

         if (item.geomZ!==undefined) {
            item.shape._geomZ = item.geomZ;
            delete item.geomZ;
         }
      }

      // take from shape (if match)
      for (var n=0;n<newlst.length;++n) {
         var item = newlst[n];

         if (item.shape._geom !== undefined) {
            item.geom = item.shape._geom;
            delete item.shape._geom;
         }

         if (item.shape._geomZ !== undefined) {
            item.geomZ = item.shape._geomZ;
            delete item.shape._geomZ;
         }
      }

      // now delete all unused geometries
      for (var n=0;n<oldlst.length;++n) {
         var item = oldlst[n];
         delete item.shape._geom;
         delete item.shape._geomZ;
      }

      return newlst;
   }

   JSROOT.GEO.ClonedNodes.prototype.BuildShapes = function(lst, limit, timelimit) {

      var created = 0,
          tm1 = new Date().getTime(),
          res = { done: false, shapes: 0, faces: 0, notusedshapes: 0 };

      for (var n=0;n<lst.length;++n) {
         var item = lst[n];

         // if enough faces are produced, nothing else is required
         if (res.done) { item.ready = true; continue; }

         if (!item.ready) {
            item._typename = "$$Shape$$"; // let reuse item for direct drawing
            item.ready = true;
            if (item.geom === undefined) {
               item.geom = JSROOT.GEO.createGeometry(item.shape);
               if (item.geom) created++; // indicate that at least one shape was created
            }
            item.nfaces = JSROOT.GEO.numGeometryFaces(item.geom);
         }

         res.shapes++;
         if (!item.used) res.notusedshapes++;
         res.faces += item.nfaces*item.refcnt;

         if (res.faces >= limit) {
            res.done = true;
         } else if ((created > 0.01*lst.length) && (timelimit!==undefined)) {
            var tm2 = new Date().getTime();
            if (tm2-tm1 > timelimit) return res;
         }
      }

      res.done = true;

      return res;
   }

   JSROOT.GEO.ObjectName = function(obj) {
      if (!obj || !obj.fName) return "";
      return obj.fName + (obj.$geo_suffix ? obj.$geo_suffix : "");
   }

   JSROOT.GEO.CheckDuplicates = function(parent, chlds) {
      if (parent) {
         if (parent.$geo_checked) return;
         parent.$geo_checked = true;
      }

      var names = [], cnts = [], obj = null;
      for (var k=0;k<chlds.length;++k) {
         var chld = chlds[k];
         if (!chld || !chld.fName) continue;
         if (!chld.$geo_suffix) {
            var indx = names.indexOf(chld.fName);
            if (indx>=0) {
               var cnt = cnts[indx] || 1;
               while(names.indexOf(chld.fName+"#"+cnt)>=0) ++cnt;
               chld.$geo_suffix = "#" + cnt;
               cnts[indx] = cnt+1;
            }
         }
         names.push(JSROOT.GEO.ObjectName(chld));
      }
   }

   JSROOT.GEO.createFlippedMesh = function(parent, shape, material) {
      // when transformation matrix includes one or several inversion of axis,
      // one should inverse geometry object, otherwise THREE.js cannot correctly draw it

      var flip =  new THREE.Vector3(1,1,-1);

      if (shape.geomZ === undefined) {

         if (shape.geom.type == 'BufferGeometry') {

            var pos = shape.geom.getAttribute('position').array,
                norm = shape.geom.getAttribute('normal').array;

            var index = shape.geom.getIndex();

            if (index) {
               // we need to unfold all points to
               var arr = index.array,
                   i0 = shape.geom.drawRange.start,
                   ilen = shape.geom.drawRange.count;
               if (i0 + ilen > arr.length) ilen = arr.length - i0;

               var dpos = new Float32Array(ilen*3), dnorm = new Float32Array(ilen*3);
               for (var ii = 0; ii < ilen; ++ii) {
                  var k = arr[i0 + ii];
                  if ((k<0) || (k*3>=pos.length)) console.log('strange index', k*3, pos.length);
                  dpos[ii*3] = pos[k*3];
                  dpos[ii*3+1] = pos[k*3+1];
                  dpos[ii*3+2] = pos[k*3+2];
                  dnorm[ii*3] = norm[k*3];
                  dnorm[ii*3+1] = norm[k*3+1];
                  dnorm[ii*3+2] = norm[k*3+2];
               }

               pos = dpos; norm = dnorm;
            }

            var len = pos.length, n, shift = 0,
                newpos = new Float32Array(len),
                newnorm = new Float32Array(len);

            // we should swap second and third point in each face
            for (n=0; n<len; n+=3) {
               newpos[n]   = pos[n+shift];
               newpos[n+1] = pos[n+1+shift];
               newpos[n+2] = -pos[n+2+shift];

               newnorm[n]   = norm[n+shift];
               newnorm[n+1] = norm[n+1+shift];
               newnorm[n+2] = -norm[n+2+shift];

               shift+=3; if (shift===6) shift=-3; // values 0,3,-3
            }

            shape.geomZ = new THREE.BufferGeometry();
            shape.geomZ.addAttribute( 'position', new THREE.BufferAttribute( newpos, 3 ) );
            shape.geomZ.addAttribute( 'normal', new THREE.BufferAttribute( newnorm, 3 ) );
            // normals are calculated with normal geometry and correctly scaled
            // geom.computeVertexNormals();

         } else {

            shape.geomZ = shape.geom.clone();

            shape.geomZ.scale(flip.x, flip.y, flip.z);

            var face, d, n = 0;
            while(n < shape.geomZ.faces.length) {
               face = geom.faces[n++];
               d = face.b; face.b = face.c; face.c = d;
            }

            // normals are calculated with normal geometry and correctly scaled
            // geom.computeFaceNormals();
         }
      }

      var mesh = new THREE.Mesh( shape.geomZ, material );
      mesh.scale.copy(flip);
      mesh.updateMatrix();

      mesh._flippedMesh = true;

      return mesh;
   }

   /** Cleanup shape entity
    * @private */
   JSROOT.GEO.cleanupShape = function(shape) {
      if (!shape) return;

      if (shape.geom && (typeof shape.geom.dispose == 'funciton'))
         shape.geom.dispose();

      if (shape.geomZ && (typeof shape.geomZ.dispose == 'funciton'))
         shape.geomZ.dispose();

      delete shape.geom;
      delete shape.geomZ;
   }

   /** @brief Set rendering order for created hierarchy
    * @desc depending from provided method sort differently objects
    * @param toplevel - top element
    * @param origin - camera position used to provide sorting
    * @param method - name of sorting method like "pnt", "ray", "size", "dflt"  */
   JSROOT.GEO.produceRenderOrder = function(toplevel, origin, method, clones) {

      var raycast = new THREE.Raycaster();

      function setdefaults(top) {
         if (!top) return;
         top.traverse(function(obj) {
            obj.renderOrder = 0;
            if (obj.material) obj.material.depthWrite = true; // by default depthWriting enabled
         });
      }

      function traverse(obj, lvl, arr) {
         // traverse hierarchy and extract all children of given level
         // if (obj.$jsroot_depth===undefined) return;

         if (!obj.children) return;

         for (var k=0;k<obj.children.length;++k) {
            var chld = obj.children[k];
            if (chld.$jsroot_order === lvl) {
               if (chld.material) {
                  if (chld.material.transparent) {
                     chld.material.depthWrite = false; // disable depth writing for transparent
                     arr.push(chld);
                  } else {
                     setdefaults(chld);
                  }
               }
            } else if ((obj.$jsroot_depth===undefined) || (obj.$jsroot_depth < lvl)) {
               traverse(chld, lvl, arr);
            }
         }
      }

      function sort(arr, minorder, maxorder) {
         // resort meshes using ray caster and camera position
         // idea to identify meshes which are in front or behind

         if (arr.length > 300) {
            // too many of them, just set basic level and exit
            for (var i=0;i<arr.length;++i) arr[i].renderOrder = (minorder + maxorder)/2;
            return false;
         }

         var tmp_vect = new THREE.Vector3();

         // first calculate distance to the camera
         // it gives preliminary order of volumes

         for (var i=0;i<arr.length;++i) {
            var mesh = arr[i],
                box3 = mesh.$jsroot_box3;

            if (!box3)
               mesh.$jsroot_box3 = box3 = JSROOT.GEO.getBoundingBox(mesh);

            if (method === 'size') {
               mesh.$jsroot_distance = box3.getSize(new THREE.Vector3());
               continue;
            }

            if (method === "pnt") {
               mesh.$jsroot_distance = origin.distanceTo(box3.getCenter(tmp_vect));
               continue;
            }

            var dist = Math.min(dist, origin.distanceTo(box3.min), origin.distanceTo(box3.max));

            var pnt = new THREE.Vector3(box3.min.x, box3.min.y, box3.max.z);
            dist = Math.min(dist, origin.distanceTo(pnt));
            pnt.set(box3.min.x, box3.max.y, box3.min.z)
            dist = Math.min(dist, origin.distanceTo(pnt));
            pnt.set(box3.max.x, box3.min.y, box3.min.z)
            dist = Math.min(dist, origin.distanceTo(pnt));

            pnt.set(box3.max.x, box3.max.y, box3.min.z)
            dist = Math.min(dist, origin.distanceTo(pnt));

            pnt.set(box3.max.x, box3.min.y, box3.max.z)
            dist = Math.min(dist, origin.distanceTo(pnt));

            pnt.set(box3.min.x, box3.max.y, box3.max.z)
            dist = Math.min(dist, origin.distanceTo(pnt));

            mesh.$jsroot_distance = dist;
         }

         arr.sort(function(a,b) { return a.$jsroot_distance - b.$jsroot_distance; });

         var resort = new Array(arr.length);

         for (var i=0;i<arr.length;++i) {
            arr[i].$jsroot_index = i;
            resort[i] = arr[i];
         }

         if (method==="ray")
         for (var i=arr.length-1;i>=0;--i) {
            var mesh = arr[i],
                box3 = mesh.$jsroot_box3,
                direction = box3.getCenter(tmp_vect);

            for(var ntry=0; ntry<2;++ntry) {

               direction.sub(origin).normalize();

               raycast.set( origin, direction );

               var intersects = raycast.intersectObjects(arr, false); // only plain array

               var unique = [];

               for (var k1=0;k1<intersects.length;++k1) {
                  if (unique.indexOf(intersects[k1].object)<0) unique.push(intersects[k1].object);
                  // if (intersects[k1].object === mesh) break; // trace until object itself
               }

               intersects = unique;

               if ((intersects.indexOf(mesh)<0) && (ntry>0))
                  console.log('MISS', clones ? clones.ResolveStack(mesh.stack).name : "???");

               if ((intersects.indexOf(mesh)>=0) || (ntry>0)) break;

               var pos = mesh.geometry.attributes.position.array;

               direction = new THREE.Vector3((pos[0]+pos[3]+pos[6])/3, (pos[1]+pos[4]+pos[7])/3, (pos[2]+pos[5]+pos[8])/3);

               direction.applyMatrix4(mesh.matrixWorld);
            }

            // now push first object in intersects to the front
            for (var k1=0;k1<intersects.length-1;++k1) {
               var mesh1 = intersects[k1], mesh2 = intersects[k1+1],
                   i1 = mesh1.$jsroot_index, i2 = mesh2.$jsroot_index;
               if (i1<i2) continue;
               for (var ii=i2;ii<i1;++ii) {
                  resort[ii] = resort[ii+1];
                  resort[ii].$jsroot_index = ii;
               }
               resort[i1] = mesh2;
               mesh2.$jsroot_index = i1;
            }

         }

         for (var i=0;i<resort.length;++i) {
            resort[i].renderOrder = maxorder - (i+1) / (resort.length+1) * (maxorder-minorder);
            delete resort[i].$jsroot_index;
            delete resort[i].$jsroot_distance;
         }

         return true;
      }

      function process(obj, lvl, minorder, maxorder) {
         var arr = [], did_sort = false;

         traverse(obj, lvl, arr);

         if (!arr.length) return;

         if (minorder === maxorder) {
            for (var k=0;k<arr.length;++k)
               arr[k].renderOrder = minorder;
         } else {
           did_sort = sort(arr, minorder, maxorder);
           if (!did_sort) minorder = maxorder = (minorder + maxorder) / 2;
         }

         for (var k=0;k<arr.length;++k) {
            var next = arr[k].parent, min = minorder, max = maxorder;

            if (did_sort) {
               max = arr[k].renderOrder;
               min = max - (maxorder - minorder) / (arr.length + 2);
            }

            process(next, lvl+1, min, max);
         }
      }

      if (!method || (method==="dflt"))
         setdefaults(toplevel);
      else
         process(toplevel, 0, 1, 1000000);
   }

   JSROOT.GEO.build = function(obj, opt, call_back) {
      // function can be used to build three.js model for TGeo object

      if (!obj) return;

      if (!opt) opt = {};
      if (!opt.numfaces) opt.numfaces = 100000;
      if (!opt.numnodes) opt.numnodes = 1000;
      if (!opt.frustum) opt.frustum = null;

      opt.res_mesh = opt.res_faces = 0;

      var shape = null, hide_top = false;

      if (('fShapeBits' in obj) && ('fShapeId' in obj)) {
         shape = obj; obj = null;
      } else if ((obj._typename === 'TGeoVolumeAssembly') || (obj._typename === 'TGeoVolume')) {
         shape = obj.fShape;
      } else if ((obj._typename === "TEveGeoShapeExtract") || (obj._typename === "ROOT::Experimental::TEveGeoShapeExtract")  ) {
         shape = obj.fShape;
      } else if (obj._typename === 'TGeoManager') {
         obj = obj.fMasterVolume;
         hide_top = !opt.showtop;
         shape = obj.fShape;
      } else if (obj.fVolume) {
         shape = obj.fVolume.fShape;
      } else {
         obj = null;
      }

      if (opt.composite && shape && (shape._typename == 'TGeoCompositeShape') && shape.fNode)
         obj = JSROOT.GEO.buildCompositeVolume(shape);

      if (!obj && shape)
         obj = JSROOT.extend(JSROOT.Create("TEveGeoShapeExtract"),
                   { fTrans: null, fShape: shape, fRGBA: [0, 1, 0, 1], fElements: null, fRnrSelf: true });

      if (!obj) return null;

      if (obj._typename.indexOf('TGeoVolume') === 0)
         obj = { _typename:"TGeoNode", fVolume: obj, fName: obj.fName, $geoh: obj.$geoh, _proxy: true };

      var clones = new JSROOT.GEO.ClonedNodes(obj);
      clones.SetVisLevel(opt.vislevel);
      clones.SetMaxVisNodes(opt.numnodes);

      var uniquevis = opt.no_screen ? 0 : clones.MarkVisibles(true);
      if (uniquevis <= 0)
         uniquevis = clones.MarkVisibles(false, false, null, hide_top);
      else
         uniquevis = clones.MarkVisibles(true, true); // copy bits once and use normal visibility bits

      clones.ProduceIdShits();

      // collect visible nodes
      var res = clones.CollectVisibles(opt.numfaces, opt.frustum);

      var draw_nodes = res.lst;

      // collect shapes
      var shapes = clones.CollectShapes(draw_nodes);

      clones.BuildShapes(shapes, opt.numfaces);

      var toplevel = new THREE.Object3D();

      for (var n=0; n < draw_nodes.length;++n) {
         var entry = draw_nodes[n];
         if (entry.done) continue;

         var shape = shapes[entry.shapeid];
         if (!shape.ready) {
            console.warn('shape marked as not ready when should');
            break;
         }
         entry.done = true;
         shape.used = true; // indicate that shape was used in building

         if (!shape.geom || (shape.nfaces === 0)) {
            // node is visible, but shape does not created
            clones.CreateObject3D(entry.stack, toplevel, 'delete_mesh');
            continue;
         }

         var prop = clones.getDrawEntryProperties(entry);

         opt.res_mesh++;
         opt.res_faces += shape.nfaces;

         var obj3d = clones.CreateObject3D(entry.stack, toplevel, opt);

         prop.material.wireframe = opt.wireframe;

         prop.material.side = opt.doubleside ? THREE.DoubleSide : THREE.FrontSide;

         var mesh = null;

         if (obj3d.matrixWorld.determinant() > -0.9) {
            mesh = new THREE.Mesh( shape.geom, prop.material );
         } else {
            mesh = JSROOT.GEO.createFlippedMesh(obj3d, shape, prop.material);
         }

         obj3d.add(mesh);
         // specify rendering order, required for transparency handling
         //if (obj3d.$jsroot_depth !== undefined)
         //   mesh.renderOrder = clones.maxdepth - obj3d.$jsroot_depth;
         //else
         //   mesh.renderOrder = clones.maxdepth - entry.stack.length;
      }

      JSROOT.CallBack(call_back, toplevel);

      return toplevel;
   }

   JSROOT.GEO.getBoundingBox = function(node, box3) {

      // extract code of Box3.expandByObject
      // Major difference - do not traverse hierarchy

      if (!node || !node.geometry) return box3;

      if (!box3) { box3 = new THREE.Box3(); box3.makeEmpty(); }

      node.updateMatrixWorld();

      var v1 = new THREE.Vector3(),
          geometry = node.geometry;

      if ( geometry.isGeometry ) {
         var vertices = geometry.vertices;
         for (var i = 0, l = vertices.length; i < l; i ++ ) {
            v1.copy( vertices[ i ] );
            v1.applyMatrix4( node.matrixWorld );
            box3.expandByPoint( v1 );
         }
      } else if ( geometry.isBufferGeometry ) {
         var attribute = geometry.attributes.position;
         if ( attribute !== undefined ) {
            for (var i = 0, l = attribute.count; i < l; i ++ ) {
               // v1.fromAttribute( attribute, i ).applyMatrix4( node.matrixWorld );
               v1.fromBufferAttribute( attribute, i ).applyMatrix4( node.matrixWorld );
               box3.expandByPoint( v1 );
            }
         }
      }

      return box3;
   }


   return JSROOT;

}));

