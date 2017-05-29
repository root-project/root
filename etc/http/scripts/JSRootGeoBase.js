/** @file JSRootGeoBase.js */
/// Basic functions for work with TGeo classes

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      // AMD. Register as an anonymous module.
      define( [ 'JSRootCore', 'threejs', 'ThreeCSG' ], factory );
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
   // === functions to create THREE.Geometry for TGeo shapes ========================

   /** @namespace JSROOT.GEO */
   /// Holder of all TGeo-related functions and classes
   JSROOT.GEO = {
         GradPerSegm: 6,     // grad per segment in cylined/spherical symetry shapes
         CompressComp: true  // use faces compression in composite shapes
    };

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.BITS = {
         kVisOverride     : JSROOT.BIT(0),           // volume's vis. attributes are overidden
         kVisNone         : JSROOT.BIT(1),           // the volume/node is invisible, as well as daughters
         kVisThis         : JSROOT.BIT(2),           // this volume/node is visible
         kVisDaughters    : JSROOT.BIT(3),           // all leaves are visible
         kVisOneLevel     : JSROOT.BIT(4),           // first level daughters are visible
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
      JSROOT.GEO.SetBit(this, JSROOT.GEO.BITS.kVisDaughters, !flag);
      JSROOT.GEO.SetBit(this, JSROOT.GEO.BITS.kVisOneLevel, false);

      if (this.fNodes)
         for (var n=0;n<this.fNodes.arr.length;++n) {
            var sub = this.fNodes.arr[n].fVolume;
            JSROOT.GEO.SetBit(sub, JSROOT.GEO.BITS.kVisThis, !flag);
            // JSROOT.GEO.SetBit(sub, JSROOT.GEO.BITS.kVisDaughters, !flag);
            //JSROOT.GEO.SetBit(sub, JSROOT.GEO.BITS.kVisOneLevel, false);
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

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.NodeKind = function(obj) {
      // return kind of the geo nodes
      // 0 - TGeoNode
      // 1 - TEveGeoNode
      // -1 - unsupported type

      if ((obj === undefined) || (obj === null) || (typeof obj !== 'object')) return -1;

      return ('fShape' in obj) && ('fTrans' in obj) ? 1 : 0;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.getNodeProperties = function(kind, node, visible) {
      // function return different properties for specified node
      // Only if node visible, material will be created

      if (kind === 1) {
         // special handling for EVE nodes

         var prop = { name: node.fName, nname: node.fName, shape: node.fShape, material: null, chlds: null };

         if (node.fElements !== null) prop.chlds = node.fElements.arr;

         if (visible) {
            var _transparent = false, _opacity = 1.0;
            if ( node.fRGBA[3] < 1.0) {
               _transparent = true;
               _opacity = node.fRGBA[3];
            }
            prop.fillcolor = new THREE.Color( node.fRGBA[0], node.fRGBA[1], node.fRGBA[2] );
            prop.material = new THREE.MeshLambertMaterial( { transparent: _transparent,
                             opacity: _opacity, wireframe: false, color: prop.fillcolor,
                             side: THREE.FrontSide /* THREE.DoubleSide*/, vertexColors: THREE.NoColors /*THREE.VertexColors */,
                             overdraw: 0. } );
            prop.material.alwaysTransparent = _transparent;
            prop.material.inherentOpacity = _opacity;
         }

         return prop;
      }

      var volume = node.fVolume;

      var prop = { name: volume.fName, nname: node.fName, volume: node.fVolume, shape: volume.fShape, material: null, chlds: null };

      if (node.fVolume.fNodes !== null) prop.chlds = node.fVolume.fNodes.arr;

      if (visible) {
         var _transparent = false, _opacity = 1.0;
         if ((volume.fFillColor > 1) && (volume.fLineColor == 1))
            prop.fillcolor = JSROOT.Painter.root_colors[volume.fFillColor];
         else
         if (volume.fLineColor >= 0)
            prop.fillcolor = JSROOT.Painter.root_colors[volume.fLineColor];

         if (volume.fMedium && volume.fMedium.fMaterial) {
            var fillstyle = volume.fMedium.fMaterial.fFillStyle;
            var transparency = (fillstyle < 3000 || fillstyle > 3100) ? 0 : fillstyle - 3000;
            if (transparency > 0) {
               _transparent = true;
               _opacity = (100.0 - transparency) / 100.0;
            }
            if (prop.fillcolor === undefined)
               prop.fillcolor = JSROOT.Painter.root_colors[volume.fMedium.fMaterial.fFillColor];
         }
         if (prop.fillcolor === undefined)
            prop.fillcolor = "lightgrey";

         prop.material = new THREE.MeshLambertMaterial( { transparent: _transparent,
                              opacity: _opacity, wireframe: false, color: prop.fillcolor,
                              side: THREE.FrontSide /* THREE.DoubleSide */, vertexColors: THREE.NoColors /*THREE.VertexColors*/,
                              overdraw: 0. } );
         prop.material.alwaysTransparent = _transparent;
         prop.material.inherentOpacity = _opacity;
      }

      return prop;
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
     // reduce has same meening and should be the same

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

   // same methods as GeometryCreator, but this different implementation

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
      // from four vertices one normaly creates two faces (1,2,3) and (1,3,4)
      // if (reduce==1), first face is reduced
      //  if (reduce==2), second face is reduced

      if (reduce === undefined) reduce = 0;

      this.v1 = new ThreeBSP.Vertex( x1, y1, z1, 0, 0, 0 );
      this.v2 = (reduce===1) ? null : new ThreeBSP.Vertex( x2, y2, z2, 0, 0, 0 );
      this.v3 = new ThreeBSP.Vertex( x3, y3, z3, 0, 0, 0 );
      this.v4 = (reduce===2) ? null : new ThreeBSP.Vertex( x4, y4, z4, 0, 0, 0 );

      this.reduce = reduce;

      if (this.multi) {
         //console.log('n',this.multi);
         //console.log('v1:' + x1.toFixed(1) + ':' + y1.toFixed(1) + ':'+ z1.toFixed(1));
         //console.log('v2:' + x2.toFixed(1) + ':' + y2.toFixed(1) + ':'+ z2.toFixed(1));
         //console.log('v3:' + x3.toFixed(1) + ':' + y3.toFixed(1) + ':'+ z3.toFixed(1));
         //console.log('v4:' + x4.toFixed(1) + ':' + y4.toFixed(1) + ':'+ z4.toFixed(1));

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

      // if (debug) console.log('NORM  x', this.cb.x.toFixed(1), '  y ',  this.cb.y.toFixed(1), '   z ' , this.cb.z.toFixed(1));

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
   JSROOT.GEO.createCube = function( shape ) {

      // instead of BoxGeometry create all vertices and faces ourself
      // reduce number of allocated objects

      //return new THREE.BoxGeometry( 2*shape.fDX, 2*shape.fDY, 2*shape.fDZ );

      var geom = new THREE.Geometry();

      geom.vertices.push( new THREE.Vector3( shape.fDX,  shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX,  shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX, -shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3( shape.fDX, -shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX,  shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX,  shape.fDY,  shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX, -shape.fDY, -shape.fDZ ) );
      geom.vertices.push( new THREE.Vector3(-shape.fDX, -shape.fDY,  shape.fDZ ) );

      var indicies = [0,2,1, 2,3,1, 4,6,5, 6,7,5, 4,5,1, 5,0,1, 7,6,2, 6,3,2, 5,7,0, 7,2,0, 1,3,4, 3,6,4];

      // normals for each  pair of faces
      var normals = [ 1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1,  0,0,-1 ];

      var color = new THREE.Color();
      var norm = null;
      for (var n=0; n < indicies.length; n+=3) {
          if (n % 6 === 0) norm = new THREE.Vector3(normals[n/2], normals[n/2+1], normals[n/2+2]);
          var face = new THREE.Face3( indicies[n], indicies[n+1], indicies[n+2], norm, color, 0);
          geom.faces.push(face);
      }

      return geom;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createCubeBuffer = function( shape, faces_limit) {

      if (faces_limit < 0) return 12;

      var dx = shape.fDX, dy = shape.fDY, dz = shape.fDZ;

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(12);

      // var creator = new JSROOT.GEO.GeometryCreator(12);

      creator.AddFace4(dx,dy,dz, dx,-dy,dz, dx,-dy,-dz, dx,dy,-dz); creator.SetNormal(1,0,0);

      creator.AddFace4(-dx,dy,-dz, -dx,-dy,-dz, -dx,-dy,dz, -dx,dy,dz); creator.SetNormal(-1,0,0);

      creator.AddFace4(-dx,dy,-dz, -dx,dy,dz, dx,dy,dz, dx,dy,-dz); creator.SetNormal(0,1,0);

      creator.AddFace4(-dx,-dy,dz, -dx,-dy,-dz, dx,-dy,-dz, dx,-dy,dz); creator.SetNormal(0,-1,0);

      creator.AddFace4(-dx,dy,dz, -dx,-dy,dz, dx,-dy,dz, dx,dy,dz); creator.SetNormal(0,0,1);

      creator.AddFace4(dx,dy,-dz, dx,-dy,-dz, -dx,-dy,-dz, -dx,dy,-dz); creator.SetNormal(0,0,-1);

      return creator.Create();
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createPara = function( shape ) {

      var txy = shape.fTxy, txz = shape.fTxz, tyz = shape.fTyz;

      var verticesOfShape = [
          -shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY-shape.fZ*tyz,  -shape.fZ,
          -shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY-shape.fZ*tyz,  -shape.fZ,
           shape.fZ*txz-txy*shape.fY-shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY-shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz+txy*shape.fY+shape.fX,  shape.fY+shape.fZ*tyz,   shape.fZ,
           shape.fZ*txz-txy*shape.fY+shape.fX, -shape.fY+shape.fZ*tyz,   shape.fZ ];

      var indicesOfFaces = [ 4,6,5,   4,7,6,   0,3,7,   7,4,0,
                             4,5,1,   1,0,4,   6,2,1,   1,5,6,
                             7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      var geom = new THREE.Geometry();

      for (var i = 0; i < verticesOfShape.length; i += 3)
         geom.vertices.push( new THREE.Vector3( verticesOfShape[i], verticesOfShape[i+1], verticesOfShape[i+2] ) );

      var color = new THREE.Color();

      for (var i = 0; i < indicesOfFaces.length; i += 3)
         geom.faces.push( new THREE.Face3( indicesOfFaces[i], indicesOfFaces[i+1], indicesOfFaces[i+2], null, color, 0 ) );

      geom.computeFaceNormals();

      return geom;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.create8edgesBuffer = function( v, faces_limit ) {

      var indicies = [ 4,7,6,5,  0,3,7,4,  4,5,1,0,  6,2,1,5,  7,3,2,6,  1,2,3,0 ];

      var creator = (faces_limit > 0) ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(12);

      // var creator = new JSROOT.GEO.GeometryCreator(12);

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
   JSROOT.GEO.createTrapezoid = function( shape ) {

      var y1, y2;
      if (shape._typename == "TGeoTrd1") {
         y1 = y2 = shape.fDY;
      } else {
         y1 = shape.fDy1; y2 = shape.fDy2;
      }

      var verticesOfShape = [
            -shape.fDx1,  y1, -shape.fDZ,
             shape.fDx1,  y1, -shape.fDZ,
             shape.fDx1, -y1, -shape.fDZ,
            -shape.fDx1, -y1, -shape.fDZ,
            -shape.fDx2,  y2,  shape.fDZ,
             shape.fDx2,  y2,  shape.fDZ,
             shape.fDx2, -y2,  shape.fDZ,
            -shape.fDx2, -y2,  shape.fDZ
         ];

      var indicesOfFaces = [
          4,6,5,   4,7,6,   0,3,7,   7,4,0,
          4,5,1,   1,0,4,   6,2,1,   1,5,6,
          7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      var geometry = new THREE.Geometry();
      for (var i = 0; i < 24; i += 3)
         geometry.vertices.push( new THREE.Vector3( verticesOfShape[i], verticesOfShape[i+1], verticesOfShape[i+2] ) );

      var color = new THREE.Color();

      for (var i = 0; i < 36; i += 3)
         geometry.faces.push( new THREE.Face3( indicesOfFaces[i], indicesOfFaces[i+1], indicesOfFaces[i+2], null, color, 0 ) );

      geometry.computeFaceNormals();
      return geometry;
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
   JSROOT.GEO.createArb8 = function( shape ) {

      var vertices = [
           shape.fXY[0][0], shape.fXY[0][1], -shape.fDZ,
           shape.fXY[1][0], shape.fXY[1][1], -shape.fDZ,
           shape.fXY[2][0], shape.fXY[2][1], -shape.fDZ,
           shape.fXY[3][0], shape.fXY[3][1], -shape.fDZ,
           shape.fXY[4][0], shape.fXY[4][1],  shape.fDZ,
           shape.fXY[5][0], shape.fXY[5][1],  shape.fDZ,
           shape.fXY[6][0], shape.fXY[6][1],  shape.fDZ,
           shape.fXY[7][0], shape.fXY[7][1],  shape.fDZ
      ];
      var indicies = [
                      4,6,5,   4,7,6,   0,3,7,   7,4,0,
                      4,5,1,   1,0,4,   6,2,1,   1,5,6,
                      7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      // detect same vertecies
      for (var n=3; n<vertices.length; n+=3) {
         if ((vertices[n-3] === vertices[n]) &&
             (vertices[n-2] === vertices[n+1]) &&
             (vertices[n-1] === vertices[n+2])) {
                for (var k=0;k<indicies.length;++k)
                   if (indicies[k] === n/3) indicies[k] = n/3-1;
            }
         }


      // detect duplicated faces or faces with same vertex
      var map = []; // list of existing faces (with all rotations)
      var usage = [0,0,0,0,0,0,0,0]; // usage counter

      for (var k=0;k<indicies.length;k+=3) {
         var id1 = indicies[k]*100   + indicies[k+1]*10 + indicies[k+2],
             id2 = indicies[k+1]*100 + indicies[k+2]*10 + indicies[k],
             id3 = indicies[k+2]*100 + indicies[k]*10   + indicies[k+1];

         if ((indicies[k] == indicies[k+1]) || (indicies[k] == indicies[k+2]) || (indicies[k+1] == indicies[k+2]) ||
             (map.indexOf(id1)>=0) || (map.indexOf(id2)>=0) || (map.indexOf(id3)>=0)) {
            indicies[k] = indicies[k+1] = indicies[k+2] = -1;
         } else {
            map.push(id1,id2,id3);
            usage[indicies[k]]++;
            usage[indicies[k+1]]++;
            usage[indicies[k+2]]++;
         }
      }

      var geometry = new THREE.Geometry();
      for (var i = 0; i < 8; ++i) {
         if (usage[i] > 0) {
            usage[i] = geometry.vertices.length; // use array to remap old vertices
            geometry.vertices.push( new THREE.Vector3( vertices[i*3], vertices[i*3+1], vertices[i*3+2] ) );
         }
         else {
            usage[i] = -1;
         }
      }

      var color = new THREE.Color();

      for (var i = 0; i < 36; i += 3) {
         if (indicies[i]<0) continue;

         var a = usage[indicies[i]],
             b = usage[indicies[i+1]],
             c = usage[indicies[i+2]];

         geometry.faces.push( new THREE.Face3( a, b, c, null, color, 0 ) );
      }

      geometry.computeFaceNormals();
      return geometry;
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
         ];

      var indicies = [
          4,7,6,   6,5,4,   0,3,7,   7,4,0,
          4,5,1,   1,0,4,   6,2,1,   1,5,6,
          7,3,2,   2,6,7,   1,2,3,   3,0,1 ];

      // var face4 = [ 4,7,6,5,  0,3,7,4,  4,5,1,0,  6,2,1,5,  7,3,2,6,  1,2,3,0 ];

      // detect same vertecies on both Z-layers
      for (var side=0;side<vertices.length;side += vertices.length/2)
         for (var n1 = side; n1 < side + vertices.length/2 - 3 ; n1+=3)
            for (var n2 = n1+3; n2 < side + vertices.length/2 ; n2+=3)
               if ((vertices[n1] === vertices[n2]) &&
                   (vertices[n1+1] === vertices[n2+1]) &&
                   (vertices[n1+2] === vertices[n2+2])) {
                      for (var k=0;k<indicies.length;++k)
                        if (indicies[k] === n2/3) indicies[k] = n1/3;
                  }


      var map = []; // list of existing faces (with all rotations)
      var numfaces = 0;

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
   JSROOT.GEO.createSphere = function( shape, faces_limit ) {
      var outerRadius = shape.fRmax,
          innerRadius = shape.fRmin,
          phiStart = shape.fPhi1 + 180,
          phiLength = shape.fPhi2 - shape.fPhi1,
          thetaStart = shape.fTheta1,
          thetaLength = shape.fTheta2 - shape.fTheta1,
          widthSegments = shape.fNseg,
          heightSegments = shape.fNz;

      var noInside = (innerRadius <= 0);

      while (phiStart >= 360) phiStart-=360;

      if (faces_limit  > 0) {

         var fact = (noInside ? 2 : 4) * widthSegments * heightSegments / faces_limit;

         if (fact > 1.) {
            widthSegments = Math.round(widthSegments/Math.sqrt(fact));
            heightSegments = Math.round(heightSegments/Math.sqrt(fact));
         }
      }

      var sphere = new THREE.SphereGeometry( outerRadius, widthSegments, heightSegments,
                                             phiStart*Math.PI/180, phiLength*Math.PI/180, thetaStart*Math.PI/180, thetaLength*Math.PI/180);
      sphere.applyMatrix( new THREE.Matrix4().makeRotationX( Math.PI / 2 ) );

      var geometry = new THREE.Geometry();
      var color = new THREE.Color();

      // add outer sphere
      for (var n=0; n < sphere.vertices.length; ++n)
         geometry.vertices.push(sphere.vertices[n]);

      // add faces
      for (var n=0; n < sphere.faces.length; ++n) {
         var face = sphere.faces[n];
         geometry.faces.push(new THREE.Face3( face.a, face.b, face.c, null, color, 0 ) );
      }

      var shift = geometry.vertices.length;

      if (noInside) {
         // simple sphere without inner cut
         if ((thetaLength === 180) && (phiLength === 360)) {
            geometry.computeFaceNormals();
            return geometry;
         }

         geometry.vertices.push(new THREE.Vector3(0, 0, 0));
      } else {
         var k = innerRadius / outerRadius;

         // add inner sphere
         for (var n=0; n < sphere.vertices.length; ++n) {
            var v = sphere.vertices[n];
            geometry.vertices.push(new THREE.Vector3(k*v.x, k*v.y, k*v.z));
         }
         for (var n=0; n < sphere.faces.length; ++n) {
            var face = sphere.faces[n];
            geometry.faces.push(new THREE.Face3( shift+face.b, shift+face.a, shift+face.c, null, color, 0 ) );
         }
      }

      if (thetaLength !== 180) {
         // add top cap
         for (var i = 0; i < widthSegments; ++i) {
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i+0, i+1, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i+0, i+1, i+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i+1, i+shift+1, i+shift, null, color, 0 ) );
            }
         }

         var dshift = sphere.vertices.length - widthSegments - 1;

         // add bottom cap
         for (var i = dshift; i < dshift + widthSegments; ++i) {
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i+0, i+1, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i+1, i+0, i+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i+shift+1, i+1, i+shift, null, color, 0 ) );
            }
         }
      }

      if (phiLength !== 360) {
         // one cuted side
         for (var j=0; j<heightSegments; j++) {
            var i1 = j*(widthSegments+1);
            var i2 = (j+1)*(widthSegments+1);
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i1, i2, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i2, i1, i1+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2+shift, i2, i1+shift, null, color, 0 ));
            }
         }
         // another cuted side
         for (var j=0;j<heightSegments;j++) {
            var i1 = (j+1)*(widthSegments+1) - 1;
            var i2 = (j+2)*(widthSegments+1) - 1;
            if (noInside) {
               geometry.faces.push( new THREE.Face3( i1, i2, shift, null, color, 0 ) );
            } else {
               geometry.faces.push( new THREE.Face3( i1, i2, i1+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift, null, color, 0));
            }
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createSphereBuffer = function( shape, faces_limit ) {
      var radius = [shape.fRmax, shape.fRmin],
          phiStart = shape.fPhi1,
          phiLength = shape.fPhi2 - shape.fPhi1,
          thetaStart = shape.fTheta1,
          thetaLength = shape.fTheta2 - shape.fTheta1,
          widthSegments = shape.fNseg,
          heightSegments = shape.fNz;

      var noInside = (radius[1] <= 0);

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

         // use direct algorithm for the sphere - here normals and position can be calculated direclty
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
   JSROOT.GEO.createTube = function( shape ) {
      var outerRadius1, innerRadius1, outerRadius2, innerRadius2;
      if ((shape._typename == "TGeoCone") || (shape._typename == "TGeoConeSeg")) {
         outerRadius1 = shape.fRmax2;
         innerRadius1 = shape.fRmin2;
         outerRadius2 = shape.fRmax1;
         innerRadius2 = shape.fRmin1;
      } else {
         outerRadius1 = outerRadius2 = shape.fRmax;
         innerRadius1 = innerRadius2 = shape.fRmin;
      }

      var hasrmin = (innerRadius1 > 0) || (innerRadius2 > 0);

      if (hasrmin) {
         if (innerRadius1 <= 0) { innerRadius1 = 0.0000001; JSROOT.GEO.warn('zero inner radius1 in tube - not yet supported'); }
         if (innerRadius2 <= 0) { innerRadius2 = 0.0000001; JSROOT.GEO.warn('zero inner radius1 in tube - not yet supported'); }
      }

      var thetaStart = 0, thetaLength = 360;
      if ((shape._typename == "TGeoConeSeg") || (shape._typename == "TGeoTubeSeg") || (shape._typename == "TGeoCtub")) {
         thetaStart = shape.fPhi1;
         thetaLength = shape.fPhi2 - shape.fPhi1;
      }

      var radiusSegments = Math.round(thetaLength/JSROOT.GEO.GradPerSegm);
      if (radiusSegments < 4) radiusSegments = 4;

      var extrapnt = (thetaLength < 360) ? 1 : 0;

      var nsegm = radiusSegments + extrapnt;

      var phi0 = thetaStart*Math.PI/180, dphi = thetaLength/radiusSegments*Math.PI/180;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(nsegm), _cos = new Float32Array(nsegm);
      for (var seg=0; seg<nsegm; ++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var geometry = new THREE.Geometry();

      // add inner tube vertices

      if (hasrmin) {
         for (var seg=0; seg<nsegm; ++seg)
            geometry.vertices.push( new THREE.Vector3( innerRadius1*_cos[seg], innerRadius1*_sin[seg], shape.fDZ));
         for (var seg=0; seg<nsegm; ++seg)
            geometry.vertices.push( new THREE.Vector3( innerRadius2*_cos[seg], innerRadius2*_sin[seg], -shape.fDZ));
      } else {
         geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDZ));
         geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDZ));
      }

      var shift = geometry.vertices.length;

      // add outer tube vertices
      for (var seg=0; seg<nsegm; ++seg)
         geometry.vertices.push( new THREE.Vector3( outerRadius1*_cos[seg], outerRadius1*_sin[seg], shape.fDZ));
      for (var seg=0; seg<nsegm; ++seg)
         geometry.vertices.push( new THREE.Vector3( outerRadius2*_cos[seg], outerRadius2*_sin[seg], -shape.fDZ));

      // recalculate Z of all vertices for ctub shape
      if (shape._typename == "TGeoCtub")
         for (var n=0;n<geometry.vertices.length;++n) {
            var vertex = geometry.vertices[n];
            if (vertex.z<0) vertex.z = -shape.fDz-(vertex.x*shape.fNlow[0]+vertex.y*shape.fNlow[1])/shape.fNlow[2];
                       else vertex.z = shape.fDz-(vertex.x*shape.fNhigh[0]+vertex.y*shape.fNhigh[1])/shape.fNhigh[2];
         }

      var color = new THREE.Color(); // make dummy color for all faces

      // add inner tube faces
      if (hasrmin)
         for (var seg=0; seg<radiusSegments; ++seg) {
            var seg1 = (extrapnt === 1) ? (seg + 1) : (seg + 1) % radiusSegments;
            geometry.faces.push( new THREE.Face3( nsegm + seg, seg,  seg1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( nsegm + seg, seg1, nsegm + seg1, null, color, 0 ) );
         }

      // add outer tube faces
      for (var seg=0; seg<radiusSegments; ++seg) {
         var seg1 = (extrapnt === 1) ? (seg + 1) : (seg + 1) % radiusSegments;
         geometry.faces.push( new THREE.Face3( shift+seg, shift + nsegm + seg, shift + seg1, null, color, 0 ) );
         geometry.faces.push( new THREE.Face3( shift + nsegm + seg, shift + nsegm + seg1, shift + seg1, null, color, 0 ) );
      }


      // add top cap
      for (var i = 0; i < radiusSegments; ++i){
         var i1 = (extrapnt === 1) ? (i+1) : (i+1) % radiusSegments;
         if (hasrmin) {
            geometry.faces.push( new THREE.Face3( i, i+shift, i1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( i+shift, i1+shift, i1, null, color, 0 ) );
         } else {
            geometry.faces.push( new THREE.Face3( 0, i+shift, i1+shift, null, color, 0 ) );
         }
      }

      // add bottom cap
      for (var i = 0; i < radiusSegments; ++i) {
         var i1 = (extrapnt === 1) ? (i+1) : (i+1) % radiusSegments;
         if (hasrmin) {
            geometry.faces.push( new THREE.Face3( nsegm+i+shift, nsegm+i,  nsegm+i1, null, color, 0 ) );
            geometry.faces.push( new THREE.Face3( nsegm+i+shift, nsegm+i1, nsegm+i1+shift, null, color, 0 ) );
         } else {
            geometry.faces.push( new THREE.Face3( nsegm+i+shift, 1, nsegm+i1+shift, null, color, 0 ) );
         }
      }

      // close cut regions
      if (extrapnt === 1) {
          if (hasrmin) {
             geometry.faces.push( new THREE.Face3( 0, nsegm, shift+nsegm, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift+nsegm, shift, null, color, 0 ) );
          } else {
             geometry.faces.push( new THREE.Face3( 0, 1, shift+nsegm, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift+nsegm, shift, null, color, 0 ) );
          }

          if (hasrmin) {
             geometry.faces.push( new THREE.Face3( radiusSegments, shift+2*radiusSegments+1, 2*radiusSegments+1, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( radiusSegments, shift + radiusSegments, shift+2*radiusSegments+1, null, color, 0 ) );
          } else {
             geometry.faces.push( new THREE.Face3( 0, shift+2*radiusSegments+1, 1, null, color, 0 ) );
             geometry.faces.push( new THREE.Face3( 0, shift + radiusSegments, shift+2*radiusSegments+1,  null, color, 0 ) );
          }
      }

      geometry.computeFaceNormals();

      return geometry;
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

      var hasrmin = (innerR[0] > 0) || (innerR[1] > 0);

      var thetaStart = 0, thetaLength = 360;
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
   JSROOT.GEO.createEltu = function( shape ) {
      var geometry = new THREE.Geometry();

      var radiusSegments = Math.round(360/JSROOT.GEO.GradPerSegm);

      // calculate all sin/cos tables in advance
      var x = new Float32Array(radiusSegments),
          y = new Float32Array(radiusSegments);
      for (var seg=0; seg<radiusSegments; ++seg) {
         var phi = seg/radiusSegments*2*Math.PI;
         x[seg] = shape.fRmin*Math.cos(phi);
         y[seg] = shape.fRmax*Math.sin(phi);
      }

      // create vertices
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.vertices.push( new THREE.Vector3( x[seg], y[seg], -shape.fDZ));
      geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDZ));

      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.vertices.push( new THREE.Vector3( x[seg], y[seg], +shape.fDZ));
      geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDZ));

      var color = new THREE.Color();

      // create tube faces
      for (var seg=0; seg<radiusSegments; ++seg) {
         var seg1 = (seg + 1) % radiusSegments;
         geometry.faces.push( new THREE.Face3( seg+radiusSegments+1, seg, seg1, null, color, 0 ) );
         geometry.faces.push( new THREE.Face3( seg+radiusSegments+1, seg1, seg1+radiusSegments+1, null, color, 0 ) );
      }

      // create bottom cap
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.faces.push( new THREE.Face3( seg, radiusSegments, (seg + 1) % radiusSegments, null, color, 0 ));

      // create upper cap
      var shift = radiusSegments + 1;
      for (var seg=0; seg<radiusSegments; ++seg)
         geometry.faces.push( new THREE.Face3( shift+seg, shift+ (seg + 1) % radiusSegments, shift+radiusSegments, null, color, 0 ));

      geometry.computeFaceNormals();
      return geometry;
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

      var creator = faces_limit ? new JSROOT.GEO.PolygonsCreator : new JSROOT.GEO.GeometryCreator(radiusSegments*4);

      var nx1 = 1, ny1 = 0, nx2 = 1, ny2 = 0;

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
   JSROOT.GEO.createTorus = function( shape, faces_limit ) {
      var radius = shape.fR,
          innerTube = shape.fRmin,
          outerTube = shape.fRmax,
          arc = shape.fDphi - shape.fPhi1,
          rotation = shape.fPhi1,
          radialSegments = 30,
          tubularSegments = Math.max(8, Math.round(arc/JSROOT.GEO.GradPerSegm)),
          hasrmin = innerTube > 0, hascut = (arc !== 360);

      if (faces_limit < 0) return (hasrmin ? 4 : 2) * (radialSegments + 1) * tubularSegments;

      if (faces_limit > 0) {
         var fact = (hasrmin ? 4 : 2) * (radialSegments + 1) * tubularSegments / faces_limit;
         if (fact > 1.) {
            radialSegments = Math.round(radialSegments/Math.sqrt(fact));
            tubularSegments = Math.round(tubularSegments/Math.sqrt(fact));
         }
      }

      var geometry = new THREE.Geometry();
      var color = new THREE.Color();

      var outerTorus = new THREE.TorusGeometry( radius, outerTube, radialSegments, tubularSegments, arc*Math.PI/180);
      outerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ(rotation*Math.PI/180) );

      // add outer torus
      for (var n=0; n < outerTorus.vertices.length; ++n)
         geometry.vertices.push(outerTorus.vertices[n]);

      for (var n=0; n < outerTorus.faces.length; ++n) {
         var face = outerTorus.faces[n];
         geometry.faces.push(new THREE.Face3( face.a, face.b, face.c, null, color, 0 ) );
      }

      var shift = geometry.vertices.length;

      if (hasrmin) {
         var innerTorus = new THREE.TorusGeometry( radius, innerTube, radialSegments, tubularSegments, arc*Math.PI/180);
         innerTorus.applyMatrix( new THREE.Matrix4().makeRotationZ(rotation*Math.PI/180) );

         // add inner torus
         for (var n=0; n < innerTorus.vertices.length; ++n)
            geometry.vertices.push(innerTorus.vertices[n]);

         for (var n=0; n < innerTorus.faces.length; ++n) {
            var face = innerTorus.faces[n];
            geometry.faces.push(new THREE.Face3( shift+face.a, shift+face.c, shift+face.b, null, color, 0 ) );
         }
      } else
      if (hascut) {
         geometry.vertices.push(new THREE.Vector3(radius*Math.cos(rotation*Math.PI/180), radius*Math.sin(rotation*Math.PI/180),0));
         geometry.vertices.push(new THREE.Vector3(radius*Math.cos((rotation+arc)*Math.PI/180), radius*Math.sin((rotation+arc)*Math.PI/180),0));
      }

      if (arc !== 360) {
         // one cuted side
         for (var j=0;j<radialSegments;j++) {
            var i1 = j*(tubularSegments+1);
            var i2 = (j+1)*(tubularSegments+1);
            if (hasrmin) {
               geometry.faces.push( new THREE.Face3( i2, i1+shift, i1, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2, i2+shift, i1+shift,  null, color, 0 ));
            } else {
               geometry.faces.push( new THREE.Face3( shift, i1, i2, null, color, 0 ));
            }
         }

         // another cuted side
         for (var j=0;j<radialSegments;j++) {
            var i1 = (j+1)*(tubularSegments+1)-1;
            var i2 = (j+2)*(tubularSegments+1)-1;
            if (hasrmin) {
               geometry.faces.push( new THREE.Face3( i2, i1, i1+shift, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( i2, i1+shift, i2+shift, null, color, 0 ));
            } else {
               geometry.faces.push( new THREE.Face3( shift+1, i2, i1, null, color, 0 ));
            }
         }
      }

      geometry.computeFaceNormals();

      return geometry;
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
   JSROOT.GEO.createPolygon = function( shape ) {

      var thetaStart = shape.fPhi1, thetaLength = shape.fDphi;

      var radiusSegments = 60;
      if ( shape._typename == "TGeoPgon" ) {
         radiusSegments = shape.fNedges;
      } else {
         radiusSegments = Math.round(thetaLength/JSROOT.GEO.GradPerSegm);
         if (radiusSegments < 4) radiusSegments = 4;
      }

      var geometry = new THREE.Geometry();

      var color = new THREE.Color();

      var hasrmin = false;
      for (var layer=0; layer < shape.fNz; ++layer)
         if (shape.fRmin[layer] > 0) hasrmin = true;

      var phi0 = thetaStart*Math.PI/180, dphi = thetaLength/radiusSegments*Math.PI/180;

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments+1), _cos = new Float32Array(radiusSegments+1);
      for (var seg=0;seg<=radiusSegments;++seg) {
         _cos[seg] = Math.cos(phi0+seg*dphi);
         _sin[seg] = Math.sin(phi0+seg*dphi);
      }

      var indxs = [[],[]], pnts = null, edges = null; // remember indexes for each layer
      var layerVerticies = radiusSegments; // how many verticies in one layer

      if (thetaLength !== 360) {
         pnts = []; // coordinate of point on cut edge (x,z)
         edges = [];  // number of layer for that points
         layerVerticies+=1; // one need one more vertice
      }

      var a,b,c,d,e; // used for face swapping

      for (var side = 0; side < 2; ++side) {

         var rside = (side === 0) ? 'fRmax' : 'fRmin';
         var prev_indx = geometry.vertices.length;

         for (var layer=0; layer < shape.fNz; ++layer) {

            indxs[side][layer] = geometry.vertices.length;

            // first create points for the layer
            var layerz = shape.fZ[layer], rad = shape[rside][layer];

            if ((layer > 0) && (layer < shape.fNz-1)) {
               if (((shape.fZ[layer-1] === layerz) && (shape[rside][layer-1] === rad)) ||
                   ((shape[rside][layer+1] === rad) && (shape[rside][layer-1] === rad))) {

                  // same Z and R as before - ignore
                  // or same R before and after
                  indxs[side][layer] = indxs[side][layer-1];
                  // if (len) len[side][layer] = len[side][layer-1];
                  continue;
               }
            }

            if (rad <= 0.) rad = 0.000001;

            var curr_indx = geometry.vertices.length;

            // create vertices for the layer (if rmin===0, only central point is included
            if ((side===0) || hasrmin)
               for (var seg=0; seg < layerVerticies; ++seg)
                  geometry.vertices.push( new THREE.Vector3( rad*_cos[seg], rad*_sin[seg], layerz ));
            else
               geometry.vertices.push( new THREE.Vector3( 0, 0, layerz ));

            if (pnts !== null) {
               if (side === 0) {
                  pnts.push(new THREE.Vector2(rad, layerz));
                  edges.push(curr_indx);
               } else
               if (rad < shape.fRmax[layer]) {
                  pnts.unshift(new THREE.Vector2(rad, layerz));
                  edges.unshift(curr_indx);
               }
            }

            if ((layer>0) && ((side===0) || hasrmin))  // create faces
               for (var seg=0;seg < radiusSegments;++seg) {
                  var seg1 = (seg + 1) % layerVerticies;
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, (side === 0) ? (prev_indx + seg1) : (curr_indx + seg) , curr_indx + seg1, null, color, 0 ) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, (side === 0) ? (curr_indx + seg) : prev_indx + seg1, null, color, 0 ));
               }

            prev_indx = curr_indx;
         }
      }

      // add faces for top and bottom side
      for (var layer = 0; layer < shape.fNz; layer+= (shape.fNz-1)) {
         if (shape.fRmin[layer] >= shape.fRmax[layer]) continue;
         var inside = indxs[1][layer], outside = indxs[0][layer];
         for (var seg=0; seg < radiusSegments; ++seg) {
            var seg1 = (seg + 1) % layerVerticies;
            if (hasrmin) {
               geometry.faces.push( new THREE.Face3( outside + seg, (layer===0) ? (inside + seg) : (outside + seg1), inside + seg1, null, color, 0 ) );
               geometry.faces.push( new THREE.Face3( outside + seg, inside + seg1, (layer===0) ? (outside + seg1) : (inside + seg), null, color, 0 ));
            } else
            if (layer==0) {
               geometry.faces.push( new THREE.Face3( outside + seg, inside, outside + seg1, null, color, 0 ));
            } else {
               geometry.faces.push( new THREE.Face3( outside + seg1, inside, outside + seg, null, color, 0 ));
            }
         }
      }

      if (pnts!==null) {
         var faces = [];
         if (pnts.length === shape.fNz * 2) {
            // special case - all layers are there, create faces ourself
            for (var layer = shape.fNz-1; layer>0; --layer) {
               if (shape.fZ[layer] === shape.fZ[layer-1]) continue;
               var right = 2*shape.fNz - 1 - layer;
               faces.push([right, layer - 1, layer]);
               faces.push([right, right + 1, layer-1]);
            }

         } else {
            // let three.js calculate our faces
            faces = THREE.ShapeUtils.triangulateShape(pnts, []);
         }

         for (var i = 0; i < faces.length; ++i) {
            var f = faces[i];
            geometry.faces.push( new THREE.Face3( edges[f[0]], edges[f[1]], edges[f[2]], null, color, 0) );
         }
         for (var i = 0; i < faces.length; ++i) {
            var f = faces[i];
            geometry.faces.push( new THREE.Face3( edges[f[0]] + radiusSegments, edges[f[2]] + radiusSegments, edges[f[1]] + radiusSegments, null, color, 0) );
         }
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createPolygonBuffer = function( shape, faces_limit ) {
      var thetaStart = shape.fPhi1,
          thetaLength = shape.fDphi,
          radiusSegments = 60;

      if ( shape._typename == "TGeoPgon" )
         radiusSegments = shape.fNedges;
      else
         radiusSegments = Math.max(5, Math.round(thetaLength/JSROOT.GEO.GradPerSegm));

      var usage = new Int16Array(2*shape.fNz), numusedlayers = 0, hasrmin = false;

      for (var layer=0; layer < shape.fNz; ++layer)
         if (shape.fRmin[layer] > 0) hasrmin = true;

      // return very rought estimation, number of faces may be much less
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
                  pnts.push(new THREE.Vector2(rad, layerz));
               } else
               if (rad < shape.fRmax[layer]) {
                  pnts.unshift(new THREE.Vector2(rad, layerz));
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
            //console.log('trinagulate ' + shape.fName);
            cut_faces = THREE.ShapeUtils.triangulateShape(pnts, []);
            //console.log('trinagulate done ' + cut_faces.length);
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

      // var creator = new JSROOT.GEO.GeometryCreator(numfaces);

      // add sides
      for (var side = 0; side < 2; ++side) {
         var rside = (side === 0) ? 'fRmax' : 'fRmin',
             z1 = shape.fZ[0], r1 = shape[rside][0],
             d1 = 1 - side, d2 = side;

         for (var layer=0; layer < shape.fNz; ++layer) {

            if (usage[layer*2+side] === 0) continue;

            var z2 = shape.fZ[layer], r2 = shape[rside][layer],
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

         var rmin = shape.fRmin[layer], rmax = shape.fRmax[layer];

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
   JSROOT.GEO.createXtru = function( shape, faces_limit ) {

      if (faces_limit < 0) return 2 * shape.fNz * shape.fNvert;

      var geometry = new THREE.Geometry(),
          fcolor = new THREE.Color(),
          prev = 0, curr = 0;

      for (var layer = 0; layer < shape.fNz; ++layer) {
         var layerz = shape.fZ[layer], scale = shape.fScale[layer];

         prev = curr;
         curr = geometry.vertices.length;

         // add vertices
         for (var vert = 0; vert < shape.fNvert; ++vert)
            geometry.vertices.push( new THREE.Vector3( scale * shape.fX[vert], scale * shape.fY[vert], layerz ));

         if (layer>0)  // create faces for sides
            for (var vert = 0; vert < shape.fNvert; ++vert) {
               var vert1 = (vert + 1) % shape.fNvert;
               geometry.faces.push( new THREE.Face3( prev + vert, curr + vert, curr + vert1, null, fcolor, 0 ) );
               geometry.faces.push( new THREE.Face3( prev + vert, curr + vert1, prev + vert1, null, fcolor, 0 ));
            }
      }

      // now try to make shape - use standard THREE.js utils

      var pnts = [];
      for (var vert = 0; vert < shape.fNvert; ++vert)
         pnts.push( new THREE.Vector2(shape.fX[vert], shape.fY[vert]));

      var faces = THREE.ShapeUtils.triangulateShape(pnts , []);

      for (var i = 0; i < faces.length; ++i) {
         face = faces[ i ];
         geometry.faces.push( new THREE.Face3( face[1], face[0], face[2], null, fcolor, 0) );
         geometry.faces.push( new THREE.Face3( face[0] + curr, face[1] + curr, face[2] + curr, null, fcolor, 0) );
      }

      geometry.computeFaceNormals();

      return geometry;
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createXtruBuffer = function( shape, faces_limit ) {
      var nfaces = (shape.fNz-1) * shape.fNvert * 2;

      if (faces_limit < 0) return nfaces + shape.fNvert*3;

      // create points
      var pnts = [];
      for (var vert = 0; vert < shape.fNvert; ++vert)
         pnts.push(new THREE.Vector2(shape.fX[vert], shape.fY[vert]));

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
             z2 = shape.fZ[layer+1], scale2 = shape.fScale[layer+1];

         for (var vert1 = 0; vert1 < shape.fNvert; ++vert1) {
            var vert2 = (vert1+1) % shape.fNvert;
            creator.AddFace4(scale1 * shape.fX[vert1], scale1 * shape.fY[vert1], z1,
                             scale2 * shape.fX[vert1], scale2 * shape.fY[vert1], z2,
                             scale2 * shape.fX[vert2], scale2 * shape.fY[vert2], z2,
                             scale1 * shape.fX[vert2], scale1 * shape.fY[vert2], z1);
            creator.CalcNormal();
         }
      }

      for (layer = 0; layer <= shape.fNz-1; layer+=(shape.fNz-1)) {
         var z = shape.fZ[layer], scale = shape.fScale[layer];

         for (var n=0;n<faces.length;++n) {
            var face = faces[n],
                pnt1 = pnts[face[0]],
                pnt2 = pnts[face[(layer===0) ? 2 : 1]],
                pnt3 = pnts[face[(layer===0) ? 1 : 2]];

            creator.AddFace3(scale * pnt1.x, scale * pnt1.y, z,
                             scale * pnt2.x, scale * pnt2.y, z,
                             scale * pnt3.x, scale * pnt3.y, z);
            creator.SetNormal(0,0,layer===0 ? -1 : 1);
         }
      }

      return creator.Create();
   }


   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.createParaboloid = function( shape, faces_limit ) {

      var radiusSegments = Math.round(360/6), heightSegments = 30;

      if (faces_limit > 0) {
         var fact = 2 * (radiusSegments+1) * (heightSegments+1) / faces_limit;
         if (fact > 1.) {
            radiusSegments = Math.round(radiusSegments/Math.sqrt(fact));
            heightSegments = Math.round(heightSegments/Math.sqrt(fact));
         }
      }

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments), _cos = new Float32Array(radiusSegments);
      for (var seg=0;seg<radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }

      var geometry = new THREE.Geometry();
      var fcolor = new THREE.Color();

      var zmin = -shape.fDZ, zmax = shape.fDZ, rmin = shape.fRlo, rmax = shape.fRhi;

      // if no radius at -z, find intersection
      if (shape.fA >= 0) {
         if (shape.fB > zmin) zmin = shape.fB;
      } else {
         if (shape.fB < zmax) zmax = shape.fB;
      }

      var ttmin = Math.atan2(zmin, rmin), ttmax = Math.atan2(zmax, rmax);

      var prev_indx = 0, prev_radius = 0;

      for (var layer = 0; layer <= heightSegments + 1; ++layer) {
         var layerz = zmax, radius = 0;

         if ((layer === heightSegments + 1) && (prev_radius === 0)) break;

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

         var curr_indx = geometry.vertices.length;

         if (radius === 0) {
            geometry.vertices.push( new THREE.Vector3( 0, 0, layerz ));
         } else {
            for (var seg=0; seg<radiusSegments; ++seg)
               geometry.vertices.push( new THREE.Vector3( radius*_cos[seg], radius*_sin[seg], layerz));
         }

         // add faces of next layer
         if (layer>0) {
            for (var seg=0; seg<radiusSegments; ++seg) {
               var seg1 = (seg+1) % radiusSegments;
               if (prev_radius === 0) {
                  geometry.faces.push( new THREE.Face3( prev_indx, curr_indx + seg1, curr_indx + seg, null, fcolor, 0) );
               } else
               if (radius == 0) {
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, prev_indx + seg1, curr_indx, null, fcolor, 0) );
               } else {
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, curr_indx + seg, null, fcolor, 0) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, prev_indx + seg1, curr_indx + seg1,  null, fcolor, 0) );
               }
            }
         }

         prev_radius = radius;
         prev_indx = curr_indx;
      }

      geometry.computeFaceNormals();

      return geometry;
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

            // use analitic normal values when open/closing parabaloid around 0
            // cutted faces (top or bottom) set with simple normal
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
   JSROOT.GEO.createHype = function( shape, faces_limit ) {

      if ((shape.fTin===0) && (shape.fTout===0))
         return JSROOT.GEO.createTubeBuffer(shape, faces_limit);

      var radiusSegments = Math.max(4, Math.round(360/JSROOT.GEO.GradPerSegm)),
          heightSegments = 30;

      if (faces_limit < 0) return ((shape.fRmin <= 0) ? 2 : 4) * (radiusSegments+1) * (heightSegments+2);

      if (faces_limit > 0) {
         var fact = ((shape.fRmin <= 0) ? 2 : 4) * (radiusSegments+1) * (heightSegments+2) / faces_limit;
         if (fact > 1.) {
            radiusSegments = Math.round(radiusSegments/Math.sqrt(fact));
            heightSegments = Math.round(heightSegments/Math.sqrt(fact));
         }
      }

      // calculate all sin/cos tables in advance
      var _sin = new Float32Array(radiusSegments), _cos = new Float32Array(radiusSegments);
      for (var seg=0;seg<radiusSegments;++seg) {
         _cos[seg] = Math.cos(seg/radiusSegments*2*Math.PI);
         _sin[seg] = Math.sin(seg/radiusSegments*2*Math.PI);
      }


      var geometry = new THREE.Geometry();
      var fcolor = new THREE.Color();

      var indexes = [[],[]];

      // in-out side
      for (var side=0;side<2;++side) {

         // add only points, no faces
         if ((side===0) && (shape.fRmin <= 0)) {
            indexes[side][0] = geometry.vertices.length;
            geometry.vertices.push( new THREE.Vector3( 0, 0, -shape.fDz ) );
            indexes[side][heightSegments] = geometry.vertices.length;
            geometry.vertices.push( new THREE.Vector3( 0, 0, shape.fDz ) );
            continue;
         }

         var prev_indx = 0;
         var r0 = (side===0) ? shape.fRmin : shape.fRmax;
         var tsq = (side===0) ? shape.fTinsq : shape.fToutsq;

         // vertical layers
         for (var layer=0;layer<=heightSegments;++layer) {
            var layerz = -shape.fDz + layer/heightSegments*2*shape.fDz;

            var radius = Math.sqrt(r0*r0+tsq*layerz*layerz);
            var curr_indx = geometry.vertices.length;

            indexes[side][layer] = curr_indx;

            for (var seg=0; seg<radiusSegments; ++seg)
               geometry.vertices.push( new THREE.Vector3( radius*_cos[seg], radius*_sin[seg], layerz));

            // add faces of next layer
            if (layer>0) {
               for (var seg=0; seg<radiusSegments; ++seg) {
                  var seg1 = (seg+1) % radiusSegments;
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, (side===0) ? (curr_indx + seg) : (prev_indx + seg1), curr_indx + seg1, null, fcolor, 0) );
                  geometry.faces.push( new THREE.Face3( prev_indx + seg, curr_indx + seg1, (side===0) ? (prev_indx + seg1) : (curr_indx + seg), null, fcolor, 0) );
               }
            }

            prev_indx = curr_indx;
         }
      }

      // add caps
      for(var layer=0; layer<=heightSegments; layer+=heightSegments) {
         var inside = indexes[0][layer], outside = indexes[1][layer];
         for (var seg=0; seg<radiusSegments; ++seg) {
            var seg1 = (seg+1) % radiusSegments;
            if (shape.fRmin <= 0) {
               geometry.faces.push( new THREE.Face3( inside, outside + (layer===0 ? seg1 : seg), outside + (layer===0 ? seg : seg1), null, fcolor, 0) );
            } else {
               geometry.faces.push( new THREE.Face3( inside + seg, (layer===0) ? (inside + seg1) : (outside + seg), outside + seg1, null, fcolor, 0) );
               geometry.faces.push( new THREE.Face3( inside + seg, outside + seg1, (layer===0) ? (outside + seg) : (inside + seg1), null, fcolor, 0) );
            }
         }
      }

      geometry.computeFaceNormals();

      return geometry;
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

         if (node.fTrans!==null) {
            matrix.set(node.fTrans[0],  node.fTrans[4],  node.fTrans[8],  0,
                       node.fTrans[1],  node.fTrans[5],  node.fTrans[9],  0,
                       node.fTrans[2],  node.fTrans[6],  node.fTrans[10], 0,
                                    0,               0,               0, 1);
            // second - set position with proper sign
            matrix.setPosition({ x: node.fTrans[12], y: node.fTrans[13], z: node.fTrans[14] });
         }
      } else
      if (('fMatrix' in node) && (node.fMatrix !== null))
         matrix = JSROOT.GEO.createMatrix(node.fMatrix);
      else
      if ((node._typename == "TGeoNodeOffset") && (node.fFinder !== null)) {
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

              switch (node.fFinder._typename.charAt[node.fFinder._typename.length-1]) {
                 case 'X': matrix.setPosition(new THREE.Vector3(_shift, 0, 0)); break;
                 case 'Y': matrix.setPosition(new THREE.Vector3(0, _shift, 0)); break;
                 case 'Z': matrix.setPosition(new THREE.Vector3(0, 0, _shift)); break;
              }
              break;

           case 'TGeoPatternCylPhi':
              var phi = (Math.PI/180)*(node.fFinder.fStart+(node.fIndex+0.5)*node.fFinder.fStep);
              var _cos = Math.cos(phi), _sin = Math.sin(phi);

              matrix = new THREE.Matrix4();

              matrix.set(_cos, -_sin, 0,  0,
                         _sin,  _cos, 0,  0,
                            0,     0, 1,  0,
                            0,     0, 0,  1);
              break;

           case 'TGeoPatternCylR':
               // seems to be, require no transformation
               matrix = new THREE.Matrix4();
               break;

           case 'TGeoPatternTrapZ':
              var dz = node.fFinder.fStart + (node.fIndex+0.5)*node.fFinder.fStep;
              matrix = new THREE.Matrix4();
              matrix.setPosition(new THREE.Vector3(node.fFinder.fTxz*dz, node.fFinder.fTyz*dz, dz)); break;
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

      if (faces_limit < 0)
         return JSROOT.GEO.createGeometry(shape.fNode.fLeft, -1) +
                JSROOT.GEO.createGeometry(shape.fNode.fRight, -1);

      var geom1, geom2, bsp1, bsp2, return_bsp = false,
          matrix1 = JSROOT.GEO.createMatrix(shape.fNode.fLeftMat),
          matrix2 = JSROOT.GEO.createMatrix(shape.fNode.fRightMat);

      // seems to be, IE has smaller stack for functions calls and ThreeCSG fails with large shapes
      if (faces_limit === 0) faces_limit = (JSROOT.browser && JSROOT.browser.isIE) ? 7000 : 10000;
                        else return_bsp = true;

      if (matrix1 && (matrix1.determinant() < -0.9))
         JSROOT.GEO.warn('Axis reflection in composite shape - not supported');

      if (matrix2 && (matrix2.determinant() < -0.9))
         JSROOT.GEO.warn('Axis reflections in composite shape - not supported');

      geom1 = JSROOT.GEO.createGeometry(shape.fNode.fLeft, faces_limit/2);

      geom2 = JSROOT.GEO.createGeometry(shape.fNode.fRight, faces_limit/2);

      if (geom1 instanceof THREE.Geometry) geom1.computeVertexNormals();
      bsp1 = new ThreeBSP.Geometry(geom1, matrix1, JSROOT.GEO.CompressComp ? 0 : undefined);

      if (geom2 instanceof THREE.Geometry) geom2.computeVertexNormals();
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


   /**
    * Creates geometry model for the provided shape
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
         return cnt>5; // only if 6 edges and more are seen, we think that box is fully visisble
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

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.numGeometryFaces = function(geom) {
      if (!geom) return 0;

      if (geom instanceof ThreeBSP.Geometry)
         return geom.tree.numPolygons();

      if (geom.type == 'BufferGeometry') {
         var attr = geom.getAttribute('position');
         return attr ? attr.count / 3 : 0;
      }

      // special array of polygons
      if (geom && geom.polygons) return geom.polygons.length;

      return geom.faces.length;
   }

   /** @memberOf JSROOT.GEO */
   JSROOT.GEO.numGeometryVertices = function(geom) {
      if (!geom) return 0;

      if (geom instanceof ThreeBSP.Geometry)
         return geom.tree.numPolygons() * 3;

      if (geom.type == 'BufferGeometry') {
         var attr = geom.getAttribute('position');
         return attr ? attr.count : 0;
      }

      if (geom && geom.polygons) return geom.polygons.length * 4;

      return geom.vertices.length;
   }

   // ====================================================================

   // class for working with cloned nodes

   JSROOT.GEO.ClonedNodes = function(obj, clones) {
      this.toplevel = true; // indicate if object creates top-level structure with Nodes and Volumes folder
      this.name_prefix = ""; // name prefix used for nodes names
      this.maxdepth = 1; // maximal hierarchy depth, required for transparancy

      if (obj) {
         if (obj.$geoh) this.toplevel = false;
         this.CreateClones(obj);
      } else
      if (clones) this.nodes = clones;
   }

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

      if (this.nodes)
         for (var n=0;n<this.nodes.length;++n)
            delete this.nodes[n].chlds;

      delete this.nodes;
      delete this.origin;

      delete this.sortmap;

   }

   JSROOT.GEO.ClonedNodes.prototype.CreateClones = function(obj, sublevel, kind) {
       if (!sublevel) {
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

       if (chlds !== null)
          for (var i = 0; i < chlds.length; ++i)
             this.CreateClones(chlds[i], sublevel+1, kind);

       if (sublevel > 1) return;

       this.nodes = [];

       var sortarr = [];

       // first create nodes objects
       for (var n=0; n<this.origin.length; ++n) {
          var obj = this.origin[n];
          var node = { id: n, kind: kind, vol: 0, nfaces: 0, numvischld: 1, idshift: 0 };
          this.nodes.push(node);
          sortarr.push(node); // array use to produce sortmap
       }

       // than fill childrens lists
       for (var n=0;n<this.origin.length;++n) {
          var obj = this.origin[n], clone = this.nodes[n];

          var chlds = null, shape = null;

          if (kind===1) {
             shape = obj.fShape;
             if (obj.fElements) chlds = obj.fElements.arr;
          } else
          if (obj.fVolume) {
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

             // if (clone.nfaces < -10) console.log('Problem  with node ' + obj.fName + ':' + obj.fMother.fName);
          }

          if (!chlds) continue;

          // in cloned object childs is only list of ids
          clone.chlds = new Int32Array(chlds.length);
          for (var k=0;k<chlds.length;++k)
             clone.chlds[k] = chlds[k]._refid;
       }

       // remove _refid identifiers from original objects
       for (var n=0;n<this.origin.length;++n)
          delete this.origin[n]._refid;

       // do sorting once
       sortarr.sort(function(a,b) { return b.vol - a.vol; });

       // rememember sort map
       this.sortmap = new Int32Array(this.nodes.length);
       for (var n=0;n<this.nodes.length;++n)
          this.sortmap[n] = sortarr[n].id;
   }


   JSROOT.GEO.ClonedNodes.prototype.MarkVisisble = function(on_screen, copy_bits, cloning) {
      if (!this.nodes) return 0;

      var res = 0, simple_copy = cloning && (cloning.length === this.nodes.length);

      if (!simple_copy && !this.origin) return 0;

      for (var n=0;n<this.nodes.length;++n) {
         var clone = this.nodes[n];

         clone.vis = false;
         clone.numvischld = 1; // reset vis counter, will be filled with next scan
         clone.idshift = 0;
         delete clone.depth;

         if (simple_copy) {
            clone.vis = cloning[n].vis;
            if (cloning[n].depth !== undefined) clone.depth = cloning[n].depth;
            if (clone.vis) res++;
            continue;
         }

         var obj = this.origin[n];

         if (clone.kind === 0) {
            if (obj.fVolume) {
               if (on_screen) {
                  clone.vis = JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisOnScreen);
                  if (copy_bits) {
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisNone, false);
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisThis, clone.vis);
                     JSROOT.GEO.SetBit(obj.fVolume, JSROOT.GEO.BITS.kVisDaughters, true);
                  }
               } else {
                  clone.vis = !JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisNone) &&
                               JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisThis);
                  if (!JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisDaughters))
                     clone.depth = JSROOT.GEO.TestBit(obj.fVolume, JSROOT.GEO.BITS.kVisOneLevel) ? 1 : 0;
               }
            }
         } else {
            clone.vis = obj.fRnrSelf;

            // when the only node is selected, draw it
            if ((n===0) && (this.nodes.length===1)) clone.vis = true;
         }

         // shape with zero volume or without faces will not be observed
         if ((clone.vol <= 0) || (clone.nfaces <= 0)) clone.vis = false;

         if (clone.vis) res++;
      }

      return res;
   }

   JSROOT.GEO.ClonedNodes.prototype.GetVisibleFlags = function() {
      // function extract only visibility flags, used to transfer them to the worker
      var res = [];
      for (var n=0;n<this.nodes.length;++n) {
         var elem = { vis: this.nodes[n].vis };
         if ('depth' in this.nodes[n]) elem.depth = this.nodes[n].depth;
         res.push(elem);
      }
      return res;
   }

   JSROOT.GEO.ClonedNodes.prototype.ScanVisible = function(arg, vislvl) {
      // Scan visible nodes in hierarchy, starting from nodeid
      // Each entry in hierarchy get its unique id, which is not changed with visibility flags

      if (!this.nodes) return 0;

      if (vislvl === undefined) {
         vislvl = 99999;
         if (!arg) arg = {};
         arg.stack = new Int32Array(100); // current stack
         arg.nodeid = 0;
         arg.counter = 0; // sequence ID of the node, used to identify it later
         arg.last = 0;
         arg.CopyStack = function(factor) {
            var entry = { nodeid: this.nodeid, seqid: this.counter, stack: new Int32Array(this.last) };
            if (factor) entry.factor = factor; // factor used to indicate importance of entry, will be build as first
            for (var n=0;n<this.last;++n) entry.stack[n] = this.stack[n+1];
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

      if (node.vis && (vislvl>=0)) {
         if (!arg.func || arg.func(node)) res++;
      }

      arg.counter++;

      if ((node.depth !== undefined) && (vislvl > node.depth)) vislvl = node.depth;

      if (arg.last > arg.stack.length - 2)
         throw 'stack capacity is not enough ' + arg.stack.length;

      if (node.chlds && (node.numvischld > 0)) {
         var currid = arg.counter, numvischld = 0;
         arg.last++;
         for (var i = 0; i < node.chlds.length; ++i) {
            arg.nodeid = node.chlds[i];
            arg.stack[arg.last] = i; // in the stack one store index of child, it is path in the hierarchy
            numvischld += this.ScanVisible(arg, vislvl-1);
         }
         arg.last--;
         res += numvischld;
         if (numvischld === 0) {
            node.numvischld = 0;
            node.idshift = arg.counter - currid;
         }
      } else {
         arg.counter += node.idshift;
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

   JSROOT.GEO.ClonedNodes.prototype.ResolveStack = function(stack, withmatrix) {

      var res = { id: 0, obj: null, node: this.nodes[0], name: this.name_prefix };

      // if (!this.toplevel || (this.nodes.length === 1) || (res.node.kind === 1)) res.name = "";

      if (withmatrix) {
         res.matrix = new THREE.Matrix4();
         if (res.node.matrix) res.matrix.fromArray(res.node.matrix);
      }

      if (this.origin) res.obj = this.origin[0];

      if (stack)
         for(var lvl=0;lvl<stack.length;++lvl) {
            res.id = res.node.chlds[stack[lvl]];
            res.node = this.nodes[res.id];
            if (this.origin) {
               res.obj = this.origin[res.id];

               if (res.obj.fName!=="") {
                  if (res.name.length>0) res.name += "/";
                  res.name += res.obj.fName;
               }
            }

            if (withmatrix && res.node.matrix)
               res.matrix.multiply(new THREE.Matrix4().fromArray(res.node.matrix));
         }

      return res;
   }

   JSROOT.GEO.ClonedNodes.prototype.FindStackByName = function(fullname) {
      if (!this.origin) return null;

      var names = fullname.split('/'),
          currid = 0, stack = [],
          top = this.origin[0];

      if (!top || (top.fName!==names[0])) return null;

      for (var n=1;n<names.length;++n) {
         var node = this.nodes[currid];
         if (!node.chlds) return null;

         for (var k=0;k<node.chlds.length;++k) {
            var chldid = node.chlds[k];
            var obj = this.origin[chldid];
            if (obj && (obj.fName === names[n])) { stack.push(k); currid = chldid; break; }
         }

         // no new entry - not found stack
         if (stack.length === n - 1) return null;
      }

      return stack;
   }

   JSROOT.GEO.ClonedNodes.prototype.CreateObject3D = function(stack, toplevel, options) {
      // create hierarchy of Object3D for given stack entry
      // such hierarchy repeats hierarchy of TGeoNodes and set matrix for the objects drawing

      var node = this.nodes[0], three_prnt = toplevel,
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
            continue;
         }

         if (!force) return null;

         obj3d = new THREE.Object3D();

         if (node.matrix) {
            obj3d.matrix.fromArray(node.matrix);
            obj3d.matrix.decompose( obj3d.position, obj3d.quaternion, obj3d.scale );
         }

         // this.accountNodes(obj3d);
         obj3d.nchld = nchld; // mark index to find it again later

         // add the mesh to the scene
         three_prnt.add(obj3d);

         // this is only for debugging - test invertion of whole geometry
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

         while (mesh && (mesh !== toplevel)) {
            three_prnt = mesh.parent;
            three_prnt.remove(mesh);
            mesh = (three_prnt.children.length == 0) ? three_prnt : null;
         }

         return null;
      }

      return three_prnt;
   }

   JSROOT.GEO.ClonedNodes.prototype.GetVolumeBoundary = function(viscnt, facelimit, nodeslimit) {
      if (!this.sortmap) {
         console.error('sorting map not exisits');
         return { min: 0, max: 1 };
      }

      var maxNode, currNode, cnt=0, facecnt = 0;

      for (var n=0; (n<this.sortmap.length) && (cnt < nodeslimit) && (facecnt < facelimit);++n) {
         var id = this.sortmap[n];
         if (viscnt[id] === 0) continue;
         currNode = this.nodes[id];
         if (!maxNode) maxNode = currNode;
         cnt += viscnt[id];
         facecnt += viscnt[id] * currNode.nfaces;
      }

      if (!currNode) {
         console.error('no volumes selected');
         return { min: 0, max: 1 };
      }

      // console.log('Volume boundary ' + currNode.vol + '  cnt ' + cnt + '  faces ' + facecnt);

      return { min: currNode.vol, max: maxNode.vol };
   }


   JSROOT.GEO.ClonedNodes.prototype.CollectVisibles = function(maxnumfaces, frustum, maxnumnodes) {
      // function collects visible nodes, using maxlimit
      // one can use map to define cut based on the volume or serious of cuts

      if (!maxnumnodes) maxnumnodes = maxnumfaces/100;

      var arg = {
         facecnt: 0,
         viscnt: new Int32Array(this.nodes.length), // counter for each node
         // nodes: this.nodes,
         func: function(node) {
            this.facecnt += node.nfaces;
            this.viscnt[node.id]++;
            return true;
         }
      };

      for (var n=0;n<arg.viscnt.length;++n) arg.viscnt[n] = 0;

      var total = this.ScanVisible(arg), minVol = 0, maxVol = 0, camVol = -1, camFact = 10;

      // console.log('Total visible nodes ' + total + ' numfaces ' + arg.facecnt);

      if (arg.facecnt > maxnumfaces) {

         var bignumfaces = maxnumfaces * (frustum ? 0.8 : 1.0),
             bignumnodes = maxnumnodes * (frustum ? 0.8 : 1.0);

         // define minimal volume, which always to shown
         var boundary = this.GetVolumeBoundary(arg.viscnt, bignumfaces, bignumnodes);

         minVol = boundary.min;
         maxVol = boundary.max;

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
         if (node.vol > minVol) {
            this.items.push(this.CopyStack());
         } else
         if ((camVol >= 0) && (node.vol > camVol))
            if (this.frustum.CheckShape(this.getmatrix(), node)) {
               this.items.push(this.CopyStack(camFact));
            }
         return true;
      }

      this.ScanVisible(arg);

      return { lst: arg.items, complete: minVol === 0 };
   }

   JSROOT.GEO.ClonedNodes.prototype.MergeVisibles = function(current, prev) {
      // merge list of drawn objects
      // in current list we should mark if object already exists
      // from previous list we should collect objects which are not there

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


   JSROOT.GEO.ClonedNodes.prototype.CollectShapes = function(lst) {
      // based on list of visible nodes, collect all uniques shapes which should be build

      var shapes = [];

      for (var i=0;i<lst.length;++i) {
         var entry = lst[i];
         var shape = this.GetNodeShape(entry.nodeid);

         if (!shape) continue; // strange, but avoid missleading

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
         entry.shapeid = entry.shape.id; // keep only id for the entry
         delete entry.shape; // remove direct references
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
            if (item.geom === undefined) {
               item.geom = JSROOT.GEO.createGeometry(item.shape);
               if (item.geom) created++; // indicate that at least one shape was created
            }
            item.nfaces = JSROOT.GEO.numGeometryFaces(item.geom);
            item.ready = true;
         }

         res.shapes++;
         if (!item.used) res.notusedshapes++;
         res.faces += item.nfaces*item.refcnt;

         if (res.faces >= limit) {
            res.done = true;
         } else
         if ((created > 0.01*lst.length) && (timelimit!==undefined)) {
            var tm2 = new Date().getTime();
            if (tm2-tm1 > timelimit) return res;
         }
      }

      res.done = true;

      return res;
   }

   return JSROOT;

}));

