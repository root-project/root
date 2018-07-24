/** @file EveElements.js */

(function( factory ) {
   if ( typeof define === "function" && define.amd ) {
      define( [ 'JSRootCore', 'threejs' ], factory );
   } else
      if (typeof exports === 'object' && typeof module !== 'undefined') {
	 factory(require("./JSRootCore.js"), require("./three.min.js"));
      } else {

	 if (typeof JSROOT == 'undefined')
            throw new Error('JSROOT is not defined', 'EveElements.js');

	 if (typeof JSROOT.EVE == 'undefined')
            throw new Error('JSROOT.EVE is not defined', 'EveElements.js');

	 if (typeof THREE == 'undefined')
            throw new Error('THREE is not defined', 'EveElements.js');

	 factory(JSROOT, THREE);
      }
} (function( JSROOT, THREE ) {

   "use strict";

   var GL = { POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4 };

   // var flat_material = new THREE.ShaderMaterial( {
   //     uniforms: { time: { value: 1.0 }, resolution: { value: new THREE.Vector2() } },
   //     vertexShader: document.getElementById( 'vertexShader' ).textContent,
   //     fragmentShader: document.getElementById( 'fragmentShader' ).textContent
   // } );

   function EveElements()
   {

   }

   EveElements.prototype.makeHit = function(hit, rnrData) {
      // console.log("drawiHt ", hit, "this type ", this.viewType);
      // console.log("marker size ", hit.fMarkerSize)
      var hit_size = 8*rnrData.fMarkerSize,
          size = rnrData.vtxBuff.length/3,
          pnts = new JSROOT.Painter.PointsCreator(size, true, hit_size);

      for (var i=0;i<size;i++) {
         pnts.AddPoint(rnrData.vtxBuff[i*3],rnrData.vtxBuff[i*3+1],rnrData.vtxBuff[i*3+2]);
         // console.log("add vertex ", rnrData.vtxBuff[i*3],rnrData.vtxBuff[i*3+1],rnrData.vtxBuff[i*3+2]);
      }
      var mesh = pnts.CreatePoints(JSROOT.Painter.root_colors[rnrData.fMarkerColor] );

      mesh.highlightMarkerSize = hit_size*3;
      mesh.normalMarkerSize = hit_size;
      

      mesh.geo_name = hit.fName;
      mesh.geo_object = hit;

      mesh.visible = hit.fRnrSelf;
      mesh.material.sizeAttenuation = false;
      mesh.material.size = hit.fMarkerSize;
      return mesh;
   }

   EveElements.prototype.makeTrack = function(track, rnrData) {
      var N = rnrData.vtxBuff.length/3;
      var track_width = track.fLineWidth || 1,
          track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

      var buf = new Float32Array(N*3*2), pos = 0;
      for (var k=0;k<(N-1);++k) {
         buf[pos]   = rnrData.vtxBuff[k*3];
         buf[pos+1] = rnrData.vtxBuff[k*3+1];
         buf[pos+2] = rnrData.vtxBuff[k*3+2];

         var breakTrack = 0;
         if (rnrData.idxBuff) {
            for (var b = 0; b < rnrData.idxBuff.length; b++)
            {
               if ( (k+1) == rnrData.idxBuff[b]) {
                  breakTrack = 1;
               }
            }
         }

         if (breakTrack) {
            buf[pos+3] = rnrData.vtxBuff[k*3];
            buf[pos+4] = rnrData.vtxBuff[k*3+1];
            buf[pos+5] = rnrData.vtxBuff[k*3+2];
         }
         else {
            buf[pos+3] = rnrData.vtxBuff[k*3+3];
            buf[pos+4] = rnrData.vtxBuff[k*3+4];
            buf[pos+5] = rnrData.vtxBuff[k*3+5];
         }

         // console.log(" vertex ", buf[pos],buf[pos+1], buf[pos+2],buf[pos+3], buf[pos+4],  buf[pos+5]);
         pos+=6;
      }
      var lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width });
      var geom = new THREE.BufferGeometry();
      geom.addAttribute( 'position', new THREE.BufferAttribute( buf, 3 )  );
      var line = new THREE.LineSegments(geom, lineMaterial);

      line.geo_name = track.fName;
      line.geo_object = track;
      line.visible = track.fRnrSelf;
      // console.log("make track ", track, line.visible);
      return line;
   }

   EveElements.prototype.makeJet = function(jet, rnrData)
   {
      // console.log("make jet ", jet);
      var jet_ro = new THREE.Object3D();
      //var geo = new EveJetConeGeometry(jet.geoBuff);
      var pos_ba = new THREE.BufferAttribute( rnrData.vtxBuff, 3 );
      var N      = rnrData.vtxBuff.length / 3;

      var geo_body = new THREE.BufferGeometry();
      geo_body.addAttribute('position', pos_ba);
      {
         var idcs = [];
         idcs.push( 0 );  idcs.push( N - 1 );  idcs.push( 1 );
         for (var i = 1; i < N - 1; ++i)
         {
            idcs.push( 0 );  idcs.push( i );  idcs.push( i + 1 );
         }
         geo_body.setIndex( idcs );
         geo_body.computeVertexNormals();
      }
      var geo_rim = new THREE.BufferGeometry();
      geo_rim.addAttribute('position', pos_ba);
      {
         var idcs = [];
         for (var i = 1; i < N; ++i)
         {
            idcs.push( i );
         }
         geo_rim.setIndex( idcs );
      }
      var geo_rays = new THREE.BufferGeometry();
      geo_rays.addAttribute('position', pos_ba);
      {
         var idcs = [];
         for (var i = 1; i < N; i += 4)
         {
            idcs.push( 0 ); idcs.push( i );
         }
         geo_rays.setIndex( idcs );
      }
      var mcol = JSROOT.Painter.root_colors[jet.fMainColor];
      var lcol = JSROOT.Painter.root_colors[jet.fLineColor];
      jet_ro.add( new THREE.Mesh        (geo_body, new THREE.MeshPhongMaterial({ depthWrite: false, color: mcol, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.LineLoop    (geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2,   color: lcol, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.LineSegments(geo_rays, new THREE.LineBasicMaterial({ linewidth: 0.5, color: lcol, transparent: true, opacity: 0.5 })) );
      jet_ro.geo_name = jet.fName;
      jet_ro.geo_object = jet;
      jet_ro.visible = jet.fRnrSelf;

      return jet_ro;
   }

   EveElements.prototype.makeJetProjected = function(jet, rnrData)
   {
      // JetProjected has 3 or 4 points. 0-th is apex, others are rim.
      // Fourth point is only present in RhoZ when jet hits barrel/endcap transition.

      // console.log("makeJetProjected ", jet);

      var jet_ro = new THREE.Object3D();
      var pos_ba = new THREE.BufferAttribute( rnrData.vtxBuff, 3 );
      var N      = rnrData.vtxBuff.length / 3;

      var geo_body = new THREE.BufferGeometry();
      geo_body.addAttribute('position', pos_ba);
      {
         var idcs = [];
         idcs.push( 0 );  idcs.push( 2 );  idcs.push( 1 );
         if (N > 3) {
            idcs.push( 0 );  idcs.push( 3 );  idcs.push( 2 );
         }
         geo_body.setIndex( idcs );
         geo_body.computeVertexNormals();
      }
      var geo_rim = new THREE.BufferGeometry();
      geo_rim.addAttribute('position', pos_ba);
      {
         var idcs = [];
         for (var i = 1; i < N; ++i) { idcs.push( i ); }
         geo_rim.setIndex( idcs );
      }
      var geo_rays = new THREE.BufferGeometry();
      geo_rays.addAttribute('position', pos_ba);
      {
         var idcs = [];
         idcs.push( 0 ); idcs.push( 1 );
         idcs.push( 0 ); idcs.push( N - 1 );
         geo_rays.setIndex( idcs );
      }
      var fcol = JSROOT.Painter.root_colors[jet.fFillColor];
      var lcol = JSROOT.Painter.root_colors[jet.fLineColor];
      // Process transparency !!!
      // console.log("cols", fcol, lcol);
      jet_ro.add( new THREE.Mesh        (geo_body, new THREE.MeshBasicMaterial({ depthWrite: false, color: fcol, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.Line        (geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2.0, color: lcol, transparent: true, opacity: 0.5 })) );
      jet_ro.add( new THREE.LineSegments(geo_rays, new THREE.LineBasicMaterial({ linewidth: 1.0, color: lcol, transparent: true, opacity: 0.5 })) );
      jet_ro.geo_name   = jet.fName;
      jet_ro.geo_object = jet;
      jet_ro.visible    = jet.fRnrSelf;

      return jet_ro;
   }

   EveElements.prototype.makeEveGeoShape = function(egs, rnr_data)
   {
      console.log("makeEveGeoShape ", egs);

      var egs_ro = new THREE.Object3D();
      var pos_ba = new THREE.BufferAttribute( rnr_data.vtxBuff, 3 );
      var idx_ba = new THREE.BufferAttribute( rnr_data.idxBuff, 1 );

      var ib_len = rnr_data.idxBuff.length;

      console.log("ib_len", ib_len, rnr_data.idxBuff[0], rnr_data.idxBuff[1], 3 * rnr_data.idxBuff[1]);

      if (rnr_data.idxBuff[0] != GL.TRIANGLES)   throw "Expect triangles first.";
      if (2 + 3 * rnr_data.idxBuff[1] != ib_len) throw "Expect single list of triangles in index buffer.";

      var body =  new THREE.BufferGeometry();
      body.addAttribute('position', pos_ba);
      body.setIndex(idx_ba);
      body.setDrawRange(2, 3 * rnr_data.idxBuff[1]);

      // XXXX Fix this. It seems we could have flat shading with usage of simple shaders.
      // XXXX Also, we could do edge detect on the server for outlines.
      // XXXX a) 3d objects - angle between triangles >= 85 degrees (or something);
      // XXXX b) 2d objects - segment only has one triangle.
      // XXXX Somewhat orthogonal - when we do tesselation, conversion from quads to
      // XXXX triangles is trivial, we could do it before invoking the big guns (if they are even needed).
      // XXXX Oh, and once triangulated, we really don't need to store 3 as number of verts in a poly each time.
      // XXXX Or do we? We might need it for projection stuff.
      body.computeVertexNormals();

      var fcol = JSROOT.Painter.root_colors[egs.fFillColor];
      // var lcol = JSROOT.Painter.root_colors[egs.fLineColor];
      egs_ro.add( new THREE.Mesh(body, new THREE.MeshPhongMaterial({// side: THREE.DoubleSide,
         depthWrite:  false,
         color:fcol, transparent: true, opacity: 0.2
      })) );

      // egs_ro.add( new THREE.LineSegments(body, new THREE.LineBasicMaterial({color:lcol })) );

      return egs_ro;
   }

   EveElements.prototype.makePolygonSetProjected = function(psp, rnr_data)
   {
      console.log("makePolygonSetProjected ", psp);

      var psp_ro = new THREE.Object3D();
      var pos_ba = new THREE.BufferAttribute( rnr_data.vtxBuff, 3 );
      var idx_ba = new THREE.BufferAttribute( rnr_data.idxBuff, 1 );

      var ib_len = rnr_data.idxBuff.length;

      var fcol = JSROOT.Painter.root_colors[psp.fMainColor];
      var line_mat = new THREE.LineBasicMaterial({color:fcol });

      for (var ib_pos = 0; ib_pos < ib_len; )
      {
         if (rnr_data.idxBuff[ib_pos] == GL.TRIANGLES)
         {
            var body = new THREE.BufferGeometry();
            body.addAttribute('position', pos_ba);
            body.setIndex(idx_ba);
            body.setDrawRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);
            body.computeVertexNormals();

            psp_ro.add( new THREE.Mesh(body, new THREE.MeshBasicMaterial
                                       ({ side: THREE.DoubleSide,
                                          depthWrite: false,
                                          color:fcol, transparent: true, opacity: 0.4
                                        })) );

            ib_pos += 2 + 3 * rnr_data.idxBuff[ib_pos + 1];
         }
         else if (rnr_data.idxBuff[ib_pos] == GL.LINE_LOOP)
         {
            var body = new THREE.BufferGeometry();
            body.addAttribute('position', pos_ba);
            body.setIndex(idx_ba);
            body.setDrawRange(ib_pos + 2, rnr_data.idxBuff[ib_pos + 1]);

            psp_ro.add( new THREE.LineLoop(body, line_mat) );

            ib_pos += 2 + rnr_data.idxBuff[ib_pos + 1];
         }
         else
         {
            throw "Unexpected primitive type", rnr_data.idxBuff[ib_pos]
         }
      }

      return psp_ro;
   }

   JSROOT.EVE.EveElements = EveElements;

   console.log("LOADING EVE ELEMENTS");

   return JSROOT;

}));
