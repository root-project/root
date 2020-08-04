sap.ui.define(['rootui5/eve7/lib/EveManager'], function(EveManager) {

   "use strict";

   // See also EveScene.js makeGLRepresentation(), there several members are
   // set for the top-level Object3D.

   //==============================================================================
   // EveElemControl
   //==============================================================================

   function EveElemControl(o3d)
   {
      // JSROOT.Painter.GeoDrawingControl.call(this);
      this.obj3d = o3d;
   }

   EveElemControl.prototype = Object.create(JSROOT.Painter.GeoDrawingControl.prototype);

   EveElemControl.prototype.invokeSceneMethod = function(fname, arg)
   {
      if ( ! this.obj3d) return false;

      var s = this.obj3d.scene;
      if (s && (typeof s[fname] == "function"))
         return s[fname](this.obj3d, arg, this.event);
      return false;
   }

   EveElemControl.prototype.separateDraw = false;

   EveElemControl.prototype.elementHighlighted = function(indx)
   {
      // default is simple selection, we ignore the indx
      this.invokeSceneMethod("processElementHighlighted"); // , indx);
   }

   EveElemControl.prototype.elementSelected = function(indx)
   {
      // default is simple selection, we ignore the indx
      this.invokeSceneMethod("processElementSelected"); //, indx);
   }


   //==============================================================================
   // EveElements
   //==============================================================================

   var GL = { POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4 };
   var RC;

   function EveElements(rc)
   {
      console.log("EveElements -- RCore");

      RC = rc;
   }

   EveElements.prototype.TestRnr = function(name, obj, rnr_data)
   {
      if (obj && rnr_data && rnr_data.vtxBuff) return false;

      var cnt = this[name] || 0;
      if (cnt++ < 5) console.log(name, obj, rnr_data);
      this[name] = cnt;
      return true;
   }


   //==============================================================================
   // makeHit
   //==============================================================================

   EveElements.prototype.makeHit = function(hit, rnr_data)
   {
      if (this.TestRnr("hit", hit, rnr_data)) return null;

      //let hit_size = 8 * rnr_data.fMarkerSize;
      //let size     = rnr_data.vtxBuff.length / 3;

      let geo = new RC.Geometry();
      geo.vertices = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );

      let col = new RC.Color(JSROOT.Painter.root_colors[hit.fMarkerColor]);

      let mat = new RC.MeshBasicMaterial;
      mat.color = col;

      let pnts = new RC.Point(geo, mat);

      // mesh.get_ctrl = function() { return new EveElemControl(this); }

      // mesh.highlightScale = 2;
      // mesh.material.sizeAttenuation = false;
      // mesh.material.size = hit.fMarkerSize;

      return pnts;
   }


   //==============================================================================
   // makeTrack
   //==============================================================================

   EveElements.prototype.makeTrack = function(track, rnr_data)
   {
      if (this.TestRnr("track", track, rnr_data)) return null;

      let N = rnr_data.vtxBuff.length/3;
      let track_width = track.fLineWidth || 1;
      let track_color = JSROOT.Painter.root_colors[track.fLineColor] || "rgb(255,0,255)";

      if (JSROOT.browser.isWin) track_width = 1;  // not supported on windows

      let buf = new Float32Array((N-1) * 6), pos = 0;
      for (let k=0;k<(N-1);++k) {
         buf[pos]   = rnr_data.vtxBuff[k*3];
         buf[pos+1] = rnr_data.vtxBuff[k*3+1];
         buf[pos+2] = rnr_data.vtxBuff[k*3+2];

         let breakTrack = false;
         if (rnr_data.idxBuff)
            for (let b = 0; b < rnr_data.idxBuff.length; b++) {
               if ( (k+1) == rnr_data.idxBuff[b]) {
                  breakTrack = true;
                  break;
               }
            }

         if (breakTrack) {
            buf[pos+3] = rnr_data.vtxBuff[k*3];
            buf[pos+4] = rnr_data.vtxBuff[k*3+1];
            buf[pos+5] = rnr_data.vtxBuff[k*3+2];
         } else {
            buf[pos+3] = rnr_data.vtxBuff[k*3+3];
            buf[pos+4] = rnr_data.vtxBuff[k*3+4];
            buf[pos+5] = rnr_data.vtxBuff[k*3+5];
         }

         pos+=6;
      }

      let style = (track.fLineStyle > 1) ? JSROOT.Painter.root_line_styles[track.fLineStyle] : "",
          dash = style ? style.split(",") : [], lineMaterial;

      if (dash && (dash.length > 1)) {
         lineMaterial = new RC.MeshBasicMaterial({ color: track_color, linewidth: track_width, dashSize: parseInt(dash[0]), gapSize: parseInt(dash[1]) });
      } else {
         lineMaterial = new RC.MeshBasicMaterial({ color: track_color, linewidth: track_width });
      }

      let geom = new RC.Geometry();
      geom.vertices = new RC.BufferAttribute( buf, 3 );

      let line = new RC.Line(geom, lineMaterial);
      line.renderingPrimitive = RC.LINES;

      // required for the dashed material
      //if (dash && (dash.length > 1))
      //   line.computeLineDistances();

      //line.hightlightWidthScale = 2;

      line.get_ctrl = function() { return new EveElemControl(this); }

      return line;
   }

   //==============================================================================
   // makeJet
   //==============================================================================

   EveElements.prototype.makeJet = function(jet, rnr_data)
   {
      if (this.TestRnr("jet", jet, rnr_data)) return null;

      // console.log("make jet ", jet);
      // let jet_ro = new RC.Object3D();
      let pos_ba = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );
      let N      = rnr_data.vtxBuff.length / 3;

      let geo_body = new RC.Geometry();
      geo_body.vertices = pos_ba;
      let idcs = new Uint16Array(3 + 3 * (N - 1)); ///[0, N-1, 1];
      idcs[0] = 0; idcs[1] = N-1; idcs[2] = 1;
      for (let i = 1; i < N - 1; ++i) {
         idcs[3*i] = 0; idcs[3*i + 1] = i; idcs[3*i + 2] = i + 1;
         // idcs.push( 0, i, i + 1 );
      }
      geo_body.indices = new RC.BufferAttribute( idcs, 1 );
      geo_body.computeVertexNormals();

      let geo_rim = new RC.Geometry();
      geo_rim.vertices = pos_ba;
      idcs = new Uint16Array(N-1);
      for (let i = 1; i < N; ++i) idcs[i-1] = i;
      geo_rim.indices = new RC.BufferAttribute( idcs, 1 );

      let geo_rays = new RC.Geometry();
      geo_rays.vertices = pos_ba;
      idcs = [];
      for (let i = 1; i < N; i += 4)
         idcs.push( 0, i );
      geo_rays.indices = idcs;

      let mcol = JSROOT.Painter.root_colors[jet.fMainColor];
      let lcol = JSROOT.Painter.root_colors[jet.fLineColor];

      let mesh = new RC.Mesh(geo_body, new RC.MeshPhongMaterial({ depthWrite: false, color: mcol, transparent: true, opacity: 0.5, side: RC.DoubleSide }));
      let line1 = new RC.Line(geo_rim,  new RC.MeshBasicMaterial({ linewidth: 2,   color: lcol, transparent: true, opacity: 0.5 }));
      let line2 = new RC.Line(geo_rays, new RC.MeshBasicMaterial({ linewidth: 0.5, color: lcol, transparent: true, opacity: 0.5 }));
      line2.renderingPrimitive = RC.LINES;

      // jet_ro.add( mesh  );
      mesh.add( line1 );
      mesh.add( line2 );

      mesh.get_ctrl = function() { return new EveElemControl(this); }

      return mesh;
   }

   EveElements.prototype.makeJetProjected = function(jet, rnr_data)
   {
      // JetProjected has 3 or 4 points. 0-th is apex, others are rim.
      // Fourth point is only present in RhoZ when jet hits barrel/endcap transition.

      // console.log("makeJetProjected ", jet);

      if (this.TestRnr("jetp", jet, rnr_data)) return null;


      let pos_ba = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );
      let N      = rnr_data.vtxBuff.length / 3;

      let geo_body = new RC.Geometry();
      geo_body.vertices = pos_ba;
      let idcs = new Uint16Array( N > 3 ? 6 : 3);
      idcs[0] = 0; idcs[1] = 2; idcs[2] = 1;
      if (N > 3) {  idcs[3] = 0; idcs[4] = 5; idcs[5] = 2; }
      geo_body.indices = new RC.BufferAttribute( idcs, 1 );
      geo_body.computeVertexNormals();

      let geo_rim = new RC.Geometry();
      geo_rim.vertices = pos_ba;
      idcs = new Uint16Array(N-1);
      for (let i = 1; i < N; ++i) idcs[i-1] = i;
      geo_rim.indices = new RC.BufferAttribute( idcs, 1 );

      let geo_rays = new RC.Geometry();
      geo_rays.vertices = pos_ba;
      idcs = new Uint16Array(4); // [ 0, 1, 0, N-1 ];
      idcs[0] = 0; idcs[1] = 1; idcs[2] = 0; idcs[3] = N-1;
      geo_rays.indices = new RC.BufferAttribute( idcs, 1 );;

      let fcol = JSROOT.Painter.root_colors[jet.fFillColor];
      let lcol = JSROOT.Painter.root_colors[jet.fLineColor];
      // Process transparency !!!
      // console.log("cols", fcol, lcol);

      // double-side material required for correct tracing of colors - otherwise points sequence should be changed
      let mesh = new RC.Mesh(geo_body, new RC.MeshBasicMaterial({ depthWrite: false, color: fcol, transparent: true, opacity: 0.5, side: RC.DoubleSide }));
      let line1 = new RC.Line(geo_rim,  new RC.MeshBasicMaterial({ linewidth: 2, color: lcol, transparent: true, opacity: 0.5 }));
      let line2 = new RC.Line(geo_rays, new RC.MeshBasicMaterial({ linewidth: 1, color: lcol, transparent: true, opacity: 0.5 }));
      line2.renderingPrimitive = RC.LINES;

      // jet_ro.add( mesh  );
      mesh.add( line1 );
      mesh.add( line2 );

      mesh.get_ctrl = function() { return new EveElemControl(this); }

      return mesh;
   }


   //==============================================================================
   // makeEveGeometry / makeEveGeoShape
   //==============================================================================

   EveElements.prototype.makeEveGeometry = function(rnr_data, force)
   {
      let nVert = rnr_data.idxBuff[1]*3;

      if (rnr_data.idxBuff[0] != GL.TRIANGLES)  throw "Expect triangles first.";
      if (2 + nVert != rnr_data.idxBuff.length) throw "Expect single list of triangles in index buffer.";

      let body = new RC.Geometry();
      body.vertices = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );
      body.indices  = new RC.BufferAttribute( rnr_data.idxBuff, 1 );
      body.setDrawRange(2, nVert);
      body.computeVertexNormalsIdxRange(2, nVert);

      // XXXX Fix this. It seems we could have flat shading with usage of simple shaders.
      // XXXX Also, we could do edge detect on the server for outlines.
      // XXXX a) 3d objects - angle between triangles >= 85 degrees (or something);
      // XXXX b) 2d objects - segment only has one triangle.
      // XXXX Somewhat orthogonal - when we do tesselation, conversion from quads to
      // XXXX triangles is trivial, we could do it before invoking the big guns (if they are even needed).
      // XXXX Oh, and once triangulated, we really don't need to store 3 as number of verts in a poly each time.
      // XXXX Or do we? We might need it for projection stuff.

      return body;
   }

   EveElements.prototype.makeEveGeoShape = function(egs, rnr_data)
   {
      let egs_ro = new RC.Group();

      let geom = this.makeEveGeometry(rnr_data);

      let fcol = new RC.Color(JSROOT.Painter.getColor(egs.fFillColor));

      // let material = new RC.MeshPhongMaterial({// side: THREE.DoubleSide,
      //                     depthWrite: false, color:fcol, transparent: true, opacity: 0.2 });
      let material = new RC.MeshPhongMaterial;
      material.color = fcol;
      material.side = 2;
      material.depthWrite = false;
      material.transparent = true;
      material.opacity = 0.2;

      let mesh = new RC.Mesh(geom, material);

      egs_ro.add(mesh);

      return egs_ro;
   }


   //==============================================================================
   // makePolygonSetProjected
   //==============================================================================

   EveElements.prototype.makePolygonSetProjected = function(psp, rnr_data)
   {
      let psp_ro = new RC.Group();
      let pos_ba = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );
      let idx_ba = new RC.BufferAttribute( rnr_data.idxBuff, 1 );

      let ib_len = rnr_data.idxBuff.length;

      let fcol = new RC.Color(JSROOT.Painter.root_colors[psp.fMainColor]);

      let material = new RC.MeshPhongMaterial;
      material.color     = fcol;
      material.specular  = new RC.Color(1, 1, 1);
      material.shininess = 50;
      material.side  = RC.FRONT_AND_BACK_SIDE;
      material.depthWrite  = false;
      material.transparent = true;
      material.opacity = 0.4;

      console.log("XXXXX", fcol, material);

      // XXXXXX Should be Mesh -> Line material ???
      let line_mat = new RC.MeshBasicMaterial;
      line_mat.color = fcol;

      for (let ib_pos = 0; ib_pos < ib_len; )
      {
         if (rnr_data.idxBuff[ib_pos] == GL.TRIANGLES)
         {
            let body = new RC.Geometry();
            body.vertices = pos_ba;
            body.indices  = idx_ba;
            body.setDrawRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);
            body.computeVertexNormalsIdxRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);

            psp_ro.add( new RC.Mesh(body, material) );

            ib_pos += 2 + 3 * rnr_data.idxBuff[ib_pos + 1];
         }
         else if (rnr_data.idxBuff[ib_pos] == GL.LINE_LOOP)
         {
            let body = new RC.Geometry();
            body.vertices = pos_ba;
            body.indices  = idx_ba;
            body.setDrawRange(ib_pos + 2, rnr_data.idxBuff[ib_pos + 1]);

            let ll = new RC.Line(body, line_mat);
            ll.renderingPrimitive = RC.LINE_LOOP;
            psp_ro.add( ll );

            ib_pos += 2 + rnr_data.idxBuff[ib_pos + 1];
         }
         else
         {
            console.error("Unexpected primitive type " + rnr_data.idxBuff[ib_pos]);
            break;
         }

      }

      return psp_ro;
   }

   //==============================================================================

   return EveElements;

});
