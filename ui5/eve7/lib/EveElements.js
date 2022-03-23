/** @file EveElements.js
 * used only together with OpenUI5 */

// TODO: add dependency from JSROOT components

sap.ui.define(['rootui5/eve7/lib/EveManager'], function(EveManager) {

   "use strict";

   let jsrp = JSROOT.Painter; // JSROOT naming convention

   // See also EveScene.js makeGLRepresentation(), there several members are
   // set for the top-level Object3D.

   //==============================================================================
   // EveElemControl
   //==============================================================================

   function EveElemControl(o3d)
   {
      // jsrp.GeoDrawingControl.call(this);
      this.obj3d = o3d;
   }

   EveElemControl.prototype = Object.create(jsrp.GeoDrawingControl.prototype);

   EveElemControl.prototype.invokeSceneMethod = function(fname, arg)
   {
      if (!this.obj3d) return false;

      let s = this.obj3d.scene;
      if (s && (typeof s[fname] == "function"))
         return s[fname](this.obj3d, arg, this.event);
      return false;
   }

   EveElemControl.prototype.separateDraw = false;

   EveElemControl.prototype.getTooltipText = function(intersect)
   {
      let el =  this.obj3d.eve_el;
      return el.fName || el.fTitle || "";
   }

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

   EveElemControl.prototype.DrawForSelection = function(sec_idcs, res)
   {
      res.geom.push(this.obj3d);
   }
   //==============================================================================
   // EveElements
   //==============================================================================

   var GL = { POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4 };

   function EveElements()
   {
   }

   /** Test if render data has vertex buffer. Make logging if not. Only for debug purposes */
   EveElements.prototype.TestRnr = function(name, obj, rnrData) {

      if (obj && rnrData && rnrData.vtxBuff) return false;

      let cnt = this[name] || 0;

      if (cnt++ < 5) console.log(name, obj, rnrData);

      this[name] = cnt;

      return true;
   }

   EveElements.prototype.makeHit = function(hit, rnrData)
   {
      if (this.TestRnr("hit", hit, rnrData)) return null;

      let hit_size = 8 * rnrData.fMarkerSize;
      let size     = rnrData.vtxBuff.length / 3;
      let pnts     = new jsrp.PointsCreator(size, true, hit_size);

      for (let i=0; i<size; i++)
         pnts.addPoint(rnrData.vtxBuff[i*3],rnrData.vtxBuff[i*3+1],rnrData.vtxBuff[i*3+2]);

      let mesh = pnts.createPoints(jsrp.getColor(hit.fMarkerColor));

      mesh.highlightScale = 2;

      mesh.material.sizeAttenuation = false;
      mesh.material.size = hit.fMarkerSize;

      mesh.get_ctrl = function() { return new EveElemControl(this); };
      mesh.dispose  = function() { this.geometry.dispose(); this.material.dispose(); };

      return mesh;
   }

   EveElements.prototype.makeTrack = function(track, rnrData)
   {
      if (this.TestRnr("track", track, rnrData)) return null;

      let N = rnrData.vtxBuff.length/3;
      let track_width = track.fLineWidth || 1;
      let track_color = jsrp.getColor(track.fLineColor) || "rgb(255,0,255)";

      let buf = new Float32Array((N-1) * 6), pos = 0;
      for (let k=0;k<(N-1);++k) {
         buf[pos]   = rnrData.vtxBuff[k*3];
         buf[pos+1] = rnrData.vtxBuff[k*3+1];
         buf[pos+2] = rnrData.vtxBuff[k*3+2];

         let breakTrack = false;
         if (rnrData.idxBuff)
            for (let b = 0; b < rnrData.idxBuff.length; b++) {
               if ( (k+1) == rnrData.idxBuff[b]) {
                  breakTrack = true;
                  break;
               }
            }

         if (breakTrack) {
            buf[pos+3] = rnrData.vtxBuff[k*3];
            buf[pos+4] = rnrData.vtxBuff[k*3+1];
            buf[pos+5] = rnrData.vtxBuff[k*3+2];
         } else {
            buf[pos+3] = rnrData.vtxBuff[k*3+3];
            buf[pos+4] = rnrData.vtxBuff[k*3+4];
            buf[pos+5] = rnrData.vtxBuff[k*3+5];
         }

         pos+=6;
      }

      let style = (track.fLineStyle > 1) ? jsrp.root_line_styles[track.fLineStyle] : "",
          dash = style ? style.split(",") : [], lineMaterial;

      if (dash && (dash.length > 1)) {
         lineMaterial = new THREE.LineDashedMaterial({ color: track_color, linewidth: track_width, dashSize: parseInt(dash[0]), gapSize: parseInt(dash[1]) });
      } else {
         lineMaterial = new THREE.LineBasicMaterial({ color: track_color, linewidth: track_width });
      }

      let geom = new THREE.BufferGeometry();
      geom.setAttribute( 'position', new THREE.BufferAttribute( buf, 3 )  );
      let line = new THREE.LineSegments(geom, lineMaterial);

      // required for the dashed material
      if (dash && (dash.length > 1))
         line.computeLineDistances();

      line.hightlightWidthScale = 2;

      line.get_ctrl = function() { return new EveElemControl(this); };
      line.dispose  = function() { this.geometry.dispose(); this.material.dispose(); };

      return line;
   }

   EveElements.prototype.makeJet = function(jet, rnrData)
   {
      if (this.TestRnr("jet", jet, rnrData)) return null;

      // console.log("make jet ", jet);
      // let jet_ro = new THREE.Object3D();
      let pos_ba = new THREE.BufferAttribute( rnrData.vtxBuff, 3 );
      let N      = rnrData.vtxBuff.length / 3;

      let geo_body = new THREE.BufferGeometry();
      geo_body.setAttribute('position', pos_ba);
      let idcs = [0, N-1, 1];
      for (let i = 1; i < N - 1; ++i)
         idcs.push( 0, i, i + 1 );
      geo_body.setIndex( idcs );
      geo_body.computeVertexNormals();

      let geo_rim = new THREE.BufferGeometry();
      geo_rim.setAttribute('position', pos_ba);
      idcs = new Uint16Array(N-1);
      for (let i = 1; i < N; ++i) idcs[i-1] = i;
      geo_rim.setIndex(new THREE.BufferAttribute( idcs, 1 ));

      let geo_rays = new THREE.BufferGeometry();
      geo_rays.setAttribute('position', pos_ba);
      idcs = [];
      for (let i = 1; i < N; i += 4)
         idcs.push( 0, i );
      geo_rays.setIndex( idcs );

      let mcol = jsrp.getColor(jet.fMainColor);
      let lcol = jsrp.getColor(jet.fLineColor);

      let mesh = new THREE.Mesh(geo_body, new THREE.MeshPhongMaterial({ depthWrite: false, color: mcol, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
      let line1 = new THREE.LineLoop(geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2,   color: lcol, transparent: true, opacity: 0.5 }));
      let line2 = new THREE.LineSegments(geo_rays, new THREE.LineBasicMaterial({ linewidth: 0.5, color: lcol, transparent: true, opacity: 0.5 }));

      // jet_ro.add( mesh  );
      mesh.add( line1 );
      mesh.add( line2 );

      mesh.get_ctrl = function() { return new EveElemControl(this); };
      mesh.dispose  = function() {
         this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
         this.geometry.dispose(); this.material.dispose();
      };

      return mesh;
   }

   EveElements.prototype.makeJetProjected = function(jet, rnrData)
   {
      // JetProjected has 3 or 4 points. 0-th is apex, others are rim.
      // Fourth point is only present in RhoZ when jet hits barrel/endcap transition.

      // console.log("makeJetProjected ", jet);

      if (this.TestRnr("jetp", jet, rnrData)) return null;


      let pos_ba = new THREE.BufferAttribute( rnrData.vtxBuff, 3 );
      let N      = rnrData.vtxBuff.length / 3;

      let geo_body = new THREE.BufferGeometry();
      geo_body.setAttribute('position', pos_ba);
      let idcs = [0, 2, 1];
      if (N > 3)
         idcs.push( 0, 3, 2 );
      geo_body.setIndex( idcs );
      geo_body.computeVertexNormals();

      let geo_rim = new THREE.BufferGeometry();
      geo_rim.setAttribute('position', pos_ba);
      idcs = new Uint16Array(N-1);
      for (let i = 1; i < N; ++i) idcs[i-1] = i;
      geo_rim.setIndex(new THREE.BufferAttribute( idcs, 1 ));

      let geo_rays = new THREE.BufferGeometry();
      geo_rays.setAttribute('position', pos_ba);
      idcs = [ 0, 1, 0, N-1 ];
      geo_rays.setIndex( idcs );

      let fcol = jsrp.getColor(jet.fFillColor);
      let lcol = jsrp.getColor(jet.fLineColor);
      // Process transparency !!!
      // console.log("cols", fcol, lcol);

      // double-side material required for correct tracing of colors - otherwise points sequence should be changed
      let mesh = new THREE.Mesh(geo_body, new THREE.MeshBasicMaterial({ depthWrite: false, color: fcol, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
      let line1 = new THREE.Line(geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2, color: lcol, transparent: true, opacity: 0.5 }));
      let line2 = new THREE.LineSegments(geo_rays, new THREE.LineBasicMaterial({ linewidth: 1, color: lcol, transparent: true, opacity: 0.5 }));

      // jet_ro.add( mesh  );
      mesh.add( line1 );
      mesh.add( line2 );

      mesh.get_ctrl = function() { return new EveElemControl(this); };
      mesh.dispose  = function() {
         this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
         this.geometry.dispose(); this.material.dispose();
      };

      return mesh;
   }


   EveElements.prototype.makeFlatBox = function(ebox, rnrData, idxBegin, idxEnd)
   {
      let fcol = jsrp.getColor(ebox.fMainColor);
      let boxMaterial = new THREE.MeshPhongMaterial({color: fcol,  flatShading: true});

      // console.log("EveElements.prototype.makeFlatBox triangulate", idxBegin, idxEnd);
      let nTriang = (idxEnd - idxBegin) -2;
      let idxBuff =  new Uint16Array(nTriang * 3);
      let nt = 0;
      for (let i = idxBegin; i < (idxEnd-2 ); ++i)
      {
         idxBuff[nt*3] = idxBegin;
         idxBuff[nt*3+1] = i + 1 ;
         idxBuff[nt*3+2] = i + 2;
         // console.log("set index ", nt,":", idxBuff[nt*3], idxBuff[nt*3+1],idxBuff[nt*3+2]);
         nt++;
      }
      let idcs = new THREE.BufferAttribute(idxBuff,1);

      let body = new THREE.BufferGeometry();
      body.setAttribute('position', new THREE.BufferAttribute( rnrData.vtxBuff, 3 ));
      body.setIndex(new THREE.BufferAttribute(idxBuff,1));
      body.computeVertexNormals();
      let mesh = new THREE.Mesh(body, boxMaterial);
      return mesh;
   }

   EveElements.prototype.makeBoxProjected = function(ebox, rnrData)
   {
      let nPnts    = parseInt(rnrData.vtxBuff.length/3);
      let breakIdx = parseInt(ebox.fBreakIdx);
      if ( ebox.fBreakIdx == 0 )
         breakIdx = nPnts;

      let mesh1 = this.makeFlatBox(ebox, rnrData, 0, breakIdx);
      let testBreak = breakIdx + 2;
      if (testBreak < nPnts)
      {
         let mesh2 = this.makeFlatBox(ebox, rnrData, breakIdx, nPnts);
         mesh2.get_ctrl = function() { return new EveElemControl(this); }
         mesh1.add(mesh2);
      }

      mesh1.get_ctrl = function() { return new EveElemControl(this); };
      mesh1.dispose  = function() {
         this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
         this.geometry.dispose(); this.material.dispose();
      };

      return mesh1;
   }

   EveElements.prototype.makeBox = function(ebox, rnr_data)
   {
      let idxBuff = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
      let vBuff = rnr_data.vtxBuff;

      let body = new THREE.BufferGeometry();
      body.setAttribute('position', new THREE.BufferAttribute( vBuff, 3 ));
      body.setIndex( idxBuff );

      let fcol = jsrp.getColor(ebox.fMainColor);
      let boxMaterial = new THREE.MeshPhongMaterial({color: fcol,  flatShading: true});
      if (ebox.fMainTransparency) {
         boxMaterial.transparent = true;
         boxMaterial.opacity = (100 - ebox.fMainTransparency)/100.0;
         boxMaterial.depthWrite = false;
      }

      let mesh = new THREE.Mesh(body, boxMaterial);
      let geo_rim = new THREE.BufferGeometry();

      geo_rim.setAttribute('position', new THREE.BufferAttribute( vBuff, 3 ));

      let nTrigs      = 6 * 2;
      let nIdcsTrings = 6 * 2 * 3 * 2;
      let idcs = new Uint16Array(nIdcsTrings);
      for (let i = 0; i < nTrigs; ++i)
      {
         let ibo = i * 3;
         let sbo = i * 6;
         idcs[sbo]     = idxBuff[ibo];
         idcs[sbo + 1] = idxBuff[ibo + 1];
         idcs[sbo + 2] = idxBuff[ibo + 1];
         idcs[sbo + 3] = idxBuff[ibo + 2];
         idcs[sbo + 4] = idxBuff[ibo + 2];
         idcs[sbo + 5] = idxBuff[ibo];
      }
      geo_rim.setIndex(new THREE.BufferAttribute( idcs, 1 ));
      let lcol = jsrp.getColor(ebox.fLineColor);
      let line = new THREE.LineSegments(geo_rim,  new THREE.LineBasicMaterial({ linewidth: 2, color: lcol, transparent: true, opacity: 0.5 }));
      mesh.add(line);

      mesh.get_ctrl = function() { return new EveElemControl(this); }
      mesh.dispose  = function() {
         this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
         this.geometry.dispose(); this.material.dispose();
      };

      return mesh;
   }


   EveElements.prototype.makeBoxSet = function(boxset, rnr_data)
   {
      let vBuff;
      if (boxset.boxType == 1) // free box
      {
         vBuff = rnr_data.vtxBuff;
      }
      else if (boxset.boxType == 2) // axis aligned
      {
         let N = rnr_data.vtxBuff.length/6;
         vBuff = new Float32Array(N*8*3);

         let off = 0;
         for (let i = 0; i < N; ++i)
         {
            let rdoff = i*6;
            let x  =  rnr_data.vtxBuff[rdoff];
            let y  =  rnr_data.vtxBuff[rdoff + 1];
            let z  =  rnr_data.vtxBuff[rdoff + 2];
            let dx =  rnr_data.vtxBuff[rdoff + 3];
            let dy =  rnr_data.vtxBuff[rdoff + 4];
            let dz =  rnr_data.vtxBuff[rdoff + 5];

            // top
            vBuff[off  ] = x;      vBuff[off + 1] = y + dy; vBuff[off + 2] = z;
            off += 3;
            vBuff[off  ] = x + dx; vBuff[off + 1] = y + dy; vBuff[off + 2] = z;
            off += 3;
            vBuff[off  ] = x + dx; vBuff[off + 1] = y;      vBuff[off + 2] = z;
            off += 3;
            vBuff[off  ] = x;      vBuff[off + 1] = y;      vBuff[off + 2] = z;
            off += 3;
            // bottom
            vBuff[off  ] = x;      vBuff[off + 1] = y + dy; vBuff[off + 2] = z + dz;
            off += 3;
            vBuff[off  ] = x + dx; vBuff[off + 1] = y + dy; vBuff[off + 2] = z + dz;
            off += 3;
            vBuff[off  ] = x + dx; vBuff[off + 1] = y;      vBuff[off + 2] = z + dz;
            off += 3;
            vBuff[off  ] = x;      vBuff[off + 1] = y;      vBuff[off + 2] = z + dz;
            off += 3;
         }
      }

      let protoSize = 6 * 2 * 3;
      let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
      let nBox = vBuff.length / 24;
      let idxBuff = [];
      for (let i = 0; i < nBox; ++i)
      {
         for (let c = 0; c < protoSize; c++) {
            let off = i * 8;
            idxBuff.push(protoIdcs[c] + off);
         }
      }

      let body = new THREE.BufferGeometry();
      body.setAttribute('position', new THREE.BufferAttribute( vBuff, 3 ));
      body.setIndex( idxBuff );

      //
      // set colors
      let material = 0;
      if (!boxset.fSingleColor)
      {
         let ci = rnr_data.idxBuff;
         let off = 0
         let colBuff = new Float32Array( nBox * 8 *3 );
         for (let x = 0; x < ci.length; ++x)
         {
            let r = (ci[x] & 0x000000FF) >>  0;
            let g = (ci[x] & 0x0000FF00) >>  8;
            let b = (ci[x] & 0x00FF0000) >> 16;
            for (let i = 0; i < 8; ++i)
            {
               colBuff[off    ] = r/256;
               colBuff[off + 1] = g/256;
               colBuff[off + 2] = b/256;
               off += 3;
            }
         }
         body.setAttribute( 'color', new THREE.BufferAttribute( colBuff, 3 ) );
         material = new THREE.MeshPhongMaterial( {
            color: 0xffffff,
            flatShading: true,
            vertexColors: true,
            shininess: 0
         } );
      }
      else
      {
         let fcol = jsrp.getColor(boxset.fMainColor);
         material = new THREE.MeshPhongMaterial({color:fcol, flatShading: true});
         if (boxset.fMainTransparency) {
            material.transparent = true;
            material.opacity = (100 - boxset.fMainTransparency)/100.0;
            material.depthWrite = false;
         }
      }

      let mesh = new THREE.Mesh(body, material);
      if (boxset.fSecondarySelect)
         mesh.get_ctrl = function() { return new BoxSetControl(mesh); };
      else
         mesh.get_ctrl = function() { return new EveElemControl(mesh); };
      mesh.dispose  = function() { this.geometry.dispose(); this.material.dispose(); };

      return mesh;
   }

   function BoxSetControl(mesh)
   {
      EveElemControl.call(this, mesh);
   }

   BoxSetControl.prototype = Object.create(EveElemControl.prototype);

   BoxSetControl.prototype.DrawForSelection = function(sec_idcs, res)
   {
      let geobox = new THREE.BufferGeometry();
      geobox.setAttribute( 'position', this.obj3d.geometry.getAttribute("position") );

      let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
      let idxBuff = new Array(sec_idcs.length * protoIdcs.length);

      let N = this.obj3d.eve_el.render_data.idxBuff.length / 2;
      for (let b = 0; b < sec_idcs.length; ++b) {
         let idx = sec_idcs[b]
         if (this.obj3d.eve_el.fDetIdsAsSecondaryIndices) {
            for (let x = 0; x < N; ++x) {
               if (this.obj3d.eve_el.render_data.idxBuff[x + N] === idx)
               {
                  idx=x;
                  break;
               }
            }
         }
         let idxOff = idx * 8;
         for (let i = 0; i < protoIdcs.length; i++)
            idxBuff.push(idxOff + protoIdcs[i]);
      }

      geobox.setIndex( idxBuff );
      let material = new THREE.MeshPhongMaterial({color:"purple", flatShading: true});
      let mesh     = new THREE.Mesh(geobox, material);
      res.geom.push(mesh);
   }

   BoxSetControl.prototype.extractIndex = function(intersect)
   {
      let idx  = Math.floor(intersect.faceIndex/12);
      return idx;
   }

   BoxSetControl.prototype.getTooltipText = function(intersect)
   {
      let t = this.obj3d.eve_el.fTitle || this.obj3d.eve_el.fName || "";
      let idx = this.extractIndex(intersect);
      if (this.obj3d.eve_el.fDetIdsAsSecondaryIndices) {
	 let N = this.obj3d.eve_el.render_data.idxBuff.length / 2;
	 let id = this.obj3d.eve_el.render_data.idxBuff[N + idx];
         return t + " idx=" + id;
      }
      return t + " idx=" + idx;
   }

   BoxSetControl.prototype.elementSelected = function(indx)
   {
       if (this.obj3d.eve_el.fDetIdsAsSecondaryIndices) {
	  let N = this.obj3d.eve_el.render_data.idxBuff.length / 2;
          indx = this.obj3d.eve_el.render_data.idxBuff[N + indx];
       }

      this.invokeSceneMethod("processElementSelected", indx);
   }

   BoxSetControl.prototype.elementHighlighted = function(indx)
   {
      this.invokeSceneMethod("processElementHighlighted", indx);
   }

   BoxSetControl.prototype.checkHighlightIndex = function(indx)
   {
      if (this.obj3d && this.obj3d.scene)
         return this.invokeSceneMethod("processCheckHighlight", indx);

      return true; // means index is different
   }


   //==============================================================================

    EveElements.prototype.makeCalo2D = function(calo2D, rnrData)
    {
       let body = new THREE.BufferGeometry();
       if (rnrData.vtxBuff) {
          let nSquares = rnrData.vtxBuff.length / 12;
          let nTriang = 2 * nSquares;

          let idxBuff = new Uint16Array(nTriang * 3);
          for (let s = 0; s < nSquares; ++s) {
             let boff = s * 6;
             let ioff = s * 4;

             // first triangle
             idxBuff[boff] = ioff;
             idxBuff[boff + 1] = ioff + 1;
             idxBuff[boff + 2] = ioff + 2;

             // second triangle
             idxBuff[boff + 3] = ioff + 2;
             idxBuff[boff + 4] = ioff + 3;
             idxBuff[boff + 5] = ioff;
          }
          let idcs = new THREE.BufferAttribute(idxBuff, 1);

          body.setAttribute('position', new THREE.BufferAttribute(rnrData.vtxBuff, 3));
          body.setIndex(new THREE.BufferAttribute(idxBuff, 1));
          body.computeVertexNormals();

          let ci = rnrData.idxBuff;
          let colBuff = new Float32Array(nSquares * 4 * 3);
          let off = 0;
          for (let x = 0; x < ci.length; ++x) {
             let slice = ci[x * 2];
             let sliceColor = calo2D.sliceColors[slice];
             let tc = new THREE.Color(jsrp.getColor(sliceColor));
             for (let i = 0; i < 4; ++i) {
                colBuff[off] = tc.r;
                colBuff[off + 1] = tc.g;
                colBuff[off + 2] = tc.b;
                off += 3;
             }
          }
          body.setAttribute('color', new THREE.BufferAttribute(colBuff, 3));
       }
       let material = new THREE.MeshPhongMaterial({
          color: 0xffffff,
          flatShading: true,
          vertexColors: true,
          shininess: 0
       });
       let mesh = new THREE.Mesh(body, material);

       mesh.get_ctrl = function () { return new Calo2DControl(mesh); };
       mesh.dispose = function () { this.geometry.dispose(); this.material.dispose(); };

       return mesh;
    }

    function Calo2DControl(mesh)
    {
        EveElemControl.call(this, mesh);
    }

    Calo2DControl.prototype = Object.create(EveElemControl.prototype);

    Calo2DControl.prototype.DrawForSelection = function(sec_idcs, res, extra)
    {
        let cells;
        for (let i = 0; i < extra.length; i++) {
            if (extra[i].caloVizId ==  this.obj3d.eve_el.fElementId) {
                cells = extra[i].cells;
                break;
            }
        }

        let ibuff = this.obj3d.eve_el.render_data.idxBuff;
        let vbuff = this.obj3d.eve_el.render_data.vtxBuff;
        let nbox = ibuff.length/2;
        let nBoxSelected = cells.length;
        let boxIdcs = [];
        for (let i = 0; i < cells.length; i++)
        {
            let bin = cells[i].b;
            let slice = cells[i].s;
            // let fraction =  cells[i].f;
           for (let r = 0; r < nbox; r++) {
              if (ibuff[r * 2] == slice) {

                 if (bin > 0 && ibuff[r * 2 + 1] == bin) {
                    boxIdcs.push(r);
                    break;
                 } else if (bin < 0 && ibuff[r * 2 + 1] == Math.abs(bin) && vbuff[r * 12 + 1] < 0) {
                    boxIdcs.push(r);
                    break;
                 }
              }
           }
        }
        let idxBuff = [];
        let vtxBuff =  new Float32Array(nBoxSelected * 4 * 3 );
        let protoIdcs = [0, 1, 2, 2, 3, 0];
        let rnr_data = this.obj3d.eve_el.render_data;
        for (let i = 0; i < nBoxSelected; ++i)
        {
            let BoxIdcs =  boxIdcs[i];
            for (let v = 0; v < 4; v++) {
                let off = i  * 12 + v * 3;
                let pos = BoxIdcs  * 12 + v *3;
                vtxBuff[off  ] = rnr_data.vtxBuff[pos  ];
                vtxBuff[off+1] = rnr_data.vtxBuff[pos+1];
                vtxBuff[off+2] = rnr_data.vtxBuff[pos+2];
            }
            {
                // fix vertex 1
                let pos = BoxIdcs  * 12;
                let v1x = rnr_data.vtxBuff[pos  ];
                let v1y = rnr_data.vtxBuff[pos + 1];
                pos += 3;
                let v2x = rnr_data.vtxBuff[pos  ];
                let v2y = rnr_data.vtxBuff[pos + 1];
                let off = i  * 12 + 3;
                vtxBuff[off  ] = v1x + cells[i].f * (v2x - v1x);
                vtxBuff[off+1] = v1y + cells[i].f * (v2y - v1y);
            }

            {
                // fix vertex 2
                let pos = BoxIdcs  * 12 + 3 * 3;
                let v1x = rnr_data.vtxBuff[pos  ];
                let v1y = rnr_data.vtxBuff[pos + 1];
                pos -= 3;
                let v2x = rnr_data.vtxBuff[pos  ];
                let v2y = rnr_data.vtxBuff[pos + 1];
                let off = i  * 12 + 3 * 2;
                vtxBuff[off  ] = v1x + cells[i].f * (v2x - v1x);
                vtxBuff[off+1] = v1y + cells[i].f * (v2y - v1y);
            }
            for (let c = 0; c < 6; c++) {
                let off = i * 4;
                idxBuff.push(protoIdcs[c] + off);
            }
        }

        let body = new THREE.BufferGeometry();
        body.setAttribute('position', new THREE.BufferAttribute( vtxBuff, 3 ));
        body.setIndex( idxBuff );

        let mesh = new THREE.Mesh(body);
        res.geom.push(mesh);
    }

   Calo2DControl.prototype.extractIndex = function(intersect)
   {
      let idx  = Math.floor(intersect.faceIndex/2);
      return idx;
   }

   Calo2DControl.prototype.getTooltipText = function(intersect)
    {
        let idx = this.extractIndex(intersect);
        let idxBuff = this.obj3d.eve_el.render_data.idxBuff;
        let bin =  idxBuff[idx*2 + 1];
        let val = this.obj3d.eve_el.render_data.nrmBuff[idx];
        let caloData =  this.obj3d.scene.mgr.GetElement(this.obj3d.eve_el.dataId);
        let slice = idxBuff[idx*2];
        let sname = caloData.sliceInfos[slice].name;

        let vbuff =  this.obj3d.eve_el.render_data.vtxBuff;
        let p = idx*12;
        let x = vbuff[p];
        let y = vbuff[p+1];
        let z = vbuff[p+2];

        if (this.obj3d.eve_el.isRPhi) {
            let phi =  Math.acos(x/Math.sqrt(x*x+y*y));
            phi *= Math.sign(y);
            return  sname + " " + Math.floor(val*100)/100 +
                " ("+  Math.floor(phi*100)/100 + ")";

        }
        else
        {
            let cosTheta = x/Math.sqrt(x*x + y*y);
            let eta = 0;
            if (cosTheta*cosTheta < 1)
            {
                eta = -0.5* Math.log( (1.0-cosTheta)/(1.0+cosTheta) );
            }

            return  sname + " " + Math.floor(val*100)/100 +
                " ("+  Math.floor(eta*100)/100 + ")";
        }

    }

    Calo2DControl.prototype.elementSelectedSendMIR = function(idx, selectionId)
    {
        let calo =  this.obj3d.eve_el;
        let idxBuff = calo.render_data.idxBuff;
        let scene = this.obj3d.scene;
        let multi = event && event.ctrlKey ? true : false;
        let bin = idxBuff[idx*2 + 1];
        let slice =  idxBuff[idx*2];
        // get sign for the case of RhoZ projection
        if (calo.render_data.vtxBuff[idx*12 + 1] < 0) bin = -bin ;

        let fcall = "NewBinPicked(" +  bin + ", " +  slice + ", " + selectionId + ", " + multi + ")"
        scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo2D");
        return true;
    }

    Calo2DControl.prototype.elementSelected = function(idx)
    {
        return this.elementSelectedSendMIR(idx, this.obj3d.scene.mgr.global_selection_id);
    }

   Calo2DControl.prototype.elementHighlighted = function(idx)
    {
        return this.elementSelectedSendMIR(idx, this.obj3d.scene.mgr.global_highlight_id);
   }

   Calo2DControl.prototype.checkHighlightIndex = function(indx)
   {
      if (this.obj3d && this.obj3d.scene)
         return this.invokeSceneMethod("processCheckHighlight", indx);

      return true; // means index is different
   }

    //==============================================================================

    EveElements.prototype.makeCalo3D = function(calo3D, rnr_data)
    {
       let body = new THREE.BufferGeometry();
       if (rnr_data.vtxBuff) {
          let vBuff = rnr_data.vtxBuff;
          let protoSize = 6 * 2 * 3;
          let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
          let nBox = vBuff.length / 24;
          let idxBuff = [];
          for (let i = 0; i < nBox; ++i) {
             for (let c = 0; c < protoSize; c++) {
                let off = i * 8;
                idxBuff.push(protoIdcs[c] + off);
             }
          }

          body.setAttribute('position', new THREE.BufferAttribute(vBuff, 3));
          body.setIndex(idxBuff);

          let ci = rnr_data.idxBuff;
          let off = 0
          let colBuff = new Float32Array(nBox * 8 * 3);
          for (let x = 0; x < nBox; ++x) {
             let slice = ci[x * 2];
             let sliceColor = calo3D.sliceColors[slice];
             let tc = new THREE.Color(jsrp.getColor(sliceColor));
             for (let i = 0; i < 8; ++i) {
                colBuff[off] = tc.r;
                colBuff[off + 1] = tc.g;
                colBuff[off + 2] = tc.b;
                off += 3;
             }
          }
          body.setAttribute('color', new THREE.BufferAttribute(colBuff, 3));
       }
       let material = new THREE.MeshPhongMaterial({
          color: 0xffffff,
          flatShading: true,
          vertexColors: true,
          shininess: 0
       });


       let mesh = new THREE.Mesh(body, material);
       mesh.get_ctrl = function () { return new Calo3DControl(mesh); };
       mesh.dispose = function () { this.geometry.dispose(); this.material.dispose(); };

       return mesh;
    }

    function Calo3DControl(mesh)
    {
        EveElemControl.call(this, mesh);
    }

    Calo3DControl.prototype = Object.create(EveElemControl.prototype);

   Calo3DControl.prototype.DrawForSelection = function (sec_idcs, res, extra) {
      console.log("CALO 3d draw for selection ", extra);
      let cells;
      for (let i = 0; i < extra.length; i++) {
         if (extra[i].caloVizId == this.obj3d.eve_el.fElementId) {
            cells = extra[i].cells;
            break;
         }
      }

      let ibuff = this.obj3d.eve_el.render_data.idxBuff;
      let nbox = ibuff.length / 2;
      let nBoxSelected = parseInt(cells.length);
      let boxIdcs = new Array;
      for (let i = 0; i < cells.length; i++) {
         let tower = cells[i].t;
         let slice = cells[i].s;

         for (let r = 0; r < nbox; r++) {
            if (ibuff[r * 2] == slice && ibuff[r * 2 + 1] == tower) {
               boxIdcs.push(r);
               break;
            }
         }
      }
      let rnr_data = this.obj3d.eve_el.render_data;
      let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
      let idxBuff = [];
      let vtxBuff = new Float32Array(nbox * 8 * 3);
      for (let i = 0; i < nBoxSelected; ++i) {
         let BoxIdcs = boxIdcs[i];
         for (let c = 0; c < 8; c++) {
            let off = i * 24 + c * 3;
            let pos = BoxIdcs * 24 + c * 3;
            vtxBuff[off] = rnr_data.vtxBuff[pos];
            vtxBuff[off + 1] = rnr_data.vtxBuff[pos + 1];
            vtxBuff[off + 2] = rnr_data.vtxBuff[pos + 2];
         }

         // fix top corners
         for (let c = 0; c < 4; c++) {
            // fix vertex 1
            let pos = BoxIdcs * 24 + c * 3;
            let v1x = rnr_data.vtxBuff[pos];
            let v1y = rnr_data.vtxBuff[pos + 1];
            let v1z = rnr_data.vtxBuff[pos + 2];
            pos += 12;
            let v2x = rnr_data.vtxBuff[pos];
            let v2y = rnr_data.vtxBuff[pos + 1];
            let v2z = rnr_data.vtxBuff[pos + 2];

            let off = i * 24 + 12 + c * 3;
            vtxBuff[off]     = v1x + cells[i].f * (v2x - v1x);
            vtxBuff[off + 1] = v1y + cells[i].f * (v2y - v1y);
            vtxBuff[off + 2] = v1z + cells[i].f * (v2z - v1z);
         }

         for (let c = 0; c < 36; c++) {
            let off = i * 8;
            idxBuff.push(protoIdcs[c] + off);
         }
      } // loop boxes

      let body = new THREE.BufferGeometry();
      body.setAttribute('position', new THREE.BufferAttribute(vtxBuff, 3));
      body.setIndex(idxBuff);

      let mesh = new THREE.Mesh(body);
      res.geom.push(mesh);
   }

    Calo3DControl.prototype.extractIndex = function(intersect)
    {
        let idx  = Math.floor(intersect.faceIndex/12);
        return idx;
    }

    Calo3DControl.prototype.getTooltipText = function(intersect)
    {
        let t = this.obj3d.eve_el.fTitle || this.obj3d.eve_el.fName || "";
        let idx = this.extractIndex(intersect);
        let val =  this.obj3d.eve_el.render_data.nrmBuff[idx];
        let idxBuff = this.obj3d.eve_el.render_data.idxBuff;
        let caloData =  this.obj3d.scene.mgr.GetElement(this.obj3d.eve_el.dataId);
        let slice = idxBuff[idx*2];

        let vbuff =  this.obj3d.eve_el.render_data.vtxBuff;
        let p = idx*24;
        let x = vbuff[p];
        let y = vbuff[p+1];
        let z = vbuff[p+2];

        let phi = Math.acos(x/Math.sqrt(x*x+y*y));
        let cosTheta = z/Math.sqrt(x*x + y*y + z*z);
        let eta = 0;
        if (cosTheta*cosTheta < 1)
        {
            eta = -0.5* Math.log( (1.0-cosTheta)/(1.0+cosTheta) );
        }

        return caloData.sliceInfos[slice].name + "\n" + Math.floor(val*100)/100 +
            " ("+  Math.floor(eta*100)/100 + ", " + Math.floor(phi*100)/100  + ")";
    }

    Calo3DControl.prototype.elementSelected = function(pidx)
    {
        let idx = pidx;
        let calo =  this.obj3d.eve_el;
        let idxBuff = calo.render_data.idxBuff;
        let scene = this.obj3d.scene;
        let selectionId = scene.mgr.global_selection_id;
        let multi = event && event.ctrlKey ? true : false;
        let fcall = "NewTowerPicked(" +  idxBuff[idx*2 + 1] + ", " +  idxBuff[idx*2] + ", "
            + selectionId + ", " + multi + ")";
        scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo3D");
        return true;
    }

    Calo3DControl.prototype.elementHighlighted = function(pidx)
   {
        let idx = pidx;
        let calo =  this.obj3d.eve_el;
        let idxBuff = calo.render_data.idxBuff;
        let scene = this.obj3d.scene;
        let selectionId = scene.mgr.global_highlight_id;
        let fcall = "NewTowerPicked(" +  idxBuff[idx*2 + 1] + ", " +  idxBuff[idx*2] + ", " + selectionId + ", false)";
        scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo3D");
    }

    Calo3DControl.prototype.checkHighlightIndex = function(indx)
    {
        if (this.obj3d && this.obj3d.scene)
        {
            console.log("check highlight idx ?????? \n");
            return this.invokeSceneMethod("processCheckHighlight", indx);

        }

        return true; // means index is different
    }

   //==============================================================================


   EveElements.prototype.makeEveGeometry = function(rnr_data, force)
   {
      if (rnr_data.idxBuff[0] != GL.TRIANGLES)  throw "Expect triangles first.";

      let nVert = 3 * rnr_data.idxBuff[1]; // number of vertices to draw

      if (rnr_data.idxBuff.length != nVert + 2) throw "Expect single list of triangles in index buffer.";

      let body = new THREE.BufferGeometry();
      body.setAttribute('position', new THREE.BufferAttribute( rnr_data.vtxBuff, 3 ));
      body.setIndex(new THREE.BufferAttribute( rnr_data.idxBuff, 1 ));
      body.setDrawRange(2, nVert);
      // this does not work correctly - draw range ignored when calculating normals
      // even worse - shift 2 makes complete logic wrong while wrong triangle are extracted
      // Let see if it will be fixed https://github.com/mrdoob/three.js/issues/15560
      if (body.computeVertexNormalsIdxRange)
         body.computeVertexNormalsIdxRange(2, nVert);

      return body;

      // XXXX Fix this. It seems we could have flat shading with usage of simple shaders.
      // XXXX Also, we could do edge detect on the server for outlines.
      // XXXX a) 3d objects - angle between triangles >= 85 degrees (or something);
      // XXXX b) 2d objects - segment only has one triangle.
      // XXXX Somewhat orthogonal - when we do tesselation, conversion from quads to
      // XXXX triangles is trivial, we could do it before invoking the big guns (if they are even needed).
      // XXXX Oh, and once triangulated, we really don't need to store 3 as number of verts in a poly each time.
      // XXXX Or do we? We might need it for projection stuff.
   }

   EveElements.prototype.makeEveGeoShape = function(egs, rnr_data)
   {
      let geom = this.makeEveGeometry(rnr_data);

      let fcol = jsrp.getColor(egs.fFillColor);

      let material = new THREE.MeshPhongMaterial({// side: THREE.DoubleSide,
                          depthWrite: false, color:fcol, transparent: true, opacity: 0.2 });

      let mesh = new THREE.Mesh(geom, material);

      mesh.get_ctrl = function() { return new EveElemControl(this); };
      mesh.dispose  = function() { this.geometry.dispose(); this.material.dispose(); };

      return mesh;
   }

   EveElements.prototype.makePolygonSetProjected = function(psp, rnr_data)
   {
      let psp_ro = new THREE.Object3D();
      let pos_ba = new THREE.BufferAttribute( rnr_data.vtxBuff, 3 );
      let idx_ba = new THREE.BufferAttribute( rnr_data.idxBuff, 1 );

      let ib_len = rnr_data.idxBuff.length;

      let fcol = jsrp.getColor(psp.fMainColor);
      var line_mat = new THREE.LineBasicMaterial({color:fcol });
      var mesh_mat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, depthWrite: false,
                                                   color:fcol, transparent: true, opacity: 0.4 });

      for (let ib_pos = 0; ib_pos < ib_len; )
      {
         if (rnr_data.idxBuff[ib_pos] == GL.TRIANGLES)
         {
            // Sergey: make check, for now here many wrong values
            let is_ok = true, maxindx = rnr_data.vtxBuff.length/3;
            for (let k=0;is_ok && (k < 3*rnr_data.idxBuff[ib_pos + 1]); ++k)
               if (rnr_data.idxBuff[ib_pos+2+k] > maxindx) is_ok = false;

            if (is_ok) {
               let body = new THREE.BufferGeometry();
               body.setAttribute('position', pos_ba);
               body.setIndex(idx_ba);
               body.setDrawRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);
               if (body.computeVertexNormalsIdxRange)
                  body.computeVertexNormalsIdxRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);

               psp_ro.add( new THREE.Mesh(body, mesh_mat) );
            } else {
               console.log('Error in makePolygonSetProjected - wrong GL.TRIANGLES indexes');
            }

            ib_pos += 2 + 3 * rnr_data.idxBuff[ib_pos + 1];
         }
         else if (rnr_data.idxBuff[ib_pos] == GL.LINE_LOOP)
         {
            let body = new THREE.BufferGeometry();
            body.setAttribute('position', pos_ba);
            body.setIndex(idx_ba);
            body.setDrawRange(ib_pos + 2, rnr_data.idxBuff[ib_pos + 1]);

            psp_ro.add( new THREE.LineLoop(body, line_mat) );

            ib_pos += 2 + rnr_data.idxBuff[ib_pos + 1];
         }
         else
         {
            console.error("Unexpected primitive type " + rnr_data.idxBuff[ib_pos]);
            break;
         }
      }

      psp_ro.get_ctrl = function() { return new EveElemControl(this); };
      psp_ro.dispose  = function() {
         this.children.forEach(c => { c.geometry.dispose(); });
         line_mat.dispose();
         mesh_mat.dispose();
      };


      return psp_ro;
   }


   ////////////////////////////////////////////////////////////////////////////////////////////

   function StraightLineSetControl(mesh)
   {
      EveElemControl.call(this, mesh);
   }

   StraightLineSetControl.prototype = Object.create(EveElemControl.prototype);

   StraightLineSetControl.prototype.separateDraw = true;

   StraightLineSetControl.prototype.cleanup = function()
   {
      if ( ! this.obj3d) return;
      this.drawSpecial(null, undefined, "h");
      this.drawSpecial(null, undefined, "s");
      delete this.obj3d;
   }

   StraightLineSetControl.prototype.extractIndex = function(intersect)
   {
      if (!intersect || intersect.index===undefined) return undefined;

      if (intersect.object.type == "LineSegments") {
         return Math.floor(intersect.index/2);
      }
      else {
         let rnr_data = this.obj3d.eve_el.render_data;
         let idx = intersect.index + this.obj3d.eve_el.fLinePlexSize;
         return rnr_data.idxBuff[idx];
      }
   }

   StraightLineSetControl.prototype.getTooltipText = function(intersect)
   {
      let t = this.obj3d.eve_el.fName || this.obj3d.eve_el.fTitle || "";
      let idx = this.extractIndex(intersect);
      return t + " idx=" + idx;
   }

   StraightLineSetControl.prototype.elementSelected = function(indx)
   {
      this.invokeSceneMethod("processElementSelected", indx);
   }

   StraightLineSetControl.prototype.elementHighlighted = function(indx)
   {
      this.invokeSceneMethod("processElementHighlighted", indx);
   }

   StraightLineSetControl.prototype.checkHighlightIndex = function(indx)
   {
      if (this.obj3d && this.obj3d.scene)
         return this.invokeSceneMethod("processCheckHighlight", indx);

      return true; // means index is different
   }

   // XXXX - DrawForSelection thing is leaky ... need dispose func for what goes into res.
   // XXXX - And then somebody needs to call it, too.

   StraightLineSetControl.prototype.DrawForSelection = function(sec_idcs, res)
    {
      let m     = this.obj3d;
      let index = sec_idcs;
      let geom = new THREE.BufferGeometry();

      geom.setAttribute( 'position', m.children[0].geometry.getAttribute("position") );
      if (index.length == 1)
      {
         geom.setDrawRange(index[0]*2, 2);
      } else if (index.length > 1)
      {
         let idcs = [];
         for (let i = 0; i < index.length; ++i)
            idcs.push(index[i]*2, index[i]*2+1);
         geom.setIndex( idcs );
      }

      let color =  jsrp.getColor(m.eve_el.fMainColor);
      let lineMaterial = new THREE.LineBasicMaterial({ color: color, linewidth: 4 });
      let line         = new THREE.LineSegments(geom, lineMaterial);
      line.matrixAutoUpdate = false;
      line.matrix.fromArray( m.matrix.toArray());
      line.updateMatrixWorld(true);
      res.geom.push(line);

      let el = m.eve_el, mindx = []

      for (let i = 0; i < index.length; ++i)
      {
         if (index[i] < el.fLinePlexSize)
         {
            let lineid = m.eve_idx_buf[index[i]];

            for (let k = 0; k < el.fMarkerPlexSize; ++k )
            {
               if (m.eve_idx_buf[ k + el.fLinePlexSize] == lineid) mindx.push(k);
            }
         }
      }

      if (mindx.length > 0)
      {
         let pnts = new jsrp.PointsCreator(mindx.length, true, 5);

         let arr = m.children[1].geometry.getAttribute("position").array;

         for (let i = 0; i < mindx.length; ++i)
         {
            let p = mindx[i]*3;
            pnts.addPoint(arr[p], arr[p+1], arr[p+2] );
         }
         let mark = pnts.createPoints(color);
         mark.material.size = m.children[1].material.size;
         mark.matrixAutoUpdate = false;
         mark.matrix.fromArray(m.matrix.toArray());
         mark.updateMatrixWorld(true);
         res.geom.push(mark);
      }
   }

   EveElements.prototype.makeStraightLineSet = function(el, rnr_data)
    {
      let obj3d = new THREE.Object3D();

      let mainColor = jsrp.getColor(el.fMainColor);

      let buf = new Float32Array(el.fLinePlexSize * 6);
      for (let i = 0; i < el.fLinePlexSize * 6; ++i)
         buf[i] = rnr_data.vtxBuff[i];
      let lineMaterial = new THREE.LineBasicMaterial({ color: mainColor, linewidth: el.fLineWidth });

      let geom = new THREE.BufferGeometry();
      geom.setAttribute( 'position', new THREE.BufferAttribute( buf, 3 ) );
      let line = new THREE.LineSegments(geom, lineMaterial);
      obj3d.add(line);

      let msize = el.fMarkerPlexSize;
      let pnts  = new jsrp.PointsCreator(msize, true, 3);

      let startIdx = el.fLinePlexSize * 6;
      let endIdx   = startIdx + msize * 3;
      for (let i = startIdx; i < endIdx; i+=3) {
         pnts.addPoint(rnr_data.vtxBuff[i], rnr_data.vtxBuff[i+1], rnr_data.vtxBuff[i+2] );
      }
      let marker = pnts.createPoints(mainColor);

      marker.material.sizeAttenuation = false;

      obj3d.add(marker);

      obj3d.eve_idx_buf = rnr_data.idxBuff;
      let octrl;
      if (el.fSecondarySelect)
         octrl = new StraightLineSetControl(obj3d);
      else
         octrl = new EveElemControl(obj3d);

      line.get_ctrl   = function() { return octrl; };
      marker.get_ctrl = function() { return octrl; };
      obj3d.get_ctrl  = function() { return octrl; };
      obj3d.dispose   = function() {
         this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
      };

      return obj3d;
   }

   //==============================================================================

   JSROOT.EVE.EveElements = EveElements;

   return EveElements;

});
