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


   //==============================================================================
   // makeEveGeometry / makeEveGeoShape
   //==============================================================================

   EveElements.prototype.makeEveGeometry = function(rnr_data, force)
   {
      var nVert = rnr_data.idxBuff[1]*3;

      if (rnr_data.idxBuff[0] != GL.TRIANGLES)  throw "Expect triangles first.";
      if (2 + nVert != rnr_data.idxBuff.length) throw "Expect single list of triangles in index buffer.";

      var body = new RC.Geometry();
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
      var egs_ro = new RC.Group();

      var geom = this.makeEveGeometry(rnr_data);

      var fcol = new RC.Color(JSROOT.Painter.getColor(egs.fFillColor));

      // var material = new RC.MeshPhongMaterial({// side: THREE.DoubleSide,
      //                     depthWrite: false, color:fcol, transparent: true, opacity: 0.2 });
      var material = new RC.MeshPhongMaterial;
      material.color = fcol;
      material.side = 2;
      material.depthWrite = false;
      material.transparent = true;
      material.opacity = 0.2;

      var mesh = new RC.Mesh(geom, material);

      egs_ro.add(mesh);

      return egs_ro;
   }


   //==============================================================================
   // makePolygonSetProjected
   //==============================================================================

   EveElements.prototype.makePolygonSetProjected = function(psp, rnr_data)
   {
      var psp_ro = new RC.Group();
      var pos_ba = new RC.BufferAttribute( rnr_data.vtxBuff, 3 );
      var idx_ba = new RC.BufferAttribute( rnr_data.idxBuff, 1 );

      var ib_len = rnr_data.idxBuff.length;

      var fcol = new RC.Color(JSROOT.Painter.root_colors[psp.fMainColor]);

      var material = new RC.MeshBasicMaterial;
      material.color = fcol;
      material.side = 2;
      material.depthWrite = false;
      material.transparent = true;
      material.opacity = 0.4;

      // XXXXXX Should be Mesh -> Line material ???
      let line_mat = new RC.MeshBasicMaterial;
      line_mat.color = fcol;

      for (var ib_pos = 0; ib_pos < ib_len; )
      {
         if (rnr_data.idxBuff[ib_pos] == GL.TRIANGLES)
         {
            var body = new RC.Geometry();
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
