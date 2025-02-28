sap.ui.define(['rootui5/eve7/lib/EveManager'], function (EveManager)
{
   "use strict";

   // See also EveScene.js makeGLRepresentation(), there several members are
   // set for the top-level Object3D.

   //==============================================================================
   // EveElemControl
   //==============================================================================

   class EveElemControl
   {

      constructor(iobj, tobj)
      {
         this.invoke_obj = iobj;
         this.top_obj = tobj ? tobj : iobj;
      }

      invokeSceneMethod(fname, arg, event)
      {
         if ( ! this.top_obj || ! this.top_obj.eve_el) return false;

         let s = this.top_obj.scene;
         if (s && (typeof s[fname] == "function"))
            return s[fname](this.top_obj.eve_el, arg, event);
         return false;
      }

      getTooltipText()
      {
         let el = this.top_obj.eve_el;
         return el.fTitle || el.fName || "";
      }

      extractIndex(instance)
      {
         return instance;
      }

      elementHighlighted(indx, event)
      {
         // default is simple selection, we ignore the indx
         return this.invokeSceneMethod("processElementHighlighted", indx, event);
      }

      elementSelected(indx, event)
      {
         // default is simple selection, we ignore the indx
         return this.invokeSceneMethod("processElementSelected", indx, event);
      }

      DrawForSelection(sec_idcs, res)
      {
         if (this.top_obj.eve_el.fSecondarySelect) {
            if (sec_idcs.length > 0) {
               res.instance_object = this.top_obj;
               res.instance_sec_idcs = sec_idcs;
               // this.invoke_obj.outlineMaterial.outline_instances_setup(sec_idcs);
            } else {
               // this.invoke_obj.outlineMaterial.outline_instances_reset();
            }
         }
         else
         {
            res.geom.push(this.top_obj);
         }
      }

   } // class EveElemControl


   // ===================================================================================
   // Digit sets control classes
   // ===================================================================================

   class BoxSetControl extends EveElemControl
   {
      DrawForSelection(xsec_idcs, res, extra)
      {
         let sec_idcs = extra.shape_idcs;

         let body = new RC.Geometry();
         body._vertices = this.top_obj.geometry._vertices;
         let origIndices = this.top_obj.geometry._indices;

         let protoIdcsLen = 3 * 12;
         let indicesPerDigit = 8;
         if (this.top_obj.eve_el.boxType == 6) {
            protoIdcsLen = 3 * 24;
            indicesPerDigit = 14;
         }

         let idxBuff = [];

         let N = this.top_obj.eve_el.render_data.idxBuff.length / 2;
         for (let b = 0; b < sec_idcs.length; ++b) {
            let idx = sec_idcs[b];
            let idxOff = idx * indicesPerDigit;
            for (let i = 0; i < protoIdcsLen; i++) {
               idxBuff.push(idxOff + origIndices.array[i]);
            }
         }

         body.indices = RC.Uint32Attribute(idxBuff, 1);
         let mesh = new RC.Mesh(body, null);
         mesh._modelViewMatrix = this.invoke_obj._modelViewMatrix;
         mesh._normalMatrix    = this.invoke_obj._normalMatrix;
         mesh._material        = this.invoke_obj._material;

         res.geom.push(mesh);
      }

      extractIndex(instance) {
         let verticesPerDigi = 8;
         if (this.top_obj.eve_el.boxType == 6)
            verticesPerDigi = 14;
         let idx = Math.floor(instance / verticesPerDigi);
         return idx;
      }

      elementSelectedSendMIR(idx, selectionId, event)
      {
         let boxset = this.top_obj.eve_el;
         let scene = this.top_obj.scene;
         let multi = event?.ctrlKey ? true : false;

         let boxIdx = idx;

         let fcall = "NewShapePicked(" + boxIdx + ", " + selectionId + ", " + multi + ")"
         scene.mgr.SendMIR(fcall, boxset.fElementId, "ROOT::Experimental::REveDigitSet");
         return true;
      }

      elementSelected(idx, event)
      {
         return this.elementSelectedSendMIR(idx, this.top_obj.scene.mgr.global_selection_id, event);
      }

      elementHighlighted(idx, event)
      {
         return this.elementSelectedSendMIR(idx, this.top_obj.scene.mgr.global_highlight_id, event);
      }

      checkHighlightIndex(idx) // XXXX ?? MT Sept-2022
      {
         if (this.top_obj && this.top_obj.scene)
            return this.invokeSceneMethod("processCheckHighlight", idx);

         return true; // means index is different
      }

   } // class BoxSetControl


   // ===================================================================================
   // Calorimeter control classes
   // ===================================================================================

   class Calo3DControl extends EveElemControl
   {
      DrawForSelection(sec_idcs, res, extra)
      {
         console.log("CALO 3d draw for selection ", extra);
         let eve_el = this.invoke_obj.eve_el;
         // locate REveCaloData cells for this object
         let cells;
         for (let i = 0; i < extra.length; i++) {
            if (extra[i].caloVizId == eve_el.fElementId) {
               cells = extra[i].cells;
               break;
            }
         }

         let rnr_data = eve_el.render_data;
         let ibuff = rnr_data.idxBuff;
         let nbox = ibuff.length / 2;
         let nBoxSelected = parseInt(cells.length);
         let boxIdcs = [];
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
         let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
         let idxBuff = [];
         let vtxBuff = new Float32Array(nBoxSelected * 8 * 3);
         for (let i = 0; i < nBoxSelected; ++i)
         {
            let box_idx = boxIdcs[i];
            for (let c = 0; c < 8; c++) {
               let off = i * 24 + c * 3;
               let pos = box_idx * 24 + c * 3;
               vtxBuff[off] = rnr_data.vtxBuff[pos];
               vtxBuff[off + 1] = rnr_data.vtxBuff[pos + 1];
               vtxBuff[off + 2] = rnr_data.vtxBuff[pos + 2];
            }

            // fix top corners, select can be partial
            for (let c = 0; c < 4; c++) {
               // fix vertex 1
               let pos = box_idx * 24 + c * 3;
               let v1x = rnr_data.vtxBuff[pos];
               let v1y = rnr_data.vtxBuff[pos + 1];
               let v1z = rnr_data.vtxBuff[pos + 2];
               pos += 12;
               let v2x = rnr_data.vtxBuff[pos];
               let v2y = rnr_data.vtxBuff[pos + 1];
               let v2z = rnr_data.vtxBuff[pos + 2];

               let off = i * 24 + 12 + c * 3;
               vtxBuff[off] = v1x + cells[i].f * (v2x - v1x);
               vtxBuff[off + 1] = v1y + cells[i].f * (v2y - v1y);
               vtxBuff[off + 2] = v1z + cells[i].f * (v2z - v1z);
            }

            for (let c = 0; c < 36; c++) {
               let off = i * 8;
               idxBuff.push(protoIdcs[c] + off);
            }
         } // loop boxes

         let body = new RC.Geometry();
         body.indices = RC.Uint32Attribute(idxBuff, 1);
         body.vertices = new RC.BufferAttribute(vtxBuff, 3); // this.invoke_obj.geometry.vertices;
         body.computeVertexNormals(); // XX should not need it when we have dFdx/y

         let mesh = new RC.Mesh(body, null);
         mesh._modelViewMatrix = this.invoke_obj._modelViewMatrix;
         mesh._normalMatrix    = this.invoke_obj._normalMatrix;
         mesh._material        = this.invoke_obj._material;

         res.geom.push(mesh);

         // console.log(body, mesh, res);
      }

      extractIndex(instance)
      {
         return Math.floor(instance / 8);
      }

      getTooltipText(idx)
      {
         // let t = this.obj3d.eve_el.fTitle || this.obj3d.eve_el.fName || "";
         let eve_el = this.top_obj.eve_el;
         let val =  eve_el.render_data.nrmBuff[idx];
         let idxBuff = eve_el.render_data.idxBuff;
         let caloData = this.top_obj.scene.mgr.GetElement(eve_el.dataId);
         let slice = idxBuff[idx*2];

         let vbuff =  eve_el.render_data.vtxBuff;
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

      elementSelected(idx, event)
      {
           let calo =  this.top_obj.eve_el;
           let idxBuff = calo.render_data.idxBuff;
           let scene = this.top_obj.scene;
           let selectionId = scene.mgr.global_selection_id;
           let multi = event?.ctrlKey ? true : false;
           let fcall = "NewTowerPicked(" +  idxBuff[idx*2 + 1] + ", " +  idxBuff[idx*2] + ", "
               + selectionId + ", " + multi + ")";
           scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo3D");
           return true;
      }

      elementHighlighted(idx, event)
      {
           let calo =  this.top_obj.eve_el;
           let idxBuff = calo.render_data.idxBuff;
           let scene = this.top_obj.scene;
           let selectionId = scene.mgr.global_highlight_id;
           let fcall = "NewTowerPicked(" +  idxBuff[idx*2 + 1] + ", " +  idxBuff[idx*2] + ", " + selectionId + ", false)";
           scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo3D");
       }

      checkHighlightIndex(idx)
      {
         if (this.top_obj && this.top_obj.scene)
         {
            console.log("check highlight idx ?????? \n");
            return this.invokeSceneMethod("processCheckHighlight", idx);

         }

         return true; // means index is different
      }

   } // class Calo3DControl


   class Calo2DControl extends EveElemControl
   {
      DrawForSelection(sec_idcs, res, extra)
      {
         let eve_el = this.invoke_obj.eve_el;
         let cells;
         for (let i = 0; i < extra.length; i++) {
            if (extra[i].caloVizId == eve_el.fElementId) {
               cells = extra[i].cells;
               break;
            }
         }
         let rnr_data = eve_el.render_data;
         let ibuff = rnr_data.idxBuff;
         let vbuff = rnr_data.vtxBuff;
         let nbox = ibuff.length / 2;
         let nBoxSelected = cells.length;
         let boxIdcs = [];
         for (let i = 0; i < cells.length; i++) {
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
         let vtxBuff = new Float32Array(nBoxSelected * 4 * 3);
         let protoIdcs = [0, 1, 2, 2, 3, 0];
         for (let i = 0; i < nBoxSelected; ++i) {
            let BoxIdcs = boxIdcs[i];
            for (let v = 0; v < 4; v++) {
               let off = i * 12 + v * 3;
               let pos = BoxIdcs * 12 + v * 3;
               vtxBuff[off] = rnr_data.vtxBuff[pos];
               vtxBuff[off + 1] = rnr_data.vtxBuff[pos + 1];
               vtxBuff[off + 2] = rnr_data.vtxBuff[pos + 2];
            }
            {
               // fix vertex 1
               let pos = BoxIdcs * 12;
               let v1x = rnr_data.vtxBuff[pos];
               let v1y = rnr_data.vtxBuff[pos + 1];
               pos += 3;
               let v2x = rnr_data.vtxBuff[pos];
               let v2y = rnr_data.vtxBuff[pos + 1];
               let off = i * 12 + 3;
               vtxBuff[off] = v1x + cells[i].f * (v2x - v1x);
               vtxBuff[off + 1] = v1y + cells[i].f * (v2y - v1y);
            }

            {
               // fix vertex 2
               let pos = BoxIdcs * 12 + 3 * 3;
               let v1x = rnr_data.vtxBuff[pos];
               let v1y = rnr_data.vtxBuff[pos + 1];
               pos -= 3;
               let v2x = rnr_data.vtxBuff[pos];
               let v2y = rnr_data.vtxBuff[pos + 1];
               let off = i * 12 + 3 * 2;
               vtxBuff[off] = v1x + cells[i].f * (v2x - v1x);
               vtxBuff[off + 1] = v1y + cells[i].f * (v2y - v1y);
            }
            for (let c = 0; c < 6; c++) {
               let off = i * 4;
               idxBuff.push(protoIdcs[c] + off);
            }
         }

         let body = new RC.Geometry();
         body.indices = RC.Uint32Attribute(idxBuff, 1);
         body.vertices = new RC.BufferAttribute(vtxBuff, 3);
         body.computeVertexNormals();

         let mesh = new RC.Mesh(body, null);
         mesh._modelViewMatrix = this.invoke_obj._modelViewMatrix;
         mesh._normalMatrix    = this.invoke_obj._normalMatrix;
         mesh._material        = this.invoke_obj._material;

         res.geom.push(mesh);
      }

      extractIndex(instance)
      {
         return Math.floor(instance / 4);
      }

      getTooltipText(idx)
      {
         let eve_el = this.top_obj.eve_el;
         let idxBuff = eve_el.render_data.idxBuff;
         // let bin =  idxBuff[idx*2 + 1];
         let val = eve_el.render_data.nrmBuff[idx];
         let slice = idxBuff[idx*2];
         let sname = "Slice " + slice;

         let vbuff =  eve_el.render_data.vtxBuff;
         let p = idx*12;
         let x = vbuff[p];
         let y = vbuff[p+1];
         // let z = vbuff[p+2];

         if (eve_el.isRPhi) {
            let phi =  Math.acos(x/Math.sqrt(x*x+y*y)) * Math.sign(y);
            return  sname + " " + Math.floor(val*100)/100 +
                  " ("+  Math.floor(phi*100)/100 + ")";
         }
         else
         {
            let cosTheta = x/Math.sqrt(x*x + y*y), eta = 0;
            if (cosTheta*cosTheta < 1)
            {
                  eta = -0.5* Math.log( (1.0-cosTheta)/(1.0+cosTheta) );
            }

            return  sname + " " + Math.floor(val*100)/100 +
                  " ("+  Math.floor(eta*100)/100 + ")";
         }
      }

      elementSelectedSendMIR(idx, selectionId, event)
      {
         let calo =  this.top_obj.eve_el;
         let idxBuff = calo.render_data.idxBuff;
         let scene = this.top_obj.scene;
         let multi = event?.ctrlKey ? true : false;
         let bin = idxBuff[idx*2 + 1];
         let slice =  idxBuff[idx*2];
         // get sign for the case of RhoZ projection
         if (calo.render_data.vtxBuff[idx*12 + 1] < 0) bin = -bin ;

         let fcall = "NewBinPicked(" +  bin + ", " +  slice + ", " + selectionId + ", " + multi + ")"
         scene.mgr.SendMIR(fcall, calo.fElementId, "ROOT::Experimental::REveCalo2D");
         return true;
      }

     elementSelected(idx, event)
     {
        return this.elementSelectedSendMIR(idx, this.top_obj.scene.mgr.global_selection_id, event);
     }

     elementHighlighted(idx, event)
     {
        return this.elementSelectedSendMIR(idx, this.top_obj.scene.mgr.global_highlight_id, event);
     }

   } // class Calo2Control

   //==============================================================================


   class GeoTopNodeControl extends EveElemControl {
     
      addMeshRec(o3, res){
         for (let c of o3.children)
         {
            if (c.material) {
               console.log("add mesh ", c);
               res.geom.push(c);
            }
            this.addMeshRec(c, res);
         }
      }
      DrawForSelection(sec_idcs, res, extra) {

         if (extra.stack.length > 0) {
            let topNode = this.top_obj;
            let stack = extra.stack;
            let clones = topNode.clones;

            // NOTE: this needs to be done diffeewnrly, this code is related to objects3d
            // TODO: make same logic fro RC objects 
            let x = topNode.clones.createRCObject3D(stack, topNode, 'force');

            // console.log("geo topnode control res = ",x);
            this.addMeshRec(x, res);
         }
      }
      getTooltipText() {
         return this.top_obj.eve_el.fName;
      }
      extractIndex(instance) {
         this.pick = instance;
      }

      sendSocketMassage(pstate_obj, t1, t2)
      {
         let topNode = this.top_obj;
         let aa = pstate_obj.stack || [];

         let mgr =  topNode.scene.mgr;
         let hbr = mgr.GetElement(topNode.eve_el.dataId);

         if (!hbr.hasOwnProperty("websocket"))
         {
            let websocket = mgr.handle.createChannel();
            mgr.handle.send("SETCHANNEL:" + hbr.fElementId + "," + websocket.getChannelId());
            hbr.websocket =  websocket;
         }

         let name = topNode.clones.getStackName(aa);
         const myArray = name.split("/");
         let msg = '[';
         let lastIdx = myArray.length - 1;
         for (let p = 0; p < myArray.length; ++p) {
            let np = "\"" + myArray[p] + "\"";
            msg += np;
            if (p == lastIdx)
               msg += ']';
            else
               msg += ",";

         }

         hbr.websocket.sendLast(t1, 200, t2 + msg);
      }

      elementSelected(idx, event, pstate_obj) {
         this.sendSocketMassage(pstate_obj, 'click', 'CLICK:');
      }

      elementHighlighted(idx, event, pstate_obj) {
         this.sendSocketMassage( pstate_obj, 'hover', 'HOVER:');
      }
   }

   //==============================================================================
   // EveElements
   //==============================================================================

   const GL = { POINTS: 0, LINES: 1, LINE_LOOP: 2, LINE_STRIP: 3, TRIANGLES: 4 };
   let RC;

   function RcCol(root_col)
   {
      return new RC.Color(EVE.JSR.getColor(root_col));
   }

   //------------------------------------------------------------------------------
   // Builder functions of this class are called by EveScene to create RCore
   // objects representing an EveElement. They can have children if multiple RCore
   // objects are required (e.g., mesh + lines + points).
   //
   // The top-level object returned by these builder functions will get additional
   // properties injected by EveScene:
   // - eve_el
   // - scene.
   //
   // Object picking functions in GlViewerRCore will navigate up the parent hierarchy
   // until an object with eve_el property is set.
   // If secondary selection is enabled on the eve_el, instance picking will be called
   // as well and the returned ID will be used as the index for secondary selection.
   // This can be overriden by setting get_ctrl property of any RCore object to a function
   // that takes a reference to the said argument and returns an instance of class
   // EveElemControl.
   // get_ctrl property needs to be set at least at the top-level object.

   class EveElements
   {
      constructor(rc, viewer)
      {
         if (viewer._logLevel >= 2)
            console.log("EveElements -- RCore instantiated.");

         RC = rc;
         this.viewer = viewer;
         this.tex_cache = viewer.tex_cache;

         this.POINT_SIZE_FAC = 1;
         this.LINE_WIDTH_FAC = 1;
         this.ColorWhite = new RC.Color(0xFFFFFF);
         this.ColorBlack = new RC.Color(0x000000);
      }

      //----------------------------------------------------------------------------
      // Helper functions
      //----------------------------------------------------------------------------

      GenerateTypeName(obj) { return "RC." + obj.type; }

      SetupPointLineFacs(ssaa, pf, lf)
      {
         this.SSAA = ssaa; // to scale down points / lines for picking and outlines
         this.POINT_SIZE_FAC = pf;
         this.LINE_WIDTH_FAC = lf;
      }

      UpdatePointPickingMaterial(obj)
      {
         let m = obj.material;
         let p = obj.pickingMaterial;
         p.usePoints = m.usePoints;
         p.pointSize = m.pointSize;
         p.pointsScale = m.pointsScale;
         p.drawCircles = m.drawCircles;
      }

      RcCol(root_col)
      {
         return RcCol(root_col);
      }

      RcPointMaterial(color, opacity, point_size, props)
      {
         let mat = new RC.PointBasicMaterial;
         mat._color = this.ColorBlack; // color;
         mat._emissive = color;
         if (opacity !== undefined && opacity < 1.0) {
            mat._opacity = opacity;
            mat._transparent = true;
            mat._depthWrite = false;
         }
         mat._pointSize = this.POINT_SIZE_FAC;
         if (point_size !== undefined) mat._pointSize *= point_size;
         if (props !== undefined) {
            mat.update(props);
         }
         return mat;
      }

      RcLineMaterial(color, opacity, line_width, props)
      {
         let mat = new RC.MeshBasicMaterial; // StripeBasicMaterial
         mat._color = this.ColorBlack;
         mat._emissive = color;
         if (opacity !== undefined && opacity < 1.0) {
            mat._opacity = opacity;
            mat._transparent = true;
            mat._depthWrite = false;
         }
         mat.lineWidth = this.LINE_WIDTH_FAC;
         if (line_width !== undefined) mat.lineWidth *= line_width;
         if (props !== undefined) {
            mat.update(props);
         }
         return mat;
      }

      RcFlatMaterial(color, opacity, props)
      {
         let mat = new RC.MeshBasicMaterial;
         mat._color = color;
         mat._emissive = color; // mat.emissive.offsetHSL(0, 0.1, 0);
         // Something is strange here. Tried also white (no change) / black (no fill -- ?).
         // mat._emissive = new RC.Color(color);
         // mat.emissive.multiplyScalar(0.1);
         // offsetHSL(0, -0.5, -0.5);

         if (opacity !== undefined && opacity < 1.0) {
            mat._opacity = opacity;
            mat._transparent = true;
            mat._depthWrite = false;
         }
         if (props !== undefined) {
            mat.update(props);
         }
         return mat;
      }

      RcFancyMaterial(color, opacity, props)
      {
         let mat = new RC.MeshPhongMaterial;
         // let mat = new RC.MeshBasicMaterial;

         mat._color = color;
         mat._specular = new RC.Color(0.3, 0.4, 0.3); // this.ColorWhite;
         mat._shininess = 64;

         if (opacity !== undefined && opacity < 1.0) {
            mat._opacity = opacity;
            mat._transparent = true;
            mat._depthWrite = false;
         }
         if (props !== undefined) {
            mat.update(props);
         }
         return mat;
      }

      RcMakeZSprite(colIdx, sSize, nInstance, vbuff, instX, instY, textureName)
      {
         let col = RcCol(colIdx);
         sSize *= this.POINT_SIZE_FAC;
         let sm = new RC.ZSpriteBasicMaterial( {
            SpriteMode: RC.SPRITE_SPACE_SCREEN, SpriteSize: [sSize, sSize],
            color: this.ColorBlack,
            emissive: col,
            diffuse: col.clone().multiplyScalar(0.5) } );
         sm.transparent = true;

         sm.addInstanceData(new RC.Texture(vbuff,
            RC.Texture.WRAPPING.ClampToEdgeWrapping, RC.Texture.WRAPPING.ClampToEdgeWrapping,
            RC.Texture.FILTER.NearestFilter, RC.Texture.FILTER.NearestFilter,
            // RC.Texture.FORMAT.R32F, RC.Texture.FORMAT.R32F, RC.Texture.TYPE.FLOAT,
            RC.Texture.FORMAT.RGBA32F, RC.Texture.FORMAT.RGBA, RC.Texture.TYPE.FLOAT,
            instX, instY));
         sm.instanceData[0].flipy = false;

         let s = new RC.ZSprite(null, sm);
         s.frustumCulled = false; // need a way to speciy bounding box/sphere !!!
         s.instanced = true;
         s.instanceCount = nInstance;

         // Now that outline and picking shaders are setup with final pixel-size,
         // scale up the main size to account for SSAA.
         sSize *= this.SSAA;
         sm.setUniform("SpriteSize", [sSize, sSize]);

         this.GetLumAlphaTexture(textureName, this.AddMapToAllMaterials.bind(this, s));

         return s;
      }

      RcMakeStripes(geom, line_width, line_color)
      {
         // Setup width for SSAA, scaled down for picking and outline materials.
         let s = new RC.Stripes(
            { geometry: new RC.StripesGeometry({ baseGeometry: geom }),
              material: new RC.StripesBasicMaterial({
                           baseGeometry: geom, mode: RC.STRIPE_SPACE_SCREEN,
                           lineWidth: line_width * this.LINE_WIDTH_FAC * this.SSAA,
                           color: this.ColorBlack,
                           emissive: line_color
                        }),
              GBufferMaterial: null
            }
         );
         s.lights = false;
         return s;
      }

      RcApplyStripesMaterials(eve_el, stripes, pick_width_scale = 2)
      {
         if (eve_el.fPickable) {
            let m = stripes.material;
            stripes.pickingMaterial = new RC.StripesBasicMaterial(
               { lineWidth: m.lineWidth * pick_width_scale / this.SSAA,
                 mode: m.mode, color: m.color });
            let pm = stripes.pickingMaterial;
            pm.programName = "custom_GBufferMini_stripes";
            pm.addSBFlag("PICK_MODE_UINT");
            pm.prevVertex = m.prevVertex;
            pm.nextVertex = m.nextVertex;
            pm.deltaOffset = m.deltaOffset;

            stripes.outlineMaterial = new RC.StripesBasicMaterial(
               { lineWidth: m.lineWidth / this.SSAA, mode: m.mode, color: m.color });
            let om = stripes.outlineMaterial;
            om.programName = "custom_GBufferMini_stripes";
            om.prevVertex = m.prevVertex;
            om.nextVertex = m.nextVertex;
            om.deltaOffset = m.deltaOffset;
         }
      }

      RcPickable(el, obj3d, do_children = true, ctrl_class = EveElemControl)
      {
         if (el.fPickable) {
            if (ctrl_class) {
               obj3d.get_ctrl = function(iobj, tobj) { return new ctrl_class(iobj, tobj); }
            }
            obj3d.pickable = true;
            if (do_children) {
               for (let i = 0; i < obj3d.children.length; ++i)
                  obj3d.children[i].pickable = true;
            }
            // using RCore auto-id to get Object3D that got picked.
            return true;
         } else {
            return false;
         }
      }

      TestRnr(name, obj, rnr_data)
      {
         if (obj && rnr_data && rnr_data.vtxBuff) {
            return false;
         }
         // console.log("test rnr failed for obj =  ", obj);
         // console.log("test rnr failed for rnr_data =  ", rnr_data);
         // console.log("test rnr failed for rnr_data vtcBuff =  ", rnr_data.vtxBuff);

         let cnt = this[name] || 0;
         if (cnt++ < 1) console.log(name, obj, rnr_data);
         this[name] = cnt;
         return true;
      }

      GetLumAlphaTexture(name, callback)
      {
         let url = this.viewer.eve_path + 'textures/' + name;

         this.tex_cache.deliver(url,
            callback,
            (image) => {
               return new RC.Texture
                  (image, RC.Texture.WRAPPING.ClampToEdgeWrapping, RC.Texture.WRAPPING.ClampToEdgeWrapping,
                          RC.Texture.FILTER.LinearFilter, RC.Texture.FILTER.LinearFilter,
                          RC.Texture.FORMAT.LUMINANCE_ALPHA, RC.Texture.FORMAT.LUMINANCE_ALPHA,
                          RC.Texture.TYPE.UNSIGNED_BYTE,
                          image.width, image.height);
            },
            () => { this.viewer.request_render() }
         );
      }

      GetRgbaTexture(name, callback)
      {
         let url = this.viewer.eve_path + 'textures/' + name;

         this.tex_cache.deliver(url,
            callback,
            (image) => {
               return new RC.Texture
                  (image, RC.Texture.WRAPPING.ClampToEdgeWrapping, RC.Texture.WRAPPING.ClampToEdgeWrapping,
                          RC.Texture.FILTER.LinearFilter, RC.Texture.FILTER.LinearFilter,
                          RC.Texture.FORMAT.RGBA, RC.Texture.FORMAT.RGBA,
                          RC.Texture.TYPE.UNSIGNED_BYTE,
                          image.width, image.height);
            },
            () => { this.viewer.request_render() }
         );
      }

      AddMapToAllMaterials(o3d, tex)
      {
         if (o3d.material) o3d.material.addMap(tex);
         if (o3d.pickingMaterial) o3d.pickingMaterial.addMap(tex);
         if (o3d.outlineMaterial) o3d.outlineMaterial.addMap(tex);
      }

      AddTextureToMaterialMap(o3d, tex)
      {
         if (o3d.material)
         {
            o3d.material.clearMaps();
            o3d.material.addMap(tex);
         }
      }

      //----------------------------------------------------------------------------
      // Builder functions
      //----------------------------------------------------------------------------

      //==============================================================================
      // makeHit
      //==============================================================================

      makeHit(hit, rnr_data)
      {
         if (this.TestRnr("hit", hit, rnr_data)) return null;

         let txName;
         if (hit.fMarkerStyle == 3)
            txName = "star5-32a.png";
         else if (hit.fMarkerStyle == 2)
            txName = "square-32a.png";
         else
            txName = "dot-32a.png"

         let s = this.RcMakeZSprite(hit.fMarkerColor, hit.fMarkerSize, hit.fSize,
            rnr_data.vtxBuff, hit.fTexX, hit.fTexY,
            txName);

         this.RcPickable(hit, s);

         return s;
      }

      //==============================================================================
      // makeTrack
      //==============================================================================

      makeTrack(track, rnr_data)
      {
         if (this.TestRnr("track", track, rnr_data)) return null;

         let N = rnr_data.vtxBuff.length / 3;
         let track_width = 2 * track.fLineWidth;
         let track_color = RcCol(track.fLineColor);

         // if (EVE.JSR.browser.isWin) track_width = 1;  // not supported on windows

         let buf = new Float32Array((N - 1) * 6), pos = 0;
         for (let k = 0; k < (N - 1); ++k)
         {
            buf[pos] = rnr_data.vtxBuff[k * 3];
            buf[pos + 1] = rnr_data.vtxBuff[k * 3 + 1];
            buf[pos + 2] = rnr_data.vtxBuff[k * 3 + 2];

            let breakTrack = false;
            if (rnr_data.idxBuff)
               for (let b = 0; b < rnr_data.idxBuff.length; b++)
               {
                  if ((k + 1) == rnr_data.idxBuff[b])
                  {
                     breakTrack = true;
                     break;
                  }
               }

            if (breakTrack)
            {
               buf[pos + 3] = rnr_data.vtxBuff[k * 3];
               buf[pos + 4] = rnr_data.vtxBuff[k * 3 + 1];
               buf[pos + 5] = rnr_data.vtxBuff[k * 3 + 2];
            }
            else
            {
               buf[pos + 3] = rnr_data.vtxBuff[k * 3 + 3];
               buf[pos + 4] = rnr_data.vtxBuff[k * 3 + 4];
               buf[pos + 5] = rnr_data.vtxBuff[k * 3 + 5];
            }

            pos += 6;
         }

         const geom = new RC.Geometry();
         geom.vertices = new RC.Float32Attribute(buf, 3);

         const line = this.RcMakeStripes(geom, track_width, track_color);
         this.RcApplyStripesMaterials(track, line, 2);
         this.RcPickable(track, line);

         return line;
      }

      //==============================================================================
      // makeZText
      //==============================================================================

      makeZText(el, rnr_data)
      {
         // if (this.TestRnr("jet", el, rnr_data)) return null;

         let text = new RC.ZText({
            text: el.fText,
            xPos: el.fPosX,
            yPos: el.fPosY,
            fontSize: el.fFontSize,
            mode: el.fMode,
            fontHinting: el.fFontHinting,
            color: RcCol(el.fTextColor),
         });
         let url_base = this.viewer.top_path + 'sdf-fonts/' + el.fFont;
         this.tex_cache.deliver_font(url_base,
            (texture, font_metrics) => {
               text.setupFrameStuff((100 - el.fMainTransparency) / 100.0, el.fDrawFrame,
                                    RcCol(el.fFillColor), el.fFillAlpha / 255.0,
                                    RcCol(el.fLineColor), el.fLineAlpha / 255.0,
                                    el.fExtraBorder, el.fLineWidth);
               text.setTextureAndFont(texture, font_metrics);
               if (el.fMode == 0) text.material.side = RC.FRONT_AND_BACK_SIDE;
            },
            (img) => RC.ZText.createDefaultTexture(img),
            () => this.viewer.request_render()
         );

        text.position.copy(new RC.Vector3(el.fPosX, el.fPosY, el.fPosZ));
        if (el.fPickable) this.RcPickable(el, text);
        return text;
      }

      //==============================================================================
      // makeJet
      //==============================================================================

      makeJet(jet, rnr_data)
      {
         if (this.TestRnr("jet", jet, rnr_data)) return null;

         // console.log("make jet ", jet);
         // let jet_ro = new RC.Object3D();
         let pos_ba = new RC.BufferAttribute(rnr_data.vtxBuff, 3);
         let N = rnr_data.vtxBuff.length / 3;

         let geo_body = new RC.Geometry();
         geo_body.vertices = pos_ba;
         let idcs = new Uint32Array(3 + 3 * (N - 2));
         idcs[0] = 0; idcs[1] = N - 1; idcs[2] = 1;
         for (let i = 1; i < N - 1; ++i)
         {
            idcs[3 * i] = 0; idcs[3 * i + 1] = i; idcs[3 * i + 2] = i + 1;
            // idcs.push( 0, i, i + 1 );
         }
         geo_body.indices = new RC.BufferAttribute(idcs, 1);
         // geo_body.computeVertexNormals();

         let geo_rim = new RC.Geometry();
         geo_rim.vertices = pos_ba;
         idcs = new Uint32Array(N - 1);
         for (let i = 1; i < N; ++i) idcs[i - 1] = i;
         geo_rim.indices = new RC.BufferAttribute(idcs, 1);

         let geo_rays = new RC.Geometry();
         geo_rays.vertices = pos_ba;
         idcs = new Uint32Array(2 * (1 + ((N - 1) / 4)));
         let p = 0;
         for (let i = 1; i < N; i += 4)
         {
            idcs[p++] = 0; idcs[p++] = i;
         }
         geo_rays.indices = new RC.BufferAttribute(idcs, 1);

         let mcol = RcCol(jet.fMainColor);
         let lcol = RcCol(jet.fLineColor);

         let mesh = new RC.Mesh(geo_body, this.RcFancyMaterial(mcol, 0.5, { side: RC.FRONT_AND_BACK_SIDE }));
         mesh.material.normalFlat = true;

         let line1 = new RC.Line(geo_rim, this.RcLineMaterial(lcol, 0.8, 4));
         line1.renderingPrimitive = RC.LINE_LOOP;

         let line2 = new RC.Line(geo_rays, this.RcLineMaterial(lcol, 0.8, 1));
         line2.renderingPrimitive = RC.LINES;

         mesh.add(line1);
         mesh.add(line2);
         this.RcPickable(jet, mesh, false);

         return mesh;
      }

      makeJetProjected(jet, rnr_data)
      {
         // JetProjected has 3 or 4 points. 0-th is apex, others are rim.
         // Fourth point is only present in RhoZ when jet hits barrel/endcap transition.

         // console.log("makeJetProjected ", jet);

         if (this.TestRnr("jetp", jet, rnr_data)) return null;

         let pos_ba = new RC.BufferAttribute(rnr_data.vtxBuff, 3);
         let N = rnr_data.vtxBuff.length / 3;

         let geo_body = new RC.Geometry();
         geo_body.vertices = pos_ba;
         let idcs = new Uint32Array(N > 3 ? 6 : 3);
         idcs[0] = 0; idcs[1] = 1; idcs[2] = 2;
         if (N > 3) { idcs[3] = 0; idcs[4] = 2; idcs[5] = 3; }
         geo_body.indices = new RC.BufferAttribute(idcs, 1);
         // geo_body.computeVertexNormals();

         let geo_rim = new RC.Geometry();
         geo_rim.vertices = pos_ba;
         idcs = new Uint32Array(N - 1);
         for (let i = 1; i < N; ++i) idcs[i - 1] = i;
         geo_rim.indices = new RC.BufferAttribute(idcs, 1);

         let geo_rays = new RC.Geometry();
         geo_rays.vertices = pos_ba;
         idcs = new Uint32Array(4); // [ 0, 1, 0, N-1 ];
         idcs[0] = 0; idcs[1] = 1; idcs[2] = 0; idcs[3] = N - 1;
         geo_rays.indices = new RC.BufferAttribute(idcs, 1);;

         let fcol = RcCol(jet.fFillColor);
         let lcol = RcCol(jet.fLineColor);
         // Process transparency !!!

         let mesh = new RC.Mesh(geo_body, this.RcFlatMaterial(fcol, 0.5));
         mesh.material.normalFlat = true;

         let line1 = this.RcMakeStripes(geo_rim,  2, lcol);
         let line2 = this.RcMakeStripes(geo_rays, 1, lcol);
         mesh.add(line1);
         mesh.add(line2);
         this.RcPickable(jet, mesh, false);

         return mesh;
      }

      makeFlatBox(ebox, rnrData, idxBegin, idxEnd)
      {
         let fcol = RcCol(ebox.fMainColor);
         let boxMaterial = this.RcFancyMaterial(fcol, 0.5, { side: RC.FRONT_AND_BACK_SIDE });

         // console.log("EveElements.prototype.makeFlatBox triangulate", idxBegin, idxEnd);
         let nTriang = (idxEnd - idxBegin) - 2;
         let idxBuff = new Uint32Array(nTriang * 3);
         let nt = 0;
         for (let i = idxBegin; i < (idxEnd - 2); ++i) {
            idxBuff[nt * 3] = idxBegin;
            idxBuff[nt * 3 + 1] = i + 1;
            idxBuff[nt * 3 + 2] = i + 2;
            // console.log("set index ", nt,":", idxBuff[nt*3], idxBuff[nt*3+1],idxBuff[nt*3+2]);
            nt++;
         }

         let body = new RC.Geometry();
         body.vertices = new RC.BufferAttribute(rnrData.vtxBuff, 3);
         body.indices = new RC.BufferAttribute(idxBuff, 1);
         //body.computeVertexNormals();
         let mesh = new RC.Mesh(body, boxMaterial);
         return mesh;
      }

      makeBoxProjected(ebox, rnrData)
      {
         let nPnts = parseInt(rnrData.vtxBuff.length / 3);
         let breakIdx = parseInt(ebox.fBreakIdx);
         if (ebox.fBreakIdx == 0)
            breakIdx = nPnts;

         let mesh1 = this.makeFlatBox(ebox, rnrData, 0, breakIdx);
         let testBreak = breakIdx + 2;
         if (testBreak < nPnts) {
            let mesh2 = this.makeFlatBox(ebox, rnrData, breakIdx, nPnts);
            mesh2.get_ctrl = function () { return new EveElemControl(this); }
            mesh1.add(mesh2);
         }

         mesh1.get_ctrl = function () { return new EveElemControl(this); };
         mesh1.dispose = function () {
            this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
            this.geometry.dispose(); this.material.dispose();
         };

         return mesh1;
      }

      makeBox(ebox, rnr_data)
      {
         let idxBuff = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
         let vBuff = rnr_data.vtxBuff;

         let body = new RC.Geometry();
         body.indices = new RC.BufferAttribute(new Uint32Array(idxBuff), 1);
         body.vertices = new RC.BufferAttribute(vBuff, 3);

         let boxMaterial = this.RcFancyMaterial(this.ColorBlack, 1.0, { side: RC.FRONT_SIDE });
         boxMaterial.normalFlat = true;
         boxMaterial.color = RcCol(ebox.fMainColor);
         if (ebox.fMainTransparency) {
            boxMaterial.transparent = true;
            boxMaterial.opacity = (100 - ebox.fMainTransparency) / 100.0;
            boxMaterial.depthWrite = false;
         }

         let mesh = new RC.Mesh(body, boxMaterial);

         let geo_rim = new RC.Geometry();
         geo_rim.vertices = new RC.BufferAttribute(vBuff, 3);

         let nTrigs = 6 * 2;
         let nIdcsTrings = 6 * 2 * 3 * 2;
         let idcs = new Uint16Array(nIdcsTrings);
         for (let i = 0; i < nTrigs; ++i) {
            let ibo = i * 3;
            let sbo = i * 6;
            idcs[sbo] = idxBuff[ibo];
            idcs[sbo + 1] = idxBuff[ibo + 1];
            idcs[sbo + 2] = idxBuff[ibo + 1];
            idcs[sbo + 3] = idxBuff[ibo + 2];
            idcs[sbo + 4] = idxBuff[ibo + 2];
            idcs[sbo + 5] = idxBuff[ibo];
         }
         geo_rim.indices = new RC.BufferAttribute(idcs, 1);//
         let lcol = RcCol(ebox.fLineColor);
         let line = new RC.Line(geo_rim, this.RcLineMaterial(lcol, 0.8, 1));
         mesh.add(line);

         mesh.get_ctrl = function () { return new EveElemControl(this); }
         mesh.dispose = function () {
            this.children.forEach(c => { c.geometry.dispose(); c.material.dispose(); });
            this.geometry.dispose(); this.material.dispose();
         };

         return mesh;
      }
      //==============================================================================
      // make Digits
      //==============================================================================
      makeBoxSetInstanced(boxset, rnr_data)
      {
         // axis aligned box
         let SN = boxset.N;
         // console.log("SN", SN, "texture dim =", boxset.texX, boxset.texY);

         let tex_insta_pos_shape = new RC.Texture(rnr_data.vtxBuff,
            RC.Texture.WRAPPING.ClampToEdgeWrapping,
            RC.Texture.WRAPPING.ClampToEdgeWrapping,
            RC.Texture.FILTER.NearestFilter,
            RC.Texture.FILTER.NearestFilter,
            RC.Texture.FORMAT.RGBA32F, RC.Texture.FORMAT.RGBA, RC.Texture.TYPE.FLOAT,
            boxset.texX, boxset.texY);

         let shm = new RC.ZShapeBasicMaterial({
            ShapeSize: [boxset.defWidth, boxset.defHeight, boxset.defDepth],
            color: RcCol(boxset.fMainColor),
            emissive: new RC.Color(0.07, 0.07, 0.06),
            diffuse: new RC.Color(0, 0.6, 0.7),
            alpha: 0.5 // AMT, what is this used for ?
         });
         if (boxset.instanceFlag == "ScalePerDigit")
            shm.addSBFlag("SCALE_PER_INSTANCE");
         else if(boxset.instanceFlag == "Mat4Trans")
            shm.addSBFlag("MAT4_PER_INSTANCE");

         if (boxset.fMainTransparency) {
            shm.transparent = true;
            shm.opacity = (100 - boxset.fMainTransparency) / 100.0;
            shm.depthWrite = false; //? AMT what does that mean
         }
         shm.addInstanceData(tex_insta_pos_shape);
         shm.instanceData[0].flipy = false;
         let geo;
         if (boxset.shapeType == 1) {
            geo = RC.ZShape.makeHexagonGeometry();
         }
         else if (boxset.shapeType == 2) {
            geo = RC.ZShape.makeConeGeometry(boxset.coneCap);
         }
         else {
            geo = RC.ZShape.makeCubeGeometry();
         }
         let zshape = new RC.ZShape(geo, shm);
         zshape.instanceCount = SN;
         zshape.frustumCulled = false;
         zshape.instanced = true;
         zshape.material.normalFlat = true;

         this.RcPickable(boxset, zshape);

         return zshape;
      }

      makeBoxSet(boxset, rnr_data)
      {
         if (this.TestRnr("boxset", boxset, rnr_data)) return null;
         // use instancing if texture coordinates
         if (boxset.instanced === true)
            return this.makeBoxSetInstanced(boxset, rnr_data);
         else
            return this.makeFreeBoxSet(boxset, rnr_data);
      }

      makeFreeBoxSet(boxset, rnr_data)
      {
         let vBuff;
         let idxBuff;
         let nVerticesPerDigit = 0;

         if (boxset.boxType == 6) // hexagon
         {
            nVerticesPerDigit = 14;
            let stepAngle = Math.PI / 3;
            let N_hex = rnr_data.vtxBuff.length / 6;
            vBuff = new Float32Array(N_hex * 7 * 2 * 3);
            for (let i = 0; i < N_hex; ++i) {
               let rdoff = i * 6;
               let R = rnr_data.vtxBuff[rdoff + 3];
               let hexRotation = rnr_data.vtxBuff[rdoff + 4];
               let hexHeight = rnr_data.vtxBuff[rdoff + 5];
               let off = i* 3 * 7 * 2;

               // position
               let pos = [rnr_data.vtxBuff[rdoff], rnr_data.vtxBuff[rdoff + 1], rnr_data.vtxBuff[rdoff + 2]];

               // center
               vBuff[off]     = pos[0];
               vBuff[off + 1] = pos[1];
               vBuff[off + 2] = pos[2];

               off += 3;
               for (let j = 0; j < 6; ++j) {
                  let angle = j*stepAngle + hexRotation;
                  let x = R * Math.cos(angle) + pos[0];
                  let y = R * Math.sin(angle) + pos[1];
                  let z = pos[2];

                  // write buffer
                  vBuff[off]     = x;
                  vBuff[off + 1] = y;
                  vBuff[off + 2] = z;
                  off += 3;
               }

               // copy for depth
               let ro = i* 3 * 7 * 2;
               for (let j = 0; j < 7; ++j)
               {
                  vBuff[ro + 21] = vBuff[ro]+ hexHeight;
                  vBuff[ro + 22] = vBuff[ro+1]+ hexHeight;
                  vBuff[ro + 23] = vBuff[ro+2] + hexHeight;
                  ro += 3;
               }
            } // end loop vertex buffer

            let protoIdcs = [0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1];
            let protoIdcs2 = [2,1,0,  3,2,0,  4,3, 0,   5,4,0,  6, 5, 0,  1, 6, 0];
            let sideIdcs = [8,1,2,2,9,8,  9,2,3,3,10,9,  10,3,4,4,11,10,
                            11,4,5,5,12,11,  5,6,13,5,13,12, 13,6,1,1,8,13 ];
            let idxBuffSize =  N_hex * (protoIdcs.length * 2 + sideIdcs.length);
            idxBuff = new Uint32Array(idxBuffSize);
            let b = 0;
            for (let i = 0; i < N_hex; ++i) {
               let off0 = i * 7 * 2;
               for (let c = 0; c < protoIdcs.length; c++) {
                  idxBuff[b++] = off0 + protoIdcs2[c];
               }
               for (let c = 0; c < protoIdcs.length; c++) {
                  idxBuff[b++] = off0 + protoIdcs[c] +7;
               }
               for (let c = 0; c < sideIdcs.length; c++) {
                  idxBuff[b++] = off0 + sideIdcs[c];
               }
            }
         }
         else {
            nVerticesPerDigit = 8;
            if (boxset.boxType == 1) // free box
            {
               vBuff = rnr_data.vtxBuff;
            }
            else if (boxset.boxType == 2) // axis aligned
            {
               let N = rnr_data.vtxBuff.length / 6;
               vBuff = new Float32Array(N * 8 * 3);

               let off = 0;
               for (let i = 0; i < N; ++i) {
                  let rdoff = i * 6;
                  let x = rnr_data.vtxBuff[rdoff];
                  let y = rnr_data.vtxBuff[rdoff + 1];
                  let z = rnr_data.vtxBuff[rdoff + 2];
                  let dx = rnr_data.vtxBuff[rdoff + 3];
                  let dy = rnr_data.vtxBuff[rdoff + 4];
                  let dz = rnr_data.vtxBuff[rdoff + 5];

                  // top
                  vBuff[off] = x; vBuff[off + 1] = y + dy; vBuff[off + 2] = z;
                  off += 3;
                  vBuff[off] = x + dx; vBuff[off + 1] = y + dy; vBuff[off + 2] = z;
                  off += 3;
                  vBuff[off] = x + dx; vBuff[off + 1] = y; vBuff[off + 2] = z;
                  off += 3;
                  vBuff[off] = x; vBuff[off + 1] = y; vBuff[off + 2] = z;
                  off += 3;
                  // bottom
                  vBuff[off] = x; vBuff[off + 1] = y + dy; vBuff[off + 2] = z + dz;
                  off += 3;
                  vBuff[off] = x + dx; vBuff[off + 1] = y + dy; vBuff[off + 2] = z + dz;
                  off += 3;
                  vBuff[off] = x + dx; vBuff[off + 1] = y; vBuff[off + 2] = z + dz;
                  off += 3;
                  vBuff[off] = x; vBuff[off + 1] = y; vBuff[off + 2] = z + dz;
                  off += 3;
               }
            }

            let protoSize = 6 * 2 * 3;
            let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
            let nBox = vBuff.length / 24;
            idxBuff = new Uint32Array(nBox * protoSize);
            let iCnt = 0;
            for (let i = 0; i < nBox; ++i) {
               for (let c = 0; c < protoSize; c++) {
                  let off = i * 8;
                  idxBuff[iCnt++] = protoIdcs[c] + off;
               }
            }
         }
         let body = new RC.Geometry();
         body.indices = new RC.BufferAttribute(idxBuff, 1);
         body.vertices = new RC.BufferAttribute(vBuff, 3);
         // body.computeVertexNormals();

         // set material and colors

         let mat = this.RcFancyMaterial(this.ColorBlack, 1.0, { side: RC.FRONT_SIDE });
         mat.normalFlat = true;
         if ( ! boxset.fSingleColor)
         {
            let ci = rnr_data.idxBuff;
            let off = 0;
            let nVert = vBuff.length /3;
            let colBuff = new Float32Array( nVert * 4 );
            for (let x = 0; x < ci.length; ++x)
            {
               let r = (ci[x] & 0x000000FF) >>  0;
               let g = (ci[x] & 0x0000FF00) >>  8;
               let b = (ci[x] & 0x00FF0000) >> 16;
               for (let i = 0; i < nVerticesPerDigit; ++i)
               {
                  colBuff[off    ] = r / 255;
                  colBuff[off + 1] = g / 255;
                  colBuff[off + 2] = b / 255;
                  colBuff[off + 3] = 1.0;
                  off += 4;
               }
            }
            body.vertColor = new RC.BufferAttribute(colBuff, 4);
            mat.useVertexColors = true;
         } else {
            mat.color = RcCol(boxset.fMainColor);
         }

         if (boxset.fMainTransparency) {
            mat.transparent = true;
            mat.opacity = (100 - boxset.fMainTransparency) / 100.0;
            mat.depthWrite = false;
         }
         let mesh = new RC.Mesh(body, mat);
         this.RcPickable(boxset, mesh, false, boxset.fSecondarySelect ? BoxSetControl : EveElemControl);

         return mesh;
      }
      //==============================================================================
      // make Calorimeters
      //==============================================================================

      makeCalo3D(calo3D, rnr_data)
      {
         if (this.TestRnr("calo3D", calo3D, rnr_data)) return null;
         let body = new RC.Geometry();
         let vBuff = rnr_data.vtxBuff;
         let protoSize = 6 * 2 * 3;
         let protoIdcs = [0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0, 1, 2, 3, 1, 3, 0, 4, 7, 6, 4, 6, 5];
         let nBox = vBuff.length / 24;
         let idxBuff = new Uint32Array(nBox * protoSize);
         let p = 0;
         for (let i = 0; i < nBox; ++i) {
            let off = i * 8;
            for (let c = 0; c < protoSize; c++) {
               idxBuff[p++] = protoIdcs[c] + off;
            }
         }

         body.indices = new RC.BufferAttribute(idxBuff, 1);
         body.vertices = new RC.BufferAttribute(rnr_data.vtxBuff, 3);
         // body.computeVertexNormals();

         let ci = rnr_data.idxBuff;
         let off = 0
         let colBuff = new Float32Array(nBox * 8 * 4);
         for (let x = 0; x < nBox; ++x) {
            let slice = ci[x * 2];
            let sliceColor = calo3D.sliceColors[slice];
            let tc = RcCol(sliceColor);
            for (let i = 0; i < 8; ++i) {
               colBuff[off] = tc.r;
               colBuff[off + 1] = tc.g;
               colBuff[off + 2] = tc.b;
               colBuff[off + 3] = 1.0;
               off += 4;
            }
         }
         body.vertColor = new RC.BufferAttribute(colBuff, 4);

         let mat = this.RcFancyMaterial(this.ColorBlack, 1.0, { side: RC.FRONT_SIDE });
         mat.useVertexColors = true;
         mat.normalFlat = true;

         let mesh = new RC.Mesh(body, mat);

         this.RcPickable(calo3D, mesh, false, Calo3DControl);

         return mesh;
      }

      makeCalo2D(calo2D, rnrData)
      {
         if (this.TestRnr("calo2D", calo2D, rnrData)) return null;
         let body = new RC.Geometry();
         let nSquares = rnrData.vtxBuff.length / 12;
         let nTriang = 2 * nSquares;

         let idxBuff = new Uint32Array(nTriang * 3);
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

         body.indices = new RC.BufferAttribute(idxBuff, 1);
         body.vertices = new RC.BufferAttribute(rnrData.vtxBuff, 3);
         // body.computeVertexNormals();

         let ci = rnrData.idxBuff;
         let colBuff = new Float32Array(nSquares * 4 * 4);
         let off = 0;
         for (let x = 0; x < nSquares; ++x) {
            let slice = ci[x * 2];
            let sliceColor = calo2D.sliceColors[slice];
            let tc = RcCol(sliceColor);
            for (let i = 0; i < 4; ++i) {
               colBuff[off] = tc.r;
               colBuff[off + 1] = tc.g;
               colBuff[off + 2] = tc.b;
               colBuff[off + 3] = 1.0;
               off += 4;
            }
         }
         body.vertColor = new RC.BufferAttribute(colBuff, 4);

         let mat = this.RcFlatMaterial(this.ColorBlack, 1);
         mat.useVertexColors = true;
         let mesh = new RC.Mesh(body, mat);

         this.RcPickable(calo2D, mesh, false, Calo2DControl);

         return mesh;
      }

      //==============================================================================
      // makeEveGeometry / makeEveGeoShape
      //==============================================================================

      makeEveGeometry(rnr_data, compute_normals)
      {
         let nVert = rnr_data.idxBuff[1] * 3;

         if (rnr_data.idxBuff[0] != GL.TRIANGLES) throw "Expect triangles first.";
         if (2 + nVert != rnr_data.idxBuff.length) throw "Expect single list of triangles in index buffer.";

         let geo = new RC.Geometry();
         geo.vertices = new RC.BufferAttribute(rnr_data.vtxBuff, 3);
         let ib = rnr_data.idxBuff;
         geo.indices = new RC.BufferAttribute(
            new Uint32Array(ib.buffer, ib.byteOffset + 8, nVert),
            1);

         if (compute_normals) {
            geo.computeVertexNormalsIdxRange(2, nVert);
         }

         // XXXX Fix this. It seems we could have flat shading with usage of simple shaders.
         // XXXX Also, we could do edge detect on the server for outlines.
         // XXXX a) 3d objects - angle between triangles >= 85 degrees (or something);
         // XXXX b) 2d objects - segment only has one triangle.
         // XXXX Somewhat orthogonal - when we do tesselation, conversion from quads to
         // XXXX triangles is trivial, we could do it before invoking the big guns (if they are even needed).
         // XXXX Oh, and once triangulated, we really don't need to store 3 as number of verts in a poly each time.
         // XXXX Or do we? We might need it for projection stuff.

         return geo;
      }

      makeEveGeoShape(egs, rnr_data)
      {
         let geom = this.makeEveGeometry(rnr_data, false);

         let fcol = RcCol(egs.fFillColor);
         let mop = 1 - egs.fMainTransparency/100;

         let mat = this.RcFancyMaterial(fcol, mop);
         // mat.side = RC.FRONT_AND_BACK_SIDE;
         mat.side = RC.FRONT_SIDE;
         mat.shininess = 50;
         mat.normalFlat = true;

         let mesh = new RC.Mesh(geom, mat);
         this.RcPickable(egs, mesh);
         return mesh;
      }

      makeGeoTopNodeProcessObject(o3, ctx, eveTopNode)
      {
         let orc;
         if (o3 instanceof THREE.Mesh) {
            if (!ctx.geomap.has(o3.geometry)) {
               let g = new RC.Geometry();
               g.vertices = new RC.BufferAttribute(o3.geometry.attributes.position.array, 3);
               g.normals = new RC.BufferAttribute(o3.geometry.attributes.normal.array, 3);
               delete o3.geometry.attributes;
               ctx.geomap.set(o3.geometry, g);
            } else {
               ++ctx.n_geo_reuse;
            }
            let m3 = o3.material;
            let mrc = this.RcFancyMaterial(new RC.Color(m3.color.r, m3.color.g, m3.color.b), m3.opacity);
            orc = new RC.Mesh(ctx.geomap.get(o3.geometry), mrc);
            this.RcPickable(eveTopNode, orc, true, GeoTopNodeControl);
            orc.material.normalFlat = true;
            // orc.amt_debug_name = "mesh" + o3.name; // set for debugging purposes
            ++ctx.n_mesh;
         } else {
            orc = new RC.Group();
            // orc.amt_debug_name = "group" + o3.name; // set for debugging purposes
            ++ctx.n_o3d;
         }

         orc.nchld = o3.nchld;
         orc.matrixAutoUpdate = false;
         orc.setMatrixFromArray(o3.matrix.elements);
         for (let c of o3.children) {
            orc.add(this.makeGeoTopNodeProcessObject(c, ctx, eveTopNode));
         }

         // selection ... remore new ...
         orc.stack = o3.stack;
         return orc;
      }

      makeGeoTopNode(tn, rnr_data) {
         // console.log("make top node ", tn);
         let json = atob(tn.geomDescription);
         let zz = EVE.JSR.parse(json);
         let o3 = EVE.JSR.build(zz);
         // console.log("tgeo painter builder o3 obj =", o3);
         let ctx = { geomap: new Map, n_o3d: 0, n_mesh: 0, n_geo_reuse: 0 };
         let orc = this.makeGeoTopNodeProcessObject(o3, ctx, tn);
         // console.log("map summary ", ctx.geomap.size, ctx.n_o3d, ctx.n_mesh, ctx.n_geo_reuse);
         orc.get_ctrl = function () { return new GeoTopNodeControl(this, orc); };

         orc.clones = o3.clones;

         // function to get stack
         orc.clones.createRCObject3D = function (stack, toplevel, options) {
            let node = this.nodes[0], three_prnt = toplevel, draw_depth = 0;

            for (let lvl = 0; lvl <= stack.length; ++lvl) {
               let nchld = (lvl > 0) ? stack[lvl - 1] : 0;
               // console.log("level ", lvl, "nchld", nchld);
               // extract current node
               if (lvl > 0) node = this.nodes[node.chlds[nchld]];
               if (!node) return null;

               let obj3d = undefined;

               if (three_prnt.children)
                  for (let i = 0; i < three_prnt.children.length; ++i) {
                     console.log(i, "<< comapre ",three_prnt.children[i].nchld, nchld );
                     if (three_prnt.children[i].nchld === nchld) {
                        console.log("createRCObject3D .... reuse obj3d .... from clones ??");
                        obj3d = three_prnt.children[i];
                        break;
                     }
                  }

               if (obj3d) {
                  three_prnt = obj3d;
                  // console.log("set three");
                  if (obj3d.$jsroot_drawable) draw_depth++;
                  continue;
               }

               // console.log("make NEW ode ", node);
               obj3d = new RC.Object3D();

               if (node.abs_matrix) {
                  obj3d.absMatrix = new RC.Matrix4();
                  obj3d.absMatrix.fromArray(node.matrix);
               } else if (node.matrix) {
                  obj3d.matrix.fromArray(node.matrix);
                  obj3d.matrix.decompose(obj3d.position, obj3d.quaternion, obj3d.scale);
               }

               // add the mesh to the scene
               three_prnt.add(obj3d);
               obj3d.updateMatrixWorld();

               three_prnt = obj3d;
            }

            return three_prnt;
         } // end clones create obj3d

         return orc;
      }

      //==============================================================================
      // makePolygonSetProjected
      //==============================================================================

      makePolygonSetProjected(psp, rnr_data)
      {
         let psp_ro = new RC.Group();
         let pos_ba = new RC.BufferAttribute(rnr_data.vtxBuff, 3);
         let idx_ba = new RC.BufferAttribute(rnr_data.idxBuff, 1);

         let ib_len = rnr_data.idxBuff.length;

         let fcol = RcCol(psp.fMainColor);
         let mop = Math.min( 1, 1 - psp.fMainTransparency/100);

         let material = this.RcFlatMaterial(fcol, mop);
         material.side = RC.FRONT_AND_BACK_SIDE;

         let line_mat = this.RcLineMaterial(fcol);

         let meshes = [];
         for (let ib_pos = 0; ib_pos < ib_len;)
         {
            if (rnr_data.idxBuff[ib_pos] == GL.TRIANGLES)
            {
               let geo = new RC.Geometry();
               geo.vertices = pos_ba;
               geo.indices = idx_ba;
               geo.setDrawRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);
               geo.computeVertexNormalsIdxRange(ib_pos + 2, 3 * rnr_data.idxBuff[ib_pos + 1]);

               let mesh = new RC.Mesh(geo, material);
               this.RcPickable(psp, mesh);
               psp_ro.add(mesh);
               meshes.push(mesh);

               ib_pos += 2 + 3 * rnr_data.idxBuff[ib_pos + 1];
            }
            else if (rnr_data.idxBuff[ib_pos] == GL.LINE_LOOP)
            {
               let geo = new RC.Geometry();
               geo.vertices = pos_ba;
               geo.indices = idx_ba;
               geo.setDrawRange(ib_pos + 2, rnr_data.idxBuff[ib_pos + 1]);

               let ll = new RC.Line(geo, line_mat);
               ll.renderingPrimitive = RC.LINE_LOOP;
               psp_ro.add(ll);

               ib_pos += 2 + rnr_data.idxBuff[ib_pos + 1];
            }
            else
            {
               console.error("Unexpected primitive type " + rnr_data.idxBuff[ib_pos]);
               break;
            }

         }
         // this.RcPickable(psp, psp_ro);
         // this.RcPickable(el, psp_ro, false, null);
         if (psp.fPickable) {
            for (let m of meshes) m.pickable = true;
         }
         psp_ro.get_ctrl = function (iobj, tobj) {
            let octrl = new EveElemControl(iobj, tobj);
            octrl.DrawForSelection = function (sec_idcs, res) {
               res.geom.push(...meshes);
               //res.geom.push(...tobj.children);
            }
            return octrl;
         }

         return psp_ro;
      }

      //==============================================================================

      makeStraightLineSet(el, rnr_data)
      {
         // console.log("makeStraightLineSet ...");

         let obj3d = new RC.Group();

         let buf = new Float32Array(el.fLinePlexSize * 6);
         for (let i = 0; i < el.fLinePlexSize * 6; ++i)
         {
            buf[i] = rnr_data.vtxBuff[i];
         }

         let geom = new RC.Geometry();
         geom.vertices = new RC.BufferAttribute(buf, 3);

         let line_color = RcCol(el.fMainColor);

         let line_width = 2 * (el.fLineWidth || 1);
         const line = this.RcMakeStripes(geom, line_width, line_color);
         this.RcApplyStripesMaterials(el, line, 2);
         this.RcPickable(el, line);
         obj3d.add(line);

         // ---------------- DUH, could share buffer attribute. XXXXX

         if (el.fMarkerPlexSize) {
            let nPnts = el.fMarkerPlexSize;
            let off = el.fLinePlexSize * 6;
            let p_buf = new Float32Array(el.fTexX*el.fTexY*4);
            for (let i = 0; i < nPnts; ++i) {
               let j = i*3;
               let k = i*4;
               p_buf[k]   = rnr_data.vtxBuff[j+off];
               p_buf[k+1] = rnr_data.vtxBuff[j+off+1];
               p_buf[k+2] = rnr_data.vtxBuff[j+off+2];
               p_buf[k+3] = 0;
            }
            let marker = this.RcMakeZSprite(el.fMainColor, el.fMarkerSize, nPnts,
               p_buf, el.fTexX, el.fTexY,
               "star5-32a.png");
            obj3d.add(marker);
         }
         // For secondary selection, see EveElements.js
         // obj3d.eve_idx_buf = rnr_data.idxBuff;
         // if (el.fSecondarySelect)
         //    octrl = new StraightLineSetControl(obj3d);
         // else
         //    octrl = new EveElemControl(obj3d);

         this.RcPickable(el, obj3d, true, null);
         obj3d.get_ctrl  = function(iobj, tobj) {
            let octrl = new EveElemControl(iobj, tobj);
            octrl.DrawForSelection = function(sec_idcs, res) {
               res.geom.push(...this.top_obj.children);
            };
            return octrl;
         }

         return obj3d;
      }

   } // class EveElements

   //==============================================================================

   EVE.EveElements = EveElements;

   return EveElements;
});
