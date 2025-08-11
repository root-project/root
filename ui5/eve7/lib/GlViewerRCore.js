sap.ui.define([
   'rootui5/eve7/lib/GlViewer',
   'rootui5/eve7/lib/EveElementsRCore'
], function(GlViewer, EveElements) {

   "use strict";

   let RC;
   let datGUI;

   class GlViewerRCore extends GlViewer {

      constructor(viewer_class)
      {
         super(viewer_class);

         const urlParams = new URLSearchParams(window.location.search);

         let mode_mm = /^(?:Direct|Simple|Full)$/.exec(urlParams.get('RQ_Mode'));
         let ssaa_mm = /^(1|2|4)$/.               exec(urlParams.get('RQ_SSAA'));
         let marker_scale = /^([\d\.]+)$/.        exec(urlParams.get('RQ_MarkerScale'));
         let line_scale = /^([\d\.]+)$/.          exec(urlParams.get('RQ_LineScale'));

         this.RQ_Mode = (mode_mm) ? mode_mm[0] : "Simple";
         this.RQ_SSAA = (ssaa_mm) ? ssaa_mm[0] : 2;
         this.RQ_MarkerScale = (marker_scale) ? marker_scale[0] : 1;
         this.RQ_LineScale   = (line_scale) ? line_scale[0] : 1;

         let jsrp = EVE.JSR.source_dir;
         // take out 'jsrootsys' and replace it with 'rootui5sys/eve7/'
         this.top_path = jsrp.substring(0, jsrp.length - 10);
         this.eve_path = this.top_path + 'rootui5sys/eve7/';

         this._logLevel = 1; // 0 - error, 1 - warning, 2 - info, 3 - debug

         if (this._logLevel > 2) {
            console.log("GlViewerRCore RQ_Mode:", this.RQ_Mode, "RQ_SSAA:", this.RQ_SSAA,
                        'RQ_MarkerScale', this.RQ_MarkerScale,  'RQ_LineScale', this.RQ_LineScale);
         }

         this._selection_map = {};
         this._selection_list = [];

         this.initialMouseX = 0;
         this.initialMouseY = 0;
         this.lastOffsetX = 0;
         this.lastOffsetY = 0;
         this.firstMouseDown = true;
         this.scale = false;
         this.pickedOverlayObj;
         this.initialSize = 0;
      }

      init(controller)
      {
         super.init(controller);

         let pthis = this;

         // For offline mode, one needs a a full URL or the request
         // gets forwarded to openi5.hana.ondemand.com.
         // This has to be understood and fixed. Loading of shaders
         // afterwards fails, too.
         // // console.log(window.location.pathname); // where are we loading from?
         // import("https://desire.physics.ucsd.edu/matevz/alja.github.io/rootui5/eve7/rnr_core/RenderCore.js").then((module) => {

         if (!RC) {
            import(this.eve_path + 'lib/RenderCore.js').then((module) => {
               if (this._logLevel >= 2)
                  console.log("GlViewerRCore.onInit - RenderCore.js loaded");
               RC = module;

               RC.Canvas.prototype.generateCanvasDOM = function(id="eve7-rc-canvas") {
                  if (RC.Canvas.prototype._xxcount === undefined) { RC.Canvas.prototype._xxcount = 0; }

                  const canvasDOM = document.createElement("canvas");
                  canvasDOM.id = id + "-" + RC.Canvas.prototype._xxcount;
                  ++RC.Canvas.prototype._xxcount;

                  //make it visually fill the positioned parent
                  //set the display size of the canvas
                  canvasDOM.style.width = "100%";
                  canvasDOM.style.height = "100%";
                  canvasDOM.style.padding = '0';
                  canvasDOM.style.margin = '0';

                  return canvasDOM;
               };

               pthis.bootstrap();
            });
         } else {
            this.bootstrap();
         }
      }

      bootstrap()
      {
         RC.GLManager.sCheckFrameBuffer = false;
         RC.Object3D.sDefaultPickable = false;
         RC.PickingShaderMaterial.DEFAULT_PICK_MODE = RC.PickingShaderMaterial.PICK_MODE.UINT;

         this.createRCoreRenderer();

         this.creator = new EveElements(RC, this);
         // this.creator.useIndexAsIs = EVE.JSR.decodeUrl().has('useindx');
         if (this.RQ_Mode != "Direct") {
            this.creator.SetupPointLineFacs(this.RQ_SSAA,
                                            this.RQ_MarkerScale * this.canvas.pixelRatio,
                                            this.RQ_LineScale   * this.canvas.pixelRatio);
         }

         this.controller.createScenes();
         this.controller.redrawScenes();
         this.setupEventHandlers();
         this.updateViewerAttributes();

         this.controller.glViewerInitDone();
      }

      cleanup() {
         if (this.controller) this.controller.removeScenes();
         super.cleanup();
      }

      //==============================================================================

      make_object(name)
      {
         let c = new RC.Group();
         c.name = name || "<no-name>";
         return c;
      }

      get_top_scene()
      {
         return this.scene;
      }
      get_overlay_scene()
      {
         return this.overlay_scene;
      }

      //==============================================================================

      createRCoreRenderer()
      {
         let canvasParentDOM = document.createElement("div");
         let vid = this.get_view().sId + "--rcore";
         canvasParentDOM.setAttribute("id", vid);
         canvasParentDOM.style.width = "100%";
         canvasParentDOM.style.height = "100%";

         // in case of openui5 rooter, the canvas element accumulates
         // destroy the old canvas element
	      let cn = this.get_view().getDomRef().childNodes;
         let oldCanvas = -1;
         for (let i =0; i < cn.length; ++i) {
           if (cn[i].id === vid)
                oldCanvas = i;
         }
         if (oldCanvas !== -1) {
            this.get_view().getDomRef().removeChild(cn[oldCanvas]);
         }

         this.get_view().getDomRef().appendChild(canvasParentDOM);
         this.canvas = new RC.Canvas(canvasParentDOM);
         let w = this.canvas.width;
         let h = this.canvas.height;
         this.fixCssSize();
         this.canvas.parentDOM.style.overflow = "hidden";
         this.canvas.canvasDOM.style.overflow = "hidden";
         // It seems SSAA of 2 is still beneficial on retina.
         // if (this.canvas.pixelRatio > 1 && this.RQ_SSAA > 1) {
         //    console.log("Correcting RQ_SSAA for pixelRatio", this.canvas.pixelRatio,
         //                "from", this.RQ_SSAA, "to", this.RQ_SSAA / this.canvas.pixelRatio);
         //    this.RQ_SSAA /= this.canvas.pixelRatio;
         // }

         this.renderer = new RC.MeshRenderer(this.canvas, RC.WEBGL2,
                                             { antialias: false, stencil: false });
         this.renderer._logLevel = 0;
         this.renderer.addShaderLoaderUrls(this.eve_path + RC.REveShaderPath);
         this.renderer.pickObject3D = true;

         RC.Cache.enabled = true;
         this.tex_cache = new RC.TextureCache;

         // add dat GUI option to set background
         let eveView = this.controller.mgr.GetElement(this.controller.eveViewerId);
         if (eveView.BlackBg)
         {
            this.bgCol =  new RC.Color(0,0,0);
            this.fgCol = new RC.Color(1,1,1);
         }
         else
         {
            this.bgCol = new RC.Color(1,1,1);
            this.fgCol = new RC.Color(0,0,0);
         }

         // always use black clear color except in tone map
         this.renderer.clearColor = "#00000000";
         this.scene = new RC.Scene();
         this.overlay_scene = new RC.Scene();

         this.lights = new RC.Group;
         this.lights.name = "Light container";
         this.scene.add(this.lights);

         let a_light = new RC.AmbientLight(new RC.Color(0xffffff), 0.05);
         this.lights.add(a_light);

         let light_3d_ctor = function(col, int, dist, decay, args) { return new RC.PointLight(col, int, dist, decay, args); };
         // let light_3d_ctor = function(col, int, dist, decay, args) { return new RC.DirectionalLight(col, int); };
         let light_2d_ctor = function(col, int) { return new RC.DirectionalLight(col, int); };

         // guides
         this.axis = new RC.Group();
         this.axis.name = "Axis";
         // this.overlay_scene.add(this.axis); // looks worse for now put to scene
         this.scene.add(this.axis);

         if (this.controller.isEveCameraPerspective())
         {
            this.camera = new RC.PerspectiveCamera(75, w / h, 20, 4000);
            this.camera.isPerspectiveCamera = true;

            let l_int = 1.4;
            let l_args = { constant: 1, linear: 0, quadratic: 0, smap_size: 0 };
            this.lights.add(light_3d_ctor(0xaa8888, l_int, 0, 1, l_args)); // R
            this.lights.add(light_3d_ctor(0x88aa88, l_int, 0, 1, l_args)); // G
            this.lights.add(light_3d_ctor(0x8888aa, l_int, 0, 1, l_args)); // B
            this.lights.add(light_3d_ctor(0xaaaa66, l_int, 0, 1, l_args)); // Y
            this.lights.add(light_3d_ctor(0x666666, l_int, 0, 1, l_args)); // gray, bottom

            // Lights are positioned in resetRenderer.

            // Markers on light positions (screws up bounding box / camera reset calculations)
            // for (let i = 1; i <= 4; ++i)
            // {
            //    let l = this.lights.children[i];
            //    l.add( new RC.IcoSphere(1, 1, 10.0, l.color.clone().multiplyScalar(0.5), false) );
            // }
         }
         else
         {
            this.camera = new RC.OrthographicCamera(-w/2, w/2, -h/2, h/2, 20, 2000);
            this.camera.isOrthographicCamera = true;

            let l_int = 0.85;
            this.lights.add(light_2d_ctor(0xffffff, l_int)); // white front
            // this.lights.add(light_2d_ctor(0xffffff, l_int)); // white back

            // Lights are positioned in resetRenderer.
         }

         // AMT, disable auto update in camera in order prevent reading quaternions in update of
         // model view  matrix in Obejct3D function updateMatrixWorld
         this.camera.matrixAutoUpdate = false;

         // Test objects
         if (this.controller.isEveCameraPerspective())
         {
            // let c = new RC.Cube(40, new RC.Color(0.2,.4,.8));
            // c.material = new RC.MeshPhongMaterial();
            // c.material.transparent = true;
            // c.material.opacity = 0.8;
            // c.material.depthWrite  = false;
            // this.scene.add(c);

            // let ss = new RC.Stripe([0,0,0, 400,0,0, 400,400,0, 400,400,400]);
            // ss.material.lineWidth = 20.0;
            // ss.material.color     = new RC.Color(0xff0000);
            // ss.material.emissive  = new RC.Color(0x008080);
            // this.scene.add(ss);
         }

         this.rot_center = new RC.Vector3(0,0,0);

         this.rqt = new RC.RendeQuTor(this.renderer, this.scene, this.camera, this.overlay_scene);
         if (this.RQ_Mode == "Direct")
         {
            this.rqt.initDirectToScreen();
         }
         else if (this.RQ_Mode == "Simple")
         {
            this.rqt.initSimple(this.RQ_SSAA);
         }
         else
         {
            this.rqt.initFull(this.RQ_SSAA);
         }
         this.rqt.updateViewport(w, h);


         // AMT secondary selection bug workaround for RenderCore PR #21
         this.rqt.pick_instance = function(state)
         {
            return this.pick_instance_low_level(this.pqueue, state);
         }
         this.rqt.pick_instance_overlay = function(state)
         {
            return this.pick_instance_low_level(this.ovlpqueue, state);
         }
      }

      setupEventHandlers()
      {
         let dome = this.canvas.canvasDOM;

         // Setup tooltip
         this.ttip = document.createElement('div');
         this.ttip.setAttribute('class', 'eve_tooltip');
         this.ttip_text = document.createElement('div');
         this.ttip.appendChild(this.ttip_text);
         this.canvas.parentDOM.appendChild(this.ttip)


         // Setup some event pre-handlers
         let glc = this;

         dome.addEventListener('pointermove', function(event) {

            if (event.movementX == 0 && event.movementY == 0)
               return;

            glc.removeMouseupListener();

            if (event.buttons === 0 && event.srcElement === glc.canvas.canvasDOM) {
               glc.removeMouseMoveTimeout();
               glc.mousemove_timeout = setTimeout(glc.onMouseMoveTimeout.bind(glc, event.offsetX, event.offsetY), glc.controller.htimeout);
            } else {
               // glc.clearHighlight();
            }
         });

         dome.addEventListener('pointerleave', function() {

            glc.removeMouseMoveTimeout();
            glc.clearHighlight();
            glc.removeMouseupListener();
         });

         dome.addEventListener('pointerdown', function(event) {

            glc.removeMouseMoveTimeout();
            if (event.button != 0 && event.button != 2)  glc.clearHighlight();
            glc.removeMouseupListener();

            // console.log("GLC::mousedown", this, glc, event, event.offsetX, event.offsetY);

            glc.mouseup_listener = function(event2)
            {
               this.removeEventListener('pointerup', glc.mouseup_listener);

               if (event2.button == 0) // Selection on mouseup without move
               {
                  glc.handleMouseSelect(event2);
               }
               else if (event2.button == 2) // Context menu on delay without move
               {
                  EVE.JSR.createMenu(event2, glc).then(menu => glc.showContextMenu(event2, menu));
               }
            }

            this.addEventListener('pointerup', glc.mouseup_listener);
         });

         dome.addEventListener('dblclick', function() {
            //if (glc.controller.dblclick_action == "Reset")
            glc.resetRenderer();
         });

         dome.addEventListener("mouseup", function() {
            glc.handleOverlayMouseUp();
         });

         dome.addEventListener("mousedown", function(event) {
            if (event.button == 0 || event.button == 2)
            {
               glc.handleOverlayMouseDown(event);

            }
         });

         dome.addEventListener("mousemove", function(event) {
            glc.handleOverlayMouseMove(event);
         });

         // Key-handlers go on window ...

         window.addEventListener('keydown', function(event) {
            // console.log("GLC::keydown", event.key, event.code, event);

            let handled = true;

            if (event.key == "t")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.lineWidth )
                  {
                     if ( ! node.lineWidth_orig) node.lineWidth_orig = node.lineWidth;

                     node.lineWidth *= 1.2;
                  }
               });
            }
            else if (event.key == "e")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.lineWidth )
                  {
                     if ( ! node.lineWidth_orig) node.lineWidth_orig = node.lineWidth;

                     node.lineWidth *= 0.8;
                  }
               });
            }
            else if (event.key == "r")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.lineWidth && node.lineWidth_orig )
                  {
                     node.lineWidth = node.lineWidth_orig;
                  }
               });
            }
            else
            {
               handled = false;
            }

            if (handled)
            {
               // // // event.stopPropagation();
               // event.preventDefault();
               // event.stopImmediatePropagation();

               glc.render();
            }
         });

         this.controls = new RC.REveCameraControls(this.camera, this.canvas.canvasDOM);
         this.controls.addEventListener('change', this.render.bind(this));

         // camera center marker
         let col = new RC.Color(0.5, 0, 0);
         const msize = this.RQ_SSAA * 8; // marker size
         let sm = new RC.ZSpriteBasicMaterial({
            SpriteMode: RC.SPRITE_SPACE_SCREEN, SpriteSize: [msize, msize],
            color: this.ColorBlack,
            emissive: col,
            diffuse: col.clone().multiplyScalar(0.5)
         }
         );
         let vtx = new Float32Array(3);
         vtx[0] = 0; vtx[1] = 0; vtx[2] = 0;
         let s = new RC.ZSprite(null, sm);
         s.instanced = false;
         s.position.set(0, 0, 0);
         s.visible = false;
         this.scene.add(s);
         this.controls.centerMarker = s;

         // This will also call render().
         this.resetRenderer();
      }

      resetRenderer()
      {
         let sbbox = new RC.Box3();
         sbbox.setFromObject( this.scene );
         if (sbbox.isEmpty())
         {
            console.error("GlViewerRenderCore.resetRenderer scene bbox empty", sbbox);
            const ext = 100;
            sbbox.expandByPoint(new RC.Vector3(-ext,-ext,-ext));
            sbbox.expandByPoint(new RC.Vector3( ext, ext, ext));
         }

         let posV = new RC.Vector3; posV.subVectors(sbbox.max, this.rot_center);
         let negV = new RC.Vector3; negV.subVectors(sbbox.min, this.rot_center);

         let extV = new RC.Vector3; extV = negV; extV.negate(); extV.max(posV);
         let extR = extV.length();

         if (this._logLevel >= 2)
            console.log("GlViewerRenderCore.resetRenderer", sbbox, posV, negV, extV, extR);

         let eveView = this.controller.mgr.GetElement(this.controller.eveViewerId);
         let v1 = eveView.camera.V1;
         let v2 = eveView.camera.V2;


         if (this.camera.isPerspectiveCamera)
         {
            this.controls.setCamBaseMtx(new RC.Vector3(v1[0], v1[1], v1[2]), new RC.Vector3(v2[0], v2[1], v2[2]));
            this.controls.screenSpacePanning = true;

            let lc = this.lights.children;
            // lights are const now -- no need to set decay and distance
            lc[1].position.set( extR, extR, -extR);
            lc[2].position.set(-extR, extR,  extR);
            lc[3].position.set( extR, extR,  extR);
            lc[4].position.set(-extR, extR, -extR);
            lc[5].position.set(0, -extR, 0);

            // console.log("resetRenderer 3D scene bbox ", sbbox, ", look_at ", this.rot_center);
         }
         else
         {
            this.controls.setCamBaseMtx(new RC.Vector3(v1[0], v1[1], v1[2]), new RC.Vector3(v2[0], v2[1], v2[2]));
            let ey = 1.02 * extV.y;
            let ex = ey / this.get_height() * this.get_width();
            this.camera._left   = -ex;
            this.camera._right  =  ex;
            this.camera._top    =  ey;
            this.camera._bottom = -ey;
            this.camera.updateProjectionMatrix();

            if (typeof this.controls.resetOrthoPanZoom == 'function')
               this.controls.resetOrthoPanZoom();

            this.controls.screenSpacePanning = true;
            this.controls.enableRotate = false;

            let lc = this.lights.children;
            lc[1].position.set( 0, 0,  extR);
            // lc[2].position.set( 0, 0, -extR);

            // console.log("resetRenderer 2D scene bbox ex ey", sbbox, ex, ey, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }

         this.controls.setFromBBox(sbbox);

         this.controls.update();
      }


      updateViewerAttributes()
      {
         let eveView = this.controller.mgr.GetElement(this.controller.eveViewerId);
         if (eveView.BlackBg)
         {
            this.fgCol = this.creator.ColorWhite;
            this.bgCol = this.creator.ColorBlack;
         }
         else
         {
            this.bgCol = this.creator.ColorWhite;
            this.fgCol = this.creator.ColorBlack;
         }

         this.axis.clear();
         if (eveView.AxesType > 0)
            this.makeAxis();

         this.request_render();
      }

      makeAxis()
      {
         function formatFloat(val) {
            let lg = Math.log10(Math.abs(val));
            let fs = "undef";

            if (lg < 0) {
                if (lg > -1) {
                    fs = val.toFixed(2);
                }
                else if (lg > -2) {
                    fs = val.toFixed(3);
                }
                else {
                    fs = val.toExponential(2);
                }
            }
            else {
                if (lg < 2)
                    fs = val.toFixed(1);
                else if (lg < 4)
                    fs = Math.round(val);
                else
                    fs = val.toExponential(2);
            }
            return val > 0 ? "+" + fs : fs;
         }

         let bb = new RC.Box3();
         bb.setFromObject(this.scene);

         let lines = [];
         lines.push({ "p": new RC.Vector3(bb.min.x, 0, 0), "c": new RC.Color(1, 0, 0), "text": "x " + formatFloat(bb.min.x) });
         lines.push({ "p": new RC.Vector3(bb.max.x, 0, 0), "c": new RC.Color(1, 0, 0), "text": "x " + formatFloat(bb.max.x) });
         lines.push({ "p": new RC.Vector3(0, bb.min.y, 0), "c": new RC.Color(0, 1, 0), "text": "y " + formatFloat(bb.min.y) });
         lines.push({ "p": new RC.Vector3(0, bb.max.y, 0), "c": new RC.Color(0, 1, 0), "text": "y " + formatFloat(bb.max.y) });
         if (this.controller.isEveCameraPerspective()) {
            lines.push({ "p": new RC.Vector3(0, 0, bb.min.z), "c": new RC.Color(0, 0, 1), "text": "z " + formatFloat(bb.min.z) });
            lines.push({ "p": new RC.Vector3(0, 0, bb.max.z), "c": new RC.Color(0, 0, 1), "text": "z " + formatFloat(bb.max.z) });
         }

         for (const ax of lines) {
            let geom = new RC.Geometry();
            let buf = new Float32Array([0, 0, 0, ax.p.x, ax.p.y, ax.p.z]);
            geom.vertices = new RC.Float32Attribute(buf, 3);
            let ss = this.creator.RcMakeStripes(geom, 2, ax.c);
            this.axis.add(ss);
         }

         let url_base = this.eve_path + 'sdf-fonts/LiberationSerif-Regular';
         this.tex_cache.deliver_font(url_base,
            (texture, font_metrics) => {
               let diag = new RC.Vector3;
               bb.getSize(diag);
               diag = diag.length() / 100;
               let ag = this.axis;
               for (const ax of lines) {
                  const text = new RC.ZText({
                     text: ax.text,
                     fontTexture: texture,
                     xPos: 0.0,
                     yPos: 0.0,
                     fontSize: 0.01,
                     mode: RC.TEXT2D_SPACE_MIXED,
                     fontHinting: 1.0,
                     color: this.fgCol,
                     font: font_metrics,
                  });
                  text.position = ax.p;
                  text.material.side = RC.FRONT_SIDE;
                  ag.add(text);
               }
            },
            (img) => RC.ZText.createDefaultTexture(img)
         );
      };

      //==============================================================================

      request_render()
      {
         // console.log("REQUEST RENDER");

         if (this.render_requested) return;
         setTimeout(this.render.bind(this), 0);
         this.render_requested = true;
      }

      render()
      {
         // console.log("RENDER", this.scene, this.camera, this.canvas, this.renderer);

         this.render_requested = false;

         if (this.canvas.width <= 0 || this.canvas.height <= 0) return;

         this.rqt.render_begin(true);

         // Render outlines for active selections.

         for (let sel_id of this._selection_list)
         {
            let sel_entry = this._selection_map[ sel_id ];

            let obj_list = this.rqt.RP_GBuffer.obj_list;
            // let instance_list = [];

            for (let el_idx in sel_entry)
            {
               let el_entry = sel_entry[ el_idx ];
               // take all geometry objects, then we have to treat them differently, depending on type.
               // and update world-matrix / check visibility
               // or setup secondary indices for sub-instance drawing

               if (el_entry.instance_object) {
                     // instance_list.push(el_entry.instance_object);
                     obj_list.push(el_entry.instance_object);
                     el_entry.instance_object.outlineMaterial.outline_instances_setup( el_entry.instance_sec_idcs );
               } else {
                  for (let geo of el_entry.geom) {
                     if (geo === undefined)
                        console.warning("Processing viewer selection, undefined object for element", this.mgr.GetElement(el_idx));
                     else
                        obj_list.push(geo);
                  }
               }
            }

            if (obj_list.length == 0)
               continue;

            // Extract edge color (note, root colors), width from selection object.
            let sel_object = this.get_manager().GetElement(sel_id);
            let c = this.creator.RcCol(sel_object.fVisibleEdgeColor);
            this.rqt.RP_Outline_mat.setUniform("edgeColor", [ 2*c.r, 2*c.g, 2*c.b, 1 ]);
            this.rqt.render_outline();

            // for (const obj of instance_list) {
            //    obj.outlineMaterial.outline_instances_reset();
            // }

            this.rqt.RP_GBuffer.obj_list = [];
         }

         this.rqt.render_main_and_blend_outline();

         // XXXX here add rendering of overlay, e.g.:
         if (this.rqt.queue.used_fail_count == 0 && this.overlay_scene.children.length > 0) {
            this.rqt.render_overlay_and_blend_it();
         }
         // YYYY Or, it might be better to render overlay after the tone-mapping.
         // Eventually, if only overlay changes, we don't need to render the base-scene but
         // only overlay and re-merge them. Need to keep base textures alive in RendeQuTor.
         // Note that rgt.render_end() releases all std textures.

         if (this.rqt.queue.used_fail_count == 0) {
            // AMT: All render passess are drawn with the black bg
            //      except of the tone map render pass
            this.renderer.clearColor = '#' +  this.bgCol.getHexString() + '00';
            this.rqt.render_tone_map_to_screen();
            this.renderer.clearColor = "#00000000";
         }

         this.rqt.render_end();

         if (this.rqt.queue.used_fail_count > 0) {
            if (this._logLevel >= 2)
               console.log("GlViewerRCore render: not all programs compiled -- setting up render timer");
            setTimeout(this.render.bind(this), 200);
         }

         // if (this.controller.kind === "3D")
         //    window.requestAnimationFrame(this.render.bind(this));
      }

      render_for_picking(x, y, detect_depth)
      {
         // console.log("RENDER FOR PICKING", this.scene, this.camera, this.canvas, this.renderer);

         if (this.canvas.width <= 0 || this.canvas.height <= 0) return null;

         this.rqt.pick_begin(x, y);

         let state = this.rqt.pick(x, y, detect_depth);

         // console.log("pick state", state);

         if (state.object === null) {
            this.rqt.pick_end();
            return null;
         }

         let top_obj = state.object;
         while (top_obj.eve_el === undefined)
            top_obj = top_obj.parent;

         state.top_object = top_obj;
         state.eve_el = top_obj.eve_el;

         if (state.eve_el.fSecondarySelect)
            this.rqt.pick_instance(state);

         this.rqt.pick_end();

         state.w = this.canvas.width;
         state.h = this.canvas.height;
         state.mouse = new RC.Vector2( ((x + 0.5) / state.w) * 2 - 1,
                                      -((y + 0.5) / state.h) * 2 + 1 );

         let ctrl_obj = state.object;
         while (ctrl_obj.get_ctrl === undefined)
            ctrl_obj = ctrl_obj.parent;

         state.ctrl = ctrl_obj.get_ctrl(ctrl_obj, top_obj);
         return state;

      }

      render_for_Overlay_picking(x, y, detect_depth)
      {
         // console.log("RENDER FOR PICKING", this.scene, this.camera, this.canvas, this.renderer);

         if (this.canvas.width <= 0 || this.canvas.height <= 0) return null;

         this.rqt.pick_begin(x, y);

         let state_overlay = this.rqt.pick_overlay(x, y, detect_depth);

         if (state_overlay.object === null) {
            this.rqt.pick_end();
            return null;
         }

         let top_obj = state_overlay.object;
            while (top_obj.eve_el === undefined)
               top_obj = top_obj.parent;

            state_overlay.top_object = top_obj;
            state_overlay.eve_el = top_obj.eve_el;

            if (state_overlay.eve_el.fSecondarySelect)
               this.rqt.pick_instance_overlay(state_overlay);

            this.rqt.pick_end();

            state_overlay.w = this.canvas.width;
            state_overlay.h = this.canvas.height;
            state_overlay.mouse = new RC.Vector2( ((x + 0.5) / state_overlay.w) * 2 - 1,
                                         -((y + 0.5) / state_overlay.h) * 2 + 1 );

            let ctrl_obj = state_overlay.object;
            while (ctrl_obj.get_ctrl === undefined)
               ctrl_obj = ctrl_obj.parent;

            state_overlay.ctrl = ctrl_obj.get_ctrl(ctrl_obj, top_obj);
            return state_overlay;
      }

      //==============================================================================

      get selection_map() { return this._selection_map; }

      remove_selection_from_list(sid)
      {
         let idx = this._selection_list.indexOf(sid);
         if (idx >= 0)
            this._selection_list.splice(idx,1);
      }

      make_selection_last_in_list(sid)
      {
         this.remove_selection_from_list(sid);
         this._selection_list.push(sid);
      }

      //==============================================================================

      fixCssSize() {
         let s = this.canvas.canvasDOM.style;
         s.width  = this.canvas.canvasDOM.clientWidth  + "px";
         s.height = this.canvas.canvasDOM.clientHeight + "px";
      }

      floatCssSize() {
         let s = this.canvas.canvasDOM.style;
         s.width = "100%";
         s.height = "100%";
      }

      onResizeTimeout()
      {
         if ( ! this.canvas) {
            if (this._logLevel >= 2)
               console.log("GlViewerRCore onResizeTimeout -- canvas is not set yet.");
            return;
         }
         {
            let dome = this.get_view().getDomRef();

            let vid = this.get_view().sId + "--rcore";
            let hasCanvas = false;
            for (const child of dome.children) {
               if (child.id === vid) {
                  hasCanvas = true;
               }
            }

            if (hasCanvas === false)
            {
               dome.appendChild(this.canvas.parentDOM);
            }
         }
         this.floatCssSize();
         this.canvas.updateSize();
         let w = this.canvas.width;
         let h = this.canvas.height;
         //console.log("GlViewerRCore onResizeTimeout", w, h, "canvas=", this.canvas, this.canvas.width, this.canvas.height);

         this.camera.aspect = w / h;
         this.rqt.updateViewport(w, h);
         this.controls.update();

         this.render();

         this.fixCssSize();
      }


      //==============================================================================
      // RCore renderer event handlers etc.
      //==============================================================================

      //------------------------------------------------------------------------------
      // Highlight & Mouse move timeout handling
      //------------------------------------------------------------------------------

      clearHighlight()
      {
         if (this.highlighted_top_object)
         {
            this.highlighted_top_object.scene.clearHighlight(); // XXXX should go through manager
            this.highlighted_top_object = null;

            this.ttip.style.display = "none";
         }
      }

      removeMouseMoveTimeout()
      {
         if (this.mousemove_timeout)
         {
            clearTimeout(this.mousemove_timeout);
            delete this.mousemove_timeout;
         }
      }

      onMouseMoveTimeout(x, y)
      {
         delete this.mousemove_timeout;

         let pstate = this.render_for_picking(x * this.canvas.pixelRatio, y * this.canvas.pixelRatio, false);

         if ( ! pstate)
            return this.clearHighlight();

         let c = pstate.ctrl;
         let idx = c.extractIndex(pstate.instance);

         c.elementHighlighted(idx, null, pstate.object)

         if (this.highlighted_top_object !== pstate.top_object)
         {
            if (pstate.object && pstate.eve_el)
               this.ttip_text.innerHTML = c.getTooltipText(idx);
            else
               this.ttip_text.innerHTML = "";
         }
         this.highlighted_top_object = pstate.top_object;

         let dome  = this.controller.getView().getDomRef();
         let mouse = pstate.mouse;
         let offs  = (mouse.x > 0 || mouse.y < 0) ? this.getRelativeOffsets(dome) : null;

         if (mouse.x <= 0) {
            this.ttip.style.left  = (x + dome.offsetLeft + 10) + "px";
            this.ttip.style.right = null;
         } else {
            this.ttip.style.right = (this.canvas.canvasDOM.clientWidth - x + offs.right + 10) + "px";
            this.ttip.style.left  = null;
         }
         if (mouse.y >= 0) {
            this.ttip.style.top    = (y + dome.offsetTop + 10) + "px";
            this.ttip.style.bottom = null;
         } else {
            this.ttip.style.bottom = (this.canvas.canvasDOM.clientHeight - y + offs.bottom + 10) + "px";
            this.ttip.style.top = null;
         }

         this.ttip.style.display= "block";
      }

      remoteToolTip(msg)
      {
         if (this.ttip_text)
            this.ttip_text.innerHTML = msg;
         if (this.highlighted_top_object && this.ttip)
            this.ttip.style.display = "block";
      }

      getRelativeOffsets(elem)
      {
         // Based on:
         // https://stackoverflow.com/questions/3000887/need-to-calculate-offsetright-in-javascript

         let r = { left: 0, right: 0, top:0, bottom: 0 };

         let parent = elem.offsetParent;

         while (parent && getComputedStyle(parent).position === 'relative')
         {
            r.top    += elem.offsetTop;
            r.left   += elem.offsetLeft;
            r.right  += parent.offsetWidth  - (elem.offsetLeft + elem.offsetWidth);
            r.bottom += parent.offsetHeight - (elem.offsetTop  + elem.offsetHeight);

            elem   = parent;
            parent = parent.offsetParent;
         }

         return r;
      }

      //------------------------------------------------------------------------------
      // Mouse button handlers, selection, context menu
      //------------------------------------------------------------------------------

      removeMouseupListener()
      {
         if (this.mouseup_listener)
         {
            this.canvas.canvasDOM.removeEventListener('pointerup', this.mouseup_listener);
            this.mouseup_listener = 0;
         }
      }

      showContextMenu(event, menu)
      {
         // console.log("GLC::showContextMenu", this, menu)

         // See js/modules/menu/menu.mjs createMenu(), menu.add()

         let x = event.offsetX * this.canvas.pixelRatio;
         let y = event.offsetY * this.canvas.pixelRatio;
         let pstate = this.render_for_picking(x, y, true);

         menu.add("header:Context Menu");

         if (pstate) {
            if (pstate.eve_el)
            menu.add("Browse to " + (pstate.eve_el.fName || "element"), pstate.eve_el.fElementId, this.controller.invokeBrowseOf.bind(this.controller));

            let data = { "p": pstate, "v": this, "cctrl": this.controls};
            menu.add("Set Camera Center", data, this.setCameraCenter.bind(data));
         }

         menu.add("Reset camera", this.resetRenderer);

         if (RC.REveDevelMode) {
            menu.add("separator");
            menu.add("Show program names", 'names', this.showShaderJson);
            menu.add("Show programs", 'progs', this.showShaderJson);
         }

         menu.show(event);
      }

      setCameraCenter(data)
      {
         let pthis = data.v;

         let fov_rad_half = pthis.camera.fov * 0.5 * (Math.PI/180);
         let ftan = Math.tan(fov_rad_half);
         let x = data.p.mouse.x * data.p.w / data.p.h * data.p.depth * ftan;
         let y = data.p.mouse.y * data.p.depth * ftan;
         let e = new RC.Vector4(-data.p.depth, x, y, 1);

         // console.log("picked point >>> ", x, y, data.p.depth);
         // console.log("picked camera vector ", e);
         // pthis.camera.testMtx.dump();

         e.applyMatrix4(pthis.camera.testMtx);
         // console.log("picked word view coordinates ", e);

         pthis.controls.setCameraCenter(e.x, e.y, e.z);
         pthis.request_render();
      }

      handleMouseSelect(event)
      {
         let x = event.offsetX * this.canvas.pixelRatio;
         let y = event.offsetY * this.canvas.pixelRatio;
         let pstate = this.render_for_picking(x, y, false);

         if (pstate) {
            let c = pstate.ctrl;
            c.elementSelected(c.extractIndex(pstate.instance), event, pstate.object);
            // WHY ??? this.highlighted_scene = pstate.top_object.scene;
         } else {
            // XXXX HACK - handlersMIR senders should really be in the mgr

            this.controller.created_scenes[0].processElementSelected(null, [], event);
         }


      }

      handleOverlayMouseUp()
      {
         // console.log("handleOverlayMouseUp");
         if(this.firstMouseDown == false)
         {
            this.firstMouseDown = true;
            //this.overlay_scene.children[0].children[0].setNewPositionOffset(this.lastOffsetX, this.lastOffsetY);
            this.pickedOverlayObj.setNewPositionOffset(this.lastOffsetX, this.lastOffsetY);
            this.lastOffsetX = 0;
            this.lastOffsetY = 0;
            this.initialMouseX = 0;
            this.initialMouseY = 0;
            this.scale = false;
            this.initialSize = 0;
            this.controls.enablePan = true;
            this.controls.enableRotate = true;
         }
      }

      handleOverlayMouseDown(event)
      {
         // console.log("handleOverlayMouseDown");
         let x = event.offsetX * this.canvas.pixelRatio;
         let y = event.offsetY * this.canvas.pixelRatio;
         let overlay_pstate = this.render_for_Overlay_picking(x, y, false);



         if(this.firstMouseDown && overlay_pstate)
         {
             this.initialMouseX = x;
             this.initialMouseY = y;
             //let c = overlay_pstate.ctrl;
             this.pickedOverlayObj = overlay_pstate.object;
             this.firstMouseDown = false;

             if(event.button == 2)
             {
               this.scale = true;
               this.controls.enablePan = false;
               //this.initialSize = this.overlay_scene.children[0].children[0].fontSize;
               this.initialSize = this.pickedOverlayObj.fontSize;
             }
             else
               this.controls.enableRotate = false;

         }
      }

      handleOverlayMouseMove(event)
      {
         //console.log("handleOverlayMouseMove");

         if(!this.firstMouseDown)
         {
            let x = event.offsetX * this.canvas.pixelRatio;
            let y = event.offsetY * this.canvas.pixelRatio;

            if(!this.scale)
            {
               this.lastOffsetX = (x - this.initialMouseX)/this.canvas.width;
               this.lastOffsetY = (this.initialMouseY - y)/this.canvas.height;
               //this.overlay_scene.children[0].children[0].setOffset([this.lastOffsetX, this.lastOffsetY]);
               this.pickedOverlayObj.setOffset([this.lastOffsetX, this.lastOffsetY]);

            }
            else
            {
               //this.overlay_scene.children[0].children[0].fontSize = this.initialSize + (x - this.initialMouseX);
               this.pickedOverlayObj.fontSize = this.initialSize + (x - this.initialMouseX);

            }
            this.render();



         }
      }

      timeStampAttributesAndTextures() {
         try {
            this.renderer.glManager._textureManager.incrementTime();
            this.renderer.glManager._attributeManager.incrementTime();
         }
         catch (e) {
            console.error("Exception caught in timeStampAttributesAndTextures.", e);
         }
      }

      clearAttributesAndTextures() {
         try {
            let delta = 2;
            this.renderer.glManager._textureManager.deleteTextures(true, delta);
            this.renderer.glManager._attributeManager.deleteBuffers(true, delta);
         }
         catch (e) {
            console.error("Exception caught in clearAttributesAndTextures.", e);
         }
      }

      static showShaderCount = 0;
      showShaderJson(arg)
      {
         let progs = RC.ShaderLoader.sAllPrograms;
         let json;
         if (arg == "names") {
            let names = [];
            for (const p of progs) names.push(p);
            json = JSON.stringify(names, null, 2);
         } else if (arg == "progs") {
            let shdrs = this.rqt.renderer._shaderLoader.resolvePrograms( progs );
            json = JSON.stringify(shdrs, null, 2);
         } else {
            json = "bad option '" + arg + "' to GlViewerRCore.showShaderJson";
         }

         let count = GlViewerRCore.showShaderCount++;
         let extra = count ? ("-" + count) : "";

         const win = window.open("", "rcore.programs.json");
         win.document.open();
         win.document.write(`<html>
<head>
   <title>programs.json</title>
   <script>function save() {
     let b = new Blob( [ \`${json}\` ], {type: 'application/json'});
     const a = document.createElement('a');
     a.href = URL.createObjectURL(b);
     a.download = document.getElementById("filename").value; // 'programs'; // filename to download
     a.click();
   }</script>
</head>
<body>
   <form name="myform">
      <input type="text" id="filename" name="Filename" value="programs${extra}">
      <input type="button" onClick="save();" value="Save">
   </form>
   <pre> ${json} </pre>
</body>
         </html>`);
         win.document.close();
      }
   } // class GlViewerRCore

   return GlViewerRCore;
});
