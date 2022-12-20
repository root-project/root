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
         this.eve_path = jsrp.substring(0, jsrp.length - 10) + 'rootui5sys/eve7/';

         this._logLevel = 1; // 0 - error, 1 - warning, 2 - info, 3 - debug

         if (this._logLevel > 2) {
            console.log("GlViewerRCore RQ_Mode:", this.RQ_Mode, "RQ_SSAA:", this.RQ_SSAA,
                        'RQ_MarkerScale', this.RQ_MarkerScale,  'RQ_LineScale', this.RQ_LineScale);
         }

         this._selection_map = {};
         this._selection_list = [];
      }

      init(controller)
      {
         super.init(controller);

         let pthis = this;

         if (!datGUI) {
            datGUI = new EVE.JSR.gui.GUI({ autoPlace: false, width: 220 });
            datGUI.domElement.id = 'posDatGUI';
            pthis.datGUIFolder = datGUI.addFolder("background")
            let dome = this.controller.getView().getParent().getParent();
            dome.getDomRef().appendChild(datGUI.domElement);
         }

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
               pthis.bootstrap();
            });
         } else {
            this.bootstrap();
         }
      }

      bootstrap()
      {
         this.creator = new EveElements(RC, this);
         // this.creator.useIndexAsIs = EVE.JSR.decodeUrl().has('useindx');

         RC.GLManager.sCheckFrameBuffer = false;
         RC.Object3D.sDefaultPickable = false;
         RC.PickingShaderMaterial.DEFAULT_PICK_MODE = RC.PickingShaderMaterial.PICK_MODE.UINT;

         this.createRCoreRenderer();
         this.controller.createScenes();
         this.controller.redrawScenes();
         this.setupEventHandlers();

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

      //==============================================================================

      createRCoreRenderer()
      {
         this.canvas = new RC.Canvas(this.get_view().getDomRef());
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
         this.renderer.clearColor = "#FFFFFF00"; // "#00000000";
         this.renderer.addShaderLoaderUrls(this.eve_path + RC.REveShaderPath);
         this.renderer.pickObject3D = true;

         // add dat GUI option to set background
         let eveView = this.controller.mgr.GetElement(this.controller.eveViewerId);
         let name = eveView.fName;
         name = name.substring(0, 3);
         let parName = name + "_WhiteBG";
         let conf = {}; conf[parName] = true;
         let pr = this;
         if (this.controller.getView().getViewData()) {
            datGUI.__folders.background.add(conf, parName).onChange(function (wbg) {
               conf[parName] = wbg;
               eveView.clearColor = wbg ? "#FFFFFF00" : "#00000000";
               pr.renderer.clearColor = eveView.clearColor;
               pr.request_render();
            });
         }
         else if (eveView.clearColor) {
            pr.renderer.clearColor = eveView.clearColor;
         }

         this.scene = new RC.Scene();

         this.lights = new RC.Group;
         this.lights.name = "Light container";
         this.scene.add(this.lights);

         let a_light = new RC.AmbientLight(new RC.Color(0xffffff), 0.05);
         this.lights.add(a_light);

         let light_3d_ctor = function(col, int, dist, decay, args) { return new RC.PointLight(col, int, dist, decay, args); };
         // let light_3d_ctor = function(col, int, dist, decay, args) { return new RC.DirectionalLight(col, int); };
         let light_2d_ctor = function(col, int) { return new RC.DirectionalLight(col, int); };

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

         this.rqt = new RC.RendeQuTor(this.renderer, this.scene, this.camera);
         if (this.RQ_Mode == "Direct")
         {
            this.rqt.initDirectToScreen();
         }
         else if (this.RQ_Mode == "Simple")
         {
            this.rqt.initSimple(this.RQ_SSAA);
            this.creator.SetupPointLineFacs(this.RQ_SSAA,
                                            this.RQ_MarkerScale * this.canvas.pixelRatio,
                                            this.RQ_LineScale   * this.canvas.pixelRatio);
         }
         else
         {
            this.rqt.initFull(this.RQ_SSAA);
            this.creator.SetupPointLineFacs(this.RQ_SSAA,
                                            this.RQ_MarkerScale * this.canvas.pixelRatio,
                                            this.RQ_LineScale   * this.canvas.pixelRatio);
         }
         this.rqt.updateViewport(w, h);
      }

      setupEventHandlers()
      {
         let dome = this.canvas.canvasDOM;

         // Setup tooltip
         this.ttip = document.createElement('div');
         this.ttip.setAttribute('class', 'eve_tooltip');
         this.ttip_text = document.createElement('div');
         this.ttip.appendChild(this.ttip_text);
         this.get_view().getDomRef().appendChild(this.ttip);

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

         if (this.camera.isPerspectiveCamera)
         {
            this.controls.setCamBaseMtx(new RC.Vector3(-1, 0, 0), new RC.Vector3(0, 1, 0)); //XOZ floor
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
            this.controls.setCamBaseMtx(new RC.Vector3(0, 0, 1), new RC.Vector3(0, 1, 0)); // XOY floor
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

         if (this.rqt.queue.used_fail_count == 0) {
            this.rqt.render_tone_map_to_screen();
            // this.rqt.render_final_to_screen();
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

         // console.log("pick result", state);
         return state;
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

         c.elementHighlighted(idx, null);

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
            c.elementSelected(c.extractIndex(pstate.instance), event);
            // WHY ??? this.highlighted_scene = pstate.top_object.scene;
         } else {
            // XXXX HACK - handlersMIR senders should really be in the mgr

            this.controller.created_scenes[0].processElementSelected(null, [], event);
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
