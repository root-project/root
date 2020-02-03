sap.ui.define([
   'sap/ui/core/Component',
   'sap/ui/core/UIComponent',
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   "sap/ui/core/ResizeHandler",
   'rootui5/eve7/lib/EveManager'
], function (Component, UIComponent, Controller, JSONModel, ResizeHandler, EveManager) {

   "use strict";

   // Use THREE.js renderer, scene, camera, etc. directly instead of through JSRootGeoPainter.
   var direct_threejs = true;

   // EveScene constructor function.
   var EveScene = null;

   let maybe_proto = Controller.extend("rootui5.eve7.Controller.GL", {


      //==============================================================================
      // Initialization, bootstrap, destruction & cleanup.
      //==============================================================================

      onInit : function()
      {
         // var id = this.getView().getId();

         let viewData = this.getView().getViewData();
         if (viewData)
         {
            this.setupManagerAndViewType(viewData);
         }
         else
         {
            UIComponent.getRouterFor(this).getRoute("View").attachPatternMatched(this.onViewObjectMatched, this);
         }

         this._load_scripts = false;
         this._render_html  = false;
         this.geo_painter   = null;

         JSROOT.AssertPrerequisites("geom", this.onLoadScripts.bind(this));
      },

      onLoadScripts: function()
      {
         var pthis = this;

         // one only can load EveScene after geometry painter
         sap.ui.define(['rootui5/eve7/lib/EveScene',    'rootui5/eve7/lib/OrbitControlsEve',
                        'rootui5/eve7/lib/OutlinePass', 'rootui5/eve7/lib/FXAAShader'],
                       function (eve_scene) {
                          EveScene = eve_scene;
                          pthis._load_scripts = true;
                          pthis.checkViewReady();
                       });
      },

      onViewObjectMatched: function(oEvent)
      {
         let args = oEvent.getParameter("arguments");

         this.setupManagerAndViewType(Component.getOwnerComponentFor(this.getView()).getComponentData(),
                                      args.viewName, JSROOT.$eve7tmp);

         delete JSROOT.$eve7tmp;
      },

      // Initialization that can be done immediately onInit or later through UI5 bootstrap callbacks.
      setupManagerAndViewType: function(data, viewName, moredata)
      {
         if (viewName)
         {
            data.standalone = viewName;
            data.kind       = viewName;
         }

         // console.log("VIEW DATA", data);

         if (moredata && moredata.mgr)
         {
            this.mgr        = moredata.mgr;
            this.eveViewerId  = moredata.eveViewerId;
            this.kind       = moredata.kind;
            this.standalone = viewName;

            this.checkViewReady();
         }
         else if (data.standalone && data.conn_handle)
         {
            this.mgr        = new EveManager();
            this.standalone = data.standalone;
            this.mgr.UseConnection(data.conn_handle);
         }
         else
         {
            this.mgr       = data.mgr;
            this.eveViewerId = data.eveViewerId;
            this.kind      = data.kind;
         }

         this.mgr.RegisterController(this);
         this.mgr.RegisterGlController(this);
      },

      // Called when HTML parent/container rendering is complete.
      onAfterRendering: function()
      {
         this._render_html = true;

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%").parent().css("overflow", "hidden");

         this.checkViewReady();
      },

      OnEveManagerInit: function()
      {
         // called when manager was updated, need only in standalone modes to detect own element id
         if (!this.standalone) return;

         let viewers = this.mgr.FindViewers();

         // first check number of views to create
         let found = null;
         for (let n = 0; n < viewers.length; ++n)
         {
            if (viewers[n].fName.indexOf(this.standalone) == 0)
            {
               found = viewers[n];
               break;
            }
         }
         if ( ! found) return;

         this.eveViewerId = found.fElementId;
         this.kind      = (found.fName == "Default Viewer") ? "3D" : "2D";

         this.checkViewReady();
      },

      // Function called from GuiPanelController.
      onExit: function()
      {
         // QQQQ EveManager does not have Unregister ... nor UnregisterController
         if (this.mgr) this.mgr.Unregister(this);
         // QQQQ plus, we should unregister this as gl-controller, too
      },

      // Checks if all initialization is performed and startup renderer.
      checkViewReady: function()
      {
         if ( ! this._load_scripts || ! this._render_html || ! this.eveViewerId)
         {
            return;
         }

         if (direct_threejs)
         {
            if (this.renderer) throw new Error("THREE renderer already created.");

            if( ! rootui5.eve7.Controller.GL.g_global_init_done)
            {
               rootui5.eve7.Controller.GL.g_global_init(this);
            }

            this.createThreejsRenderer();
            this.createScenes();
            this.redrawScenes();
            this.setupThreejsDomAndEventHandlers();
         }
         else
         {
            if (this.geo_painter) throw new Error("GeoPainter already created.");

            this.createGeoPainter();
         }
      },


      //==============================================================================
      // JSRoot GeoPainter creation etc
      //==============================================================================

      createGeoPainter: function()
      {
         let options = "outline";
         // options += " black, ";
         if (this.kind != "3D") options += ", ortho_camera";

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         this.geo_painter = JSROOT.Painter.CreateGeoPainter(this.getView().getDomRef(), null, options);

         // function used by TGeoPainter to create OutlineShader - for the moment remove from JSROOT
         this.geo_painter.createOutline = function(w,h)
         {
            this.outline_pass = new THREE.OutlinePass( new THREE.Vector2( w, h ), this._scene, this._camera );
            this.outline_pass.edgeStrength = 5.5;
            this.outline_pass.edgeGlow = 0.7;
            this.outline_pass.edgeThickness = 1.5;
            this.outline_pass.usePatternTexture = false;
            this.outline_pass.downSampleRatio = 1;
            this.outline_pass.glowDownSampleRatio = 3;

            // const sh = THREE.OutlinePass.selection_enum["select"]; // doesnt stand for spherical harmonics :P
            // THREE.OutlinePass.selection_atts[sh].visibleEdgeColor.set('#dd1111');
            // THREE.OutlinePass.selection_atts[sh].hiddenEdgeColor.set('#1111dd');

            this._effectComposer.addPass( this.outline_pass );

            this.fxaa_pass = new THREE.ShaderPass( THREE.FXAAShader );
            this.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / w, 1 / h );
            this.fxaa_pass.renderToScreen = true;
            this._effectComposer.addPass( this.fxaa_pass );
         }

         // assign callback function - when needed
         this.geo_painter.WhenReady(this.onGeoPainterReady.bind(this));

         this.geo_painter.AssignObject(null);

         this.geo_painter.prepareObjectDraw(null); // and now start everything
      },

      onGeoPainterReady: function(painter)
      {
         console.log("GL_controller::onGeoPainterReady");

         // AMT temporary here, should be set in camera instantiation time
         if (this.geo_painter._camera.type == "OrthographicCamera")
         {
            this.geo_painter._camera.left   = -this.getView().$().width();
            this.geo_painter._camera.right  =  this.getView().$().width();
            this.geo_painter._camera.top    =  this.getView().$().height();
            this.geo_painter._camera.bottom = -this.getView().$().height();
            this.geo_painter._camera.updateProjectionMatrix();
         }

         painter.eveGLcontroller = this;
         painter._controls.ProcessMouseMove = function(intersects)
         {
            var active_mesh = null, tooltip = null, resolve = null, names = [], geo_object, geo_index;

            // try to find mesh from intersections
            for (var k=0;k<intersects.length;++k)
            {
               var obj = intersects[k].object, info = null;
               if (!obj) continue;
               if (obj.geo_object) info = obj.geo_name; else
                  if (obj.stack) info = painter.GetStackFullName(obj.stack);
               if (info===null) continue;

               if (info.indexOf("<prnt>")==0)
                  info = painter.GetItemName() + info.substr(6);

               names.push(info);

               if (!active_mesh) {
                  active_mesh = obj;
                  tooltip = info;
                  geo_object = obj.geo_object;
                  if (obj.get_ctrl) {
                     geo_index = obj.get_ctrl().extractIndex(intersects[k]);
                     if ((geo_index !== undefined) && (typeof tooltip == "string")) tooltip += " indx:" + JSON.stringify(geo_index);
                  }
                  if (active_mesh.stack) resolve = painter.ResolveStack(active_mesh.stack);
               }
            }

            // painter.HighlightMesh(active_mesh, undefined, geo_object, geo_index); AMT override
            if (active_mesh && active_mesh.get_ctrl()){
               active_mesh.get_ctrl().elementHighlighted( 0xffaa33, geo_index);
            }
            else {
               var sl = painter.eveGLcontroller.created_scenes;
               for (var k=0; k < sl.length; ++k)
                   sl[k].clearHighlight();
            }


            if (painter.options.update_browser) {
               if (painter.options.highlight && tooltip) names = [ tooltip ];
               painter.ActivateInBrowser(names);
            }

            if (!resolve || !resolve.obj) return tooltip;

            var lines = JSROOT.GEO.provideInfo(resolve.obj);
            lines.unshift(tooltip);

            return { name: resolve.obj.fName, title: resolve.obj.fTitle || resolve.obj._typename, lines: lines };
         }

         // this.geo_painter._highlight_handlers = [ this ]; // register ourself for highlight handling
         this.last_highlight = null;

         // outline_pass passthrough
         this.outline_pass = this.geo_painter.outline_pass;

         var sz = this.geo_painter.size_for_3d();
         this.geo_painter._effectComposer.setSize( sz.width, sz.height);
         this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / sz.width, 1 / sz.height );

         // create only when geo painter is ready
         this.createScenes();
         this.redrawScenes();
         this.geo_painter.adjustCameraPosition(true);
         this.render();
      },


      //==============================================================================
      // THREE renderer creation, DOM/event handler setup, reset
      //==============================================================================

      createThreejsRenderer: function()
      {
         var w = this.getView().$().width();
         var h = this.getView().$().height();

         // console.log("createThreejsRenderer", this.kind, "w=", w, "h=", h);

         this.scene  = new THREE.Scene();
         // this.scene.fog = new THREE.FogExp2( 0xaaaaaa, 0.05 );

         if (this.kind === "3D")
         {
            this.camera = new THREE.PerspectiveCamera(75, w / h, 1, 5000);
         }
         else
         {
            this.camera = new THREE.OrthographicCamera(-w/2, w/2, -h/2, h/2, 0, 2000);
         }
         this.scene.add(this.camera);

         this.rot_center = new THREE.Vector3(0,0,0);

         this.renderer = new THREE.WebGLRenderer();
         this.renderer.setPixelRatio( window.devicePixelRatio );
         this.renderer.setSize( w, h );

         this.renderer.setClearColor( 0xffffff, 1 );

         // -------- Raycaster, lights, composer & FXAA and Outline passes.

         this.raycaster = new THREE.Raycaster();
         this.raycaster.params.Points.threshold = 4;   // ???
         this.raycaster.linePrecision           = 2.5; // ???

         // Lights are positioned in resetRenderer

         this.point_lights = new THREE.Object3D;
         this.point_lights.add( new THREE.PointLight( 0xff5050, 0.7 )); // R
         this.point_lights.add( new THREE.PointLight( 0x50ff50, 0.7 )); // G
         this.point_lights.add( new THREE.PointLight( 0x5050ff, 0.7 )); // B
         this.scene.add(this.point_lights);

         // var plane = new THREE.GridHelper(20, 20, 0x80d080, 0x8080d0);
         // this.scene.add(plane);

         this.composer = new THREE.EffectComposer(this.renderer);
         this.composer.addPass(new THREE.RenderPass(this.scene, this.camera));

         this.outline_pass = new THREE.OutlinePass(new THREE.Vector2(w, h), this.scene, this.camera);
         this.outline_pass.edgeStrength        = 5.5;
         this.outline_pass.edgeGlow            = 0.7;
         this.outline_pass.edgeThickness       = 1.5;
         this.outline_pass.usePatternTexture   = false;
         this.outline_pass.downSampleRatio     = 1;
         this.outline_pass.glowDownSampleRatio = 3;

         // This does not work ... seems it is not standard pass?
         // this.outline_pass.renderToScreen = true;
         // Tried hacking with this, but would apparently need to load it somehow, sigh.
         // var copyPass = new ShaderPass( CopyShader );
         // this.composer.addPass( new THREE.ShaderPass(CopyShader) );

         this.composer.addPass(this.outline_pass);

         this.fxaa_pass = new THREE.ShaderPass( THREE.FXAAShader );
         this.fxaa_pass.uniforms.resolution.value.set(0.5 / w, 0.5 / h);
         this.fxaa_pass.renderToScreen = true;

         this.composer.addPass( this.fxaa_pass );
      },

      setupThreejsDomAndEventHandlers: function()
      {
         this.getView().getDomRef().appendChild( this.renderer.domElement );

         // Setup tooltip
         this.ttip = document.createElement('div');
         this.ttip.setAttribute('class', 'eve_tooltip');
         this.ttip_text = document.createElement('div');
         this.ttip.appendChild(this.ttip_text);
         this.getView().getDomRef().appendChild(this.ttip);

         // Setup controls
         this.controls = new THREE.OrbitControlsEve( this.camera, this.getView().getDomRef() );

         this.controls.addEventListener('change', this.render.bind(this));

         // Setup some event pre-handlers
         var glc = this;

         this.renderer.domElement.addEventListener('mousemove', function(event) {

            if (event.movementX == 0 && event.movementY == 0)
               return;

            glc.removeMouseMoveTimeout();
            glc.clearHighlight();
            glc.removeMouseupListener();

            if (event.buttons === 0)
            {
               glc.mousemove_timeout = setTimeout(glc.onMouseMoveTimeout.bind(glc, event), 250);
            }
         });

         this.renderer.domElement.addEventListener('mouseleave', function(event) {

            glc.removeMouseMoveTimeout();
            glc.clearHighlight();
            glc.removeMouseupListener();
         });

         this.renderer.domElement.addEventListener('mousedown', function(event) {

            glc.removeMouseMoveTimeout();
            if (event.buttons != 1 && event.buttons != 2)  glc.clearHighlight();
            glc.removeMouseupListener();

            // console.log("GLC::mousedown", this, glc, event, event.offsetX, event.offsetY);

            glc.mouseup_listener = function(event2)
            {
               this.removeEventListener('mouseup', glc.mouseup_listener);

               if (event.buttons == 1) // Selection on mouseup without move
               {
                  glc.handleMouseSelect(event2);
               }
               else if (event.buttons == 2) // Context menu on delay without move
               {
                  // Was needed for "on press with timeout"
                  // glc.controls.resetMouseDown(event);

                  JSROOT.Painter.createMenu(glc, glc.showContextMenu.bind(glc, event2));
               }
            }

            this.addEventListener('mouseup', glc.mouseup_listener);
         });

         this.renderer.domElement.addEventListener('dblclick', function(event) {

            // console.log("GLC::dblclick", glc, event);
            glc.resetThreejsRenderer();
         });

         // Key-handlers go on window ...

         window.addEventListener('keydown', function(event) {

            // console.log("GLC::keydown", event.key, event.code, event);

            let handled = true;

            if (event.key == "t")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.material && node.material.linewidth )
                  {
                     if ( ! node.material.linewidth_orig) node.material.linewidth_orig = node.material.linewidth;

                     node.material.linewidth *= 1.2;
                  }
               });
            }
            else if (event.key == "e")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.material && node.material.linewidth )
                  {
                     if ( ! node.material.linewidth_orig) node.material.linewidth_orig = node.material.linewidth;

                     node.material.linewidth *= 0.8;
                  }
               });
            }
            else if (event.key == "r")
            {
               glc.scene.traverse( function( node ) {

                  if ( node.material && node.material.linewidth && node.material.linewidth_orig )
                  {
                     node.material.linewidth = node.material.linewidth_orig;
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

         // This will also call render().
         this.resetThreejsRenderer();

         ResizeHandler.register(this.getView(), this.onResize.bind(this));
      },

      /** Reset camera, lights based on scene bounding box. */
      resetThreejsRenderer: function()
      {
         let sbbox = new THREE.Box3();
         sbbox.setFromObject( this.scene );

         let posV = new THREE.Vector3; posV.subVectors(sbbox.max, this.rot_center);
         let negV = new THREE.Vector3; negV.subVectors(sbbox.min, this.rot_center);

         let extV = new THREE.Vector3; extV = negV; extV.negate(); extV.max(posV);
         let extR = extV.length();

         let lc = this.point_lights.children;
         lc[0].position.set( extR, extR, -extR);
         lc[1].position.set(-extR, extR,  extR);
         lc[2].position.set( extR, extR,  extR);

         if (this.camera.isPerspectiveCamera)
         {
            let posC = new THREE.Vector3(-0.7 * extR, 0.5 * extR, -0.7 * extR);

            this.camera.position.copy(posC);

            this.controls.screenSpacePanning = true;

            // console.log("resetThreejsRenderer 3D scene bbox ", sbbox, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         else
         {
            let posC = new THREE.Vector3(0, 0, 1000);

            this.camera.position.copy(posC);

            let ey = 1.02 * extV.y;
            let ex = ey / this.getView().$().height() * this.getView().$().width();
            this.camera.left   = -ex;
            this.camera.right  =  ex;
            this.camera.top    =  ey;
            this.camera.bottom = -ey;

            this.controls.resetOrthoPanZoom();

            this.controls.screenSpacePanning = true;
            this.controls.enableRotate = false;

            // console.log("resetThreejsRenderer 2D scene bbox ex ey", sbbox, ex, ey, ", camera_pos ", posC, ", look_at ", this.rot_center);
         }
         this.controls.target.copy( this.rot_center );

         this.composer.reset();

         this.controls.update();
      },


      //==============================================================================
      // Common functions between THREE and GeoPainter
      //==============================================================================

      /** returns container for 3d objects */
      // XXXX rename to getEveSceneContainer - fix also in EveScene.js
      getThreejsContainer: function(scene_name)
      {
         let parent = direct_threejs ? this.scene : this.geo_painter.getExtrasContainer();

         for (let k = 0; k < parent.children.length; ++k)
         {
            if (parent.children[k]._eve_name === scene_name)
            {
               return parent.children[k];
            }
         }

         let obj3d = new THREE.Object3D();
         obj3d._eve_name = scene_name;
         parent.add(obj3d);
         return obj3d;
      },

      /** Render (use GeoPainter or not, initialize dom / controls on first entry)
          XXXX Init stuff needs to go elsewhere.
          XXXX Camera pos etc should also be reset on first arrival of scenes.
       */
      render: function()
      {
         if (direct_threejs)
         {
            // console.log(this.camera, this.controls, this.controls.target);

            // Render through composer:
            this.composer.render( this.scene, this.camera );

            // or directly through renderer:
            // this.renderer.render( this.scene, this.camera );
         }
         else
         {
            if (this.geo_painter) {
               this.geo_painter.Render3D();
            }
         }
      },

      createScenes: function()
      {
         if (this.created_scenes !== undefined) return;
         this.created_scenes = [];

         // only when rendering completed - register for modify events
         let element = this.mgr.GetElement(this.eveViewerId);

         // loop over scene and add dependency
         for (let scene of element.childs)
         {
            this.created_scenes.push(new EveScene(this.mgr, scene, this));
         }
      },

      redrawScenes: function()
      {
         for (let s of this.created_scenes)
         {
            s.redrawScene();
         }
      },


      /// invoked from ResizeHandler
      onResize: function(event)
      {
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 250); // minimal latency
      },

      onResizeTimeout: function()
      {
         delete this.resize_tmout;

         // console.log("onResizeTimeout", this.camera);

         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");

         if (this.geo_painter)
         {
            this.geo_painter.CheckResize();
            if (this.geo_painter.fxaa_pass)
               this.geo_painter.fxaa_pass.uniforms[ 'resolution' ].value.set( 1 / this.geo_painter._scene_width, 1 / this.geo_painter._scene_height );
            return;
         }

         let w = this.getView().$().width();
         let h = this.getView().$().height();

         if (this.camera.isPerspectiveCamera)
         {
            this.camera.aspect = w / h;
         }
         else
         {
            this.camera.left  =  this.camera.bottom / h * w;
            this.camera.right = -this.camera.left;
            this.camera.updateProjectionMatrix();
         }
         this.camera.updateProjectionMatrix();

         this.renderer.setSize(w, h);
         this.outline_pass.setSize(w, h);
         this.fxaa_pass.uniforms.resolution.value.set(0.5 / w, 0.5 / h);

         this.composer.reset();
         this.controls.update();
         this.render();
      },


      //==============================================================================
      // THREE renderer event handlers etc.
      //==============================================================================

      //------------------------------------------------------------------------------
      // Highlight & Mouse move timeout handling
      //------------------------------------------------------------------------------

      clearHighlight: function()
      {
         if (this.highlighted_scene)
         {
            this.highlighted_scene.clearHighlight(); // XXXX should go through manager
            this.highlighted_scene = 0;

            this.ttip.style.display = "none";
        }
      },

      removeMouseMoveTimeout: function()
      {
         if (this.mousemove_timeout)
         {
            clearTimeout(this.mousemove_timeout);
            this.mousemove_timeout = 0;
         }
      },

      onMouseMoveTimeout: function(event)
      {
         this.mousemove_timeout = 0;

         let x = event.offsetX;
         let y = event.offsetY;
         let w = this.getView().$().width();
         let h = this.getView().$().height();

         // console.log("GLC::onMouseMoveTimeout", this, event, x, y);

         let mouse = new THREE.Vector2( ((x + 0.5) / w) * 2 - 1, -((y + 0.5) / h) * 2 + 1 );

         this.raycaster.setFromCamera(mouse, this.camera);

         let intersects = this.raycaster.intersectObjects(this.scene.children, true);

         let o = null, c = null;

         for (let i = 0; i < intersects.length; ++i)
         {
            o = intersects[i].object;
            if (o.get_ctrl)
            {
               c = o.get_ctrl();
               c.elementHighlighted(c.extractIndex(intersects[i]));

               this.highlighted_scene = c.obj3d.scene;

               break;
            }
         }

         if (c)
         {
            if (c.obj3d && c.obj3d.eve_el)
               this.ttip_text.innerHTML = c.obj3d.eve_el.fTitle || c.obj3d.eve_el.fName || "";
            else
               this.ttip_text.innerHTML = "";

            let del  = this.getView().getDomRef();
            let offs = (mouse.x > 0 || mouse.y < 0) ? this.getRelativeOffsets(del) : null;

            if (mouse.x <= 0) {
               this.ttip.style.left  = (x + del.offsetLeft + 10) + "px";
               this.ttip.style.right = null;
            } else {
               this.ttip.style.right = (w - x + offs.right + 10) + "px";
               this.ttip.style.left  = null;
            }
            if (mouse.y >= 0) {
               this.ttip.style.top    = (y + del.offsetTop + 10) + "px";
               this.ttip.style.bottom = null;
            } else {
               this.ttip.style.bottom = (h - y + offs.bottom + 10) + "px";
               this.ttip.style.top = null;
            }
            this.ttip.style.display= "block";
         }
      },

      getRelativeOffsets: function(elem)
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
      },

      //------------------------------------------------------------------------------
      // Mouse button handlers, selection, context menu
      //------------------------------------------------------------------------------

      removeMouseupListener: function()
      {
         if (this.mouseup_listener)
         {
            this.renderer.domElement.removeEventListener('mouseup', this.mouseup_listener);
            this.mouseup_listener = 0;
         }
      },

      showContextMenu: function(event, menu)
      {
         // console.log("GLC::showContextMenu", this, menu)

         // See js/scripts/JSRootPainter.jquery.js JSROOT.Painter.createMenu(), menu.add()

         menu.add("header:Context Menu");

         menu.add("Reset camera", this.resetThreejsRenderer);

         menu.add("separator");

         let fff = this.defaultContextMenuAction;
         menu.add("sub:Sub Test");
         menu.add("Foo",     'foo', fff);
         menu.add("Bar",     'bar', fff);
         menu.add("Baz",     'baz', fff);
         menu.add("endsub:");

         menu.show(event);
      },

      defaultContextMenuAction: function(arg)
      {
         console.log("GLC::defaultContextMenuAction", this, arg);
      },

      handleMouseSelect: function(event)
      {
         let x = event.offsetX;
         let y = event.offsetY;
         let w = this.getView().$().width();
         let h = this.getView().$().height();

         // console.log("GLC::handleMouseSelect", this, event, x, y);

         let mouse = new THREE.Vector2( ((x + 0.5) / w) * 2 - 1, -((y + 0.5) / h) * 2 + 1 );

         this.raycaster.setFromCamera(mouse, this.camera);

         let intersects = this.raycaster.intersectObjects(this.scene.children, true);

         let o = null, c = null;

         for (let i = 0; i < intersects.length; ++i)
         {
            o = intersects[i].object;

            if (o.get_ctrl)
            {
               c = o.get_ctrl();
               c.event = event;

               c.elementSelected(c.extractIndex(intersects[i]));

               this.highlighted_scene = o.scene;

               break;
            }
         }

         if ( ! c)
         {
            // XXXX HACK - handlersMIR senders should really be in the mgr

            this.created_scenes[0].processElementSelected(null, [], event);
         }
      },

   });


   //==============================================================================
   // Global / non-prototype members
   //==============================================================================

   let GLC = rootui5.eve7.Controller.GL;

   GLC.g_highlight_update = function(mgr)
   {
      let sa = THREE.OutlinePass.selection_atts;
      let gs = mgr.GetElement(mgr.global_selection_id);
      let gh = mgr.GetElement(mgr.global_highlight_id);

      sa[0].visibleEdgeColor.setStyle(JSROOT.Painter.root_colors[gs.fVisibleEdgeColor]);
      sa[0].hiddenEdgeColor .setStyle(JSROOT.Painter.root_colors[gs.fHiddenEdgeColor]);
      sa[1].visibleEdgeColor.setStyle(JSROOT.Painter.root_colors[gh.fVisibleEdgeColor]);
      sa[1].hiddenEdgeColor .setStyle(JSROOT.Painter.root_colors[gh.fHiddenEdgeColor]);
   }

   GLC.g_global_init = function(glc)
   {
      glc.mgr.RegisterSelectionChangeFoo(GLC.g_highlight_update);

      GLC.g_highlight_update(glc.mgr);

      GLC.g_global_init_done = true;
   }

   return maybe_proto;
});
