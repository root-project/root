import * as RC from "../rnr_core/RenderCore.js";

function iterateSceneR(object, callback)
{
    // if (object === null || object === undefined) return;

    if (object.children.length > 0) {
        for (let i = 0; i < object.children.length; i++) {
            iterateSceneR(object.children[i], callback);
        }
    }

    callback(object);
}

export class RendeQuTor
{
    constructor(renderer, scene, camera)
    {
        this.renderer = renderer;
        this.scene    = scene;
        this.camera   = camera;
        this.queue    = new RC.RenderQueue(renderer);
        this.pqueue   = new RC.RenderQueue(renderer);
        this.make_PRP_plain();

        // Depth extraction somewhat works , get float but in some unknown coordinates :)
        // If you enable this, also enable EXT_color_buffer_float in GlViewerRCore.createRCoreRenderer
        // this.make_PRP_depth2r();
        // See also comments in shaders/custom/copyDepth2RReve.frag

        this.SSAA_value = 1;

        const nearPlane = 0.0625; // XXXX - pass to view_setup(vport, nfclip)
        const farPlane  = 8192;   // XXXX

        // Why object.pickable === false in Initialize functions ???
        // How is outline supposed to work ???
        // Picking ???

        this.OriginalMats = [];
        this.MultiMats    = [];

        // RenderQueue ... subclass or envelop?
        // For every pass, store object + resize behaviour
    }

    initDirectToScreen()
    {
        this.make_RP_DirectToScreen();
    }

    initSimple(ssaa_val)
    {
        this.SSAA_value = ssaa_val;

        this.make_RP_SSAA_Super();
        //this.make_RP_SSAA_Down();
        this.make_RP_ToScreen();
        this.RP_ToScreen.input_texture = "color_ssaa_super";
    }

    initFull(ssaa_val)
    {
        this.SSAA_value = ssaa_val;

        this.make_RP_SSAA_Super();
        this.make_RP_HighPassGaussBloom();
        // this.make_RP_SSAA_Down(); this.RP_SSAA_Down.input_texture = "color_bloom";
        this.make_RP_ToScreen();
        this.RP_ToScreen.input_texture = "color_bloom";
    }

    updateViewport(w, h)
    {
        let vp = { width: w, height: h };
        let rq = this.queue._renderQueue;
        for (let i = 0; i < rq.length; i++)
        {
            rq[i].view_setup(vp);
        }
        rq = this.pqueue._renderQueue;
        for (let i = 0; i < rq.length; i++)
        {
            rq[i].view_setup(vp);
        }
    }

    render()
    {
        this.queue.render();
    }

    pick()
    {
        let foo = this.pqueue.render();
        console.log(foo);

        {
            let glman  = this.renderer.glManager;
            let gl     = this.renderer.gl;
            let texref = this.pqueue._textureMap["depthr32f_picking"];
            let tex    = glman.getTexture(texref);

            console.log("Dumper:", glman, gl, texref, tex);

            const fb = gl.createFramebuffer();
			gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
			gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

            let x = this.renderer._pickCoordinateX;
            let y = this.renderer._canvas.height - this.renderer._pickCoordinateY;
            console.log(x, y);

            let d = new Float32Array(9);
            gl.readPixels(x-1, y-1, 3, 3, gl.RED, gl.FLOAT, d);
            console.log("Pick depth at", x, ",", y, ":", d);
            /*
            let d = new Uint32Array(9);
            gl.readPixels(x-1, y-1, 3, 3, gl.RED, gl.UNSIGNED_INT, d);
            console.log("Pick depth:", d;
            */

            gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
            gl.deleteFramebuffer(fb);
        }
    }


    //=============================================================================
    // Picking RenderPasses
    //=============================================================================

    make_PRP_plain()
    {
        var pthis = this;

        this.PRP_plain = new RC.RenderPass(
            RC.RenderPass.BASIC,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) { return { scene: pthis.scene, camera: pthis.camera }; },
            RC.RenderPass.TEXTURE,
            null,
            "depth_picking",
            [ { id: "color_picking", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG } ]
        );
        this.PRP_plain.view_setup = function (vport) { this.viewport = vport; };

        this.pqueue.pushRenderPass(this.PRP_plain);
    }

    make_PRP_depth2r()
    {
        this.PRP_depth2r_mat = new RC.CustomShaderMaterial("copyDepth2RReve");
        this.PRP_depth2r_mat.lights = false;
        var pthis = this;

        this.PRP_depth2r = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.PRP_depth2r_mat, textures: [ textureMap["depth_picking"] ] };
            },
            RC.RenderPass.TEXTURE,
            null,
            null,
            [ { id: "depthr32f_picking", textureConfig: RC.RenderPass.FULL_FLOAT_R32F_TEXTURE_CONFIG } ]
            // [ { id: "depthr32f_picking", textureConfig: RC.RenderPass.DEFAULT_R32UI_TEXTURE_CONFIG } ]
        );
        this.PRP_depth2r.view_setup = function (vport) { this.viewport = vport; };

        this.pqueue.pushRenderPass(this.PRP_depth2r);
    }

    //=============================================================================

    make_RP_DirectToScreen()
    {
        var pthis = this;

        this.RP_DirectToScreen = new RC.RenderPass(
            RC.RenderPass.BASIC,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) { return { scene: pthis.scene, camera: pthis.camera }; },
            RC.RenderPass.SCREEN,
            null
        );
        this.RP_DirectToScreen.view_setup = function (vport) { this.viewport = vport; };

        this.queue.pushRenderPass(this.RP_DirectToScreen);
    }

    //=============================================================================

    make_RP_SSAA_Super()
    {
        var pthis = this;

        this.RP_SSAA_Super = new RC.RenderPass(
            // Rendering pass type
            RC.RenderPass.BASIC,

            // Initialize function
            function (textureMap, additionalData) {
                iterateSceneR(pthis.scene, function(object){
                    if (object.pickable === false || object instanceof RC.Text2D || object instanceof RC.IcoSphere) {
                        object.visible = true;
                        return;
                    }
                    pthis.OriginalMats.push(object.material);
                });
            },

            // Preprocess function
            function (textureMap, additionalData) {
                let m_index = 0;

                iterateSceneR(pthis.scene, function(object) {
                    if(object.pickable === false || object instanceof RC.Text2D || object instanceof RC.IcoSphere) {
                        object.visible = true;
                        return;
                    }
                    object.material = pthis.OriginalMats[m_index];
                    m_index++;
                });

                return { scene: pthis.scene, camera: pthis.camera };
            },

            // Target
            RC.RenderPass.TEXTURE,

            // Viewport
            null,

            // Bind depth texture to this ID
            "depthDefaultDefaultMaterials",

            [ { id: "color_ssaa_super", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG } ]
        );
        this.RP_SSAA_Super.view_setup = function (vport) { this.viewport = { width: vport.width*pthis.SSAA_value, height: vport.height*pthis.SSAA_value }; };

        this.queue.pushRenderPass(this.RP_SSAA_Super);
    }

    make_RP_SSAA_Down()
    {
        this.RP_SSAA_Down_mat = new RC.CustomShaderMaterial("copyTexture");
        this.RP_SSAA_Down_mat.lights = false;
        var pthis = this;

        this.RP_SSAA_Down = new RC.RenderPass(
            // Rendering pass type
            RC.RenderPass.POSTPROCESS,

            // Initialize function
            function (textureMap, additionalData) {},

            // Preprocess function
            function (textureMap, additionalData) {
                return { material: pthis.RP_SSAA_Down_mat, textures: [textureMap[pthis.input_texture]] };
            },

            // Target
            RC.RenderPass.TEXTURE,

            // Viewport
            null,

            // Bind depth texture to this ID
            null,

            [ { id: "color_ssaa_down", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG } ]
        );
        this.RP_SSAA_Down.input_texture = "color_ssaa_super";
        this.RP_SSAA_Down.view_setup = function(vport) { this.viewport = vport; };

        this.queue.pushRenderPass(this.RP_SSAA_Down);
    }

    //=============================================================================

    make_RP_ToScreen()
    {
        this.RP_ToScreen_mat = new RC.CustomShaderMaterial("copyTexture");
        this.RP_ToScreen_mat.lights = false;
        var pthis = this;

        this.RP_ToScreen = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.RP_ToScreen_mat, textures: [ textureMap[this.input_texture] ] }; // XXXX pthis or this ????
            },
            RC.RenderPass.SCREEN,
            null
        );
        this.RP_ToScreen.input_texture = "color_ssaa_down";
        this.RP_ToScreen.view_setup = function(vport) { this.viewport = vport; };

        this.queue.pushRenderPass(this.RP_ToScreen);
    }

    //=============================================================================

    make_RP_HighPassGaussBloom()
    {
        var pthis = this;
        // let hp = new RC.CustomShaderMaterial("highPass", {MODE: RC.HIGHPASS_MODE_BRIGHTNESS, targetColor: [0.2126, 0.7152, 0.0722], threshold: 0.75});
        let hp = new RC.CustomShaderMaterial("highPass", { MODE: RC.HIGHPASS_MODE_DIFFERENCE,
                                             targetColor: [0x0/255, 0x0/255, 0xff/255], threshold: 0.1});
        console.log("XXXXXXXX", hp);
        // let hp = new RC.CustomShaderMaterial("highPassReve");
        this.RP_HighPass_mat = hp;
        this.RP_HighPass_mat.lights = false;

        this.RP_HighPass = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.RP_HighPass_mat, textures: [textureMap["color_ssaa_super"]] };
            },
            RC.RenderPass.TEXTURE,
            null,
            // XXXXXX MT: this was "dt", why not null ????
            null, // "dt",
            [ {id: "color_high_pass", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG} ]
        );
        this.RP_HighPass.view_setup = function (vport) { this.viewport = { width: vport.width*pthis.SSAA_value, height: vport.height*pthis.SSAA_value }; };
        this.queue.pushRenderPass(this.RP_HighPass);

        this.RP_Gauss1_mat = new RC.CustomShaderMaterial("gaussBlur", {horizontal: true, power: 1.0});
        this.RP_Gauss1_mat.lights = false;

        this.RP_Gauss1 = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.RP_Gauss1_mat, textures: [textureMap["color_high_pass"]] };
            },
            RC.RenderPass.TEXTURE,
            null,
            null,
            [ {id: "color_gauss_half", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG} ]
        );
        this.RP_Gauss1.view_setup = function (vport) { this.viewport = { width: vport.width*pthis.SSAA_value, height: vport.height*pthis.SSAA_value }; };
        this.queue.pushRenderPass(this.RP_Gauss1);

        this.RP_Gauss2_mat = new RC.CustomShaderMaterial("gaussBlur", {horizontal: false, power: 1.0});
        this.RP_Gauss2_mat.lights = false;

        this.RP_Gauss2 = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.RP_Gauss2_mat, textures: [textureMap["color_gauss_half"]] };
            },
            RC.RenderPass.TEXTURE,
            null,
            null,
            [ {id: "color_gauss_full", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG} ]
        );
        this.RP_Gauss2.view_setup = function (vport) { this.viewport = { width: vport.width*pthis.SSAA_value, height: vport.height*pthis.SSAA_value }; };
        this.queue.pushRenderPass(this.RP_Gauss2);

        this.RP_Bloom_mat = new RC.CustomShaderMaterial("bloom");
        this.RP_Bloom_mat.lights = false;

        this.RP_Bloom = new RC.RenderPass(
            RC.RenderPass.POSTPROCESS,
            function (textureMap, additionalData) {},
            function (textureMap, additionalData) {
                return { material: pthis.RP_Bloom_mat, textures: [textureMap["color_gauss_full"], textureMap["color_ssaa_super"]] };
            },
            RC.RenderPass.TEXTURE,
            null,
            null,
            [ {id: "color_bloom", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG} ]
        );
        this.RP_Bloom.view_setup = function (vport) { this.viewport = { width: vport.width*pthis.SSAA_value, height: vport.height*pthis.SSAA_value }; };
        this.queue.pushRenderPass(this.RP_Bloom);
    }
};


/*
export const RenderPass_MainMulti = new RC.RenderPass(
    // Rendering pass type
    RC.RenderPass.BASIC,

    // Initialize function
    function (textureMap, additionalData) {
        iterateSceneR(scene, function(object){
            if(object.pickable === false || object instanceof RC.Text2D || object instanceof RC.IcoSphere) {
                object.visible = false;
                //GL_INVALID_OPERATION : glDrawElementsInstancedANGLE: buffer format and fragment output variable type incompatible
                //Program has no frag output at location 1, but destination draw buffer has an attached image.
                return;
            }
            const multi = new RC.CustomShaderMaterial("multi", {near: nearPlane, far: farPlane});
            multi.side = RC.FRONT_AND_BACK_SIDE; //reather use depth from default materials
            MultiMats.push(multi);
        });
    },

    // Preprocess function
    function (textureMap, additionalData) {
        let m_index = 0;

        iterateSceneR(scene, function(object){
            if(object.pickable === false || object instanceof RC.Text2D || object instanceof RC.IcoSphere) {
                object.visible = false;
                return;
            }
            object.material = MultiMats[m_index];
            m_index++;
        });


        return { scene: scene, camera: camera };
    },

    // Target
    RC.RenderPass.TEXTURE,

    // Viewport
    { width: predef_width*SSAA_value, height: predef_height*SSAA_value },

    // Bind depth texture to this ID
    "depthDefaultMultiMaterials",

    [
        {id: "depth", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG},
        {id: "normal", textureConfig: RC.RenderPass.DEFAULT_RGB_TEXTURE_CONFIG},
        {id: "viewDir", textureConfig: RC.RenderPass.DEFAULT_RGB_TEXTURE_CONFIG},
        {id: "camDist", textureConfig: RC.RenderPass.DEFAULT_RGBA16F_TEXTURE_CONFIG}
    ]
    );
*/

/*
const outline = new RC.CustomShaderMaterial("outline", {scale: 1.0*SSAA_value, edgeColor: [1.0, 1.0, 1.0, 1.0]});
outline.lights = false;
export const RenderPass_Outline = new RC.RenderPass(
    // Rendering pass type
    RC.RenderPass.POSTPROCESS,

    // Initialize function
    function (textureMap, additionalData) {
    },

    // Preprocess function
    function (textureMap, additionalData) {
        return {material: outline, textures: [textureMap["depthDefaultMultiMaterials"], textureMap["normal"], textureMap["viewDir"], textureMap["color_bloom"]]};
    },

    // Target
    RC.RenderPass.TEXTURE,

    // Viewport
    { width: predef_width*SSAA_value, height: predef_height*SSAA_value },

    // Bind depth texture to this ID
    null,

    [
        {id: "color_outline", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG}
    ]
    );

const fog = new RC.CustomShaderMaterial("fog", {MODE: 1, fogColor: [0.5, 0.4, 0.45, 0.8]});
fog.lights = false;
export const RenderPass_Fog = new RC.RenderPass(
    // Rendering pass type
    RC.RenderPass.POSTPROCESS,

    // Initialize function
    function (textureMap, additionalData) {
    },

    // Preprocess function
    function (textureMap, additionalData) {
        //return {material: fog, textures: [textureMap["color_outline"], textureMap["depthDefaultDefaultMaterials"]]}; //grid jumps on depth buffer
        return {material: fog, textures: [textureMap["color_outline"], textureMap["camDist"]]}; //grid has specific shader for extruding geometry, even if implemented, it would jump around
    },

    // Target
    RC.RenderPass.TEXTURE,

    // Viewport
    { width: predef_width*SSAA_value, height: predef_height*SSAA_value },

    // Bind depth texture to this ID
    null,

    [
        {id: "color_fog", textureConfig: RC.RenderPass.DEFAULT_RGBA_TEXTURE_CONFIG}
    ]
    );
*/