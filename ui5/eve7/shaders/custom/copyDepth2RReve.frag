#version 300 es
precision highp float;
precision highp usampler2D;

struct Material {
    #if (TEXTURE)
        // usampler2D texture0; // this fails in MeshRenderer, uniform setter for material
        sampler2D texture0;
    #fi
};

uniform Material material;

#if (TEXTURE)
    in vec2 fragUV;
#fi

out vec4 color;


void main() {
    #if (TEXTURE)
        // color.r = 0.5370;
        // color.r = fragUV.x;
        // Logarithmic: (?)
        // color.r = texture(material.texture0, fragUV).r;
        // Linearize: (?)
        color.r = pow(2.0, texture(material.texture0, fragUV).r) - 1.0;

        // Or is this something with 1/z or w or what. Argh.
        // Perhaphs the best way forward is to:
        // 1. Implement picking on reduced target, say 16x16 or even 8x8
        // 2. Attach float buffer to picking shader and write z that you want in there.
        //    Or be a total pig and attach 3 buffers and store world xyz :)
    #fi
}