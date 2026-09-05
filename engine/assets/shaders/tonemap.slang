import fullscreen;

// Final post-process pass: turns the linear HDR scene into the pixels the display gets
//
// Everything before writes unbounded linear radiance, so this is the one place deciding how that maps to a
// screen - which is why it is a pass rather than the tail of every material shader

struct TonemapParams
{
    // Linear multiplier applied before the curve. 1 is neutral
    float exposure;

    // 0 = Reinhard, 1 = ACES (fitted). Reinhard is what the material shaders used to do inline, so it is the
    // default that reproduces the previous look
    uint mode;

    // Encode gamma. 2.2 for an UNORM swapchain, which is what this engine picks
    float gamma;

    // Grading, applied after the tone curve and before the encode - the curve decides what fits on screen,
    // grading decides how it looks. 1 is neutral for both
    float contrast;
    float saturation;

    // Corner darkening. 0 disables it entirely
    float vignette;

    // Film grain amount. 0 disables it
    float grain;

    // Seconds, so the grain pattern moves rather than sitting as a static overlay
    float time;

    // std140 rounds a uniform block up to a multiple of 16, so this block is 48 bytes even though its
    // fields end at 36. The C++ struct pads to match - a mismatch here is silent, since the fields still
    // line up and only the descriptor range comes out short
    float _pad0;
};

[[vk::binding(0,0)]]
ConstantBuffer<TonemapParams> gParams;

[[vk::binding(1,0)]]
Sampler2D gSceneColor;

typedef FullscreenVSOutput VSOutput;

// Full-screen triangle from the vertex id alone - no vertex buffer, no index buffer, and one fewer triangle
// than a quad with no diagonal seam through the middle
[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    return fullscreenTriangle(vertexId);
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Reinhard: x / (1 + x). Cheap, desaturates highlights, and is what the material shaders applied before the
// tonemap moved here
float3 tonemapReinhard(float3 color)
{
    return color / (color + float3(1.0, 1.0, 1.0));
}

// Narkowicz's fit of the ACES filmic curve. Holds saturation into the highlights far better than Reinhard,
// at the cost of a slightly darker, contrastier midtone
float3 tonemapACES(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

// Ordered dither, ±half a code value at 8-bit. The output is 8-bit UNORM and a tonemapped gradient
// quantizes into visible bands without it; the noise is below the quantization step, so it costs nothing
// perceptually and breaks the banding up
float ditherOffset(float2 pixel)
{
    // Interleaved gradient noise, same function the shadow filter rotates its kernel with
    float noise = frac(52.9829189 * frac(dot(pixel, float2(0.06711056, 0.00583715))));
    return (noise - 0.5) / 255.0;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 color = gSceneColor.Sample(input.uv).rgb * gParams.exposure;

    color = gParams.mode == 1 ? tonemapACES(color) : tonemapReinhard(color);

    // Contrast pivots around middle grey so raising it darkens shadows and lifts highlights, rather than
    // brightening the whole image the way a plain multiply would
    const float kMiddleGrey = 0.18;
    color = max((color - kMiddleGrey) * gParams.contrast + kMiddleGrey, float3(0.0, 0.0, 0.0));

    color = max(lerp(float3(luminance(color), luminance(color), luminance(color)), color, gParams.saturation),
                float3(0.0, 0.0, 0.0));

    // Encode for the display. The swapchain is a UNORM format, so nothing does this for us
    color = pow(max(color, float3(0.0, 0.0, 0.0)), float3(1.0 / gParams.gamma, 1.0 / gParams.gamma, 1.0 / gParams.gamma));

    // Vignette, applied in display space so it darkens what the viewer actually sees rather than being
    // re-shaped by the tone curve afterwards. Squared distance from centre keeps the falloff smooth
    if (gParams.vignette > 0.0)
    {
        float2 centered = input.uv - 0.5;
        float falloff = 1.0 - saturate(dot(centered, centered) * gParams.vignette * 2.0);
        color *= falloff * falloff;
    }

    // Grain last: it is a property of the image being displayed, not of the scene, so nothing downstream
    // should filter or grade it. Animated by time, or it reads as dirt on the lens instead of film
    if (gParams.grain > 0.0)
    {
        float noise = frac(sin(dot(input.uv + frac(gParams.time), float2(12.9898, 78.233))) * 43758.5453);
        color += (noise - 0.5) * gParams.grain;
    }

    color += ditherOffset(input.position.xy);

    return float4(color, 1.0);
}
