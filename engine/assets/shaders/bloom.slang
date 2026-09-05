import fullscreen;

// Bloom: the wide, soft glow around anything bright enough to overwhelm a lens
//
// Linear HDR, before the tonemap, because that is the only place "bright" still means anything - afterwards
// every highlight is squashed into [0,1] and the information is gone. It is what makes an emissive material
// read as a light source rather than as bright paint
//
// Three stages: threshold + downsample, upsample with additive accumulation, composite back over the scene

struct BloomParams
{
    // 1 / source resolution, so filter taps land on texel centres of whatever mip is being read
    float2 inverseSourceSize;

    // Luminance where bloom starts. Below it a pixel contributes nothing
    float threshold;

    // Width of the soft shoulder below the threshold - a hard cut-off makes bloom pop on and off as a
    // highlight crosses it, which reads as flickering on any moving specular
    float knee;

    // Upsample filter width, in source texels. Larger spreads the glow wider per step
    float filterRadius;

    // How much bloom is mixed back over the scene at the end
    float strength;

    float2 _pad0;
};

[[vk::push_constant]]
BloomParams gParams;

[[vk::binding(0,0)]]
Sampler2D gSource;

// Only bound for the composite stage, where the original scene has to be added back
[[vk::binding(1,0)]]
Sampler2D gSceneColor;

typedef FullscreenVSOutput VSOutput;

[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    return fullscreenTriangle(vertexId);
}

float luminance(float3 color)
{
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

// Soft-knee threshold: fades in over `knee` rather than switching on at `threshold`
float3 applyThreshold(float3 color)
{
    float brightness = max(color.r, max(color.g, color.b));
    float knee = max(gParams.knee, 1e-4);
    float soft = clamp(brightness - gParams.threshold + knee, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee);
    float contribution = max(soft, brightness - gParams.threshold) / max(brightness, 1e-4);
    return color * contribution;
}

// 13-tap downsample with a partial Karis average on the first step
//
// A box filter lets one very bright pixel dominate its neighbourhood and flicker as the camera moves - the
// classic firefly. Weighting the inner groups by inverse luminance averages them perceptually instead
//
// Deliberately *not* [shader("fragment")]: as an entry point Slang promotes its `uniform` parameter into a
// second push block alongside gParams, and an entry point may only have one
float4 fragmentDownsample(VSOutput input, uniform bool applyKaris) : SV_Target0
{
    float2 texel = gParams.inverseSourceSize;
    float2 uv = input.uv;

    float3 a = gSource.Sample(uv + float2(-2.0, 2.0) * texel).rgb;
    float3 b = gSource.Sample(uv + float2( 0.0, 2.0) * texel).rgb;
    float3 c = gSource.Sample(uv + float2( 2.0, 2.0) * texel).rgb;
    float3 d = gSource.Sample(uv + float2(-2.0, 0.0) * texel).rgb;
    float3 e = gSource.Sample(uv).rgb;
    float3 f = gSource.Sample(uv + float2( 2.0, 0.0) * texel).rgb;
    float3 g = gSource.Sample(uv + float2(-2.0,-2.0) * texel).rgb;
    float3 h = gSource.Sample(uv + float2( 0.0,-2.0) * texel).rgb;
    float3 i = gSource.Sample(uv + float2( 2.0,-2.0) * texel).rgb;

    float3 j = gSource.Sample(uv + float2(-1.0, 1.0) * texel).rgb;
    float3 k = gSource.Sample(uv + float2( 1.0, 1.0) * texel).rgb;
    float3 l = gSource.Sample(uv + float2(-1.0,-1.0) * texel).rgb;
    float3 m = gSource.Sample(uv + float2( 1.0,-1.0) * texel).rgb;

    // Five overlapping 2x2 groups, weighted 0.5 for the centre and 0.125 for the corners
    float3 group0 = (j + k + l + m) * 0.25;
    float3 group1 = (a + b + d + e) * 0.25;
    float3 group2 = (b + c + e + f) * 0.25;
    float3 group3 = (d + e + g + h) * 0.25;
    float3 group4 = (e + f + h + i) * 0.25;

    float3 result;
    if (applyKaris)
    {
        // Karis average: weight each group by 1/(1+luma) so one blown-out pixel cannot carry the result
        float w0 = 1.0 / (1.0 + luminance(group0));
        float w1 = 1.0 / (1.0 + luminance(group1));
        float w2 = 1.0 / (1.0 + luminance(group2));
        float w3 = 1.0 / (1.0 + luminance(group3));
        float w4 = 1.0 / (1.0 + luminance(group4));
        float total = w0 * 0.5 + (w1 + w2 + w3 + w4) * 0.125;
        result = (group0 * w0 * 0.5 + (group1 * w1 + group2 * w2 + group3 * w3 + group4 * w4) * 0.125) / max(total, 1e-4);

        // Thresholding belongs on this first step only - later mips are already filtered bloom
        result = applyThreshold(result);
    }
    else
    {
        result = group0 * 0.5 + (group1 + group2 + group3 + group4) * 0.125;
    }

    return float4(result, 1.0);
}

// The first downsample reads the scene and needs both the Karis average and the threshold
[shader("fragment")]
float4 fragmentDownsampleFirst(VSOutput input) : SV_Target0
{
    return fragmentDownsample(input, true);
}

[shader("fragment")]
float4 fragmentDownsampleNext(VSOutput input) : SV_Target0
{
    return fragmentDownsample(input, false);
}

// 9-tap tent upsample. Blended additively onto the next larger mip, so each level's glow accumulates into
// the one above it and the result widens smoothly instead of in visible rings
[shader("fragment")]
float4 fragmentUpsample(VSOutput input) : SV_Target0
{
    float2 offset = gParams.inverseSourceSize * gParams.filterRadius;
    float2 uv = input.uv;

    float3 a = gSource.Sample(uv + float2(-1.0,  1.0) * offset).rgb;
    float3 b = gSource.Sample(uv + float2( 0.0,  1.0) * offset).rgb;
    float3 c = gSource.Sample(uv + float2( 1.0,  1.0) * offset).rgb;
    float3 d = gSource.Sample(uv + float2(-1.0,  0.0) * offset).rgb;
    float3 e = gSource.Sample(uv).rgb;
    float3 f = gSource.Sample(uv + float2( 1.0,  0.0) * offset).rgb;
    float3 g = gSource.Sample(uv + float2(-1.0, -1.0) * offset).rgb;
    float3 h = gSource.Sample(uv + float2( 0.0, -1.0) * offset).rgb;
    float3 i = gSource.Sample(uv + float2( 1.0, -1.0) * offset).rgb;

    float3 result = e * 4.0;
    result += (b + d + f + h) * 2.0;
    result += (a + c + g + i);
    result *= 1.0 / 16.0;

    return float4(result, 1.0);
}

// Scene + bloom, still in HDR - the tonemap runs after this and sees the sum
[shader("fragment")]
float4 fragmentComposite(VSOutput input) : SV_Target0
{
    float3 scene = gSceneColor.Sample(input.uv).rgb;
    float3 bloom = gSource.Sample(input.uv).rgb;
    return float4(scene + bloom * gParams.strength, 1.0);
}
