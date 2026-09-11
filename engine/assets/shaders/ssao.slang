import fullscreen;

// Screen-space ambient occlusion
//
// Occlusion belongs to the *indirect* light: it approximates nearby geometry blocking the environment, and
// says nothing about a light the surface can plainly see. Forward shading has already summed everything by
// the time this runs, so the world stage hands the indirect diffuse out on its own attachment:
//
//     final = color - indirect + indirect * ao
//
// Multiplying the finished image would darken direct light and emission too, which reads as murk

struct SsaoParams
{
    float4x4 viewProjection;
    float4x4 inverseViewProjection;
    // xy = target size in pixels, zw = its reciprocal
    float4 screenSize;
    // x = world-space sample radius, y = strength, z = depth-range cutoff, w = sample count
    float4 tuning;
    // x = render mode, yzw = spare
    float4 misc;
};

[[vk::push_constant]]
ConstantBuffer<SsaoParams> gParams;

[[vk::binding(0, 0)]]
Sampler2D gSceneColor;

[[vk::binding(1, 0)]]
Sampler2D gDepth;

[[vk::binding(2, 0)]]
Sampler2D gNormalRoughness;

[[vk::binding(3, 0)]]
Sampler2D gIndirectDiffuse;

typedef FullscreenVSOutput VSOutput;

// No Y flip - see the note in ssr.slang. Camera::getProjection() already negates proj[1][1]
[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID)
{
    return fullscreenTriangle(vertexID);
}

float3 worldPositionFromDepth(float2 uv, float depth)
{
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 world = mul(gParams.inverseViewProjection, clip);
    return world.xyz / world.w;
}

// Cheap per-pixel hash, used to rotate the sample pattern
//
// Without it every pixel samples the same directions and bands into arcs. Rotating turns that structured
// error into noise, which a blur can remove - banding survives one
float hash(float2 p)
{
    return frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
}

// Sample directions on a hemisphere, pushed toward the centre so more samples land close to the surface
//
// Occlusion falls off with distance, so uniform spacing spends most of its samples where they matter least
float3 hemisphereSample(uint index, uint count, float rotation)
{
    float golden = 2.39996323;
    float angle = float(index) * golden + rotation;
    float radius = sqrt((float(index) + 0.5) / float(count));

    float2 disk = float2(cos(angle), sin(angle)) * radius;
    float height = sqrt(max(1.0 - dot(disk, disk), 0.0));
    return float3(disk, height);
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 sceneColor = gSceneColor.Sample(input.uv).rgb;
    float3 indirect = gIndirectDiffuse.Sample(input.uv).rgb;
    float4 normalRoughness = gNormalRoughness.Sample(input.uv);

    uint renderMode = uint(gParams.misc.x);
    bool aoOnly = (renderMode == 15);

    // Nothing wrote geometry here - sky, an overlay, a blended surface. There is no surface to occlude
    if (dot(normalRoughness.xyz, normalRoughness.xyz) < 1e-4)
    {
        return float4(aoOnly ? float3(1.0, 1.0, 1.0) : sceneColor, 1.0);
    }

    float depth = gDepth.Sample(input.uv).r;
    if (depth >= 1.0)
    {
        return float4(aoOnly ? float3(1.0, 1.0, 1.0) : sceneColor, 1.0);
    }

    float3 N = normalize(normalRoughness.xyz);
    float3 worldPosition = worldPositionFromDepth(input.uv, depth);

    float radius = gParams.tuning.x;
    float strength = gParams.tuning.y;
    float rangeCutoff = gParams.tuning.z;
    uint sampleCount = uint(gParams.tuning.w);

    // An orthonormal basis around the normal. The arbitrary axis is chosen to be the one least aligned with
    // N, because crossing against a nearly-parallel vector produces a degenerate tangent
    float3 helper = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T = normalize(cross(helper, N));
    float3 B = cross(N, T);

    float rotation = hash(input.uv * gParams.screenSize.xy) * 6.2831853;

    float occlusion = 0.0;
    float validSamples = 0.0;

    for (uint i = 0; i < sampleCount; ++i)
    {
        float3 hemisphere = hemisphereSample(i, sampleCount, rotation);
        float3 offset = (T * hemisphere.x + B * hemisphere.y + N * hemisphere.z) * radius;

        float3 samplePosition = worldPosition + offset;

        // Where that point lands on screen, projected through the same matrix the geometry was rasterized
        // with, so the sample and the depth buffer agree on where things are
        float4 projected = mul(gParams.viewProjection, float4(samplePosition, 1.0));
        if (projected.w <= 0.0)
        {
            continue;
        }

        float3 ndc = projected.xyz / projected.w;
        float2 sampleUv = ndc.xy * 0.5 + 0.5;
        if (any(sampleUv < float2(0.0, 0.0)) || any(sampleUv > float2(1.0, 1.0)))
        {
            continue;
        }

        float sceneDepth = gDepth.Sample(sampleUv).r;
        if (sceneDepth >= 1.0)
        {
            validSamples += 1.0;
            continue;
        }

        float3 scenePosition = worldPositionFromDepth(sampleUv, sceneDepth);

        // Occluded when real geometry sits in front of where the sample was placed
        float3 toSample = scenePosition - worldPosition;
        float distanceToScene = length(toSample);

        // Beyond the cutoff the occluder is a different surface entirely - a wall across the room behind a
        // railing, say - and counting it produces dark haloes around silhouettes rather than contact shadows
        float rangeWeight = saturate(1.0 - distanceToScene / max(rangeCutoff, 1e-4));

        if (distanceToScene > 1e-5)
        {
            float3 direction = toSample / distanceToScene;
            // Only geometry above the tangent plane occludes; anything below is behind the surface
            float facing = saturate(dot(N, direction));
            occlusion += facing * rangeWeight;
        }

        validSamples += 1.0;
    }

    float ao = 1.0;
    if (validSamples > 0.0)
    {
        ao = saturate(1.0 - (occlusion / validSamples) * strength);
    }

    if (aoOnly)
    {
        return float4(ao, ao, ao, 1.0);
    }

    // Subtract what this pixel contributed, add it back weighted. Direct, indirect specular and emission
    // pass through untouched - the whole reason that term is separate
    //
    // Clamped because a blended surface scaled the opaque contribution by (1 - alpha) while this attachment
    // holds it unscaled, so subtracting in full over-darkens and could push radiance negative
    return float4(max(sceneColor - indirect + indirect * ao, float3(0.0, 0.0, 0.0)), 1.0);
}
