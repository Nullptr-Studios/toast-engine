import fullscreen;

// Screen-space reflections
//
// Top tier of the reflection fallback chain. A probe's cubemap can only pin geometry on its proxy box;
// anything else slides under camera motion. This pass has the depth buffer, so it reflects what is there
//
// Only what is *on screen* - everything else falls through to the probe and environment tiers already in
// the scene colour, which is why this adds rather than replaces

struct SsrParams
{
    float4x4 viewProjection;
    float4x4 inverseViewProjection;
    float4 cameraPosition;
    // x = intensity, y = max roughness that still reflects, z = thickness in world units, w = step count
    float4 tuning;
    // x = stride in world units, y = render mode, zw = spare
    float4 marching;
};

[[vk::push_constant]]
ConstantBuffer<SsrParams> gParams;

[[vk::binding(0, 0)]]
Sampler2D gSceneColor;

[[vk::binding(1, 0)]]
Sampler2D gDepth;

[[vk::binding(2, 0)]]
Sampler2D gNormalRoughness;

typedef FullscreenVSOutput VSOutput;

// Fullscreen triangle from the vertex id, so this pass needs no vertex buffer
//
// No Y flip: Camera::getProjection() already negates proj[1][1], so NDC +Y and texture +V both run down.
// Flipping again inverts the image and silently mirrors every depth lookup below
[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID)
{
    return fullscreenTriangle(vertexID);
}

// Rebuilds a world position from the depth buffer
//
// A position attachment would be three more channels restating what depth and the inverse projection give
float3 worldPositionFromDepth(float2 uv, float depth)
{
    // Straight remap, no Y flip - see vertexMain for why both axes map directly
    float4 clip = float4(uv * 2.0 - 1.0, depth, 1.0);
    float4 world = mul(gParams.inverseViewProjection, clip);
    return world.xyz / world.w;
}

// Projects a world position back to UV, returning false when it lands off screen
bool projectToScreen(float3 worldPosition, out float2 uv, out float depth)
{
    float4 clip = mul(gParams.viewProjection, float4(worldPosition, 1.0));
    if (clip.w <= 0.0)
    {
        uv = float2(0.0, 0.0);
        depth = 0.0;
        return false;
    }

    float3 ndc = clip.xyz / clip.w;
    uv = ndc.xy * 0.5 + 0.5;
    depth = ndc.z;
    return all(uv >= float2(0.0, 0.0)) && all(uv <= float2(1.0, 1.0));
}

// How much a hit near the edge of the screen is trusted
//
// The fade *is* the feature: cutting off hard puts a moving line down the image wherever the reflection
// runs out, which is worse than the missing reflection
float screenEdgeFade(float2 uv)
{
    float2 fromEdge = min(uv, float2(1.0, 1.0) - uv);
    float fade = saturate(min(fromEdge.x, fromEdge.y) / 0.1);
    return fade * fade;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 sceneColor = gSceneColor.Sample(input.uv).rgb;
    float4 normalRoughness = gNormalRoughness.Sample(input.uv);

    uint renderMode = uint(gParams.marching.y);

    // Debug view: the geometry buffer itself, remapped out of its signed range. This is the view that proves
    // the buffer before anything trusts it - a normal map reading as flat colour, or as the interpolated
    // vertex normal, is invisible in the lit image and fatal to every trace below
    if (renderMode == 12)
    {
        if (dot(normalRoughness.xyz, normalRoughness.xyz) < 1e-4)
        {
            // Nothing wrote here: sky, an overlay, or a transparent surface that deliberately abstained
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        return float4(normalize(normalRoughness.xyz) * 0.5 + 0.5, 1.0);
    }

    if (renderMode == 13)
    {
        // Roughness alone, the other half of the buffer
        return float4(normalRoughness.www, 1.0);
    }

    // Every non-contributing path returns black, so the view is exactly this pass's output. Handing back
    // scene colour would make "traced and found nothing" look like "never traced"
    bool ssrOnly = (renderMode == 14);
    float3 passthrough = ssrOnly ? float3(0.0, 0.0, 0.0) : sceneColor;

    // A zero-length normal means the world stage wrote nothing here - sky, grid, or a blended surface that
    // does not own the pixel. Nothing to reflect off
    float normalLength = dot(normalRoughness.xyz, normalRoughness.xyz);
    if (normalLength < 1e-4)
    {
        return float4(passthrough, 1.0);
    }

    float3 N = normalize(normalRoughness.xyz);
    float roughness = normalRoughness.w;

    // Rough surfaces scatter the reflection so widely that a single mirror ray is the wrong model for them,
    // and marching one produces a sharp reflection where the material says there should be none. Those are
    // the surfaces the prefiltered probe chain already handles correctly
    float maxRoughness = gParams.tuning.y;
    if (roughness > maxRoughness)
    {
        return float4(passthrough, 1.0);
    }

    float depth = gDepth.Sample(input.uv).r;
    if (depth >= 1.0)
    {
        return float4(passthrough, 1.0);
    }

    float3 worldPosition = worldPositionFromDepth(input.uv, depth);
    float3 V = normalize(gParams.cameraPosition.xyz - worldPosition);
    float3 R = reflect(-V, N);

    // A ray heading back toward the camera is behind the near plane almost immediately and only ever finds
    // the surface it started from
    if (dot(R, V) > 0.999)
    {
        return float4(passthrough, 1.0);
    }

    uint stepCount = uint(gParams.tuning.w);
    float stride = gParams.marching.x;
    float thickness = gParams.tuning.z;

    // Start off the surface. Without this the first sample lands on the originating pixel, whose depth
    // trivially matches, and every surface reflects itself
    float3 rayPosition = worldPosition + N * stride * 0.5;

    float3 hitColor = float3(0.0, 0.0, 0.0);
    float confidence = 0.0;
    float previousDelta = 0.0;
    float3 previousPosition = rayPosition;

    for (uint i = 0; i < stepCount; ++i)
    {
        rayPosition += R * stride;

        float2 sampleUv;
        float sampleDepth;
        if (!projectToScreen(rayPosition, sampleUv, sampleDepth))
        {
            break;
        }

        float sceneDepth = gDepth.Sample(sampleUv).r;
        if (sceneDepth >= 1.0)
        {
            // Sky along the ray. The environment is already in the scene colour, so there is nothing to add
            continue;
        }

        float3 scenePosition = worldPositionFromDepth(sampleUv, sceneDepth);

        // Compared along the view direction rather than in depth-buffer units, which are non-linear and make
        // one thickness value behave completely differently near and far
        float delta = dot(rayPosition - scenePosition, normalize(scenePosition - gParams.cameraPosition.xyz));

        if (delta > 0.0)
        {
            // The ray is behind a surface. The depth buffer is a heightfield with no notion of how thick
            // anything is, so a ray passing far behind a thin object would otherwise "hit" its back face -
            // beyond this many world units the surface is treated as something the ray went past, not into
            if (delta > thickness)
            {
                break;
            }

            // One bisection between the last two samples. The linear march overshoots by up to a full stride,
            // which on a shallow-angle reflection is a visibly displaced hit; refining the crossing costs a
            // handful of samples and is the difference between a reflection that lines up and one that does not
            float3 lo = previousPosition;
            float3 hi = rayPosition;
            for (uint j = 0; j < 4; ++j)
            {
                float3 mid = (lo + hi) * 0.5;
                float2 midUv;
                float midDepth;
                if (!projectToScreen(mid, midUv, midDepth))
                {
                    break;
                }

                float midSceneDepth = gDepth.Sample(midUv).r;
                float3 midScenePosition = worldPositionFromDepth(midUv, midSceneDepth);
                float midDelta =
                    dot(mid - midScenePosition, normalize(midScenePosition - gParams.cameraPosition.xyz));

                if (midDelta > 0.0)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid;
                }
            }

            float2 hitUv;
            float hitDepth;
            if (projectToScreen(hi, hitUv, hitDepth))
            {
                hitColor = gSceneColor.Sample(hitUv).rgb;

                // A hit on a surface facing away from the ray is geometry the ray struck from behind, which
                // the depth buffer cannot tell apart from a genuine front-face hit
                float3 hitNormal = gNormalRoughness.Sample(hitUv).xyz;
                float facing = dot(hitNormal, hitNormal) > 1e-4 ? saturate(-dot(normalize(hitNormal), R)) : 0.0;

                confidence = screenEdgeFade(hitUv) * step(0.01, facing);
            }
            break;
        }

        previousDelta = delta;
        previousPosition = rayPosition;
    }

    if (renderMode == 14)
    {
        // The reflection on its own, before it is weighted into the scene - separates "the march found
        // nothing" from "the march found something and the weighting threw it away"
        return float4(hitColor * confidence, 1.0);
    }

    if (confidence <= 0.0)
    {
        return float4(sceneColor, 1.0);
    }

    // Grazing angles reflect far more strongly than head-on ones, which is what makes a wet floor read as wet
    float NdotV = saturate(dot(N, V));
    float fresnel = 0.04 + 0.96 * pow(1.0 - NdotV, 5.0);

    // Sharper surfaces get more of it: this pass traces a single mirror ray, so its answer is only right in
    // the limit of a smooth surface and has to hand back to the prefiltered chain as roughness climbs
    float roughnessWeight = saturate(1.0 - roughness / max(maxRoughness, 1e-4));

    float weight = confidence * fresnel * roughnessWeight * gParams.tuning.x;

    // Added, not blended over. What is already in the scene colour is the probe/environment answer for this
    // pixel, and there is no F0 or albedo in the geometry buffer to reconstruct how much of it was specular -
    // so the two overlap slightly on metals. See the roadmap: separating them needs a fuller G-buffer
    return float4(sceneColor + hitColor * weight, 1.0);
}
