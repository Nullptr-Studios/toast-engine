import fullscreen;

// Traced directional shadows - the first consumer of the TLAS
//
// Every shadow artifact here comes from approximating visibility with a depth map: cascade seams,
// peter-panning, acne, bias tuning. A ray asks the actual question and needs no bias
//
// A debug view, not a replacement: the raster path is what ships without ray query, and stays the reference

struct TracedShadowParams
{
    float4x4 inverseViewProjection;
    // xyz world-space direction the light travels, w unused
    float4 lightDirection;
    // x = 0 for the visibility view, 1 for the applied one - the pass normalizes it so this shader never
    // depends on the editor's RenderMode ordinals; y = normal bias, z = max trace distance, w = spare
    float4 tuning;
};

[[vk::push_constant]]
ConstantBuffer<TracedShadowParams> gParams;

[[vk::binding(0,0)]]
Sampler2D gSceneColor;

[[vk::binding(1,0)]]
Sampler2D gDepth;

[[vk::binding(2,0)]]
Sampler2D gNormalRoughness;

[[vk::binding(3,0)]]
RaytracingAccelerationStructure gScene;

typedef FullscreenVSOutput VSOutput;

// No Y flip - Camera::getProjection() already negates proj[1][1], so uv maps straight onto clip space. See
// the note in ssr.slang; getting this wrong mirrors every depth lookup rather than announcing itself
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

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 sceneColor = gSceneColor.Sample(input.uv).rgb;

    // The pass only records at all in one of the two traced views, so there is no mode to reject here - this
    // just picks which of them to produce
    bool applied = gParams.tuning.x > 0.5;

    float4 normalRoughness = gNormalRoughness.Sample(input.uv);
    float depth = gDepth.Sample(input.uv).r;

    // Nothing wrote geometry here - sky, an overlay, a blended surface. Unshadowed by definition
    if (dot(normalRoughness.xyz, normalRoughness.xyz) < 1e-4 || depth >= 1.0)
    {
        return float4(applied ? sceneColor : float3(1.0, 1.0, 1.0), 1.0);
    }

    float3 N = normalize(normalRoughness.xyz);
    float3 worldPosition = worldPositionFromDepth(input.uv, depth);

    // Toward the light, against the direction it travels
    float3 L = normalize(-gParams.lightDirection.xyz);

    // A surface facing away from the light is shadowed by its own geometry; tracing would return a hit on
    // itself or miss into the sky, and neither is the right answer
    float NdotL = dot(N, L);
    if (NdotL <= 0.0)
    {
        return float4(float3(0.0, 0.0, 0.0), 1.0);
    }

    RayDesc ray;
    // Offset along the normal, not along the ray. This is the only bias a traced shadow needs, and it exists
    // to escape the surface's own triangle rather than to compensate for depth quantisation - which is why it
    // is a fixed small distance instead of the slope-scaled, cascade-scaled pair the shadow maps require
    ray.Origin = worldPosition + N * gParams.tuning.y;
    ray.Direction = L;
    ray.TMin = 0.0;
    ray.TMax = gParams.tuning.z;

    RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xFF, ray);

    // No Proceed() loop: with ACCEPT_FIRST_HIT_AND_END_SEARCH a shadow ray only needs to know whether
    // anything is in the way, so traversal stops at the first hit and there are no candidates to resolve
    query.Proceed();

    float visibility = query.CommittedStatus() == COMMITTED_NOTHING ? 1.0 : 0.0;

    if (!applied)
    {
        return float4(visibility, visibility, visibility, 1.0);
    }
    return float4(sceneColor * visibility, 1.0);
}
