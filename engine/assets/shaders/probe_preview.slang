// A mirror ball at a reflection probe's position, for inspecting what it captured
//
// Faces are stored separately, so a rotation or mirroring between them only reads as wrong once they are
// seen joined up. A sphere shows the whole cube at once, and a misaligned seam is obvious as the camera moves

struct ProbePreviewParams
{
    float4x4 viewProjection;

    // Sphere centre (the probe's position) and its world-space radius in w
    float4 centerRadius;

    // Camera position, for the view vector. w unused
    float4 cameraPosition;
};

[[vk::push_constant]]
ProbePreviewParams gParams;

[[vk::binding(0,0)]]
SamplerCube gProbeCube;

struct VSOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float3 worldPosition : TEXCOORD0;

    [[vk::location(1)]]
    float3 normal : NORMAL;
};

static const uint kRings = 24;
static const uint kSegments = 48;

/// @brief A point on the unit sphere for grid coordinates (@p ring, @p segment)
float3 spherePoint(uint ring, uint segment)
{
    float v = float(ring) / float(kRings);
    float u = float(segment) / float(kSegments);

    float phi = v * 3.14159265359;          // 0..pi, pole to pole
    float theta = u * 2.0 * 3.14159265359;  // around

    float sinPhi = sin(phi);
    // Z-up, matching the engine's world convention
    return float3(sinPhi * cos(theta), sinPhi * sin(theta), cos(phi));
}

// Generated from the vertex id rather than a vertex buffer: the sphere is fixed, so uploading and binding
// geometry for it would be more moving parts than the arithmetic it replaces
[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    uint quad = vertexId / 6u;
    uint corner = vertexId % 6u;

    uint ring = quad / kSegments;
    uint segment = quad % kSegments;

    // Two triangles per grid cell
    uint2 offsets[6] = {
        uint2(0u, 0u), uint2(0u, 1u), uint2(1u, 1u),
        uint2(0u, 0u), uint2(1u, 1u), uint2(1u, 0u)
    };
    uint2 offset = offsets[corner];

    float3 normal = spherePoint(ring + offset.x, segment + offset.y);
    float3 world = gParams.centerRadius.xyz + normal * gParams.centerRadius.w;

    VSOutput output;
    output.position = mul(gParams.viewProjection, float4(world, 1.0));
    output.worldPosition = world;
    output.normal = normal;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float3 N = normalize(input.normal);
    float3 V = normalize(gParams.cameraPosition.xyz - input.worldPosition);

    // Mip 0 always: this is an inspection tool, and the roughness chain would hide exactly the misalignment
    // it exists to reveal
    return float4(gProbeCube.SampleLevel(reflect(-V, N), 0.0).rgb, 1.0);
}
