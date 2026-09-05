// Camera-facing textured quad, used for editor/debug icons (light gizmos and the like)
//
// Takes no vertex buffer - the quad's six vertices are generated from SV_VertexID, so a draw is just
// vkCmdDraw(6, ...) plus the push constants below

struct CameraUBO
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewProjection;

    float3 cameraPosition;
    float time;
};

[[vk::binding(0,0)]]
ConstantBuffer<CameraUBO> gCamera;

[[vk::binding(0,1)]]
Sampler2D gIcon;

struct PushConstants
{
    float4 centerSize;  // xyz world-space center, w world-space edge length
    float4 tint;
};

[[vk::push_constant]]
PushConstants gPush;

struct VSOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOutput vertexMain(uint vertexId : SV_VertexID)
{
    // Two triangles worth of corner offsets in the billboard's own 2D plane, and the matching UVs.
    // +y is up on screen, so v is flipped to keep the image the right way round
    const float2 offsets[6] = {
        float2(-1.0, -1.0), float2( 1.0, -1.0), float2( 1.0,  1.0),
        float2(-1.0, -1.0), float2( 1.0,  1.0), float2(-1.0,  1.0),
    };

    float2 offset = offsets[vertexId];

    float3 center = gPush.centerSize.xyz;
    float halfSize = gPush.centerSize.w * 0.5;

    // Build the quad's basis from the camera position rather than the view matrix: glm hands the engine
    // column-major matrices while Slang indexes row-major, so pulling basis vectors out of gCamera.view
    // means depending on that mapping. cameraPosition is a plain float3 with no such ambiguity, and facing
    // the eye point (rather than the view plane) is what a spherical billboard wants anyway
    float3 forward = center - gCamera.cameraPosition;
    float forwardLen = length(forward);
    forward = forwardLen > 0.0001 ? forward / forwardLen : float3(0.0, 1.0, 0.0);

    // Engine is Z-up. Looking at a billboard from directly above/below makes forward parallel to up and
    // the cross product degenerate, so swap in another axis for that case
    float3 up = float3(0.0, 0.0, 1.0);
    if (abs(dot(forward, up)) > 0.999)
    {
        up = float3(0.0, 1.0, 0.0);
    }

    float3 right = normalize(cross(forward, up));
    up = cross(right, forward);

    float3 worldPos = center + right * (offset.x * halfSize) + up * (offset.y * halfSize);

    VSOutput output;
    output.position = mul(gCamera.viewProjection, float4(worldPos, 1.0));
    output.uv = float2(offset.x * 0.5 + 0.5, 0.5 - offset.y * 0.5);
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    float4 icon = gIcon.Sample(input.uv) * gPush.tint;

    // Icons are mostly empty space; dropping the fully transparent texels keeps them from darkening what's
    // behind the quad and stops them writing anything at all where there's no glyph
    if (icon.a < 0.01)
    {
        discard;
    }

    return icon;
}
