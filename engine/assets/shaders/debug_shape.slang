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

struct PushConstants
{
    float4x4 model;
    float4 tint;
};

[[vk::push_constant]]
PushConstants gPush;

struct VSInput
{
    [[vk::location(0)]]
    float3 position : POSITION;

    [[vk::location(1)]]
    float4 color : COLOR;
};

struct VSOutput
{
    float4 position : SV_Position;

    [[vk::location(0)]]
    float4 color : COLOR;
};

[shader("vertex")]
VSOutput vertexMain(VSInput input)
{
    VSOutput output;

    float4 worldPos =
        mul(
            gPush.model,
            float4(input.position, 1.0));

    output.position =
        mul(
            gCamera.viewProjection,
            worldPos);

    output.color = input.color * gPush.tint;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target0
{
    return input.color;
}

// Editor furniture drawn from a real mesh rather than generated line geometry - see DebugPass's mesh
// pipeline. Reads the renderer::Vertex stream, so the locations below are that struct's attribute indices
// and not a layout of this shader's own choosing
struct MeshVSInput
{
    [[vk::location(0)]]
    float3 position : POSITION;

    [[vk::location(1)]]
    float3 normal : NORMAL;

    [[vk::location(4)]]
    float3 color : COLOR;
};

[shader("vertex")]
VSOutput vertexMesh(MeshVSInput input)
{
    VSOutput output;

    float4 worldPos =
        mul(
            gPush.model,
            float4(input.position, 1.0));

    output.position =
        mul(
            gCamera.viewProjection,
            worldPos);

    // Lit off a fixed direction rather than the scene's lights: a gizmo has to stay readable in a dark scene,
    // and it is furniture, not geometry that belongs to the world
    float3 worldNormal =
        normalize(
            mul(
                (float3x3)gPush.model,
                input.normal));

    float lambert = saturate(dot(worldNormal, normalize(float3(0.35, 0.5, 0.8))));
    // Floors at 0.45 so a face pointing away is shaded, never black
    float shade = lerp(0.45, 1.0, lambert);

    output.color = float4(input.color * gPush.tint.rgb * shade, gPush.tint.a);

    return output;
}

[shader("fragment")]
float4 fragmentMesh(VSOutput input) : SV_Target0
{
    return input.color;
}
