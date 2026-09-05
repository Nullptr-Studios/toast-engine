// Toast inspector attributes
//
// Import this module (`import toast;`) to tag material parameters for the editor
//
// Only parameters marked [Reflect] show up in the material inspector; everything
// else is engine-owned data (matrices, light info...)
//
//   [Reflect]                  expose this parameter in the material editor
//   [Range(min, max)]          clamp/slider range for numeric parameters
//   [Color]                    treat a float3/float4 as a color
//   [Name("Display Name")]     override the inspector display name
//   [Group("group name")]      group parameters together; when the group name matches
//                              a texture's name (or its [Name]) the parameters live in
//                              that texture's TOML table
//   [Subgroup("sub name")]     nested grouping inside a [Group]
//   [Unit("m/s")]              display unit suffix
//   [Default("black")]        fallback texture when a Sampler2D parameter has no texture bound;
//                              one of "white" (1,1,1,1 - the implicit default), "black" (0,0,0,1),
//                              or "flat_normal" (0.5,0.5,1,1 - tangent-space up)
//   [Linear]                   this texture carries data, not colour (normals, metalness, roughness,
//                              occlusion). Binding an sRGB-encoded texture to it corrupts the values -
//                              0.5 decodes to 0.21 - so the engine warns when one is

[__AttributeUsage(_AttributeTargets.Var)]
public struct ReflectAttribute {};

[__AttributeUsage(_AttributeTargets.Var)]
public struct RangeAttribute
{
    float min;
    float max;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct ColorAttribute {};

[__AttributeUsage(_AttributeTargets.Var)]
public struct NameAttribute
{
    string name;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct GroupAttribute
{
    string name;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct SubgroupAttribute
{
    string name;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct UnitAttribute
{
    string unit;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct DefaultAttribute
{
    string fallback;
};

[__AttributeUsage(_AttributeTargets.Var)]
public struct LinearAttribute {};
