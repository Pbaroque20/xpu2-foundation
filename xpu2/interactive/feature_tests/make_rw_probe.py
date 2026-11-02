"""make_rw_probe.py <out folder>: one random-walk sphere (radius 0.7) under a white environment of radiance 1, seen from
straight above (64x64, the centre pixel on its pole), no other geometry and no bounces: the pixel is the walk's mean
throughput. Writes grid-8 and grid-32 scenes and a fixtures report."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_material_scenes import mesh_rdla, sphere  # noqa: E402

out = Path(sys.argv[1]).resolve()
report = {'cases': {}}
for name, material in {
    'rw-pole': 'DwaBaseMaterial("/s0") { ["albedo"]=Rgb(0.9,0.75,0.65), ["scattering_color"]=Rgb(0.95,0.45,0.3), '
               '["scattering_radius"]=0.25, ["specular"]=0.0, ["bssrdf"]=2 }',
    'nd-pole': 'DwaBaseMaterial("/s0") { ["albedo"]=Rgb(0.9,0.75,0.65), ["scattering_color"]=Rgb(0.95,0.45,0.3), '
               '["scattering_radius"]=0.25, ["specular"]=0.0, ["bssrdf"]=0 }',
}.items():
    entry = report['cases'][name] = {'resolution': [64, 64], 'domain': 'probe', 'scenes': {}}
    for grid in (8, 32):
        d = out / name / f'grid-{grid}' / 'frozen'
        (d / 'assets').mkdir(parents=True)
        lines = [
            'SceneVariables { ["image_width"]=64, ["image_height"]=64, ["pixel_samples"]=%d, ["sampling_mode"]=0, ["frame"]=0,'
            ' ["pixel_filter"]=0, ["pixel_filter_width"]=1.0, ["max_depth"]=5, ["max_diffuse_depth"]=0, ["max_glossy_depth"]=0,'
            ' ["max_mirror_depth"]=0, ["lights_visible_in_camera"]=false, ["two_stage_output"]=false, ["enable_motion_blur"]=false,'
            ' ["enable_dof"]=false }' % grid,
            # Looking straight down from above the pole: local -Z is world -Y.
            'PerspectiveCamera("/camera") { ["node_xform"]=Mat4(1,0,0,0, 0,0,-1,0, 0,1,0,0, 0,3,0,1), ["focal"]=40.0,'
            ' ["film_width_aperture"]=36.0, ["near"]=0.01, ["far"]=1000.0 }',
            'SceneVariables { ["camera"]=PerspectiveCamera("/camera") }',
            'EnvLight("/sky") { ["intensity"]=1.0, ["color"]=Rgb(1,1,1) }',
            'LightSet("/lights") { EnvLight("/sky") }',
            'RenderOutput("/beauty") { ["file_name"]="beauty.exr", ["result"]=0, ["channel_format"]=0, ["compression"]=1 }',
            'RenderOutput("/alpha") { ["file_name"]="beauty.exr", ["result"]=1, ["channel_format"]=0, ["compression"]=1 }',
            material,
            mesh_rdla('ball0', *sphere(0, 0, 0, 0.7, 96, 48)),
            'GeometrySet("/geometry") { RdlMeshGeometry("/ball0") }',
            'Layer("/layer") { { RdlMeshGeometry("/ball0"), "", DwaBaseMaterial("/s0"), LightSet("/lights") } }',
            'SceneVariables { ["layer"]=Layer("/layer") }',
        ]
        (d / 'scene.rdla').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        entry['scenes'][str(grid)] = str(d / 'scene.rdla')
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(out / 'report.json')
