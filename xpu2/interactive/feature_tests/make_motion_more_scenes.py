"""More motion blur for the GPU: make_motion_more_scenes.py <folder>. Fog in meshes moving over the shutter (by a turning
node transform, two overlapping, and beside a moving surface), and many lights moving at once, more than a launch
thread's records of them (GpuLane::MovedLightSlots), uniformly sampled and through the light tree."""
import math
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
folder = Path(sys.argv[1]).resolve()
folder.mkdir(parents=True, exist_ok=True)


def load(name, stop):
    source = (here / name).read_text(encoding='utf-8')
    space = {'__file__': str(here / name)}
    argv = sys.argv
    sys.argv = [argv[0], str(folder)]
    exec(compile(source[:source.index(stop)], name, 'exec'), space)
    sys.argv = argv
    return space


emissive = load('make_emissive_scenes.py', 'GLOW = ')
motion = load('make_motion_scenes.py', 'cases = {')
blur, translate, mul, rot_y, HERE, TURN, STILL, PERSP = (motion[n] for n in ('blur', 'translate', 'mul', 'rot_y', 'HERE', 'TURN', 'STILL', 'PERSP'))


def fog(name, density, color='0.9,0.85,0.8'):
    return 'BaseVolume("%s") { ["attenuation_color"]=Rgb(1,1,1), ["attenuation_intensity"]=%g, ["diffuse_color"]=Rgb(%s) }' % (name, density, color)


def moving(volumes, movers):
    """The volumes' scene under motion blur, with a key light for them to scatter, each mover's node transform blurred."""
    text = emissive['scene'](volumes)
    text = text.replace('["enable_motion_blur"]=false', '["enable_motion_blur"]=true, ["motion_steps"]={-0.25, 0.25}', 1)
    text = text.replace('EnvLight("/fill") { ["intensity"]=0.05 }', 'EnvLight("/fill") { ["intensity"]=0.3 }\n'
                        'SphereLight("/key") { ["node_xform"]=%s, ["radius"]=0.5, ["intensity"]=30 }' % motion['mat'](translate(1.5, 4, 2)), 1)
    text = text.replace('LightSet("/lights") { EnvLight("/fill") }', 'LightSet("/lights") { EnvLight("/fill"), SphereLight("/key") }', 1)
    for name, a, b in movers:
        text = text.replace('RdlMeshGeometry("%s") {' % name, 'RdlMeshGeometry("%s") { ["node_xform"]=%s,' % (name, blur(a, b)), 1)
    return text


ONE = translate(0, 0, 0)
cases = {
    'vm-fog-slide': moving([('/v1', (-0.4, 1.3, -0.5), 0.9, fog('/f1', 1.5))], [('/v1', ONE, translate(1.2, 0.2, 0))]),
    'vm-fog-turn': moving([('/v1', (-0.8, 1.3, -0.5), 0.9, fog('/f1', 2.5))], [('/v1', ONE, mul(rot_y(35), translate(0.3, 0, 0)))]),
    'vm-two': moving([('/v1', (-0.6, 1.3, -0.5), 0.8, fog('/f1', 1.2)), ('/v2', (0.2, 1.4, -0.6), 0.7, fog('/f2', 3.0, '0.5,0.7,0.9'))],
                     [('/v1', ONE, translate(0.8, 0, 0)), ('/v2', ONE, translate(-0.6, 0.3, 0))]),
    'vm-with-surface': moving([('/v1', (-0.4, 1.3, -0.5), 0.9, fog('/f1', 1.5))], [('/v1', ONE, translate(0.9, 0, 0)), ('/ball', ONE, translate(-1.2, 0.5, 0))]),
}


def many_lights(count):
    lights, names = [], []
    for k in range(count):
        a = 2 * math.pi * k / count
        x, z = 3.5 * math.cos(a), 3.5 * math.sin(a)
        name = '/k%d' % k
        lights.append('SphereLight("%s") { ["node_xform"]=%s, ["radius"]=0.2, ["intensity"]=%g, ["color"]=Rgb(%.2f,%.2f,%.2f), ["mb"]=true }'
                      % (name, blur(translate(x, 2.5, z), translate(x + 1.2 * math.sin(a), 2.8, z - 1.2 * math.cos(a))), 240.0 / count,
                         0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.cos(a + 2.1), 0.5 + 0.5 * math.cos(a + 4.2)))
        names.append('SphereLight("%s")' % name)
    return '\n'.join(lights), ', '.join(names)


cases['ml-17'] = motion['scene'](STILL, motion['mesh'](motion['mat'](HERE)), camera_extra=PERSP, lights=many_lights(17))
cases['ml-40'] = motion['scene'](STILL, motion['mesh'](motion['mat'](HERE)), camera_extra=PERSP, lights=many_lights(40))
cases['ml-40-tree'] = motion['scene'](STILL, motion['mesh'](motion['mat'](HERE)), camera_extra=PERSP, lights=many_lights(40), sv_extra=', ["light_sampling_mode"]=1')
for name, text in cases.items():
    (folder / (name + '.rdla')).write_text(text, encoding='utf-8')
print(len(cases), 'cases')
