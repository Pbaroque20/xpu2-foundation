"""The multi-device simulation app's own parts (xpu2/tools/simulate_devices.py): its cases, its realtime scene file and
its verdicts. No session is started."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('simulate_devices', Path(__file__).resolve().parents[1] / 'tools' / 'simulate_devices.py')
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def result(case, picture, log='', fallback=None, done=True):
    return {'case': case, 'picture': picture, 'log': log, 'harness_exit': 0, 'render': None, 'seconds': 1.0,
            'report': {'fallback': fallback, 'exit_code': 0, 'events': [[1.0, 'DONE', 1]] if done else []}}


class Cases(unittest.TestCase):
    def test_quick_suite(self):
        names = [c.name for c in sim.cases('quick', 3, 3)]
        for name in ('base-a', 'base-b', 'cards-2', 'groups-2', 'cards-2-groups-2', 'uneven-cards', 'card-fails',
                     'denoise-optix', 'denoise-cpu', 'denoise-next-card'):
            self.assertIn(name, names)
        self.assertNotIn('cards-3', names)

    def test_full_suite_counts(self):
        names = [c.name for c in sim.cases('full', 4, 3)]
        for name in ('cards-2', 'cards-3', 'cards-4', 'groups-2', 'groups-3', 'denoise-all-fail'):
            self.assertIn(name, names)

    def test_switches(self):
        by = {c.name: c for c in sim.cases('full', 3, 3)}
        self.assertEqual(by['cards-3'].env, {'XPU2_SIMULATE_CARDS': 3})
        self.assertEqual(by['card-fails'].env['XPU2_SIMULATE_CARD_FAIL'], 2)
        self.assertEqual(by['denoise-next-card'].env['XPU2_SIMULATE_DENOISE_FAIL'], 1)
        self.assertTrue(by['denoise-cpu'].realtime)

    def test_realtime_scene(self):
        with tempfile.TemporaryDirectory() as d:
            text = sim.realtime_scene(Path(d), samples=8).read_text()
        self.assertIn('UserData("/modo/xpu2/generations")', text)
        self.assertIn('{2, 1, 8, 0, 0, 0, 0, 1, 1}', text)
        self.assertIn('RenderOutput("/modo/preview/denoise_albedo")', text)
        self.assertIn('["material_aov"] = "albedo"', text)
        self.assertIn('["state_variable"] = 2', text)


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.by = {c.name: c for c in sim.cases('quick', 3, 3)}
        base = (2, 1, (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        noisy = (2, 1, (1.01, 0.99, 1.0, 1.0, 1.0, 1.0))
        self.results = {'base-a': result(self.by['base-a'], base), 'base-b': result(self.by['base-b'], noisy)}
        self.results['base-b']['comparison'] = sim.compare(base, noisy)

    def judge(self, name, picture, log='', **kw):
        return sim.judge(result(self.by[name], picture, log, **kw), self.results)

    def test_within_noise_passes(self):
        r = self.judge('cards-2', (2, 1, (1.005, 1.0, 0.995, 1.0, 1.0, 1.0)), 'MoonRay XPU2: graphics card 0 (lane 1) renders with the first')
        self.assertTrue(r['passed'], r['why'])

    def test_too_different_fails(self):
        r = self.judge('cards-2', (2, 1, (1.2, 1.0, 1.0, 1.0, 1.0, 1.0)), 'MoonRay XPU2: graphics card 0 (lane 1) renders with the first')
        self.assertFalse(r['passed'])

    def test_missing_line_fails(self):
        r = self.judge('cards-2', (2, 1, (1.0,) * 6))
        self.assertFalse(r['passed'])
        self.assertIn('lacks', r['why'][0])

    def test_fallback_fails(self):
        r = self.judge('groups-2', (2, 1, (1.0,) * 6), 'MoonRay XPU2: 2 processor groups', fallback='no GPU')
        self.assertFalse(r['passed'])

    def test_left_out_only_where_expected(self):
        r = self.judge('cards-2', (2, 1, (1.0,) * 6), 'renders with the first (lane 1)\nMoonRay XPU2: graphics card 0 left out: x\n'
                       'MoonRay XPU2: graphics card 0 (lane 1) renders with the first')
        self.assertFalse(r['passed'])
        r = self.judge('card-fails', (2, 1, (1.0,) * 6), 'graphics card 0 left out: failed (simulated, XPU2_SIMULATE_CARD_FAIL)')
        self.assertTrue(r['passed'], r['why'])


if __name__ == '__main__':
    unittest.main()
