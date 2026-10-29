import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from optimizer_research import AdaSym, Config, coherence, effective_window
from optimizer_research.synthetic import init, evaluate, dataset


class ReferenceTests(unittest.TestCase):
    def test_adamw_limit_matches_independent_recurrence(self):
        rng = np.random.default_rng(15)
        for config in (Config(alpha=0), Config(cap=1)):
            expected = rng.normal(size=12)
            actual = expected.copy()
            opt = AdaSym([actual], config)
            m, v = np.zeros_like(actual), np.zeros_like(actual)
            for t in range(1, 25):
                g = rng.normal(size=12)
                m = config.beta1*m + (1-config.beta1)*g
                v = config.beta2*v + (1-config.beta2)*g*g
                expected *= 1-config.lr*config.weight_decay
                expected -= config.lr*(m/(1-config.beta1**t))/(np.sqrt(v/(1-config.beta2**t))+config.eps)
                opt.step([g])
                np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_consistency_distinguishes_cancellation_and_magnitude_variation(self):
        stable = coherence(np.array([1.]), np.array([1.]))[0]
        alternating = coherence(np.array([0.]), np.array([1.]))[0]
        variable = coherence(np.array([2.]), np.array([5.]))[0]
        self.assertGreater(stable, variable)
        self.assertGreater(variable, alternating)
        self.assertEqual(effective_window(0.9, 1), 1)
        self.assertAlmostEqual(effective_window(0.9), 19)

    def test_gain_bound_random_and_zeros(self):
        rng = np.random.default_rng(100)
        for rule in ('linear','square','sqrt','rational','tanh','historical'):
            p = np.zeros(20)
            o = AdaSym([p], Config(rule=rule, cap=8))
            for i in range(50):
                o.step([rng.normal(size=20) if i else np.zeros(20)])
                self.assertGreaterEqual(o.diagnostics['gain'], 1-1e-12)
                self.assertLessEqual(o.diagnostics['gain'], np.sqrt(8)+1e-12)
                self.assertTrue(np.isfinite(p).all())

    def test_invalid_input_does_not_mutate(self):
        p = np.ones(3)
        o = AdaSym([p])
        with self.assertRaises(ValueError):
            o.step([np.full(3, np.nan)])
        np.testing.assert_array_equal(p, np.ones(3))
        self.assertEqual(o.states[0]['t'], 0)
        o.step([None])
        np.testing.assert_array_equal(p, np.ones(3))

    def test_sign_and_scale(self):
        rng = np.random.default_rng(10)
        p, n, scaled = np.zeros(4), np.zeros(4), np.zeros(4)
        config = Config(eps=1e-12, weight_decay=0)
        a, b, c = [AdaSym([v], config) for v in (p,n,scaled)]
        for _ in range(15):
            g = rng.normal(size=4)
            a.step([g]); b.step([-g]); c.step([7*g])
        np.testing.assert_allclose(p, -n, atol=1e-13)
        np.testing.assert_allclose(p, scaled, atol=1e-11)

    def test_mlp_gradient_finite_difference(self):
        p = init(15)
        x, y = dataset(42)[0]
        x, y = x[:7], y[:7]
        _, grads = evaluate(p, x, y, gradient=True)
        for array, grad in zip(p, grads):
            index = (0,)*array.ndim
            old = array[index]
            array[index] = old+1e-6
            plus = evaluate(p, x, y)[0]
            array[index] = old-1e-6
            minus = evaluate(p, x, y)[0]
            array[index] = old
            self.assertAlmostEqual(grad[index], (plus-minus)/2e-6, places=7)

    def test_sigmoid_requires_explicit_progress(self):
        o = AdaSym([np.zeros(2)], Config(schedule='sigmoid'))
        with self.assertRaises(ValueError):
            o.step([np.ones(2)])
        o.step([np.ones(2)], progress=0)
        first = o.diagnostics['gain']
        o.step([np.ones(2)], progress=1)
        self.assertLess(o.diagnostics['gain'], first)


try:
    import torch
    from optimizer_research.torch_optimizer import AdaSymTorch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional torch not installed')
class TorchTests(unittest.TestCase):
    def test_matches_numpy_all_rules(self):
        rng = np.random.default_rng(7)
        for rule in ('linear','square','sqrt','rational','tanh','historical'):
            for schedule in ('constant','sigmoid'):
                p = rng.normal(size=5)
                t = torch.nn.Parameter(torch.tensor(p.copy(), dtype=torch.float64))
                config = Config(rule=rule, schedule=schedule)
                a, b = AdaSym([p], config), AdaSymTorch([t], config)
                for step in range(25):
                    g = rng.normal(size=5)
                    t.grad = torch.tensor(g, dtype=torch.float64)
                    b.set_progress(step/24)
                    a.step([g], progress=step/24)
                    b.step()
                    np.testing.assert_allclose(t.detach().numpy(), p, atol=1e-13)

    def test_off_matches_native_adamw(self):
        c = Config(alpha=0)
        a = torch.nn.Parameter(torch.ones(7, dtype=torch.float64))
        b = torch.nn.Parameter(a.detach().clone())
        ours = AdaSymTorch([a], c)
        native = torch.optim.AdamW([b], lr=c.lr, betas=(c.beta1,c.beta2), eps=c.eps, weight_decay=c.weight_decay, foreach=False)
        for step in range(20):
            g = torch.sin(torch.arange(7, dtype=torch.float64)+step)
            a.grad, b.grad = g.clone(), g.clone()
            ours.step(); native.step()
        torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)

    def test_checkpoint_resume(self):
        p = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
        a = AdaSymTorch([p])
        p.grad = torch.ones_like(p)
        a.step()
        q = torch.nn.Parameter(p.detach().clone())
        b = AdaSymTorch([q])
        b.load_state_dict(copy.deepcopy(a.state_dict()))
        p.grad = torch.full_like(p, -0.5)
        q.grad = p.grad.clone()
        a.step(); b.step()
        torch.testing.assert_close(p, q)


if __name__ == '__main__':
    unittest.main()
