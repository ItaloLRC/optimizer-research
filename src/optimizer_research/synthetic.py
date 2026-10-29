"""A small, reproducible nonlinear classification task and tanh MLP.

Train, validation, and test samples are independent conditional on the same
teacher. Data IDs are independent from initialization/minibatch RNG seeds.
"""
import numpy as np
from .numpy_optimizer import AdaSym


def dataset(task_seed, count=1400, features=12, noise=0.15):
    rng = np.random.default_rng(task_seed)
    x = rng.normal(size=(count, features))
    score = x[:, 0]*x[:, 1] + 0.6*np.sin(2*x[:, 2]) + 0.5*x[:, 3] - 0.3*x[:, 4]**2
    y = (score + noise*rng.normal(size=count) > 0).astype(np.int64)
    return (x[:900], y[:900]), (x[900:1150], y[900:1150]), (x[1150:], y[1150:])


def init(seed, features=12, hidden=24):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(features, hidden))/np.sqrt(features), np.zeros(hidden), rng.normal(size=(hidden, 2))/np.sqrt(hidden), np.zeros(2)]


def evaluate(params, x, y, gradient=False):
    w, b, v, c = params
    h = np.tanh(x@w+b)
    logits = h@v+c
    logits -= logits.max(axis=1, keepdims=True)
    logprob = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = -float(logprob[np.arange(len(y)), y].mean())
    accuracy = float((logprob.argmax(axis=1) == y).mean())
    if not gradient:
        return loss, accuracy
    delta = np.exp(logprob)
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    dh = (delta@v.T)*(1-h*h)
    return loss, [x.T@dh, dh.sum(axis=0), h.T@delta, delta.sum(axis=0)]


def train(config, seed, task_seed, epochs=35, batch_size=64, include_test=False):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    training, validation, test = dataset(task_seed)
    p = init(seed)
    optimizer = AdaSym(p, config)
    rng = np.random.default_rng(seed + 100000)
    curve = []
    total_steps = epochs * int(np.ceil(len(training[0])/batch_size))
    step = 0
    best = (float('inf'), None, None)
    for epoch in range(epochs):
        order = rng.permutation(len(training[0]))
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            _, grads = evaluate(p, training[0][ids], training[1][ids], gradient=True)
            optimizer.step(grads, progress=step/max(total_steps-1, 1))
            step += 1
        train_loss, _ = evaluate(p, *training)
        val_loss, val_accuracy = evaluate(p, *validation)
        if not np.isfinite(train_loss+val_loss):
            raise FloatingPointError("divergent run")
        curve.append(dict(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy, **optimizer.diagnostics))
        if val_loss < best[0]:
            best = (val_loss, [a.copy() for a in p], epoch+1)
    result = dict(seed=seed, task_seed=task_seed, config=config.dictionary(), val_auc=float(np.mean([r['val_loss'] for r in curve])), best_val_loss=best[0], best_epoch=best[2], curve=curve)
    if include_test:
        result['test_loss'], result['test_accuracy'] = evaluate(best[1], *test)
    return result
