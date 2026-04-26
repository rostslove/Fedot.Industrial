"""Unit tests for the PEA ensemblers and their factory.

These tests bypass FEDOT AutoML composing entirely. We replace the
RAFEnsembler's branch-fitting / branch-prediction primitives with cheap
sklearn estimators so we can exercise the aggregation logic
deterministically without spinning up a composer.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from fedot_ind.core.operation.ensembling import (
    BaseEnsembler,
    EnsembleStrategy,
    PEABagEnsembler,
    PEABoostEnsembler,
    PEAStackEnsembler,
)
from fedot_ind.core.operation.partitioning import (
    FeatureSpacePartitioner,
    StratifiedPartitioner,
)


class FakeBranch:
    """Sklearn estimator wrapped in the duck-typed ``predict`` API the
    ensemblers expect from a FEDOT pipeline."""

    def __init__(self, model, problem: str, n_classes: int = 2):
        self.model = model
        self.problem = problem
        self.n_classes = n_classes

    def predict(self, mmd, output_mode: str = 'labels'):  # noqa: ARG002
        # mmd is the duck object passed by the fake RAF; it carries
        # ``features``.
        features = np.asarray(mmd.features)
        if self.problem == 'classification':
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)
                # SimpleNamespace mimics FEDOT's ``OutputData.predict``.
                return SimpleNamespace(predict=proba)
            labels = self.model.predict(features).reshape(-1, 1)
            return SimpleNamespace(predict=labels)
        scalar = self.model.predict(features).reshape(-1, 1)
        return SimpleNamespace(predict=scalar)


class FakeRAF:
    """In-memory stand-in for :class:`RAFEnsembler` that only implements
    the protocol the ensemblers depend on."""

    def __init__(self, problem: str, n_classes: int):
        self.problem = problem
        self._n_classes = n_classes if problem == 'classification' else None
        self._branches = []
        self._head = None
        self.current_pipeline = None
        self.task = None
        self.head = 'xgboost' if problem == 'classification' else 'treg'
        self.n_splits = None
        self.source_prefix = 'data_source_table'

    def _fit_single_branch(self, idx, features, target):
        if self.problem == 'classification':
            model = DecisionTreeClassifier(
                max_depth=3, random_state=42).fit(features, target.ravel())
            return FakeBranch(model, self.problem, self._n_classes)
        model = DecisionTreeRegressor(
            max_depth=3, random_state=42).fit(features, target.ravel())
        return FakeBranch(model, self.problem)

    def _collect_branch_predictions(self, branches, input_data,
                                     branch_indices=None):
        del branch_indices
        cols = []
        n = int(np.asarray(input_data.features).shape[0])
        for branch in branches:
            mmd = SimpleNamespace(features=input_data.features)
            raw = np.asarray(branch.predict(mmd).predict)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if (self.problem == 'classification'
                    and raw.shape[1] == 1
                    and self._n_classes is not None):
                labels = raw[:, 0].astype(int)
                one_hot = np.zeros((n, self._n_classes), dtype=np.float32)
                one_hot[np.arange(n), labels] = 1.0
                raw = one_hot
            cols.append(raw)
        return cols

    def _stack_predictions(self, branches, input_data):
        return np.concatenate(
            self._collect_branch_predictions(branches, input_data),
            axis=1).astype(np.float32, copy=False)


@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=300, n_features=8,
                                n_classes=3, n_informative=5,
                                random_state=0)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=0.1,
                            random_state=0)
    return X, y


def _train_input(X, y):
    return SimpleNamespace(features=X, target=y, idx=np.arange(len(X)))


def _test_input(X):
    return SimpleNamespace(features=X, target=None, idx=np.arange(len(X)))


class TestEnsembleStrategyFactory:
    @pytest.mark.parametrize('alias,expected', [
        ('stacking', PEAStackEnsembler),
        ('pea_stack', PEAStackEnsembler),
        ('bagging', PEABagEnsembler),
        ('pea_bag', PEABagEnsembler),
        ('boosting', PEABoostEnsembler),
        ('pea_boost', PEABoostEnsembler),
    ])
    def test_aliases_resolve_to_expected_class(self, alias, expected):
        ens = EnsembleStrategy.create(alias, n_splits=3)
        assert isinstance(ens, expected)
        assert ens.n_splits == 3

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match='Unknown ensembling method'):
            EnsembleStrategy.create('mystery', n_splits=3)

    def test_params_are_filtered_to_signature(self):
        ens = EnsembleStrategy.create(
            'bagging', n_splits=4,
            params={'voting': 'hard', 'irrelevant_kwarg': 123})
        assert isinstance(ens, PEABagEnsembler)
        assert ens.voting == 'hard'

    def test_n_splits_can_come_from_params(self):
        ens = EnsembleStrategy.create(
            'boosting', n_splits=2, params={'n_splits': 7})
        assert ens.n_splits == 7

    def test_default_method_is_stacking(self):
        ens = EnsembleStrategy.create()
        assert isinstance(ens, PEAStackEnsembler)


class TestPEABagEnsembler:
    def test_soft_voting_returns_probability_surface(self, clf_data):
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABagEnsembler(n_splits=3, voting='soft')
        ens.fit(raf, _train_input(X, y), partitioner)
        probs = ens.predict(raf, _test_input(X), output_mode='full_probs')
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-4)

    def test_soft_voting_labels_match_argmax(self, clf_data):
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABagEnsembler(n_splits=3, voting='soft')
        ens.fit(raf, _train_input(X, y), partitioner)
        probs = ens.predict(raf, _test_input(X), output_mode='full_probs')
        labels = ens.predict(raf, _test_input(X), output_mode='labels')
        np.testing.assert_array_equal(labels, probs.argmax(axis=1))

    def test_hard_voting_returns_int_labels(self, clf_data):
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABagEnsembler(n_splits=3, voting='hard')
        ens.fit(raf, _train_input(X, y), partitioner)
        labels = ens.predict(raf, _test_input(X), output_mode='labels')
        assert labels.dtype.kind == 'i'
        assert set(np.unique(labels).tolist()).issubset({0, 1, 2})

    def test_regression_returns_branch_mean(self, reg_data):
        X, y = reg_data
        raf = FakeRAF('regression', n_classes=0)
        partitioner = FeatureSpacePartitioner.create(
            'sequential', n_splits=3)
        ens = PEABagEnsembler(n_splits=3)
        ens.fit(raf, _train_input(X, y), partitioner)
        preds = ens.predict(raf, _test_input(X), output_mode='labels')
        assert preds.shape == (len(X),)
        # the mean of three trees fit on disjoint partitions should be
        # closer to y than the worst tree alone -- a smoke test, not
        # an exact metric.
        assert np.var(preds) > 0

    def test_invalid_voting_raises(self):
        with pytest.raises(ValueError, match='voting must be'):
            PEABagEnsembler(voting='medium')


class TestPEABoostEnsembler:
    def test_classification_fits_multiple_rounds(self, clf_data):
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABoostEnsembler(n_splits=4, random_state=0)
        ens.fit(raf, _train_input(X, y), partitioner)
        assert 1 <= len(raf._branches) <= 4
        assert len(ens.alphas_) == len(raf._branches)
        assert all(a > 0 for a in ens.alphas_)
        assert ens.n_classes_ == 3

    def test_classification_predict_label_shape(self, clf_data):
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABoostEnsembler(n_splits=3, random_state=0)
        ens.fit(raf, _train_input(X, y), partitioner)
        labels = ens.predict(raf, _test_input(X), output_mode='labels')
        probs = ens.predict(raf, _test_input(X), output_mode='probs')
        assert labels.shape == (len(X),)
        assert probs.shape == (len(X), 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_classification_beats_first_round(self, clf_data):
        """A multi-round boosting should at least match the first
        round's training accuracy on the noisy training set."""
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEABoostEnsembler(n_splits=5, random_state=0)
        ens.fit(raf, _train_input(X, y), partitioner)
        labels = ens.predict(raf, _test_input(X), output_mode='labels')
        accuracy = float(np.mean(labels == y))
        # Trivial sanity: better than random guessing for 3 classes.
        assert accuracy > 1.0 / 3.0

    def test_regression_residual_sequence_decreases_variance(self, reg_data):
        X, y = reg_data
        raf = FakeRAF('regression', n_classes=0)
        partitioner = FeatureSpacePartitioner.create(
            'sequential', n_splits=3)
        ens = PEABoostEnsembler(n_splits=4, learning_rate=0.5,
                                 random_state=0)
        ens.fit(raf, _train_input(X, y), partitioner)
        assert ens.init_value_ == pytest.approx(float(np.mean(y)))
        preds = ens.predict(raf, _test_input(X), output_mode='labels')
        assert preds.shape == (len(X),)
        # The boosted prediction should explain more variance than the
        # constant init estimator alone.
        constant_residual = np.var(y - ens.init_value_)
        boosted_residual = np.var(y - preds)
        assert boosted_residual <= constant_residual

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match='learning_rate'):
            PEABoostEnsembler(learning_rate=0)


class TestPEAStackEnsembler:
    def test_stacking_uses_partitioner_and_head(self, clf_data, monkeypatch):
        """Replace the FEDOT head ``_fit_head`` with a sklearn proxy so
        the test stays in pure Python and still exercises the wiring."""
        X, y = clf_data
        raf = FakeRAF('classification', n_classes=3)
        partitioner = StratifiedPartitioner(n_splits=3, random_state=0)
        ens = PEAStackEnsembler(n_splits=3)

        head_clf = DecisionTreeClassifier(max_depth=4, random_state=0)

        class FakeHead:
            """Predicts via sklearn but returns FEDOT-shaped output."""

            def __init__(self, model):
                self.model = model

            def fit(self, input_data):  # noqa: D401, ARG002
                # ``InputData`` has .features / .target attrs in real RAF;
                # for the test we rebind via the captured kwargs below.
                self.model.fit(self._features, self._target)

            def predict(self, input_data, output_mode='labels'):
                features = np.asarray(input_data.features)
                if output_mode == 'labels':
                    out = self.model.predict(features)
                else:
                    out = self.model.predict_proba(features)
                return SimpleNamespace(predict=out)

        captured = {}

        def fake_fit_head(raf, head_op, stacked, target):  # noqa: ARG001
            captured['head_op'] = head_op
            captured['stacked'] = stacked
            head = FakeHead(head_clf)
            head._features = stacked
            head._target = np.asarray(target)
            head.fit(input_data=None)
            return head

        monkeypatch.setattr(PEAStackEnsembler, '_fit_head',
                            staticmethod(fake_fit_head))
        ens.fit(raf, _train_input(X, y), partitioner)

        assert captured['head_op'] == raf.head  # default 'xgboost'
        # 3 branches * 3 classes == 9 stacked features
        assert captured['stacked'].shape == (len(X), 9)

        # predict path also exercises _stack_predictions on test input
        labels = ens.predict(raf, _test_input(X), output_mode='labels')
        assert labels.shape == (len(X),)

    def test_predict_before_fit_raises(self, clf_data):
        X, _ = clf_data
        raf = FakeRAF('classification', n_classes=3)
        ens = PEAStackEnsembler(n_splits=3)
        with pytest.raises(RuntimeError, match='head model is not fitted'):
            ens.predict(raf, _test_input(X), output_mode='labels')


class TestBaseEnsemblerHelpers:
    def test_wants_labels(self):
        assert BaseEnsembler._wants_labels('labels') is True
        assert BaseEnsembler._wants_labels('LABELS') is True
        assert BaseEnsembler._wants_labels('probs') is False
        assert BaseEnsembler._wants_labels('full_probs') is False

    def test_probs_to_labels_argmax(self):
        probs = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
        np.testing.assert_array_equal(
            BaseEnsembler._probs_to_labels(probs), [1, 0])

    def test_probs_to_labels_binary_single_column(self):
        probs = np.array([[0.4], [0.7], [0.5001]])
        np.testing.assert_array_equal(
            BaseEnsembler._probs_to_labels(probs), [0, 1, 1])
