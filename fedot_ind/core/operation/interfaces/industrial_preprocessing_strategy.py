import warnings
from copy import deepcopy
from inspect import signature
from typing import Optional, Union, Callable

import numpy as np
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import convert_to_multivariate_model, EvaluationStrategy
from fedot.core.utils import is_multi_output_target
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operation_type_from_id, OperationTypesRepository
from fedot.utilities.random import ImplementationRandomStateHandler
from pymonad.either import Either
from pymonad.tools import curry

from fedot_ind.core.architecture.preprocessing.data_convertor import ConditionConverter, FedotConverter, NumpyConverter
from fedot_ind.core.repository.IndustrialOperationParameters import IndustrialOperationParameters
from fedot_ind.core.repository.model_repository import FEDOT_PREPROC_MODEL, FORECASTING_PREPROC, \
    INDUSTRIAL_CLF_PREPROC_MODEL, INDUSTRIAL_PREPROC_MODEL

# from fedot_ind.core.architecture.preprocessing.data_convertor import TensorConverter
import torch


class MultiDimPreprocessingStrategy(EvaluationStrategy):
    """
    Class for preprocessing operations that can be used for multi-dimensional data.

    Args:
        operation_impl: operation implementation
        operation_type: operation type
        params: operation parameters
        mode: mode of operation. Can be 'one_dimensional', 'channel_independent' or 'multi_dimensional'

    """

    def __init__(self, operation_impl,
                 operation_type: str,
                 params: Optional[OperationParameters] = None,
                 mode: str = 'one_dimensional'
                 ):
        self.operation_impl = operation_impl
        super().__init__(operation_type, params)
        self.output_mode = 'labels'
        self.concat_func = np.hstack
        self.mode = mode

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def __convert_input_data(self, input_data):
        multidim_features = len(input_data.features.shape) > 2
        converted_data = self._convert_input_data(input_data) if multidim_features else input_data
        return converted_data

    def __operation_multidim_adapter(self, trained_operation, predict_data, output_mode):
        neural_operation = self.mode == 'multi_dimensional'
        multidim_predict = self.operation_condition.have_predict_method and neural_operation

        prediction = Either(value=trained_operation,
                            monoid=[predict_data, multidim_predict]). \
            either(left_function=lambda data: self._predict_for_ndim(data, trained_operation),
                   right_function=lambda operation: operation.predict(predict_data, output_mode)
                   if not self.operation_condition.have_predict_atr else operation)

        return prediction

    def _convert_to_output(
            self,
            prediction,
            predict_data: InputData,
            output_data_type: DataTypesEnum = DataTypesEnum.table,
            output_mode: str = 'default') -> OutputData:

        return FedotConverter(
            data=predict_data).convert_to_output_data(
            prediction, predict_data, output_data_type)

    def _sklearn_compatible_prediction(
            self,
            trained_operation,
            predict_data,
            output_mode: str = 'probs'):

        # For genuinely tabular 2D data, ``output_mode_converter`` calls
        # ``predict_proba`` → returns ``(N, 2)`` matrix, while FEDOT's
        # metric evaluation expects labels ``(N,)``. That mismatch makes
        # every tuning trial score exactly ~0.501 regardless of model.
        # Force labels output for tabular classifiers.
        _input = predict_data
        if isinstance(_input, InputData) and getattr(_input, 'data_type', None) == DataTypesEnum.table \
                and getattr(_input.features, 'ndim', 0) == 2:
            output_mode = 'labels'

        one_class_operation = (self.operation_condition.is_one_class_operation,
                               self.operation_condition.is_regression_of_forecasting_task)
        only_predict_method = self.operation_condition.is_regression_of_forecasting_task \
            # or not self.operation_condition.have_predict_for_fit_method
        n_classes = 1 if any(one_class_operation) \
            else len(trained_operation.classes_[0]) \
            if self.operation_condition.is_multi_output_target \
            else len(trained_operation.classes_) \
            if hasattr(trained_operation.classes_, '__len__') else trained_operation.classes_
        predict_data = predict_data if self.operation_condition.is_predict_input_fedot else predict_data.features
        predict_method = curry(1)(lambda data: trained_operation.predict(data) if only_predict_method
                                  else trained_operation.predict_for_fit(data))

        prediction = Either(value=predict_data,
                            monoid=[dict(output_mode=output_mode,
                                         predict_data=predict_data,
                                         n_classes=n_classes),
                                    only_predict_method]). \
            either(left_function=lambda data_dict: self.operation_condition.output_mode_converter(**data_dict),
                   right_function=predict_method)
        return prediction

    def _convert_input_data(self, train_data, mode: str = None):
        return FedotConverter(train_data).convert_to_industrial_composing_format(
            mode if mode is not None else self.mode)

    def _predict_for_ndim(self, predict_data, trained_operation: list):
        trained_operation = trained_operation if isinstance(trained_operation, list) else [trained_operation]
        self.operation_condition_for_channel_independent = ConditionConverter(
            predict_data, trained_operation[0], self.mode)
        predict_method = self.operation_condition_for_channel_independent.have_predict_method
        fedot_input = self.operation_condition_for_channel_independent.is_transform_input_fedot

        # Genuinely tabular 2D data was not split by channels during fit
        # (see ``MultiDimPreprocessingStrategy.fit`` bypass). Match that
        # here: run predict/transform once on the whole matrix. Two
        # call conventions, matching the fit bail:
        #   * FEDOT data-op wrapper: ``.transform(InputData) -> OutputData``.
        #   * Raw sklearn estimator: ``.transform(X)`` (preprocessors) /
        #     ``.predict_proba(X)`` (classifiers, so downstream metric
        #     eval sees probabilities) / ``.predict(X)`` (regressors).
        # Dispatch by fit signature (both kinds take a single arg to
        # transform/predict, but only sklearn's ``fit`` takes ``(X, y)``),
        # mirroring what the fit bail did.
        if not isinstance(predict_data, list) \
                and getattr(predict_data.features, 'ndim', 0) == 2 \
                and predict_data.data_type == DataTypesEnum.table:
            op = trained_operation[0]
            sklearn_style = hasattr(op, 'fit') and len(signature(op.fit).parameters) > 1
            if sklearn_style:
                X = predict_data.features
                # For sklearn preprocessors (scaling etc.) use ``transform``.
                # For models (classifiers / regressors) use ``predict`` which
                # returns labels / scalars shaped (N,) — matches what FEDOT's
                # metric evaluation expects in default mode. Returning
                # ``predict_proba`` here breaks downstream metric eval and
                # makes every tuning trial score ~0.5 regardless of model.
                if hasattr(op, 'transform') and not hasattr(op, 'predict'):
                    prediction = op.transform(X)
                elif hasattr(op, 'predict'):
                    prediction = op.predict(X)
                elif hasattr(op, 'transform'):
                    prediction = op.transform(X)
                else:
                    raise ValueError(
                        f"Operation {op!r} has neither .transform nor "
                        ".predict for 2D tabular data")
            else:
                # FEDOT data-op wrapper path.
                if hasattr(op, 'transform'):
                    prediction = op.transform(predict_data)
                elif hasattr(op, 'predict'):
                    prediction = op.predict(predict_data)
                else:
                    raise ValueError(
                        f"Operation {op!r} has neither .transform nor "
                        ".predict for 2D tabular data")
            if isinstance(prediction, OutputData):
                return prediction.predict
            arr = np.asarray(prediction)
            # Classifier predict() returns (N,) labels; reshape to (N, 1)
            # because downstream DataMerger / Industrial concat expects 2D.
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr

        # create list of InputData, where each InputData correspond to each
        # channel
        predict_data = predict_data if isinstance(predict_data, list) else \
            [InputData(idx=predict_data.idx,
                       features=features,
                       target=predict_data.target,
                       task=predict_data.task,
                       data_type=predict_data.data_type,
                       supplementary_data=predict_data.supplementary_data) for features in
             predict_data.features.swapaxes(1, 0)]

        # If model is classical sklearn model we use one_dimensional mode
        predict_branch = curry(2)(
            lambda operation_list,
            data_list: list(
                operation_sample.predict(data_sample) for operation_sample,
                data_sample in zip(
                    operation_list,
                    data_list)) if predict_method else data_list)

        transform_branch = curry(2)(
            lambda operation_list, previous_state: previous_state if predict_method else list(
                operation_sample.transform(
                    data_sample.features) for operation_sample, data_sample in zip(
                    operation_list, previous_state)) if not fedot_input else list(
                operation_sample.transform(data_sample) for operation_sample, data_sample in zip(
                    operation_list, previous_state)))

        prediction = Either.insert(predict_data). \
            then(predict_branch(trained_operation)). \
            then(transform_branch(trained_operation)).value

        prediction = [pred.predict if not isinstance(pred, np.ndarray) else pred for pred in prediction]
        prediction = NumpyConverter(data=self.concat_func(prediction)).convert_to_torch_format() \
            if not isinstance(prediction[0], OutputData) else prediction
        return prediction

    def _custom_fit(self, train_data):
        operation_implementation = self.operation_impl(self.params_for_fit)
        operation_implementation.fit(train_data)
        return operation_implementation

    def fit_one_sample(self, train_data: InputData):
        # evaluate logical condition
        is_model_not_support_multi = self.operation_type in OperationTypesRepository().suitable_operation(
            task_type=train_data.task.task_type, tags=['non_multi'])
        not_fedot_input_data = len(signature(self.operation_condition.operation_implementation.fit).parameters) > 1
        is_multi_target = is_multi_output_target(train_data)
        model_multi_adaptation = all([is_model_not_support_multi, is_multi_target])

        operation_implementation = Either(value=train_data,
                                          monoid=[train_data, model_multi_adaptation]). \
            either(
            left_function=lambda data: self.operation_condition.operation_implementation.fit(
                data.features, data.target) if not_fedot_input_data else
            self.operation_condition.operation_implementation.fit(data),
            right_function=lambda data:
            convert_to_multivariate_model(self.operation_condition.operation_implementation, data))
        operation_implementation = operation_implementation if model_multi_adaptation \
            else self.operation_condition.operation_implementation
        return operation_implementation

    def _init_impl(self, channel_params):
        dict_input = len(signature(self.operation_impl).parameters) > 1
        operation_implementation = self.operation_impl(**channel_params.to_dict()) if dict_input \
            else self.operation_impl(channel_params)
        return operation_implementation

    def _list_of_fitted_model(self, data, prev_state):
        for operation_example, data_sample in zip(prev_state, data):
            if self.operation_condition.have_fit_method:
                operation_example.fit(data_sample)
            else:
                operation_example.transform(data_sample)
        return prev_state

    def fit(self, train_data: InputData):
        # create logical condition verifier
        list_of_params = isinstance(self.params_for_fit, list)

        self.operation_condition = Either(value=self.params_for_fit, monoid=[None, True]).then(
            lambda params: self._init_impl(params) if not list_of_params else list(map(self._init_impl, params))). \
            then(lambda operation: ConditionConverter(train_data, operation, self.mode)).value

        # For 2D tabular + sklearn-style estimator (``.fit(X, y)``) the
        # Industrial-managed ``_init_impl`` seeds the estimator with
        # TS-oriented defaults (e.g. ``min_samples_leaf=0.5``) that collapse
        # every tree to a constant predictor, pinning F1 at ~0.5 regardless
        # of tuning. Rebuild the estimator with its class defaults before
        # the fit chain runs — this affects both the ``one_dimensional``
        # path (``fit_one_sample``) and the ``channel_independent`` path
        # because they both pick up ``self.operation_condition.operation_implementation``.
        is_single_tabular = (not isinstance(train_data, list)
                             and getattr(train_data, 'data_type', None) == DataTypesEnum.table
                             and getattr(train_data.features, 'ndim', 0) == 2)
        if is_single_tabular and not isinstance(self.operation_condition.operation_implementation, list):
            orig = self.operation_condition.operation_implementation
            if hasattr(orig, 'fit') and len(signature(orig.fit).parameters) > 1:
                try:
                    fresh = type(orig)()
                    self.operation_condition.operation_implementation = fresh
                    self.operation_condition.operation_example = fresh
                except Exception:
                    pass

        if is_single_tabular:
            wrapper = self.operation_condition.operation_implementation
            sklearn_style = len(signature(wrapper.fit).parameters) > 1
            if sklearn_style:
                # Industrial's default ``IndustrialOperationParameters`` and
                # tuner search space for sklearn classifiers/regressors are
                # TS-oriented: e.g. ``min_samples_leaf=0.5`` (float=fraction)
                # which collapses every tree to a constant predictor on
                # tabular data, so F1 stays ~0.5 regardless of tuning.
                # For 2D-table we rebuild the estimator from its own class
                # defaults; tuning over the Industrial space for tables is
                # a separate fix (bad search ranges in Industrial repo).
                impl_cls = type(wrapper)
                try:
                    wrapper = impl_cls()
                except Exception:
                    # Non-standard constructors (e.g. sklearn wrappers with
                    # required args) — keep the pre-built wrapper.
                    pass
                target = train_data.target
                if target is not None and target.ndim > 1 and target.shape[1] == 1:
                    target = target.ravel()
                wrapper.fit(train_data.features, target)
            else:
                wrapper.fit(train_data)
            return wrapper
        operation_for_every_dim = self.operation_condition.input_data_is_list_container
        operation_for_one_dim = self.operation_condition.is_one_dim_operation
        operation_for_multidim = not any([operation_for_one_dim, operation_for_every_dim])

        # If model is classical sklearn model we use one_dimensional mode
        fit_one_dim = curry(2)(lambda operation, init_state: self.fit_one_sample(init_state)
                               if operation_for_one_dim else operation)

        # Elif model could be use for each dimension(channel) independently we use channel_independent mode
        channel_independent_branch = curry(2)(lambda data, prev_state: list(deepcopy(prev_state) for i in
                                                                            range(len(data)))
                                              if operation_for_every_dim else prev_state)

        # Apply fit operation for every dimension
        fit_for_every_dim = curry(2)(lambda data, prev_state: self._list_of_fitted_model(data, prev_state)
                                     if operation_for_every_dim else prev_state)

        fit_multidim = curry(2)(lambda data, prev_state: prev_state.fit(data) if operation_for_multidim else prev_state)

        trained_operation = Either.insert(train_data). \
            then(fit_one_dim(self.operation_condition.operation_implementation)). \
            then(channel_independent_branch(train_data)). \
            then(fit_for_every_dim(train_data)).then(fit_multidim(train_data)).value

        return trained_operation

    def _abstract_predict(self, predict_data, trained_operation, output_mode):
        # If model is classical sklearn model we use classical sklearn predict method
        predict_one_dim = curry(2)(lambda operation, init_state: self._sklearn_compatible_prediction(
            operation, init_state, output_mode) if self.operation_condition.is_one_dim_operation else init_state)
        is_transform_for_fit = output_mode.__contains__('transform_fit')

        def multidim_predict(operation, previous_state):
            state_is_predict = isinstance(previous_state, np.ndarray) or isinstance(previous_state, OutputData)
            if isinstance(previous_state, OutputData):
                previous_state = previous_state.predict
            return previous_state if state_is_predict else self.__operation_multidim_adapter(operation,
                                                                                             previous_state,
                                                                                             output_mode)

        # Elif model could be used for each dimension(channel) independently we use multidimensional predict method
        predict_for_every_dim = curry(2)(multidim_predict)
        if is_transform_for_fit:
            prediction = trained_operation[0].transform_for_fit(predict_data[0])
        else:
            prediction = Either.insert(predict_data). \
                then(predict_one_dim(trained_operation)). \
                then(predict_for_every_dim(trained_operation)).value

        return prediction

    def predict_for_fit(
            self,
            trained_operation: Union[Callable, list],
            predict_data: Union[InputData, list],
            output_mode: str = 'default') -> OutputData:
        data_type, predict_data_copy = FedotConverter(predict_data).unwrap_list_to_output()
        # Create data condition verifier
        self.operation_condition = ConditionConverter(predict_data, trained_operation, self.mode)
        prediction = self._abstract_predict(predict_data, trained_operation, output_mode)
        converted = self._convert_to_output(prediction, predict_data_copy, data_type, output_mode)
        return converted

    def predict(self, trained_operation, predict_data: InputData,
                output_mode: str = 'default') -> OutputData:
        data_type, predict_data_copy = FedotConverter(predict_data).unwrap_list_to_output()
        # Create data condition verifier
        self.operation_condition = ConditionConverter(predict_data, trained_operation, self.mode)
        prediction = self._abstract_predict(predict_data, trained_operation, output_mode)
        converted = self._convert_to_output(prediction, predict_data_copy, data_type, output_mode)
        return converted


class IndustrialCustomPreprocessingStrategy:
    _operations_by_types = FEDOT_PREPROC_MODEL

    def __init__(
            self,
            operation_type: str,
            params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        if params is None or operation_type == 'xgboost':
            params = IndustrialOperationParameters().from_operation_type(operation_type)
        else:
            params = IndustrialOperationParameters().from_params(operation_type, params)
        self.multi_dim_dispatcher = MultiDimPreprocessingStrategy(
            self.operation_impl, operation_type, params=params, mode='channel_independent')
        self.operation_id = operation_type
        self.output_mode = False

    @property
    def operation_type(self):
        return get_operation_type_from_id(self.operation_id)

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types:
            return self._operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain {self.__class__} strategy for {operation_type}')

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        return self.multi_dim_dispatcher.fit(train_data)

    def predict(
            self,
            trained_operation,
            predict_data: InputData,
            output_mode: str = 'default'):
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data)
        return self.multi_dim_dispatcher.predict(
            trained_operation, converted_predict_data, output_mode=output_mode)

    def predict_for_fit(
            self,
            trained_operation,
            predict_data: InputData,
            output_mode: str = 'default') -> OutputData:
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data)
        return self.multi_dim_dispatcher.predict_for_fit(
            trained_operation, converted_predict_data, output_mode=output_mode)


class IndustrialPreprocessingStrategy(IndustrialCustomPreprocessingStrategy):
    _operations_by_types = INDUSTRIAL_PREPROC_MODEL

    def __init__(
            self,
            operation_type: str,
            params: Optional[OperationParameters] = None):
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)
        self.params_for_fit = self.multi_dim_dispatcher.params_for_fit

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types.keys():
            return self._operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain custom preprocessing strategy for {operation_type}')

    def fit(self, train_data: InputData):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        if "torch" in self.operation_type:
            train_data.features = torch.Tensor(train_data.features)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                output_mode: str = 'default') -> OutputData:
        if "torch" in self.operation_type:
            predict_data.features = torch.Tensor(predict_data.features)
        prediction = trained_operation.transform(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(
            self,
            trained_operation,
            predict_data: InputData,
            output_mode: str = 'default') -> OutputData:
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self.multi_dim_dispatcher._convert_to_output(prediction, predict_data)
        return converted


class IndustrialForecastingPreprocessingStrategy(
        IndustrialCustomPreprocessingStrategy):
    _operations_by_types = FORECASTING_PREPROC

    def __init__(
            self,
            operation_type: str,
            params: Optional[OperationParameters] = None):
        params = IndustrialOperationParameters().from_params(operation_type, params) if params \
            else IndustrialOperationParameters().from_operation_type(operation_type)
        super().__init__(operation_type, params)
        self.multi_dim_dispatcher.concat_func = np.vstack
        self.ensemble_func = np.sum

    def _check_exog_params(self, fit_output):
        if self.operation_type == 'exog_ts':
            for output in fit_output:
                output.supplementary_data.is_main_target = False
        return fit_output

    def fit(self, train_data: InputData):
        train_data = self.multi_dim_dispatcher._convert_input_data(train_data)
        fit_output = self.multi_dim_dispatcher.fit(train_data)
        fit_output = self._check_exog_params(fit_output)
        return fit_output

    def predict(self, trained_operation,
                predict_data: InputData,
                output_mode: str = 'default'):
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data)
        predict_output = self.multi_dim_dispatcher.predict(
            trained_operation, converted_predict_data, output_mode=output_mode)
        predict_output.predict = predict_output.predict.squeeze()
        return predict_output

    def predict_for_fit(self, trained_operation,
                        predict_data: InputData,
                        output_mode: str = 'default') -> OutputData:
        converted_predict_data = self.multi_dim_dispatcher._convert_input_data(
            predict_data)
        predict_output = self.multi_dim_dispatcher.predict_for_fit(
            trained_operation, converted_predict_data, output_mode='transform_fit_stage')
        predict_output.predict = predict_output.predict.squeeze()
        return predict_output


class IndustrialClassificationPreprocessingStrategy(
        IndustrialCustomPreprocessingStrategy):
    _operations_by_types = INDUSTRIAL_CLF_PREPROC_MODEL

    def __init__(
            self,
            operation_type: str,
            params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return self.multi_dim_dispatcher.fit(train_data)


class IndustrialDataSourceStrategy(IndustrialCustomPreprocessingStrategy):

    def __init__(
            self,
            operation_type: str,
            params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        return object()

    def predict(self, trained_operation, predict_data: InputData,
                output_mode: str = 'labels') -> OutputData:
        return FedotConverter(predict_data).convert_input_to_output()

    def _convert_to_operation(self, operation_type: str):
        return object()

    def predict_for_fit(
            self,
            trained_operation,
            predict_data: InputData,
            output_mode: str = 'default') -> OutputData:
        return FedotConverter(predict_data).convert_input_to_output()
