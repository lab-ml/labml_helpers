from typing import Optional, Dict, List, Union, Callable, Tuple

import torch.optim
import torch.utils.data
from torch import nn

import labml.utils.pytorch as pytorch_utils
from labml import tracker, monit
from labml.configs import option, meta_config, BaseConfigs
from .training_loop import TrainingLoopConfigs


class StateModule:
    def __init__(self):
        pass

    # def __call__(self):
    #     raise NotImplementedError

    def create_state(self) -> any:
        raise NotImplementedError

    def set_state(self, data: any):
        raise NotImplementedError

    def on_epoch_start(self):
        raise NotImplementedError

    def on_epoch_end(self):
        raise NotImplementedError


class ModeState:
    def __init__(self):
        self._rollback_stack = []

        self.is_train = False
        self.is_log_activations = False
        self.is_log_parameters = False
        self.is_optimize = False

    def _enter(self, mode: Dict[str, any]):
        rollback = {}
        for k, v in mode.items():
            if v is None:
                continue
            rollback[k] = getattr(self, k)
            setattr(self, k, v)

        self._rollback_stack.append(rollback)

        return len(self._rollback_stack)

    def _exit(self, n: int):
        assert n == len(self._rollback_stack)

        rollback = self._rollback_stack[-1]
        self._rollback_stack.pop(-1)

        for k, v in rollback.items():
            setattr(self, k, v)

    def update(self, *,
               is_train: Optional[bool] = None,
               is_log_parameters: Optional[bool] = None,
               is_log_activations: Optional[bool] = None,
               is_optimize: Optional[bool] = None):
        return Mode(self,
                    is_train=is_train,
                    is_log_parameters=is_log_parameters,
                    is_log_activations=is_log_activations,
                    is_optimize=is_optimize)


MODE_STATE = ModeState()


class Mode:
    def __init__(self, mode: ModeState, **kwargs: any):
        self.mode = mode
        self.update = {}
        for k, v in kwargs.items():
            if v is not None:
                self.update[k] = v

        self.idx = -1

    def __enter__(self):
        self.idx = self.mode._enter(self.update)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mode._exit(self.idx)


class ForwardHook:
    def __init__(self, mode: ModeState, model_name, name: str, module: torch.nn.Module):
        self.mode = mode
        self.model_name = model_name
        self.name = name
        self.module = module
        module.register_forward_hook(self)

    def save(self, name: str, output):
        if isinstance(output, torch.Tensor):
            pytorch_utils.store_l1_l2(name, output)
        elif isinstance(output, tuple):
            for i, o in enumerate(output):
                self.save(f"{name}.{i}", o)

    def __call__(self, module, i, o):
        if not self.mode.is_log_activations:
            return

        self.save(f"module.{self.model_name}.{self.name}", o)


def hook_model_outputs(mode: ModeState, model: torch.nn.Module, model_name: str = "model"):
    for name, module in model.named_modules():
        if name == '':
            name = 'full'
        ForwardHook(mode, model_name, name, module)


class Trainer:
    def __init__(self, *,
                 name: str,
                 data_loader: torch.utils.data.DataLoader,
                 is_increment_global_step: bool,
                 log_interval: Optional[int],
                 update_interval: Optional[int],
                 inner_iterations: int,
                 state_modules: List[StateModule],
                 prepare_for_iteration: Callable,
                 step: Callable[[any, int], Tuple[torch.Tensor, int]],
                 backward: Callable[[torch.Tensor], None],
                 optimizer_step: Callable[[torch.Tensor], None]):
        self.backward = backward
        self.optimizer_step = optimizer_step
        self.step = step
        self.prepare_for_iteration = prepare_for_iteration
        self.log_interval = log_interval
        self.update_interval = update_interval
        self.is_increment_global_step = is_increment_global_step
        self.data_loader = data_loader
        self.name = name
        self.state_modules = state_modules
        self.__total_steps = len(data_loader)
        self.__iteration_idx = -1
        self.__iterable = None
        self.__n_iteration = -1
        self.inner_iterations = inner_iterations
        self.__states = [sm.create_state() for sm in self.state_modules]

    def __call__(self):
        for sm, s in zip(self.state_modules, self.__states):
            sm.set_state(s)

        if self.__iterable is None or self.__n_iteration >= self.inner_iterations:
            self.__iterable = iter(self.data_loader)
            self.__iteration_idx = 0
            self.__n_iteration = 0
            for sm in self.state_modules:
                sm.on_epoch_start()
        self.prepare_for_iteration()
        with torch.set_grad_enabled(MODE_STATE.is_train):
            self.__iterate()

    def __iterate(self):
        self.__n_iteration += 1
        if self.__iteration_idx >= self.__n_iteration * self.__total_steps / self.inner_iterations:
            return

        is_updated = True
        is_activations_logged = False
        is_parameters_logged = False

        with monit.section(self.name, is_partial=True):
            if self.__iteration_idx == 0:
                monit.progress(0)
            while True:
                i = self.__iteration_idx
                batch = next(self.__iterable)

                # TODO: Calculate
                # is_update
                # is_log_activations
                # is_log_parameters
                # TODO: Change
                # is_logged = is_logged or is_log
                is_update = MODE_STATE.is_train and (
                        self.update_interval is not None and (i + 1) % self.update_interval == 0)
                is_log_activations = MODE_STATE.is_log_activations and not is_activations_logged
                # TODO: do this on validation also
                is_log_parameters = is_update and MODE_STATE.is_log_parameters and not is_parameters_logged
                #
                with Mode(is_log_activations=(MODE_STATE.is_log_activations and not is_activations_logged)):
                    is_activations_logged = is_activations_logged or MODE_STATE.is_log_activations
                    with Mode(is_log_parameters=(MODE_STATE.is_log_parameters and not is_parameters_logged)):
                        if self.update_interval is not None and (i + 1) % self.update_interval == 0:
                            is_parameters_logged = is_parameters_logged or MODE_STATE.is_log_parameters
                            if MODE_STATE.is_train:
                                self.optimizer_step(loss)
                        is_updated = True
                    self.step(batch, i)

                is_updated = False

                if self.update_interval is not None and (i + 1) % self.update_interval == 0:
                    with Mode(is_log_parameters=(MODE_STATE.is_log_parameters and not is_parameters_logged)):
                        is_parameters_logged = is_parameters_logged or MODE_STATE.is_log_parameters
                        if MODE_STATE.is_train:
                            self.optimizer_step(loss)
                    is_updated = True

                if self.log_interval is not None and (i + 1) % self.log_interval == 0:
                    tracker.save()

                self.__iteration_idx += 1
                monit.progress(self.__iteration_idx / self.__total_steps)

                if self.__iteration_idx >= self.__n_iteration * self.__total_steps / self.inner_iterations:
                    break

        if not is_updated:
            with Mode(is_log_parameters=(MODE_STATE.is_log_parameters and not is_parameters_logged)):
                if MODE_STATE.is_train:
                    self.optimizer_step(loss)

        if self.__n_iteration >= self.inner_iterations:
            for sm in self.state_modules:
                sm.on_epoch_end()


class TrainerConfigs(BaseConfigs):
    name: str
    log_interval: int = 10
    update_interval: int = 1
    data_loader: torch.utils.data.DataLoader
    is_increment_global_step: bool
    inner_iterations: int
    state_modules: List[StateModule]
    prepare_for_iteration: Callable
    step: Callable[[any, int], Tuple[torch.Tensor, int]]
    backward: Callable[[torch.Tensor], None]
    optimizer_step: Callable[[torch.Tensor], None]

    trainer: Trainer


@option(TrainerConfigs.trainer)
def _trainer(c: TrainerConfigs):
    return Trainer(name=c.name,
                   data_loader=c.data_loader,
                   is_increment_global_step=c.is_increment_global_step,
                   log_interval=c.log_interval,
                   update_interval=c.update_interval,
                   inner_iterations=c.inner_iterations,
                   state_modules=c.state_modules,
                   prepare_for_iteration=c.prepare_for_iteration,
                   step=c.step,
                   backward=c.backward,
                   optimizer_step=c.optimizer_step)


class BatchIndex:
    idx: int
    total: int
    iteration: int
    total_iterations: int

    @property
    def total_inner(self) -> int:
        total_inner = (self.total + self.total_iterations - 1) // self.total_iterations
        # last iteration
        if self.iteration == self.total_iterations - 1:
            total_inner = self.total - self.iteration * total_inner

        return total_inner

    @property
    def inner(self) -> int:
        total_inner = (self.total + self.total_iterations - 1) // self.total_iterations
        return self.idx % total_inner

    def is_interval(self, interval: int):
        if self.idx + 1 == self.total:
            return True
        else:
            return (self.idx + 1) % interval == 0

    def should_update(self, update_interval: Optional[int]):
        if self.idx + 1 == self.total:
            return True
        elif update_interval is None:
            return False
        else:
            return (self.idx + 1) % update_interval == 0


class TrainValidConfigs(TrainingLoopConfigs):
    state_modules: List[StateModule] = []
    optimizer: torch.optim.Adam
    model: Union[nn.Module, Dict[str, nn.Module]]
    device: torch.device

    loss_func: nn.Module
    mode: ModeState

    epochs: int = 10

    trainer: TrainerConfigs
    validator: TrainerConfigs
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader

    loop_count = 'data_loop_count'
    loop_step = None

    inner_iterations: int = 1

    is_log_parameters: bool = True
    is_log_activations: bool = True

    update_batches: int = 1
    log_params_updates: int = 2 ** 32
    log_activations_batches: int = 2 ** 32

    def init_tracker(self):
        tracker.set_queue("loss.*", 20, True)

    def prepare_for_iteration(self):
        if isinstance(self.model, dict):
            for m in self.model.values():
                m.train(MODE_STATE.is_train)
        else:
            self.model.train(MODE_STATE.is_train)

    def step(self, batch: any, batch_idx: BatchIndex):
        """Override"""
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        with monit.section("model"):
            with self.mode.update(is_log_activations=batch_idx.is_interval(self.log_activations_batches)):
                output = self.model(data)

        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

            with monit.section('backward'):
                loss.backward()

            if batch_idx.is_interval(self.update_batches):
                with monit.section('optimize'):
                    self.optimizer.step()
                if batch_idx.is_interval(self.log_params_updates):
                    pytorch_utils.store_model_indicators(self.model)
                self.optimizer.zero_grad()

    def run_step(self):
        for i in range(self.inner_iterations):
            with tracker.namespace('sample'):
                self.sample()
            with self.mode.update(is_train=True):
                with tracker.namespace('train'):
                    self.trainer.trainer()
            if self.validator:
                with tracker.namespace('valid'):
                    self.validator.trainer()

    def run(self):
        self.init_tracker()
        _ = self.validator.trainer
        _ = self.trainer.trainer
        for _ in self.training_loop:
            self.run_step()

    def sample(self):
        pass


@option(TrainValidConfigs.trainer)
def trainer(c: TrainValidConfigs):
    conf = TrainerConfigs()
    conf.name = 'Train'
    conf.data_loader = c.train_loader
    conf.is_increment_global_step = True
    conf.inner_iterations = c.inner_iterations
    conf.state_modules = c.state_modules
    conf.prepare_for_iteration = c.prepare_for_iteration
    conf.step = c.step
    conf.backward = c.backward
    conf.optimizer_step = c.optimizer_step

    return conf


@option(TrainValidConfigs.validator)
def validator(c: TrainValidConfigs):
    conf = TrainerConfigs()
    conf.name = 'Valid'
    conf.data_loader = c.valid_loader
    conf.is_increment_global_step = False
    conf.log_interval = None
    conf.update_interval = None
    conf.inner_iterations = c.inner_iterations
    conf.state_modules = c.state_modules
    conf.prepare_for_iteration = c.prepare_for_iteration
    conf.step = c.step
    conf.backward = None
    conf.optimizer_step = None

    return conf


@option(TrainValidConfigs.optimizer)
def _default_optimizer(c: TrainValidConfigs):
    from labml_helpers.optimizer import OptimizerConfigs
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


@option(TrainValidConfigs.loop_count)
def data_loop_count(c: TrainValidConfigs):
    return c.epochs


meta_config(TrainerConfigs.log_interval,
            TrainerConfigs.update_interval)
meta_config(TrainValidConfigs.is_log_parameters,
            TrainValidConfigs.is_log_activations)
