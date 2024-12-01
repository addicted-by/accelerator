import hydra
import omegaconf
from .base_wrapper import BaseModelWrapper
import copy
from .utils import compare_models, model_info
from logger import log_info, log_warning
from textwrap import dedent


class ModelWrapper(BaseModelWrapper):
    def __init__(self, model_cfg: omegaconf.OmegaConf):
        self._cfg = model_cfg

        self.instantiate_model()
        self.save_original_model()

        log_info(model_info(self._model, self._cfg.verbosity))


    def preload_ops(self):
        if self._cfg.get('checkpoint'):
            self._ckpt = hydra.utils.instantiate(
                self._cfg.load_operation
            )(self._cfg.checkpoint) # default -- torch.load
        else:
            self._ckpt = None

        for operation in self._cfg.preload_ops.order:
            if operation not in self._cfg.preload_ops:
                log_warning(
                    f'Operation {operation} cannot be found among {self.preload_ops}!'
                )
                continue
        
            preload_fn = hydra.utils.instantiate(self._cfg.preload_ops.operation.body)
            
            self._ckpt = preload_fn(
                self._ckpt,
                *self._cfg.preload_ops.operation.args, 
                **self._cfg.preload_ops.kwargs
            )
        
        return self._ckpt

    def postload_ops(self):
        for operation in self._cfg.postload_ops.order:
            if operation not in self._cfg.postload_ops:
                log_warning(
                    f'Operation {operation} cannot be found among {self._cfg.postload_ops}!'
                )
                continue
                
            postload_fn = hydra.utils.instantiate(self._cfg.postload_ops.operation.body)
            postload_fn(
                self._model,
                *self._cfg.preload_ops.args,
                **self._cfg.preload_ops.kwargs
            )

    @property
    def model(self):
        return self._model
    
    @property
    def original_model(self):
        return self._original_model

    def is_model_changed(self):
        return compare_models(self._original_model, self._model)
    
    def save_original_model(self):
        self._original_model = copy.deepcopy(self._model)

    def instantiate_model(self):
        self._model = self.preload_ops()
        if self._model is None: # change to callable check
            if self._cfg.get('body', None) is None:
                msg = dedent(
                    """
                    Please, provide the class to be instantiated or
                    the provided preload_ops must return model itself!
                    """
                )
                raise NotImplementedError(msg) 
            self._model = hydra.utils.instantiate(self._cfg.body)
        self.postload_ops()
        log_info('Model was initialized successfully')
        