from typing import *
import torch
import tilelang
from tilelang.jit import JITImpl
from .. import save_autotune_cache
from ... import AUTOSAVE_AUTOTUNE_CACHE


class TileLangPersistentCacheAutotunerImpl:
    def __init__(
        self,
        jit_impl: JITImpl,
        configs: dict,
        key_fn: Callable,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 100,
    ):
        self.jit_impl = jit_impl
        self.configs = configs
        self.key_fn = key_fn
        self.warmup = warmup
        self.rep = rep
        self.timeout = timeout
        self.config_cache = {}
        self.kernel_cache = {}
        
    def get_cache(self):
        return self.config_cache
    
    def set_cache(self, cache):
        self.config_cache = cache

    def _args_to_kwargs(self, args, kwargs):
        # Convert args to kwargs
        arg_names = self.jit_impl.signature.parameters.keys()
        arg_dict = dict(zip(arg_names, args))
        arg_dict.update(kwargs)
        return arg_dict
    
    def __call__(self, *args, **kwargs):
        compile_arg_dict = self._args_to_kwargs(args, kwargs)
        for k, v in compile_arg_dict.items():
            if isinstance(v, torch.dtype):
                compile_arg_dict[k] = v.__str__().replace('torch.', '')

        def autotuned_kernel_fn(*kernel_args):
            key = self.key_fn(compile_kwargs=compile_arg_dict, kernel_args=kernel_args)

            if key not in self.config_cache:
                tuner = tilelang.autotuner.AutoTuner.from_kernel(self.jit_impl.func, configs=self.configs)
                tuner.jit_compile = lambda **config: self.jit_impl(*args, **kwargs, __tune_params=config)
                tuner.set_profile_args(
                    warmup=self.warmup, rep=self.rep, timeout=self.timeout,
                    supply_prog=lambda _: list(kernel_args),
                )
                tuner.set_compile_args(
                    out_idx=self.jit_impl.out_idx,
                    execution_backend=self.jit_impl.execution_backend,
                    target=self.jit_impl.target,
                    target_host=self.jit_impl.target_host,
                    verbose=self.jit_impl.verbose,
                    pass_configs=self.jit_impl.pass_configs,
                )
                
                TILELANG_AUTO_TUNING_DISABLE_CACHE_ = tilelang.env.TILELANG_AUTO_TUNING_DISABLE_CACHE
                tilelang.env.TILELANG_AUTO_TUNING_DISABLE_CACHE = '1'
                artifact = tuner.run()             # compiles + runs + validates all configs
                tilelang.env.TILELANG_AUTO_TUNING_DISABLE_CACHE = TILELANG_AUTO_TUNING_DISABLE_CACHE_

                best_kernel = artifact.kernel      # JITKernel
                best_config = artifact.config
                self.config_cache[key] = best_config
                self.kernel_cache[key] = best_kernel

                if AUTOSAVE_AUTOTUNE_CACHE:
                    save_autotune_cache()

            elif key not in self.kernel_cache:
                best_config = self.config_cache[key]
                best_kernel = self.jit_impl(*args, **kwargs, __tune_params=best_config)
                self.kernel_cache[key] = best_kernel
            
            # Reuse best kernel
            best_kernel = self.kernel_cache[key]
            best_kernel(*kernel_args)

        return autotuned_kernel_fn
