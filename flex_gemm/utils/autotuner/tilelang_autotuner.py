from typing import *
import inspect
import traceback
import concurrent.futures
from tqdm.auto import tqdm
import torch
import tilelang
from tilelang import env
from tilelang.jit import JITImpl
from tilelang.autotuner.tuner import _init_logger_handlers, logger, get_available_cpu_count, run_with_timeout, TimeoutException
from tilelang.autotuner.param import AutotuneResult
from .. import save_autotune_cache
from ... import AUTOSAVE_AUTOTUNE_CACHE


class TileLangAutoTunerWithInputFn(tilelang.autotuner.AutoTuner):
    input_fn: Callable
    
    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()

        sig = inspect.signature(self.fn)
        parameters = sig.parameters

        # NOTE(chaofan):  We need to extract some parameters from the closure.
        # Consider the case:
        #   def gemm(M, N, K):
        #       def kernel(...)
        # If we only extract source, M/N/K will be symbolic and there will be cache problem.
        extra_parameters: dict[str, Any] = {}
        cells = self.fn.__closure__
        var_names = self.fn.__code__.co_freevars
        if cells is not None:
            assert len(var_names) == len(cells), "Number of free variables does not match"
            for var_name, cell in zip(var_names, cells):
                if var_name in parameters:
                    continue
                # Cell content must be serializable
                assert isinstance(cell.cell_contents, (int, float, str, bool, type(None))), (
                    f"Cell contents {cell.cell_contents} is not serializable: {type(cell.cell_contents)}"
                )
                extra_parameters[var_name] = cell.cell_contents

        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)

        best_latency: float = 1e8
        best_config: dict[str, Any] | None = None
        best_kernel: tilelang.JITKernel | None = None

        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))

        if self.jit_compile is None:
            self.jit_compile = _compile

        def target_fn(jit_kernel: tilelang.JITKernel, config):
            profiler = jit_kernel.get_profiler()
            latency = profiler.do_bench(warmup=warmup, rep=rep, input_tensors=self.input_fn(**config))
            return latency

        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)

        if len(config_args) == 0:
            raise ValueError("No configurations to tune, please check your `@autotune` decorator")

        # get the cpu count
        available_cpu_count = get_available_cpu_count()
        cpu_utilizations = float(env.TILELANG_AUTO_TUNING_CPU_UTILITIES)
        cpu_counts = int(env.TILELANG_AUTO_TUNING_CPU_COUNTS)
        max_cpu_count = int(env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT)
        if cpu_counts > 0:
            num_workers = min(cpu_counts, available_cpu_count)
            logger.info(f"Auto-tuning with {cpu_counts} CPU counts, {available_cpu_count} CPUs available, {num_workers} CPUs will be used")
        else:
            num_workers = max(1, int(available_cpu_count * cpu_utilizations))
            logger.info(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used"
            )

        if max_cpu_count > 0 and num_workers > max_cpu_count:
            logger.warning(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used, but the max CPU count is {max_cpu_count}, so we will use {max_cpu_count} CPUs"
            )
            num_workers = max_cpu_count

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        future_to_index = {}

        def cuda_device_wrapper(func, device):
            def inner(**config_arg):
                torch.cuda.set_device(device)
                return func(**config_arg)

            return inner

        for i, config_arg in enumerate(config_args):
            compile_func = self.jit_compile

            if torch.cuda.is_available():
                device = torch.cuda.current_device()

                compile_func = cuda_device_wrapper(self.jit_compile, device)

            future = pool.submit(
                compile_func,
                **config_arg,
            )
            futures.append(future)
            future_to_index[future] = i

        results_with_configs = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Compiling configurations"):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                result = future.result()
                results_with_configs.append((result, config))
            except Exception as e:
                logger.debug(f"Compilation failed for config {config} at index {idx} with error: {e}")
                continue

        progress_bar = tqdm(range(len(results_with_configs)), desc="Bench configurations")
        for i in progress_bar:
            jit_kernel, config = results_with_configs[i]
            try:
                latency = run_with_timeout(target_fn, timeout, jit_kernel, config)
            except TimeoutException:
                logger.warning(f"A timeout occurred while testing config {config}, checkout autotuner.log for more details")
                continue
            except Exception:
                logger.warning(f"An error occurred while testing config {config}, checkout autotuner.log for more details")
                logger.debug(f"Error: {traceback.format_exc()}")
                continue

            if latency < best_latency:
                best_latency = latency
                best_config = config
                best_kernel = jit_kernel

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()

        if best_kernel is None:
            error_msg = "Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation."
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        best_kernel: tilelang.JITKernel = best_kernel.update_tuner_result(
            latency=best_latency,
            config=best_config,
            ref_latency=None,
        )

        autotuner_result = AutotuneResult(
            latency=best_latency,
            config=best_config,
            libcode=best_kernel.get_kernel_source(),
            func=best_kernel.prim_func,
            kernel=best_kernel,
        )

        return autotuner_result


class TileLangPersistentCacheAutotunerImpl:
    def __init__(
        self,
        jit_impl: JITImpl,
        configs: dict,
        key_fn: Callable,
        heuristic_fn: Callable = None,
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 100,
    ):
        self.jit_impl = jit_impl
        self.configs = configs
        self.key_fn = key_fn
        self.heuristic_fn = heuristic_fn
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
                tuner = TileLangAutoTunerWithInputFn.from_kernel(self.jit_impl.func, configs=self.configs)
                if self.heuristic_fn is not None:
                    tuner.input_fn = lambda **config: self.heuristic_fn(compile_kwargs={**compile_arg_dict, **config}, kernel_args=kernel_args)
                else:
                    tuner.input_fn = lambda **config: kernel_args
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
            best_config = self.config_cache[key]
            best_kernel = self.kernel_cache[key]
            if self.heuristic_fn is not None:
                kernel_args = self.heuristic_fn(compile_kwargs={**compile_arg_dict, **best_config}, kernel_args=kernel_args)
            best_kernel(*kernel_args)

        return autotuned_kernel_fn
