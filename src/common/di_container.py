# src/common/di_container.py
from __future__ import annotations
from dependency_injector import containers, providers

from src.common.managers.amp_manager import AMPManager
from src.common.managers.batch_manager import BatchManager
from src.common.managers.cuda_manager import CUDAManager
from src.common.managers.data_manager import DataManager
from src.common.managers.dataloader_manager import DataLoaderManager
from src.common.managers.directory_manager import DirectoryManager
from src.common.managers.metrics_manager import MetricsManager
from src.common.managers.model_manager import ModelManager
from src.common.managers.optuna_manager import OptunaManager
from src.common.managers.parameter_manager import ParameterManager
from src.common.managers.resource_manager import ProcessResourceManager
from src.common.managers.storage_manager import StorageManager
from src.common.managers.tensor_manager import TensorManager
from src.common.managers.tokenizer_manager import TokenizerManager
from src.common.managers.wandb_manager import WandbManager
from src.common.managers.worker_manager import WorkerManager

from src.common.resource.resource_factory import ResourceFactory
from src.common.resource.resource_pool import ResourcePool
from src.common.resource.resource_initializer import ResourceInitializer

class CoreContainer(containers.DeclarativeContainer):
    """
    Core dependency injection container.
    
    This container handles:
    - Manager instantiation and dependencies
    - Resource class instantiation
    - Configuration management
    - Factory creation
    """
    
    # Configuration
    config = providers.Configuration()

    # Primary Managers (No Dependencies)
    cuda_manager = providers.Singleton(
        CUDAManager,
        config=config
    )

    directory_manager = providers.Singleton(
        DirectoryManager,
        config=config
    )

    tokenizer_manager = providers.Singleton(
        TokenizerManager,
        config=config
    )

    # Secondary Managers (Single Dependencies)
    amp_manager = providers.Singleton(
        AMPManager,
        cuda_manager=cuda_manager,
        config=config
    )

    tensor_manager = providers.Singleton(
        TensorManager,
        cuda_manager=cuda_manager,
        config=config
    )

    dataloader_manager = providers.Singleton(
        DataLoaderManager,
        cuda_manager=cuda_manager,
        config=config
    )

    storage_manager = providers.Singleton(
        StorageManager,
        directory_manager=directory_manager,
        config=config
    )

    # Complex Managers (Multiple Dependencies)
    batch_manager = providers.Singleton(
        BatchManager,
        cuda_manager=cuda_manager,
        tensor_manager=tensor_manager,
        config=config
    )

    metrics_manager = providers.Singleton(
        MetricsManager,
        cuda_manager=cuda_manager,
        config=config
    )

    model_manager = providers.Singleton(
        ModelManager,
        cuda_manager=cuda_manager,
        tokenizer_manager=tokenizer_manager,
        config=config
    )

    parameter_manager = providers.Singleton(
        ParameterManager,
        config=config
    )

    optuna_manager = providers.Singleton(
        OptunaManager,
        storage_manager=storage_manager,
        config=config
    )

    wandb_manager = providers.Singleton(
        WandbManager,
        config=config
    )

    data_manager = providers.Singleton(
        DataManager,
        tokenizer_manager=tokenizer_manager,
        dataloader_manager=dataloader_manager,
        config=config
    )

    worker_manager = providers.Singleton(
        WorkerManager,
        cuda_manager=cuda_manager,
        model_manager=model_manager,
        tokenizer_manager=tokenizer_manager,
        config=config
    )

    resource_manager = providers.Singleton(
        ProcessResourceManager,
        cuda_manager=cuda_manager,
        model_manager=model_manager,
        tokenizer_manager=tokenizer_manager,
        data_manager=data_manager,
        config=config
    )

    # Resource Classes
    resource_pool = providers.Singleton(
        ResourcePool,
        cuda_manager=cuda_manager,
        memory_limit_gb=config.resources.max_memory_gb,
        cleanup_interval=0.1
    )

    resource_factory = providers.Singleton(
        ResourceFactory,
        dataloader_manager=dataloader_manager
    )

    resource_initializer = providers.Singleton(
        ResourceInitializer,
        cuda_manager=cuda_manager,
        amp_manager=amp_manager,
        data_manager=data_manager,
        dataloader_manager=dataloader_manager,
        tensor_manager=tensor_manager,
        tokenizer_manager=tokenizer_manager,
        model_manager=model_manager,
        metrics_manager=metrics_manager,
        parameter_manager=parameter_manager,
        storage_manager=storage_manager,
        directory_manager=directory_manager,
        worker_manager=worker_manager,
        wandb_manager=wandb_manager,
        optuna_manager=optuna_manager
    )

    # Factory for getting all managers and resources
    factory = providers.Factory(
        lambda: {
            # Managers
            "directory_manager": directory_manager.provided,
            "storage_manager": storage_manager.provided,
            "cuda_manager": cuda_manager.provided,
            "tensor_manager": tensor_manager.provided,
            "amp_manager": amp_manager.provided,
            "batch_manager": batch_manager.provided,
            "metrics_manager": metrics_manager.provided,
            "parameter_manager": parameter_manager.provided,
            "resource_manager": resource_manager.provided,
            "tokenizer_manager": tokenizer_manager.provided,
            "dataloader_manager": dataloader_manager.provided,
            "wandb_manager": wandb_manager.provided,
            "optuna_manager": optuna_manager.provided,
            "worker_manager": worker_manager.provided,
            "data_manager": data_manager.provided,
            "model_manager": model_manager.provided,
            
            # Resources
            "resource_pool": resource_pool.provided,
            "resource_factory": resource_factory.provided,
            "resource_initializer": resource_initializer.provided
        }
    )


__all__ = ['CoreContainer']
