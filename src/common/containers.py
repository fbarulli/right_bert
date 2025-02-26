class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    cuda_manager = providers.Singleton(CUDAManager, config=config)
    
    tensor_manager = providers.Singleton(
        TensorManager,
        config=config,
        cuda_manager=cuda_manager
    )
