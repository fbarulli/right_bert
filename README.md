# Process-Safe Machine Learning Framework

## Architecture Overview

This framework provides a robust, process-safe infrastructure for distributed machine learning experiments, particularly designed to work correctly with Python's multiprocessing in 'spawn' mode.

## Key Components

### 1. Process Management

The system uses a hierarchical process model:
- **Main Process**: Coordinates execution and manages trials
- **Trial Processes**: Run individual experiments (managed by Optuna)
- **Worker Processes**: Handle data loading and preprocessing

Process tracking is done through `src/common/process_registry.py`, which:
- Tracks process hierarchies
- Manages initialization across process boundaries
- Provides safe cleanup during termination

### 2. Manager System

The framework follows a manager-based architecture:

- **Base Managers**: 
  - `BaseManager`: Thread-local state management
  - `BaseProcessManager`: Process-aware state management

- **Core Managers**:
  - `CUDAManager`: Device handling and CUDA availability
  - `TensorManager`: Memory management and tensor creation
  - `TokenizerManager`: Tokenization services
  - `DirectoryManager`: File and directory operations

- **Mid-level Managers**:
  - `DataManager`: Dataset creation
  - `ModelManager`: Model loading and saving
  - `AMPManager`: Automatic mixed precision

- **High-level Managers**:
  - `BatchManager`: Batch processing
  - `MetricsManager`: Metric tracking
  - `WandbManager`: Weights & Biases integration

### 3. Dependency Injection

The system uses dependency injection via `src/common/containers.py`:
- Managers declare dependencies
- The container resolves and initializes in the right order
- Each process gets its own initialized instances

### 4. Multiprocessing Safety

Critical features for multiprocessing safety:
- Process-local state in managers
- Explicit initialization in child processes
- Signal handling for graceful termination
- Ordered cleanup to handle dependencies

## Usage Guide

### Initializing the Framework

```python
from src.common.managers import initialize_factory
initialize_factory(config)
```

### Accessing Managers

```python
from src.common.managers import get_model_manager, get_data_manager
model_manager = get_model_manager()
data_manager = get_data_manager()
```

### Starting Child Processes

When starting new processes, always ensure proper initialization:

```python
from src.common.managers.process_init import ensure_process_initialized
ensure_process_initialized(config)
```

### Safe Resource Cleanup

```python
from src.common.managers import cleanup_managers
cleanup_managers()
```

## Troubleshooting

### Process Boundary Issues

If you encounter errors about managers not being initialized:
1. Ensure `ensure_process_initialized()` is called at the start of the process
2. Check that you're accessing managers in the right order

### Memory Management

If you see memory leaks:
1. Add explicit calls to `clear_memory()`
2. Ensure all tensors are properly moved to CPU before passing to child processes
3. Check for circular references

## Architecture Diagram

```
┌─────────────────────┐  
│  Main Process       │  
│  ┌───────────────┐  │  
│  │ManagerFactory │  │  
│  └───────────────┘  │  
└────────┬────────────┘  
         │                
         ▼                
┌─────────────────────┐  
│  Child Processes    │  
│  ┌───────────────┐  │  
│  │ ProcessInit   │──┼─────► Manager Initialization 
│  └───────────────┘  │      in correct order
│                     │  
│  ┌───────────────┐  │  
│  │ BaseManager   │  │  
│  └───────────────┘  │  
└────────┬────────────┘  
         │                
         ▼                
┌─────────────────────┐  
│  Cleanup            │  
│  ┌───────────────┐  │  
│  │ProcessRegistry│──┼─────► Safe Resource Release
│  └───────────────┘  │  
└─────────────────────┘  
```
