# Accelerator Framework Architecture Diagram

```mermaid
graph TB
    %% Styling definitions
    classDef highLevel fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    classDef middleLevel fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000
    classDef lowLevel fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px,color:#000
    classDef crossDomain fill:#fff3e0,stroke:#e65100,stroke-width:3px,color:#000
    classDef acceleration fill:#ffebee,stroke:#b71c1c,stroke-width:2px,color:#000
    classDef mlflow fill:#f1f8e9,stroke:#33691e,stroke-width:2px,color:#000
    classDef config fill:#fafafa,stroke:#424242,stroke-width:1px,color:#000

    %% High-Level API Layer
    subgraph HighAPI ["🔵 High-Level API Layer"]
        direction TB
        
        %% CLI Interface Components
        subgraph CLITools ["CLI Interface"]
            direction TB
            CLIExperiment["accelerator.cli.experiment<br/>• train<br/>• analyze_model<br/>• analyze_data"]
            CLIAdd["accelerator.cli.add<br/>• model<br/>• optimizer<br/>• scheduler<br/>• loss<br/>• transform"]
            CLIConfigure["accelerator.cli.configure<br/>• configure<br/>• init<br/>• status<br/>• clean"]
        end
        
        %% Pipeline Execution Components
        subgraph PipelineExec ["Pipeline Execution"]
            direction TB
            RunPipe["run_pipe.py<br/>Multi-step Training<br/>Pipeline Orchestration"]
            MLProject["MLProject<br/>MLflow Entry Points<br/>• accelerator<br/>• model_analysis<br/>• data_analysis"]
        end
        
        %% Configuration Entry Points
        subgraph ConfigEntry ["Configuration Entry Points"]
            direction TB
            UserConfigs["configs/<br/>User-Level Configs<br/>• main.yaml<br/>• model/<br/>• training_components/"]
            HydraComposition["Hydra Configuration<br/>Composition System<br/>• defaults<br/>• overrides<br/>• indices"]
        end
    end

    %% Middle-Level API Layer  
    subgraph MiddleAPI ["🟣 Middle-Level API Layer"]
        direction TB
        
        %% Runtime System Components
        subgraph RuntimeSystem ["Runtime System"]
            direction TB
            
            %% Runtime Context and Component Management
            subgraph RuntimeContext ["Runtime Context & Component Management"]
                direction TB
                TrainingContext["runtime.context.training<br/>Training Context<br/>• Component lifecycle<br/>• State management<br/>• Resource coordination"]
                ComponentMgmt["runtime.context.component<br/>Component Management<br/>• Registry integration<br/>• Dynamic loading<br/>• Dependency resolution"]
                ContextBase["runtime.context.context<br/>Context Base<br/>• Configuration binding<br/>• Environment setup<br/>• Resource allocation"]
            end
            
            %% Training Engines
            subgraph TrainingEngines ["Training Engines"]
                direction TB
                VanillaTorch["runtime.engine.vanilla_torch<br/>Vanilla PyTorch Engine<br/>• Standard training loop<br/>• Single device support<br/>• Basic optimization"]
                AccelerateEngine["runtime.engine.accelerate<br/>Accelerate Engine<br/>• HuggingFace Accelerate<br/>• Multi-device support<br/>• Mixed precision"]
                SingleGPU["runtime.engine.single_gpu<br/>Single GPU Engine<br/>• GPU-optimized training<br/>• Memory management<br/>• CUDA operations"]
                EngineBase["runtime.engine.base<br/>Engine Base Class<br/>• Common interface<br/>• Lifecycle hooks<br/>• Error handling"]
            end
            
            %% Training Loops and Orchestration
            subgraph TrainingOrchestration ["Training Loops & Orchestration"]
                direction TB
                TrainLoop["runtime.loop.train<br/>Training Loop<br/>• Forward/backward pass<br/>• Loss computation<br/>• Gradient updates"]
                ValidationLoop["runtime.loop.validation<br/>Validation Loop<br/>• Model evaluation<br/>• Metric computation<br/>• Performance tracking"]
                TestLoop["runtime.loop.test<br/>Test Loop<br/>• Final evaluation<br/>• Result aggregation<br/>• Report generation"]
                LoopBase["runtime.loop.base<br/>Loop Base Class<br/>• Common patterns<br/>• State management<br/>• Hook system"]
            end
            
            %% Callbacks System
            subgraph CallbackSystem ["Callbacks System"]
                direction TB
                CallbackManager["runtime.callbacks.manager<br/>Callback Manager<br/>• Callback orchestration<br/>• Event dispatching<br/>• Priority handling"]
                LoggerCallback["runtime.callbacks.logger<br/>Logger Callback<br/>• Training metrics<br/>• Loss tracking<br/>• Performance logging"]
                ProgressCallback["runtime.callbacks.progress<br/>Progress Callback<br/>• Training progress<br/>• ETA estimation<br/>• Visual feedback"]
                AlwaysOnCallback["runtime.callbacks.always_on<br/>Always-On Callbacks<br/>• System monitoring<br/>• Resource tracking<br/>• Health checks"]
                CallbackBase["runtime.callbacks.base<br/>Callback Base<br/>• Event interface<br/>• State access<br/>• Hook definitions"]
            end
        end
        
        %% Configuration Management System
        subgraph ConfigManagement ["Configuration Management"]
            direction TB
            
            %% Hydra Integration
            subgraph HydraIntegration ["Hydra Integration"]
                direction TB
                HydraUtils["utilities.hydra_utils<br/>Hydra Utilities<br/>• Configuration composition<br/>• Override handling<br/>• Environment resolution"]
                HydraCallbacks["utilities.hydra_utils.callbacks<br/>Hydra Callbacks<br/>• Config validation<br/>• Parameter logging<br/>• Error handling"]
                HydraLogger["utilities.hydra_utils.logger<br/>Hydra Logger<br/>• Configuration logging<br/>• Change tracking<br/>• Debug support"]
                HydraMonitor["utilities.hydra_utils.monitor<br/>Hydra Monitor<br/>• Config changes<br/>• Runtime updates<br/>• Validation checks"]
            end
            
            %% Component Registry System
            subgraph ComponentRegistry ["Component Registry System"]
                direction TB
                BaseRegistry["utilities.base_registry<br/>Base Registry<br/>• Component registration<br/>• Type validation<br/>• Discovery patterns"]
                LossRegistry["runtime.loss.registry<br/>Loss Registry<br/>• Loss function registry<br/>• Dynamic loading<br/>• Validation rules"]
                TransformRegistry["runtime.transform.registry<br/>Transform Registry<br/>• Transform registration<br/>• Pipeline composition<br/>• Type checking"]
                AccelRegistry["acceleration.registry<br/>Acceleration Registry<br/>• Acceleration techniques<br/>• Method registration<br/>• Compatibility checks"]
            end
            
            %% Configuration Resolution
            subgraph ConfigResolution ["Configuration Resolution"]
                direction TB
                ConfigComposition["Configuration Composition<br/>• Hierarchical merging<br/>• Default inheritance<br/>• Override application"]
                IndexManagement["Index Management<br/>• Component indices<br/>• Auto-generation<br/>• Dependency tracking"]
                EnvResolution["Environment Resolution<br/>• Variable substitution<br/>• Path resolution<br/>• Runtime binding"]
            end
        end
        
        %% Experiment Tracking Integration
        subgraph ExperimentTracking ["🔬 Experiment Tracking & MLflow Integration"]
            direction TB
            
            %% MLflow Core Integration Points
            subgraph MLflowCore ["🟤 MLflow Core Integration"]
                direction TB
                MLflowPatches["utilities.patches.mlflow<br/>🔧 MLflow Patches<br/>• Framework integration<br/>• Custom logging extensions<br/>• Compatibility fixes<br/>• Enhanced tracking APIs"]
                
                MLflowTracking["🔬 MLflow Tracking Server<br/>• Experiment management<br/>• Run lifecycle<br/>• Remote tracking URI<br/>• Authentication & access"]
                
                MLflowUI["🖥️ MLflow UI<br/>• Experiment comparison<br/>• Metric visualization<br/>• Artifact browsing<br/>• Model registry"]
            end
            
            %% Automatic Parameter Logging Flow
            subgraph ParamLoggingFlow ["📊 Automatic Parameter Logging Flow"]
                direction TB
                ConfigFlattening["🔄 Configuration Flattening<br/>• flatten_dict integration<br/>• Nested parameter extraction<br/>• Hierarchical key generation<br/>• Type conversion & serialization"]
                
                HydraMLflowBridge["🌉 Hydra ↔ MLflow Bridge<br/>• Automatic config logging<br/>• Override tracking<br/>• Composition logging<br/>• Runtime parameter updates"]
                
                ParamLogging["📝 Parameter Logging<br/>• Hyperparameter tracking<br/>• Nested parameter support<br/>• Configuration versioning<br/>• Change detection & logging"]
                
                MetricLogging["📈 Metric Logging<br/>• Training metrics<br/>• Validation metrics<br/>• Custom metrics<br/>• Real-time streaming"]
            end
            
            %% Artifact Management Flow
            subgraph ArtifactFlow ["📦 Artifact Management Flow"]
                direction TB
                ArtifactMgmt["🗃️ Artifact Management<br/>• Model checkpoints<br/>• Training artifacts<br/>• Configuration snapshots<br/>• Result storage & versioning"]
                
                ModelRegistry["🏛️ Model Registry<br/>• Model versioning<br/>• Stage management<br/>• Model metadata<br/>• Deployment tracking"]
                
                ArtifactStorage["💾 Artifact Storage<br/>• Local storage<br/>• Remote storage (S3, etc.)<br/>• Artifact organization<br/>• Access control"]
            end
            
            %% Experiment Organization
            subgraph ExpOrganizationFlow ["🗂️ Experiment Organization"]
                direction TB
                ExpOrganization["📋 Experiment Organization<br/>• Run management<br/>• Experiment grouping<br/>• Tag-based organization<br/>• Search & filtering"]
                
                RunLifecycle["🔄 Run Lifecycle Management<br/>• Run creation & initialization<br/>• Active run tracking<br/>• Run completion & cleanup<br/>• Error handling & recovery"]
                
                ExperimentMetadata["📊 Experiment Metadata<br/>• Run information<br/>• Environment tracking<br/>• Git integration<br/>• Reproducibility data"]
            end
            
            %% Checkpoint Management with MLflow Integration
            subgraph CheckpointMLflowIntegration ["💾 Checkpoint & MLflow Integration"]
                direction TB
                CheckpointHandler["runtime.checkpoint.checkpoint_handler<br/>💾 Checkpoint Handler<br/>• Save/load operations<br/>• MLflow artifact logging<br/>• State management<br/>• Recovery support"]
                
                MetaHandler["runtime.checkpoint.meta_handler<br/>📋 Meta Handler<br/>• Metadata management<br/>• MLflow metadata logging<br/>• Version tracking<br/>• Compatibility checks"]
                
                CheckpointOps["runtime.checkpoint.operation<br/>⚙️ Checkpoint Operations<br/>• Transformations<br/>• MLflow artifact updates<br/>• Migration support<br/>• Validation & logging"]
            end
        end
    end

    %% Low-Level API Layer
    subgraph LowAPI ["🟢 Low-Level API Layer"]
        direction TB
        
        %% Pluggable Domain Architecture
        subgraph DomainComponents ["🔌 Pluggable Domain Architecture"]
            direction TB
            
            %% Cross-Domain Components (Reusable)
            subgraph CrossDomainComponents ["🟠 Cross-Domain (Reusable)"]
                direction TB
                CrossDomainLosses["domain.cross.loss.*<br/>🔄 Generic Loss Functions<br/>• MAE, MSE, etc.<br/>• Domain-agnostic<br/>• Registerable & plug-and-play"]
                CrossDomainTransforms["domain.cross.transform.*<br/>🔄 Generic Transforms<br/>• Input transforms<br/>• Loss transforms<br/>• Registerable & plug-and-play"]
                CrossDomainUtils["domain.cross.utils.*<br/>🔄 Generic Utilities<br/>• Base implementations<br/>• Helper functions<br/>• Registerable & plug-and-play"]
            end
            
            %% Computer Vision Domain
            subgraph CVDomain ["📷 Computer Vision Domain"]
                direction TB
                CVLosses["domain.cv.loss.*<br/>📷 CV Loss Functions<br/>• Structural, perceptual, color<br/>• Image-specific<br/>• Registerable & plug-and-play"]
                CVInputTransforms["domain.cv.transform.input.*<br/>📷 CV Input Transforms<br/>• Image preprocessing<br/>• Data augmentation<br/>• Registerable & plug-and-play"]
                CVLossTransforms["domain.cv.transform.loss.*<br/>📷 CV Loss Transforms<br/>• Color space conversion<br/>• Multi-scale processing<br/>• Registerable & plug-and-play"]
                CVUtils["domain.cv.utils.*<br/>📷 CV Utilities<br/>• Image processing tools<br/>• Vision-specific helpers<br/>• Registerable & plug-and-play"]
            end
            
            %% Future Domain Extensions (Extensible)
            subgraph FutureDomains ["🚀 Future Domain Extensions"]
                direction TB
                NLPDomain["domain.nlp.*<br/>📝 NLP Domain (Future)<br/>• Text loss functions<br/>• Language transforms<br/>• NLP utilities<br/>• Registerable & plug-and-play"]
                AudioDomain["domain.audio.*<br/>🔊 Audio Domain (Future)<br/>• Audio loss functions<br/>• Signal transforms<br/>• Audio utilities<br/>• Registerable & plug-and-play"]
                SignalsDomain["domain.signals.*<br/>📡 Signals Domain (Future)<br/>• Signal loss functions<br/>• Signal transforms<br/>• Signal utilities<br/>• Registerable & plug-and-play"]
                CustomDomain["domain.custom.*<br/>⚙️ Custom Domains<br/>• User-defined domains<br/>• Domain-specific components<br/>• Custom implementations<br/>• Registerable & plug-and-play"]
            end
        end
        
        %% Acceleration Framework
        subgraph AccelFramework ["Acceleration Framework"]
            direction TB
            AccelPruning[Pruning Techniques]
            AccelQuantization[Quantization Methods]
            AccelOptimization[Optimization Approaches]
            AccelReparametrization[Reparametrization Techniques]
        end
        
        %% Utilities & Base Classes
        subgraph Utilities ["Utilities & Base Classes"]
            direction TB
            BaseRegistry[Registry Patterns]
            DistributedUtils[Distributed Training Support]
            ModelUtils[Model Utilities]
            HydraUtils[Hydra Integration]
        end
    end



    %% MLflow Integration Data Flow Arrows
    %% Configuration to MLflow Flow
    HydraComposition --> HydraMLflowBridge
    HydraUtils --> HydraMLflowBridge
    HydraCallbacks --> ConfigFlattening
    ConfigFlattening --> ParamLogging
    HydraMLflowBridge --> ParamLogging
    
    %% CLI to MLflow Integration
    CLIExperiment --> ExpOrganization
    RunPipe --> MLflowTracking
    MLProject --> MLflowTracking
    
    %% Runtime to MLflow Integration
    TrainingContext --> RunLifecycle
    TrainLoop --> MetricLogging
    ValidationLoop --> MetricLogging
    TestLoop --> MetricLogging
    LoggerCallback --> MetricLogging
    
    %% Checkpoint to MLflow Integration
    CheckpointHandler --> ArtifactMgmt
    MetaHandler --> ExperimentMetadata
    CheckpointOps --> ArtifactStorage
    
    %% MLflow Internal Flow
    ParamLogging --> MLflowTracking
    MetricLogging --> MLflowTracking
    ArtifactMgmt --> ArtifactStorage
    ArtifactMgmt --> ModelRegistry
    MLflowTracking --> MLflowUI
    ExpOrganization --> MLflowUI
    RunLifecycle --> ExperimentMetadata
    
    %% Apply styling to components
    class CLIExperiment,CLIAdd,CLIConfigure,RunPipe,MLProject highLevel
    class UserConfigs,HydraComposition config
    
    %% Middle-Level API styling
    class TrainingContext,ComponentMgmt,ContextBase middleLevel
    class VanillaTorch,AccelerateEngine,SingleGPU,EngineBase middleLevel
    class TrainLoop,ValidationLoop,TestLoop,LoopBase middleLevel
    class CallbackManager,LoggerCallback,ProgressCallback,AlwaysOnCallback,CallbackBase middleLevel
    class HydraUtils,HydraCallbacks,HydraLogger,HydraMonitor middleLevel
    class BaseRegistry,LossRegistry,TransformRegistry,AccelRegistry middleLevel
    class ConfigComposition,IndexManagement,EnvResolution config
    
    %% MLflow Integration styling
    class MLflowPatches,MLflowTracking,MLflowUI mlflow
    class ConfigFlattening,HydraMLflowBridge,ParamLogging,MetricLogging mlflow
    class ArtifactMgmt,ModelRegistry,ArtifactStorage mlflow
    class ExpOrganization,RunLifecycle,ExperimentMetadata mlflow
    class CheckpointHandler,MetaHandler,CheckpointOps middleLevel
    
    %% Low-Level API styling
    %% Cross-Domain Components (Reusable) - Special orange styling
    class CrossDomainLosses,CrossDomainTransforms,CrossDomainUtils crossDomain
    
    %% CV-Specific Components - Standard low-level styling
    class CVLosses,CVInputTransforms,CVLossTransforms,CVUtils lowLevel
    
    %% Future Domain Extensions - Highlighted for extensibility
    class NLPDomain,AudioDomain,SignalsDomain,CustomDomain acceleration
    
    %% Framework components
    class AccelPruning,AccelQuantization,AccelOptimization,AccelReparametrization acceleration
    class BaseRegistry,DistributedUtils,ModelUtils,HydraUtils lowLevel
```

## Architecture Overview

This diagram represents the three-tier architecture of the Accelerator ML training framework:

### 🔵 High-Level API Layer

#### CLI Interface
- **accelerator.cli.experiment**: Core experiment management commands
  - `train`: Execute training with configuration
  - `analyze_model`: Model analysis and inspection
  - `analyze_data`: Dataset analysis and validation
- **accelerator.cli.add**: Component configuration generation
  - `model`, `optimizer`, `scheduler`: Training component configs
  - `loss`, `transform`: Domain-specific component configs
- **accelerator.cli.configure**: Project setup utilities
  - `configure`: Copy default configurations to project
  - `init`: Initialize project with minimal configs
  - `status`: Show configuration status and structure
  - `clean`: Remove configurations from project

#### Pipeline Execution
- **run_pipe.py**: Multi-step training pipeline orchestration
  - Hydra configuration composition and resolution
  - MLflow experiment tracking integration
  - Distributed training coordination
  - Step-by-step execution with checkpointing
- **MLProject**: MLflow entry points definition
  - `accelerator`: Main training entry point with distributed support
  - `model_analysis`: Model inspection and analysis
  - `data_analysis`: Dataset validation and statistics

#### Configuration Entry Points
- **configs/**: User-level configuration directory
  - `main.yaml`: Primary configuration composition
  - `model/`: Neural network model configurations
  - `training_components/`: Optimizer, scheduler, loss, transform configs
- **Hydra Configuration System**: Hierarchical configuration composition
  - Default configuration inheritance
  - Override mechanisms and parameter substitution
  - Component indices for dynamic loading

### 🟣 Middle-Level API Layer

#### Runtime System
- **Runtime Context & Component Management**
  - `runtime.context.training`: Training context with component lifecycle and state management
  - `runtime.context.component`: Component management with registry integration and dynamic loading
  - `runtime.context.context`: Context base providing configuration binding and resource allocation
- **Training Engines**
  - `runtime.engine.vanilla_torch`: Standard PyTorch training with single device support
  - `runtime.engine.accelerate`: HuggingFace Accelerate integration for multi-device training
  - `runtime.engine.single_gpu`: GPU-optimized training with memory management
  - `runtime.engine.base`: Common engine interface with lifecycle hooks
- **Training Loops & Orchestration**
  - `runtime.loop.train`: Training loop with forward/backward pass and gradient updates
  - `runtime.loop.validation`: Validation loop for model evaluation and metric computation
  - `runtime.loop.test`: Test loop for final evaluation and result aggregation
  - `runtime.loop.base`: Base loop class with common patterns and hook system
- **Callbacks System**
  - `runtime.callbacks.manager`: Callback orchestration with event dispatching
  - `runtime.callbacks.logger`: Training metrics and loss tracking
  - `runtime.callbacks.progress`: Training progress with ETA estimation
  - `runtime.callbacks.always_on`: System monitoring and health checks
  - `runtime.callbacks.base`: Callback base interface with event hooks

#### Configuration Management
- **Hydra Integration**
  - `utilities.hydra_utils`: Configuration composition and override handling
  - `utilities.hydra_utils.callbacks`: Config validation and parameter logging
  - `utilities.hydra_utils.logger`: Configuration logging and change tracking
  - `utilities.hydra_utils.monitor`: Config changes and runtime validation
- **Component Registry System**
  - `utilities.base_registry`: Base registry with component registration and discovery
  - `runtime.loss.registry`: Loss function registry with dynamic loading
  - `runtime.transform.registry`: Transform registration and pipeline composition
  - `acceleration.registry`: Acceleration techniques and method registration
- **Configuration Resolution**
  - Configuration composition with hierarchical merging and default inheritance
  - Index management with auto-generation and dependency tracking
  - Environment resolution with variable substitution and path resolution

#### 🔬 Experiment Tracking & MLflow Integration
- **MLflow Core Integration**
  - `utilities.patches.mlflow`: Framework integration with custom logging extensions and compatibility fixes
  - MLflow Tracking Server: Experiment management with run lifecycle and remote tracking support
  - MLflow UI: Experiment comparison, metric visualization, and model registry interface
- **Automatic Parameter Logging Flow**
  - Configuration Flattening: `flatten_dict` integration for nested parameter extraction and hierarchical key generation
  - Hydra ↔ MLflow Bridge: Automatic config logging with override tracking and composition logging
  - Parameter Logging: Hyperparameter tracking with nested parameter support and configuration versioning
  - Metric Logging: Training, validation, and custom metrics with real-time streaming
- **Artifact Management Flow**
  - Artifact Management: Model checkpoints, training artifacts, and configuration snapshots with versioning
  - Model Registry: Model versioning with stage management and deployment tracking
  - Artifact Storage: Local and remote storage (S3, etc.) with artifact organization
- **Experiment Organization**
  - Experiment Organization: Run management with experiment grouping and tag-based organization
  - Run Lifecycle Management: Run creation, active tracking, completion, and error handling
  - Experiment Metadata: Run information with environment tracking and Git integration
- **Checkpoint & MLflow Integration**
  - `runtime.checkpoint.checkpoint_handler`: Save/load operations with MLflow artifact logging
  - `runtime.checkpoint.meta_handler`: Metadata management with MLflow metadata logging
  - `runtime.checkpoint.operation`: Checkpoint transformations with MLflow artifact updates

### 🟢 Low-Level API Layer

#### 🔌 Pluggable Domain Architecture
The framework implements a pluggable domain architecture where different domains can be easily added with the same API structure:

#### 🟠 Cross-Domain Components (Reusable)
- **Cross-Domain Loss Functions** (`domain.cross.loss.*`): Generic loss functions like MAE, MSE that work across any domain
- **Cross-Domain Transforms** (`domain.cross.transform.*`): Generic input and loss transforms usable in any domain
- **Cross-Domain Utilities** (`domain.cross.utils.*`): Base implementations and helper functions that work across domains
- **All components are registerable and plug-and-play**

#### 📷 Computer Vision Domain (Currently Implemented)
- **CV Loss Functions** (`domain.cv.loss.*`): Structural, perceptual, color, and custom losses specific to computer vision
- **CV Input Transforms** (`domain.cv.transform.input.*`): Image preprocessing and data augmentation transforms
- **CV Loss Transforms** (`domain.cv.transform.loss.*`): Color space conversions and multi-scale processing for losses
- **CV Utilities** (`domain.cv.utils.*`): Image processing tools and vision-specific helper functions
- **All components are registerable and plug-and-play**

#### 🚀 Future Domain Extensions (Extensible Architecture)
- **NLP Domain** (`domain.nlp.*`): Text loss functions, language transforms, and NLP utilities
- **Audio Domain** (`domain.audio.*`): Audio loss functions, signal transforms, and audio utilities  
- **Signals Domain** (`domain.signals.*`): Signal loss functions, signal transforms, and signal utilities
- **Custom Domains** (`domain.custom.*`): User-defined domains with custom implementations
- **All future domains follow the same registerable and plug-and-play pattern**

#### Acceleration Framework
- **Pruning Techniques**: Neural network pruning methods for model compression
- **Quantization Methods**: Model quantization approaches for efficiency
- **Optimization Approaches**: Model optimization techniques for performance
- **Reparametrization Techniques**: Model reparametrization for structural optimization

#### Utilities & Base Classes
- **Registry Patterns**: Component registration and discovery mechanisms
- **Distributed Training Support**: Multi-node, multi-GPU training utilities
- **Model Utilities**: Model manipulation and analysis tools
- **Hydra Integration**: Configuration system integration helpers

## MLflow Integration Data Flows

The diagram shows comprehensive MLflow integration throughout all architecture layers:

### � Coenfiguration → MLflow Flow
- **Hydra Configuration** → **Hydra-MLflow Bridge** → **Parameter Logging**
- **Configuration Flattening** processes nested configs for MLflow compatibility
- **Automatic parameter logging** captures all hyperparameters and overrides
- **Real-time configuration tracking** during experiment execution

### 🚀 CLI → MLflow Integration
- **CLI Experiment Commands** → **Experiment Organization** → **MLflow UI**
- **Pipeline Execution** → **MLflow Tracking Server** for run management
- **MLProject Entry Points** → **MLflow Tracking** for distributed execution

### ⚙️ Runtime → MLflow Integration
- **Training Context** → **Run Lifecycle Management** for experiment state
- **Training/Validation/Test Loops** → **Metric Logging** for real-time metrics
- **Logger Callbacks** → **Metric Logging** for comprehensive tracking

### 💾 Checkpoint → MLflow Integration
- **Checkpoint Handler** → **Artifact Management** for model storage
- **Meta Handler** → **Experiment Metadata** for version tracking
- **Checkpoint Operations** → **Artifact Storage** for transformation logging

### 🔗 MLflow Internal Flow
- **Parameter/Metric Logging** → **MLflow Tracking Server**
- **Artifact Management** → **Model Registry** and **Artifact Storage**
- **All tracking data** → **MLflow UI** for visualization and comparison

## Color Coding Legend

- **🔵 Blue**: High-Level API components (user-facing interfaces)
- **🟣 Purple**: Middle-Level API components (runtime orchestration)
- **🟢 Green**: Low-Level API components (domain implementations)
- **🟠 Orange**: Cross-domain features (reusable across domains) - **HIGHLIGHTED**
- **🔴 Red**: Acceleration approaches (model optimization)
- **🟤 Brown**: MLflow integration points - **ENHANCED**
- **⚫ Gray**: Configuration components

## Pluggable Domain Architecture

The diagram demonstrates the framework's extensible domain architecture:

### 🔌 Pluggable Design Principles
- **Unified API**: All domains implement the same API structure for losses, transforms, and utilities
- **Registry-Based**: All components are registerable through the framework's registry system
- **Plug-and-Play**: Components can be easily added, removed, or swapped without framework changes
- **Extensible**: New domains can be added following the established patterns

### 🟠 Cross-Domain Components (Orange highlighting)
- **Reusable Across Domains**: Generic implementations that work with any data type
- **Base Functionality**: Fundamental operations like MAE, MSE losses and basic transforms
- **Foundation Layer**: Provides common functionality that domain-specific components can build upon

### 📷 Domain-Specific Components (Green highlighting)
- **Computer Vision**: Currently implemented with comprehensive CV-specific losses, transforms, and utilities
- **Specialized Knowledge**: Incorporates domain expertise (image processing, color theory, perceptual models)
- **Optimized Performance**: Tailored for specific data types and use cases

### 🚀 Future Extensions (Red highlighting)
- **NLP Domain**: Text processing, language models, linguistic transforms
- **Audio Domain**: Signal processing, audio analysis, acoustic transforms  
- **Signals Domain**: Time series, signal analysis, frequency transforms
- **Custom Domains**: User-defined domains for specialized applications

### Key Benefits
- **Modularity**: Each domain is self-contained but follows common patterns
- **Extensibility**: New domains can be added without modifying existing code
- **Consistency**: Unified API across all domains ensures predictable behavior
- **Flexibility**: Mix and match components from different domains as needed
- **Comprehensive Tracking**: MLflow integration provides automatic experiment tracking, parameter logging, and artifact management across all architecture layers
- **Reproducibility**: Complete experiment tracking with configuration versioning, environment capture, and Git integration ensures reproducible research
- **Scalability**: MLflow tracking server supports remote tracking, distributed experiments, and team collaboration