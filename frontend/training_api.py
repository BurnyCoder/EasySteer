from flask import Blueprint, request, jsonify
import os
import sys
import threading
import logging
import json

from core.gpu_utils import normalize_requested_gpu_devices

# Create a blueprint for training-related endpoints
training_bp = Blueprint('training', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Global training status tracking
training_status = {
    'is_training': False,
    'current_epoch': 0,
    'current_step': 0,
    'status_message': '',
    'error_message': '',
    'logs': []
}

from transformers.trainer_callback import TrainerCallback

class TrainingProgressCallback(TrainerCallback):
    """Custom training callback to track progress"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when a log is created during training"""
        global training_status
        import time
        
        if logs:
            # Update training status
            training_status['current_step'] = state.global_step
            training_status['current_epoch'] = state.epoch
            
            # Format the log message
            log_message = f"[{time.strftime('%H:%M:%S')}] "
            
            # Process all possible log fields
            log_parts = []
            
            if 'epoch' in logs:
                log_parts.append(f"Epoch: {logs['epoch']:.2f}")
            
            # Prioritize loss-related information
            if 'loss' in logs:
                log_parts.append(f"Loss: {logs['loss']:.4f}")
            elif 'train_loss' in logs:
                log_parts.append(f"Loss: {logs['train_loss']:.4f}")
                
            if 'grad_norm' in logs:
                log_parts.append(f"Grad: {logs['grad_norm']:.4f}")
                
            if 'learning_rate' in logs:
                log_parts.append(f"LR: {logs['learning_rate']:.2e}")
                
            if 'train_runtime' in logs:
                log_parts.append(f"Runtime: {logs['train_runtime']:.2f}s")
                
            if 'train_samples_per_second' in logs:
                log_parts.append(f"Speed: {logs['train_samples_per_second']:.2f} samples/s")
                
            if 'eval_loss' in logs:
                log_parts.append(f"Eval Loss: {logs['eval_loss']:.4f}")
                
            # Assemble the complete log message
            if log_parts:
                log_message += " | ".join(log_parts)
            else:
                # If no recognized fields, display the raw log
                log_message += str(logs)
                
            # Add to the log list
            if len(training_status['logs']) > 100:  # Keep only the last 100 logs
                training_status['logs'] = training_status['logs'][-50:]
            
            training_status['logs'].append(log_message)
            
            # Update status message
            if 'loss' in logs:
                training_status['status_message'] = f"Training - Step {state.global_step}, Loss: {logs['loss']:.4f}"
            elif 'train_loss' in logs:
                training_status['status_message'] = f"Training - Step {state.global_step}, Loss: {logs['train_loss']:.4f}"

def get_message(key, lang='zh', **kwargs):
    """Get a message in the specified language"""
    error_messages = {
        'zh': {
            'missing_field': '缺少必填字段: {field}',
            'server_error': '服务器错误: {error}',
        },
        'en': {
            'missing_field': 'Missing required field: {field}',
            'server_error': 'Server error: {error}',
        }
    }
    messages = error_messages.get(lang, error_messages['zh'])
    message = messages.get(key, key)
    return message.format(**kwargs)

@training_bp.route('/api/train', methods=['POST'])
def train():
    """Start training"""
    try:
        data = request.json
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        
        # Validate required fields
        required_fields = ['model_path', 'output_dir', 'training_examples']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': get_message('missing_field', lang, field=field)}), 400
        
        # Validate training data format
        try:
            training_examples = data['training_examples']
            if not isinstance(training_examples, list) or len(training_examples) == 0:
                return jsonify({'error': 'Training data must be a non-empty array'}), 400
            
            for i, example in enumerate(training_examples):
                if not isinstance(example, list) or len(example) != 2:
                    return jsonify({'error': f'Training example {i} has incorrect format. Must be an array of two elements [input, output]'}), 400
        except Exception as e:
            return jsonify({'error': f'Incorrect training data format: {str(e)}'}), 400
        
        normalized_gpu_devices = normalize_requested_gpu_devices(
            data.get('gpu_devices', '0')
        )
        data['gpu_devices'] = normalized_gpu_devices
        
        # Start training (using asynchronous method)
        def train_model():
            global training_status
            try:
                import torch
                import transformers
                import time

                os.environ["CUDA_VISIBLE_DEVICES"] = normalized_gpu_devices
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    raise RuntimeError(
                        "No CUDA GPUs are visible after applying "
                        f"CUDA_VISIBLE_DEVICES={normalized_gpu_devices}."
                    )
                
                # Initialize training status
                training_status.update({
                    'is_training': True,
                    'current_epoch': 0,
                    'current_step': 0,
                    'status_message': f'Initializing training on GPU(s): {normalized_gpu_devices}...',
                    'error_message': '',
                    'logs': []
                })
                
                # Temporarily add project root to Python path to import local reft module
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                original_path = sys.path.copy()
                
                try:
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    # Import pyreft from local reft module
                    from easysteer.reft import pyreft
                finally:
                    # Restore original sys.path to avoid conflicts with pip-installed packages
                    sys.path[:] = original_path
                
                device = "cuda"
                
                logger.info(f"Starting to load model: {data['model_path']}")
                training_status['status_message'] = f"Loading model: {data['model_path']}"
                
                # Load model
                model_name_or_path = data['model_path']
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device
                )
                
                # Get tokenizer
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name_or_path, model_max_length=2048, padding_side="right", use_fast=False
                )
                tokenizer.pad_token = tokenizer.eos_token
                
                logger.info("Model loaded, setting up ReFT config")
                training_status['status_message'] = "Model loaded, setting up ReFT config"
                
                # Set ReFT config
                reft_config_data = data.get('reft_config', {})
                reft_config = pyreft.ReftConfig(
                    representations={
                        "layer": reft_config_data.get('layer', 8),
                        "component": reft_config_data.get('component', 'block_output'),
                        "low_rank_dimension": reft_config_data.get('low_rank_dimension', 4),
                        "intervention": pyreft.LoreftIntervention(
                            embed_dim=model.config.hidden_size, 
                            low_rank_dimension=reft_config_data.get('low_rank_dimension', 4)
                        ),
                    }
                )
                reft_model = pyreft.get_reft_model(model, reft_config)
                reft_model.set_device(device)
                reft_model.print_trainable_parameters()
                
                logger.info("ReFT model configured, preparing training data")
                training_status['status_message'] = "ReFT model configured, preparing training data"
                
                # Prepare training data
                prompt_no_input_template = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
                
                data_module = pyreft.make_last_position_supervised_data_module(
                    tokenizer,
                    model,
                    [prompt_no_input_template % e[0] for e in training_examples],
                    [e[1] for e in training_examples],
                )
                
                # Training arguments
                training_args_data = data.get('training_args', {})
                training_args = transformers.TrainingArguments(
                    num_train_epochs=training_args_data.get('num_train_epochs', 100.0),
                    output_dir=data['output_dir'],
                    per_device_train_batch_size=training_args_data.get('per_device_train_batch_size', 10),
                    learning_rate=training_args_data.get('learning_rate', 4e-3),
                    logging_steps=training_args_data.get('logging_steps', 40),
                    report_to=[],
                    save_strategy="no",
                    # save_steps=training_args_data.get('save_steps', 500),  # Save every 500 steps
                    # save_total_limit=3,  # Keep at most 3 checkpoints
                    # evaluation_strategy="no",  # No evaluation
                    # dataloader_drop_last=False,
                    # remove_unused_columns=False,
                )
                
                logger.info(f"Training arguments configured, starting training - Epochs: {training_args.num_train_epochs}, Batch Size: {training_args.per_device_train_batch_size}")
                training_status['status_message'] = f"Preparing to start training - Total {int(training_args.num_train_epochs)} epochs"
                
                # Create training callback
                progress_callback = TrainingProgressCallback()
                
                # Create trainer
                trainer = pyreft.ReftTrainerForCausalLM(
                    model=reft_model, tokenizer=tokenizer, args=training_args, **data_module
                )
                
                # Add callback
                trainer.add_callback(progress_callback)
                
                # Start training
                logger.info("Starting training...")
                _ = trainer.train()
                
                # Save model
                logger.info("Training complete, saving model...")
                training_status['status_message'] = "Training complete, saving model..."
                reft_model.set_device("cpu")
                reft_model.save(
                    save_directory=data['output_dir'],
                    save_to_hf_hub=False,
                )
                
                # Training finished
                training_status.update({
                    'is_training': False,
                    'status_message': f"Training complete! Model saved to: {data['output_dir']}"
                })
                
                logger.info(f"Training complete, model saved to: {data['output_dir']}")
                
            except Exception as e:
                # Training failed
                training_status.update({
                    'is_training': False,
                    'error_message': str(e),
                    'status_message': f"Training failed: {str(e)}"
                })
                logger.error(f"Training failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Start training in a background thread
        train_thread = threading.Thread(target=train_model)
        train_thread.daemon = True
        train_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training has started',
            'output_dir': data['output_dir'],
            'gpu_devices': normalized_gpu_devices,
            'training_examples_count': len(training_examples),
            'reft_config': data.get('reft_config', {}),
            'training_args': data.get('training_args', {}),
            'note': 'Training is running in the background. Check server logs for progress.'
        }), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        lang = request.headers.get('Accept-Language', 'zh').split(',')[0].split('-')[0]
        return jsonify({'error': get_message('server_error', lang, error=str(e))}), 500

@training_bp.route('/api/train-configs', methods=['GET'])
def list_train_configs():
    """List all available training configuration files"""
    try:
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs', 'training')
        if not os.path.exists(configs_dir):
            return jsonify({"configs": []})
        
        # Define friendly names for training config files
        config_display_names = {
            'emoji_loreft': 'Emoji LoReft Training Configuration',
            'emoji_bias': 'Emoji Bias Training Configuration'
        }
        
        config_files = []
        for filename in os.listdir(configs_dir):
            if filename.endswith('.json'):
                config_name = filename[:-5]  # Remove .json extension
                display_name = config_display_names.get(config_name, config_name.replace('_', ' ').title())
                config_files.append({
                    "name": config_name,
                    "display_name": display_name
                })
        
        return jsonify({"configs": config_files})
    
    except Exception as e:
        logger.error(f"Failed to list training configs: {e}")
        return jsonify({"error": f"Failed to list training configs: {str(e)}"}), 500

@training_bp.route('/api/train-config/<config_name>', methods=['GET'])
def get_train_config(config_name):
    """Get a training configuration file"""
    try:
        # Validate config name
        allowed_configs = ['emoji_loreft', 'emoji_bias']
        if config_name not in allowed_configs:
            return jsonify({"error": f"Training config {config_name} not found"}), 404
        
        # Read config file
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'training', f'{config_name}.json')
        if not os.path.exists(config_path):
            return jsonify({"error": f"Training config file {config_path} not found"}), 404
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return jsonify(config)
    
    except Exception as e:
        logger.error(f"Failed to get training config: {e}")
        return jsonify({"error": f"Failed to get training config: {str(e)}"}), 500

@training_bp.route('/api/train-status', methods=['GET'])
def get_train_status():
    """Get training status"""
    global training_status
    return jsonify(training_status), 200

@training_bp.route('/api/train-restart', methods=['POST'])
def restart_training_backend():
    """Fully restart the training backend process"""
    try:
        import sys
        import threading
        import time
        
        global training_status
        training_status.update({
            'is_training': False,
            'status_message': 'Preparing to fully restart the backend process...',
            'logs': []
        })
        
        def delayed_restart():
            """Delayed restart to allow response to be sent"""
            time.sleep(1)  # Wait 1 second for the response to be sent
            logger.info("Restarting backend process...")
            
            # Get current Python executable and script arguments
            python_executable = sys.executable
            script_args = sys.argv
            
            # Use os.execv to restart the process
            import os
            os.execv(python_executable, [python_executable] + script_args)
        
        # Execute restart in a new thread to avoid blocking the response
        restart_thread = threading.Thread(target=delayed_restart)
        restart_thread.daemon = True
        restart_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Backend is restarting, please try again in a few seconds..."
        })
    
    except Exception as e:
        logger.error(f"Failed to restart backend: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Failed to restart backend: {str(e)}"
        }), 500 
