from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(model_path: str, device_map: str):
    """
    Load a sequence-to-sequence model from HuggingFace.
    
    Args:
        model_path (str): Path or name of the model on HuggingFace Hub
        device_map (str): Device configuration for model loading ('auto', 'cpu', 'cuda', etc.)
        
    Returns:
        AutoModelForSeq2SeqLM: Loaded seq2seq model in evaluation mode
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map=device_map
    )
    model.eval()
    
    return model


def load_tokenizer(model_path: str, add_bos_token: bool = True):
    """
    Load a tokenizer suitable for sequence-to-sequence tasks.
    
    Args:
        model_path (str): Path or name of the model on HuggingFace Hub
        add_bos_token (bool): Whether to add beginning-of-sequence token
        
    Returns:
        AutoTokenizer: Configured tokenizer for seq2seq tasks
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",  # Left padding is often preferred for seq2seq tasks
        add_bos_token=add_bos_token,
    )
    
    # Handle special tokens for seq2seq models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure we have start and end tokens for both encoder and decoder
    special_tokens = {
        'additional_special_tokens': []
    }
    
    if tokenizer.bos_token is None:
        special_tokens['bos_token'] = '<s>'
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = '</s>'
    if tokenizer.pad_token is None:
        special_tokens['pad_token'] = '<pad>'
    if tokenizer.sep_token is None:
        special_tokens['sep_token'] = '</s>'
    
    # Only add special tokens if we needed to define any
    if any(special_tokens.values()):
        tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer