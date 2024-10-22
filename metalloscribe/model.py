import torch
import torch.nn as nn
import math
from transformers import SwinForImageClassification, SwinConfig, BartForConditionalGeneration, BartConfig
from metalloscribe.tokenizer import NodeTokenizer
from .tokenizer import SOS_ID, EOS_ID, PAD_ID, MASK_ID
from transformers.modeling_outputs import BaseModelOutput

class Encoder(nn.Module):
    def __init__(self, pretrained_model):
        super(Encoder, self).__init__()
        self.model = pretrained_model

    def forward(self, x):
        # Get the outputs from the model
        outputs = self.model(x)
        
        # Access hidden states if available
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        
        # If hidden states are available, use the last one
        if hidden_states is not None:
            #w("Using Last Hidden State")
            features = hidden_states[-1]
            #print(f"Encoder Features: {features}")
        else:
            # Otherwise, fallback to using logits
            #print("Using Logits")
            features = outputs.logits
        
        return features

class BARTDecoder(nn.Module):
    def __init__(self, num_layers, num_heads, hidden_dim, vocab_size, encoder_dim, max_len):
        super(BARTDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.max_len = max_len
        self.eos_token_id = EOS_ID

        # BART Configuration
        bart_config = BartConfig(
            d_model=hidden_dim,
            encoder_attention_heads=num_heads,
            decoder_attention_heads=num_heads,
            encoder_ffn_dim=hidden_dim * 4,
            decoder_ffn_dim=hidden_dim * 4,
            encoder_layers=num_layers,
            decoder_layers=num_layers,
            max_position_embeddings=max_len,
            vocab_size=vocab_size + 300
        )

        # BART model for Conditional Generation
        self.bart = BartForConditionalGeneration(bart_config)
        self.enc_pos_emb = nn.Embedding(144, encoder_dim)
        self.enc_trans_layer = nn.Linear(encoder_dim, hidden_dim)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        pos_emb = self.enc_pos_emb(torch.arange(144).to(encoder_out.device)).unsqueeze(0)
        encoder_out = encoder_out + pos_emb
        encoder_out = self.enc_trans_layer(encoder_out)
        return encoder_out

    def forward(self, encoder_out, labels=None):
        encoder_out = self.enc_transform(encoder_out)
        #encoder_out = encoder_out.permute(1, 0, 2)

        #print(f"Encoder_out after transformation: {encoder_out.shape}")
        #print(f"Labels: {labels.shape}")
        #print(f"Attention Mask: {(labels != PAD_ID).long().shape}")

        if labels is not None:
            decoder_inputs = labels
            attention_mask = (labels != PAD_ID).long()
            outputs = self.bart(
                input_ids=decoder_inputs, 
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out), 
                #attention_mask=attention_mask
            )
            return outputs.logits
        else:
            outputs = self.bart.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out), 
                max_length=self.max_len, 
                eos_token_id=self.eos_token_id
            )
            return outputs

    def inference(self, encoder_out):
        encoder_out = self.enc_transform(encoder_out)
        #encoder_out = encoder_out.permute(1, 0, 2)

        generated_sequence = self.bart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_out),
            max_length=self.max_len,
            num_beams=1,
            early_stopping=True,
            decoder_start_token_id=SOS_ID,
            eos_token_id=self.eos_token_id
        )

        return generated_sequence

# Load the Swin Transformer model and feature extractor
model_name = "microsoft/swin-base-patch4-window12-384-in22k"
config = SwinConfig.from_pretrained(model_name)
config.output_hidden_states = True
swin_model = SwinForImageClassification.from_pretrained(model_name, config=config)

# Instantiate Encoder
encoder = Encoder(swin_model)

# Load NodeTokenizer
tokenizer = NodeTokenizer(path='../vocab/vocab_periodic_table.json') # Not sure if there's a better way to get the vocab size

# Instantiate Decoder Parameters
vocab_size = len(tokenizer)
hidden_dim = 256
num_layers = 6
num_heads = 8
max_len = 1024 # So <= 250 atoms, covers 95% of organometallic complexes

decoder = BARTDecoder(num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, 
                      vocab_size=vocab_size, encoder_dim=config.hidden_size, max_len=max_len)

class EncoderDecoderModel(nn.Module):
    def __init__(self):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = decoder.vocab_size

    def forward(self, images, labels=None):
        encoder_out = self.encoder(images)
        #print(f"Encoder_out after encoder: {encoder_out.shape}")
        logits = self.decoder(encoder_out, labels)
        return logits
    
    def inference(self, images, labels=None):
        encoder_out = self.encoder(images)
        #print(f"Encoder_out after encoder: {encoder_out.shape}")
        generated_sequence = self.decoder.inference(encoder_out)
        return generated_sequence
