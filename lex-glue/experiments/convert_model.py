import copy
import json
import os.path
import warnings
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
warnings.filterwarnings("ignore")


def convert_roberta_to_lf(pretrained_model_name_or_path, output_dir, max_text_length=4096, use_auth_token=None):
    # load pre-trained bert model and tokenizer
    roberta_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token,
                                              model_max_length=max_text_length)

    # load dummy config and change specifications
    roberta_config = roberta_model.config
    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # Text length parameters
    lf_config.max_position_embeddings = max_text_length + 2
    lf_config.model_max_length = max_text_length
    lf_config.num_hidden_layers = roberta_config.num_hidden_layers
    # Transformer parameters
    lf_config.hidden_size = roberta_config.hidden_size
    lf_config.intermediate_size = roberta_config.intermediate_size
    lf_config.num_attention_heads = roberta_config.num_attention_heads
    lf_config.hidden_act = roberta_config.hidden_act
    lf_config.attention_window = [512] * roberta_config.num_hidden_layers
    # Vocabulary parameters
    lf_config.vocab_size = roberta_config.vocab_size
    lf_config.pad_token_id = roberta_config.pad_token_id
    lf_config.bos_token_id = roberta_config.bos_token_id
    lf_config.eos_token_id = roberta_config.eos_token_id
    lf_config.cls_token_id = tokenizer.cls_token_id
    lf_config.sep_token_id = tokenizer.sep_token_id
    lf_config.type_vocab_size = roberta_config.type_vocab_size

    # load dummy hi-transformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    k = 2
    step = roberta_config.max_position_embeddings - 2
    lf_model.longformer.embeddings.position_embeddings.weight.data[:2] = roberta_model.roberta.embeddings.position_embeddings.weight[:2]
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:] = roberta_model.roberta.embeddings.position_embeddings.weight[2:(roberta_config.max_position_embeddings + 2)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:(k + step)] = roberta_model.roberta.embeddings.position_embeddings.weight[2:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(roberta_model.roberta.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(roberta_model.roberta.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(roberta_model.roberta.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    roberta_model.roberta.encoder.layer = roberta_model.roberta.encoder.layer[:roberta_config.num_hidden_layers]
    for i in range(len(roberta_model.roberta.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].output.LayerNorm)
        # attention output
        lf_model.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.dense)
        lf_model.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.output.LayerNorm)
        # local q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)
        # global q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            roberta_model.roberta.encoder.layer[i].attention.self.value)

    # copy lm_head
    lf_model.lm_head.load_state_dict(roberta_model.lm_head.state_dict())

    # save model
    lf_model.save_pretrained(output_dir)

    # save tokenizer
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'tokenizer_config.json')) as in_file:
        config = json.load(in_file)
        config['model_max_length'] = max_text_length
    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as out_file:
        json.dump(config, out_file)


def convert_bert_to_lf(pretrained_model_name_or_path, output_dir, max_text_length=4096, use_auth_token=None):
    # load pre-trained bert model and tokenizer
    bert_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_auth_token=use_auth_token,
                                              model_max_length=max_text_length)

    # load dummy config and change specifications
    bert_config = bert_model.config
    lf_config = AutoConfig.from_pretrained('allenai/longformer-base-4096')
    # Text length parameters
    lf_config.max_position_embeddings = max_text_length + 2
    lf_config.model_max_length = max_text_length
    lf_config.num_hidden_layers = bert_config.num_hidden_layers
    # Transformer parameters
    lf_config.hidden_size = bert_config.hidden_size
    lf_config.intermediate_size = bert_config.intermediate_size
    lf_config.num_attention_heads = bert_config.num_attention_heads
    lf_config.hidden_act = bert_config.hidden_act
    lf_config.attention_window = [512] * bert_config.num_hidden_layers
    # Vocabulary parameters
    lf_config.vocab_size = bert_config.vocab_size
    lf_config.pad_token_id = bert_config.pad_token_id
    lf_config.bos_token_id = bert_config.bos_token_id
    lf_config.eos_token_id = bert_config.eos_token_id
    lf_config.cls_token_id = tokenizer.cls_token_id
    lf_config.sep_token_id = tokenizer.sep_token_id
    lf_config.type_vocab_size = bert_config.type_vocab_size

    # load dummy hi-transformer model
    lf_model = AutoModelForMaskedLM.from_config(lf_config)

    # copy embeddings
    k = 2
    step = bert_config.max_position_embeddings - 2
    lf_model.longformer.embeddings.position_embeddings.weight.data[:2] = bert_model.bert.embeddings.position_embeddings.weight[:2]
    while k < lf_config.max_position_embeddings - 1:
        if k + step >= lf_config.max_position_embeddings:
            max_pos = lf_config.max_position_embeddings - k
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:] = bert_model.bert.embeddings.position_embeddings.weight[2:(max_pos + 2)]
        else:
            lf_model.longformer.embeddings.position_embeddings.weight.data[k:(k + step)] = bert_model.bert.embeddings.position_embeddings.weight[2:]
        k += step
    lf_model.longformer.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
    lf_model.longformer.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
    lf_model.longformer.embeddings.LayerNorm.load_state_dict(bert_model.bert.embeddings.LayerNorm.state_dict())

    # copy transformer layers
    bert_model.bert.encoder.layer = bert_model.bert.encoder.layer[:bert_config.num_hidden_layers]
    for i in range(len(bert_model.bert.encoder.layer)):
        # generic
        lf_model.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].intermediate.dense)
        lf_model.longformer.encoder.layer[i].output.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.dense)
        lf_model.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(
            bert_model.bert.encoder.layer[i].output.LayerNorm)
        # attention output
        lf_model.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.output.dense)
        lf_model.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.output.LayerNorm)
        # local q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)
        # global q,k,v
        lf_model.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.query)
        lf_model.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.key)
        lf_model.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(
            bert_model.bert.encoder.layer[i].attention.self.value)

    # save model
    lf_model.save_pretrained(output_dir)

    # save tokenizer
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, 'tokenizer_config.json')) as in_file:
        config = json.load(in_file)
        config['model_max_length'] = max_text_length
    with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as out_file:
        json.dump(config, out_file)


