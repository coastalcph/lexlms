import torch
import copy
import numpy as np


def mod_roberta(teacher_model, student_model, student_teacher_mapping_ids):
    # copy positional and token type embeddings
    student_model.roberta.embeddings.position_embeddings. \
        load_state_dict(teacher_model.roberta.embeddings.position_embeddings.state_dict())
    student_model.roberta.embeddings.token_type_embeddings.weight[0] = \
        teacher_model.roberta.embeddings.token_type_embeddings.weight.detach()[0]
    student_model.roberta.embeddings.LayerNorm. \
        load_state_dict(teacher_model.roberta.embeddings.LayerNorm.state_dict())

    # Extract teacher word embeddings
    word_embeddings_matrix = copy.deepcopy(teacher_model.roberta.embeddings.word_embeddings.weight.detach())
    word_embeddings = [word_embeddings_matrix[teacher_id] if isinstance(teacher_id, int)
                       else word_embeddings_matrix[teacher_id].mean(dim=0)
                       for student_id, teacher_id in student_teacher_mapping_ids.items()]
    word_embeddings = torch.stack(word_embeddings)

    # replace student's word embeddings matrix
    student_model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(word_embeddings)

    # Copy transformer block
    student_model.roberta.encoder.load_state_dict(teacher_model.roberta.encoder.state_dict())

    # copy embeddings to lm_head
    student_model.lm_head.decoder.weight = torch.nn.Parameter(word_embeddings)
    student_model.lm_head.layer_norm.load_state_dict(teacher_model.lm_head.layer_norm.state_dict())
    student_model.lm_head.dense.load_state_dict(teacher_model.lm_head.dense.state_dict())

    # Extract teacher word embeddings, use the centroid for compound tokens
    lm_head_biases = copy.deepcopy(teacher_model.lm_head.bias.detach())
    lm_head_biases = [lm_head_biases[teacher_id] if isinstance(teacher_id, int)
                      else lm_head_biases[teacher_id].mean(dim=0)
                      for student_id, teacher_id in student_teacher_mapping_ids.items()]
    lm_head_biases = torch.as_tensor(np.array(lm_head_biases))

    # replace student's word embeddings matrix
    student_model.lm_head.decoder.bias = torch.nn.Parameter(lm_head_biases)

    return student_model


def mod_bert(teacher_model, student_model, student_teacher_mapping_ids):
    # copy positional and token type embeddings
    student_model.roberta.embeddings.position_embeddings.weight[2:] =\
        teacher_model.bert.embeddings.position_embeddings.weight.detach()
    student_model.roberta.embeddings.token_type_embeddings.weight =\
        teacher_model.bert.embeddings.token_type_embeddings.weight.detach()
    student_model.roberta.embeddings.LayerNorm. \
        load_state_dict(teacher_model.bert.embeddings.LayerNorm.state_dict())

    # Extract teacher word embeddings
    word_embeddings_matrix = copy.deepcopy(teacher_model.bert.embeddings.word_embeddings.weight.detach())
    word_embeddings = [word_embeddings_matrix[teacher_id] if isinstance(teacher_id, int)
                       else word_embeddings_matrix[teacher_id].mean(dim=0)
                       for student_id, teacher_id in student_teacher_mapping_ids.items()]
    word_embeddings = torch.stack(word_embeddings)

    # replace student's word embeddings matrix
    student_model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(word_embeddings)

    # Copy transformer block
    student_model.roberta.encoder.load_state_dict(teacher_model.bert.encoder.state_dict())

    # copy embeddings to lm_head
    student_model.lm_head.decoder.weight = torch.nn.Parameter(word_embeddings)
    student_model.lm_head.layer_norm.load_state_dict(teacher_model.cls.predictions.transform.LayerNorm.state_dict())
    student_model.lm_head.dense.load_state_dict(teacher_model.cls.predictions.transform.dense.state_dict())

    # Extract teacher word embeddings, use the centroid for compound tokens
    lm_head_biases = copy.deepcopy(teacher_model.cls.predictions.bias.detach())
    lm_head_biases = [lm_head_biases[teacher_id] if isinstance(teacher_id, int)
                      else lm_head_biases[teacher_id].mean(dim=0)
                      for student_id, teacher_id in student_teacher_mapping_ids.items()]
    lm_head_biases = torch.as_tensor(np.array(lm_head_biases))

    # replace student's word embeddings matrix
    student_model.lm_head.decoder.bias = torch.nn.Parameter(lm_head_biases)

    return student_model


def mod_gpt2(teacher_model, student_model, student_teacher_mapping_ids):
    # copy positional and token type embeddings
    student_model.roberta.embeddings.position_embeddings. \
        load_state_dict(teacher_model.roberta.embeddings.position_embeddings.state_dict())
    student_model.roberta.embeddings.token_type_embeddings. \
        load_state_dict(teacher_model.roberta.embeddings.token_type_embeddings.state_dict())
    student_model.roberta.embeddings.LayerNorm. \
        load_state_dict(teacher_model.roberta.embeddings.LayerNorm.state_dict())

    # Extract teacher word embeddings
    word_embeddings_matrix = copy.deepcopy(teacher_model.roberta.embeddings.word_embeddings.weight.detach())
    word_embeddings = [word_embeddings_matrix[teacher_id] if isinstance(teacher_id, int)
                       else word_embeddings_matrix[teacher_id].mean(dim=0)
                       for student_id, teacher_id in student_teacher_mapping_ids.items()]
    word_embeddings = torch.stack(word_embeddings)

    # replace student's word embeddings matrix
    student_model.roberta.embeddings.word_embeddings.weight = torch.nn.Parameter(word_embeddings)

    # Copy transformer block
    student_model.roberta.encoder.load_state_dict(teacher_model.roberta.encoder.state_dict())

    # copy embeddings to lm_head
    student_model.lm_head.decoder.weight = torch.nn.Parameter(word_embeddings)
    student_model.lm_head.layer_norm.load_state_dict(teacher_model.lm_head.layer_norm.state_dict())
    student_model.lm_head.dense.load_state_dict(teacher_model.lm_head.dense.state_dict())

    # Extract teacher word embeddings, use the centroid for compound tokens
    lm_head_biases = copy.deepcopy(teacher_model.lm_head.bias.detach())
    lm_head_biases = [lm_head_biases[teacher_id] if isinstance(teacher_id, int)
                      else lm_head_biases[teacher_id].mean(dim=0)
                      for student_id, teacher_id in student_teacher_mapping_ids.items()]
    lm_head_biases = torch.as_tensor(np.array(lm_head_biases))

    # replace student's word embeddings matrix
    student_model.lm_head.decoder.bias = torch.nn.Parameter(lm_head_biases)

    return student_model