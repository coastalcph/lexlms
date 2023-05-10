import os
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import unidecode
import copy
import argparse

from data import DATA_DIR, AUTH_KEY
from flota_tokenizer import FlotaTokenizer
from mod_helpers import mod_roberta, mod_bert

SPECIAL_TOKENS_MAPPING_ROBERTA = ['<mask>', '<s>', '</s>', '<unk>', '<pad>']
SPECIAL_TOKENS_MAPPING_BERT = ['[MASK]', '[CLS]', '[SEP]', '[UNK]', '[PAD]']


def warm_start_model():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--teacher_model_path', default='roberta-base')
    parser.add_argument('--teacher_start', default='Ġ')
    parser.add_argument('--student_model_path', default=os.path.join(DATA_DIR, 'PLMs', 'lex-lm-base-uncased'))
    parser.add_argument('--student_start', default='Ġ')
    parser.add_argument('--use_flota', default=True)
    parser.add_argument('--flota_mode', default='longest', choices=['flota', 'longest', 'first'])
    parser.add_argument('--auth_token', default=AUTH_KEY)
    parser.add_argument('--output_dir', default=os.path.join(DATA_DIR, 'PLMs', 'lex-lm-base-uncased-v2'))
    config = parser.parse_args()

    # load tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_path,
                                                      use_auth_token=config.auth_token)
    student_tokenizer = AutoTokenizer.from_pretrained(config.student_model_path,
                                                      use_auth_token=config.auth_token)
    if config.use_flota:
        # use FLOTA tokenizer of Hofmann et al. (2022) to decrease over-fragmentation
        teacher_flota_tokenizer = FlotaTokenizer(config.teacher_model_path,
                                                 mode=config.flota_mode,
                                                 k=1 if config.flota_mode in ['longest', 'first'] else 4)

    # define student-teacher mappings
    student_teacher_mapping_ids = {}
    student_teacher_mapping_tokens = {}
    student_teacher_mapping_compound_tokens = {}

    # Build teacher vocab dict
    teacher_vocab = [(token_id, token) for token, token_id in teacher_tokenizer.vocab.items()]
    teacher_vocab.sort()
    teacher_vocab = {token: token_id for (token_id, token) in teacher_vocab}
    teacher_vocab_lowercased = {token.lower(): token_id for (token, token_id) in teacher_vocab.items()}
    teacher_vocab_ids = {token_id: token for token, token_id in teacher_vocab.items()}
    TEACHER_START = config.teacher_start
    TEACHER_SPECIAL_TOKENS = SPECIAL_TOKENS_MAPPING_BERT if 'bert-' in config.teacher_model_path \
        else SPECIAL_TOKENS_MAPPING_ROBERTA

    # Build student vocab dict
    student_vocab = [(token_id, token) for token, token_id in student_tokenizer.vocab.items()]
    student_vocab.sort()
    student_vocab = {token: token_id for (token_id, token) in student_vocab}
    STUDENT_START =config.student_start
    STUDENT_SPECIAL_TOKENS = SPECIAL_TOKENS_MAPPING_BERT if 'bert-' in config.student_model_path \
        else SPECIAL_TOKENS_MAPPING_ROBERTA

    # statistics counter
    identical_tokens = 0
    semi_identical_tokens = 0
    semi_identical_normalized_tokens = 0
    longest_first_tokens = 0
    compound_tokens = 0
    unk_tokens = 0
    flota_diffs = []

    # look up for student tokens in teacher's vocabulary
    for original_token, token_id in student_vocab.items():
        if token_id in student_tokenizer.all_special_ids:
            token = TEACHER_SPECIAL_TOKENS[STUDENT_SPECIAL_TOKENS.index(original_token)]
            student_teacher_mapping_ids[token_id] = teacher_vocab[token]
            student_teacher_mapping_tokens[original_token] = token
            continue
        if 'bert-' in config.teacher_model_path and re.match('[a-z]{2,}', original_token):
            token = '##' + copy.deepcopy(original_token)
        else:
            token = copy.deepcopy(original_token)
        token = token.replace(STUDENT_START, TEACHER_START)
        # perfect match (e.g., "document" --> "document")
        if token in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[token]
            student_teacher_mapping_tokens[original_token] = token
            identical_tokens += 1
        # perfect match with cased version of token (e.g., "paris" --> "Paris")
        elif token.lower() in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[token.lower()]
            student_teacher_mapping_tokens[original_token] = teacher_vocab_ids[teacher_vocab_lowercased[token.lower()]]
            semi_identical_tokens += 1
        # match with non-starting token (e.g., "concat" --> "_concat")
        elif token.replace(TEACHER_START, '') in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[token.replace(TEACHER_START, '')]
            student_teacher_mapping_tokens[original_token] = token.replace(TEACHER_START, '')
            semi_identical_tokens += 1
        # match with cased version of non-starting token (e.g., "ema" --> "_EMA")
        elif token.lower().replace(TEACHER_START, '') in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[token.lower().replace(TEACHER_START, '')]
            student_teacher_mapping_tokens[original_token] = teacher_vocab_ids[teacher_vocab_lowercased[token.lower().replace(TEACHER_START, '')]]
            semi_identical_normalized_tokens += 1
        # normalized version of token in vocab -> map to the normalized version of token
        # (e.g., 'garçon' --> 'garcon')
        elif unidecode.unidecode(token) in teacher_vocab:
            student_teacher_mapping_ids[token_id] = teacher_vocab[unidecode.unidecode(token)]
            student_teacher_mapping_tokens[original_token] = unidecode.unidecode(unidecode.unidecode(token))
            semi_identical_normalized_tokens += 1
        # normalized version of uncased token in vocab -> map to the normalized version of token
        # (e.g., 'Garçon' --> 'garçon' --> 'garcon')
        elif unidecode.unidecode(token).lower() in teacher_vocab_lowercased:
            student_teacher_mapping_ids[token_id] = teacher_vocab_lowercased[unidecode.unidecode(token).lower()]
            student_teacher_mapping_tokens[original_token] = unidecode.unidecode(unidecode.unidecode(token).lower())
            semi_identical_normalized_tokens += 1
        else:
            # tokenize token (e.g., "unprecedented" --> ['_un', 'prec', 'edent', 'ed'])
            if 'bert-' in config.teacher_model_path:
                token = token.replace(STUDENT_START, ' ').replace('##', '')
            else:
                token = token.replace(STUDENT_START, ' ').replace(TEACHER_START, ' ')
            sub_words = teacher_tokenizer.encode(' ' + token, add_special_tokens=False)
            if set(sub_words) == teacher_tokenizer.unk_token_id:
                sub_words = teacher_tokenizer.encode(' ' + unidecode.unidecode(token), add_special_tokens=False)
            sub_words_tokens = [teacher_vocab_ids[sub_word] for sub_word in sub_words]
            if config.use_flota:
                # tokenize token with FLOTA (e.g., "unprecedented" --> ['_un', 'precedented'])
                flota_sub_words = teacher_flota_tokenizer.encode(token)
                flota_sub_words = [sub_word for sub_word in flota_sub_words 
                                   if sub_word not in teacher_tokenizer.all_special_ids]
                flota_sub_words_tokens = [teacher_vocab_ids[sub_word] for sub_word in flota_sub_words
                                          if sub_word not in teacher_tokenizer.all_special_ids]
            # keep the list with the fewer sub-words
            if config.use_flota and len(flota_sub_words_tokens) and len(flota_sub_words_tokens) <= len(sub_words_tokens):
                flota_diffs.append((sub_words_tokens, flota_sub_words_tokens))
                sub_words = flota_sub_words
                sub_words_tokens = flota_sub_words_tokens
            # sub-word token -> map to the sub-word (e.g., "_μ" --> ["μ"] --> "μ")
            if len(sub_words) == 1 and sub_words[0] != teacher_tokenizer.unk_token_id:
                student_teacher_mapping_ids[token_id] = sub_words[0]
                student_teacher_mapping_tokens[original_token] = sub_words_tokens[0]
                if sub_words_tokens[0].replace(STUDENT_START, '').replace(TEACHER_START, '') == \
                        original_token.replace(STUDENT_START, '').replace(TEACHER_START, ''):
                    semi_identical_tokens += 1
                else:
                    longest_first_tokens += 1
            # list of sub-words w/o <unk> -> map to the list (e.g., 'overqualified' --> ['over', '_qualified'] )
            elif len(sub_words) >= 2 and teacher_tokenizer.unk_token_id not in sub_words:
                student_teacher_mapping_ids[token_id] = sub_words
                student_teacher_mapping_tokens[original_token] = sub_words_tokens
                student_teacher_mapping_compound_tokens[token] = sub_words_tokens
                compound_tokens += 1
            else:
                # list of sub-words w/ <unk> -> map to the list (e.g., 'Ω-power' --> [<unk>, '-power'] --> '-power')
                if len(sub_words) > 1 and set(sub_words) != {teacher_tokenizer.unk_token_id}:
                    student_teacher_mapping_ids[token_id] = sub_words.remove(teacher_tokenizer.unk_token_id)
                    student_teacher_mapping_tokens[original_token] = sub_words_tokens.remove(teacher_tokenizer.unk_token)
                    student_teacher_mapping_compound_tokens[token] = sub_words_tokens
                    compound_tokens += 1
                # <unk> -> map to <unk>
                else:
                    # No hope use <unk> (e.g., '晚上好' --> <unk>)
                    student_teacher_mapping_ids[token_id] = teacher_tokenizer.unk_token_id
                    student_teacher_mapping_tokens[original_token] = teacher_tokenizer.unk_token
                    print(f'Token "{token}" not in vocabulary, replaced with UNK.')
                    unk_tokens += 1

    # Fix issue with None tokens
    for key, value in student_teacher_mapping_ids.items():
        if value is None:
            student_teacher_mapping_ids[key] = teacher_tokenizer.unk_token_id

    # print mapping statistics
    print(f'The student-teacher mapping algorithm led to:')
    print(f'- ({str(identical_tokens):>5}) ({identical_tokens/len(student_vocab)*100:.1f}%) identical tokens ')
    print(f'- ({str(semi_identical_normalized_tokens):>5}) ({semi_identical_normalized_tokens/len(student_vocab)*100:.1f}%) semi-identical normalized tokens.')
    print(f'- ({str(semi_identical_tokens):>5}) ({semi_identical_tokens/len(student_vocab)*100:.1f}%) semi-identical tokens.')
    print(f'- ({str(longest_first_tokens):>5}) ({longest_first_tokens/len(student_vocab)*100:.1f}%) semi-identical tokens.')
    print(f'- ({str(compound_tokens):>5}) ({compound_tokens/len(student_vocab)*100:.1f}%) compound tokens.')
    print(f'- ({str(unk_tokens):>5}) ({unk_tokens/len(student_vocab)*100:.1f}%) unknown tokens.')
    if config.use_flota:
        avg_flota_chunks = sum([len(tokens[1]) for tokens in flota_diffs]) / len(flota_diffs)
        avg_standard_chunks = sum([len(tokens[0]) for tokens in flota_diffs]) / len(flota_diffs)
        flota_diff = sum([1 for tokens in flota_diffs if len(tokens[1]) < len(tokens[0])])
        print(f'FLOTA: Decreased fragmentation for {str(flota_diff):>5} ({flota_diff/len(flota_diffs)*100:.1f}%) tokens '
              f'with an average of {avg_standard_chunks - avg_flota_chunks:.1f} sub-words.')

    # load dummy student model
    student_model_config = AutoConfig.from_pretrained(config.student_model_path,
                                                      use_auth_token=config.auth_token)
    student_model = AutoModelForMaskedLM.from_config(student_model_config)

    # load teacher model
    teacher_model = AutoModelForMaskedLM.from_pretrained(config.teacher_model_path)

    for param in student_model.base_model.parameters():
        param.requires_grad = False

    if teacher_model.config.model_type == 'bert':
        student_model = mod_bert(teacher_model=teacher_model, student_model=student_model,
                                 student_teacher_mapping_ids=student_teacher_mapping_ids)
    else:
        student_model = mod_roberta(teacher_model=teacher_model, student_model=student_model,
                                    student_teacher_mapping_ids=student_teacher_mapping_ids)

    # save frankenstein model
    student_model.save_pretrained(os.path.join(DATA_DIR, config.output_dir))


if __name__ == '__main__':
    warm_start_model()
