import numpy as np

def convert_single_example(text_a, max_seq_length,
                            tokenizer):

    tokens_a = tokenizer.tokenize(text_a)

    seg_id_a = 0
    seg_id_cls = 0
    seg_id_pad = 0

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(seg_id_cls)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(seg_id_a)
    tokens.append("[SEP]")
    segment_ids.append(seg_id_a)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(seg_id_pad)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return (np.array(input_ids), np.array(input_mask), np.array(segment_ids))

