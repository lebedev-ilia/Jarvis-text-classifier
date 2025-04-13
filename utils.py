import torch


class BertInputItem(object):

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, label2idx, max_seq_length, tokenizer):
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        segment_ids = [0] * len(input_ids)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label2idx[label]

        input_items.append(
            BertInputItem(text=text,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))

        
    return input_items

def convert_example_to_input_infer(text, max_seq_length, tokenizer):
    
    input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]

    segment_ids = [0] * len(input_ids)

    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    input_ids = (torch.tensor(input_ids, dtype=torch.long)).unsqueeze(0)
    input_mask = (torch.tensor(input_mask, dtype=torch.long)).unsqueeze(0)
    segment_ids = (torch.tensor(segment_ids, dtype=torch.long)).unsqueeze(0)

    return BertInputItem(text=text,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=None)
