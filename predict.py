import dataclasses
import json
import logging

import numpy as np
import torch
from transformers import BertTokenizer

from layoutlm import LayoutlmConfig, LayoutlmForTokenClassification

LABELS = ['date', 'company', 'address', 'total']
max_position_embeddings = 512
log = logging.getLogger(__name__)


@dataclasses.dataclass
class LayoutLMInput:
    input_ids: torch.LongTensor
    bbox: torch.LongTensor


def load_models(device: torch.device):
    config = LayoutlmConfig.from_pretrained('model/config.json', num_labels=5, cache_dir=None)

    model = LayoutlmForTokenClassification.from_pretrained(f'model/pytorch_model.bin', from_tf=False, config=config,
                                                           cache_dir=None)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained('model/', do_lower_case=True, cache_dir=None)

    return model, tokenizer


def processing_ocr(ocr_lines, tokenizer, img_height, img_width, device):
    boxes = []
    input_ids = []
    seq_id_to_words = dict()
    for line in ocr_lines:

        x0 = line['upper_left_x']
        y0 = line['upper_left_y']
        x2 = line['lower_right_x']
        y2 = line['lower_right_y']
        word = line['words']
        word_tokens = tokenizer.encode(word)[1:-1]
        x_min = 1000 * (int(x0)) / img_width
        y_min = 1000 * (int(y0)) / img_height
        x_max = 1000 * (int(x2)) / img_width
        y_max = 1000 * (int(y2)) / img_height

        if x_max <= x_min:
            log.info(f"Illegal X coordinated ({x_min},{x_max}) for word {word}. skip word...")
            continue
        if y_max <= y_min:
            log.info(f"Illegal y coordinated ({y_min},{y_max}) for word {word}. skip word...")
            continue

        for id_ in word_tokens:
            seq_id_to_words[len(boxes)] = word
            boxes.append([x_min, y_min, x_max, y_max])
            input_ids.append(id_)

    input_ids = [101] + input_ids + [102]  # add start and end tokens.
    if len(input_ids) > max_position_embeddings:
        msg = (f"Invoice too long."
               f"The max length is {max_position_embeddings} and current length is {len(input_ids)}"
               f"Token list : {tokenizer.decode(input_ids)}")
        log.error(msg)
        return None, {}

    boxes = [[0, 0, 0, 0]] + boxes + [[1000, 1000, 1000, 1000]]  # add start and end tokens.
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    boxes = torch.LongTensor(boxes).unsqueeze(0).to(device)
    return LayoutLMInput(input_ids, boxes), seq_id_to_words


def get_words_in_label(pred_labels, seq_id_to_words, label_id):
    context_pred_labels = pred_labels[1:-1]  # remove start and end tokens.
    words = [seq_id_to_words[i] for i in range(len(context_pred_labels)) if context_pred_labels[i] == label_id]
    return list(set(words))


def get_text_labels(pred_labels, seq_id_to_words):
    text_lables = dict()
    for i in range(len(LABELS)):
        text_lables[LABELS[i]] = get_words_in_label(pred_labels, seq_id_to_words, i)
    return text_lables


def check_input(input_ocr):
    assert "img_height" in input_ocr
    assert "img_width" in input_ocr
    assert "tokens" in input_ocr
    assert isinstance(input_ocr["tokens"], list)

    for token in input_ocr["tokens"]:
        assert isinstance(token, dict)
        assert 'upper_left_x' in token
        assert 'upper_left_y' in token
        assert 'lower_right_x' in token
        assert 'lower_right_y' in token
        assert 'words' in token


def main():
    input_ocr_str = input()
    input_ocr = json.loads(input_ocr_str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_models(device)

    img_height, img_width = input_ocr["img_height"], input_ocr["img_width"]

    model_input, seq_id_to_words = processing_ocr(input_ocr["tokens"], tokenizer, img_height, img_width, device)
    if model_input is None:
        return {}

    outputs = model(
        input_ids=model_input.input_ids,
        bbox=model_input.bbox
    )
    pred_labels = np.argmax(outputs[0].detach().cpu().numpy(), axis=-1).flatten()
    results = get_text_labels(pred_labels, seq_id_to_words)
    return results


if __name__ == "__main__":
    print(main())
