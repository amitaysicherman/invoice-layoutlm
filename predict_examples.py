import time

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from predict import LABELS, load_models, LayoutLMInput, get_text_labels


def processing_ocr_example(ocr_lines, tokenizer, img_height, img_width, device):
    boxes = []
    input_ids = []
    seq_id_to_words = dict()
    for line in ocr_lines:
        x0, y0, _, _, x2, y2, _, _, *word = line.split(",")
        word = " ".join(word)

        word_tokens = tokenizer.encode(word)[1:-1]
        x_min = 1000 * (int(x0)) / img_width
        y_min = 1000 * (int(y0)) / img_height
        x_max = 1000 * (int(x2)) / img_width
        y_max = 1000 * (int(y2)) / img_height
        for id_ in word_tokens:
            seq_id_to_words[len(boxes)] = word
            boxes.append([x_min, y_min, x_max, y_max])
            input_ids.append(id_)

    input_ids = [101] + input_ids + [102]  # add start and end tokens.
    boxes = [[0, 0, 0, 0]] + boxes + [[1000, 1000, 1000, 1000]]  # add start and end tokens.
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
    boxes = torch.LongTensor(boxes).unsqueeze(0).to(device)
    return LayoutLMInput(input_ids, boxes), seq_id_to_words


def plot_results(pred_labels, model_input, input_img_file):
    img = cv2.imread(input_img_file)
    img_height, img_width, c = img.shape

    color_map = {0: (0.12, 0.47, 0.71, 1), 1: (1.0, 0.5, 0.05, 1), 2: (0.17, 0.63, 0.17, 1), 3: (0.84, 0.15, 0.16, 1)}

    for p, b in zip(pred_labels, model_input.bbox[0]):
        if p <= 3:
            rgb = [int(x * 255) for x in color_map[p][:-1]]
            img = cv2.rectangle(img, (int(img_width * b[0] / 1000), int(img_height * b[1] / 1000)),
                                (int(img_width * b[2] / 1000), int(img_height * b[3] / 1000)), rgb, 2)

    plt.figure(figsize=(6, 10), dpi=200)
    plt.imshow(img)

    handles = []
    for i, l in enumerate(LABELS):
        handles.append(mpatches.Patch(color=color_map[i], label=l.upper()))
    plt.legend(handles=handles, loc='upper left', prop={'size': 7})
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_models(device)

    for img_input_file in ['examples/X51009568881.jpg',
                           'examples/X51009453729.jpg',
                           'examples/X51009447842.jpg',
                           'examples/X51009008095.jpg',
                           'examples/X510056849111.jpg']:
        ocr_input_file = img_input_file.replace("jpg", "txt")

        img = cv2.imread(img_input_file)
        img_height, img_width, c = img.shape

        with open(ocr_input_file) as f:
            ocr_lines = f.read().splitlines()

        model_input, seq_id_to_words = processing_ocr_example(ocr_lines, tokenizer, img_height, img_width, device)

        outputs = model(
            input_ids=model_input.input_ids,
            bbox=model_input.bbox
        )
        pred_labels = np.argmax(outputs[0].detach().cpu().numpy(), axis=-1).flatten()
        prediction_time = time.time() - start_time
        results = get_text_labels(pred_labels, seq_id_to_words)
        print(results)
        print(f"prediction time {prediction_time:.2f} sec")

        plot_results(pred_labels, model_input, img_input_file)
