import torch
import torch.nn.functional as F

import onnx
import onnxruntime
import numpy as np
from transformers import (WEIGHTS_NAME, BertConfig,
                          BertForSequenceClassification, BertTokenizer)
from sklearn.metrics import accuracy_score
import time

"""
This script converts pretrained pytorch bert model to onnx
and compares the performance between pytorch and onnx model
"""


def preprocess(tokenizer, text):

    max_seq_length = 128
    tokens = tokenizer.tokenize(text)
    # insert "[CLS]"
    tokens.insert(0, "[CLS]")
    # insert "[SEP]"
    tokens.append("[SEP]")
    segment_ids = []
    for i in range(len(tokens)):
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print("input ids ", input_ids)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids = torch.tensor([input_ids], dtype=torch.long).to("cpu")
    input_mask = torch.tensor([input_mask], dtype=torch.long).to("cpu")
    segment_ids = torch.tensor([segment_ids], dtype=torch.long).to("cpu")

    return input_ids, input_mask, segment_ids

"""
Load the test dataset
"""

def load_data(file_):

    read_file = open(file_)
    examples = []
    labels = []

    for line in read_file:
        example, label = line.split("\t")
        label = label.strip("\n")
        if label == "0":
            examples.append(example)
            labels.append(0)
        elif label == "1":
            examples.append(example)
            labels.append(1)


    print("total examples ", len(examples))
    print("total labels ", len(labels))

    return examples, labels

"""
Inference on pretrained pytorch model
"""

def inference_pytorch(model, input_ids, input_mask, segment_ids):

    with torch.no_grad():
        outputs = model(input_ids, input_mask, segment_ids)

    logits = outputs[0]
    logits = F.softmax(logits, dim=1)
    return logits

"""
This function stores pretrained bert model
into onnx format
"""

def convert_bert_to_onnx(text):

    model_dir = "/home/ramesh/git_repos/transformers/models"
    config = BertConfig.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.to("cpu")
    input_ids, input_mask, segment_ids = preprocess(tokenizer, text)

    torch.onnx.export(model, (input_ids, input_mask, segment_ids), "bert.onnx",  input_names = ["input_ids", "input_mask", "segment_ids"],
    output_names = ["output"])

    print("model convert to onnx format successfully")


def inference(model_name, examples):

    onnx_inference = []
    pytorch_inference = []
    model_dir = "/home/ramesh/git_repos/transformers/models"
    #onnx session
    ort_session = onnxruntime.InferenceSession(model_name)
    #pytorch pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    config = BertConfig.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.to("cpu")

    for example in examples:
        """
        Onnx inference
        """
        input_ids, input_mask, segment_ids = preprocess(tokenizer, examples)
        #
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_ids),
                        ort_session.get_inputs()[1].name: to_numpy(input_mask),
                        ort_session.get_inputs()[2].name: to_numpy(segment_ids)}
        ort_outs = ort_session.run(["output"], ort_inputs)
        torch_onnx_output = torch.tensor(ort_outs[0], dtype=torch.float32)
        onnx_logits = F.softmax(torch_onnx_output, dim=1)

        logits_label = torch.argmax(onnx_logits, dim=1)
        label = logits_label.detach().cpu().numpy()
        onnx_inference.append(label[0])

        """
        Pretrained bert pytorch model
        """
        #

        torch_out = inference_pytorch(model, input_ids, input_mask, segment_ids)

        logits_label = torch.argmax(torch_out, dim=1)
        label = logits_label.detach().cpu().numpy()
        pytorch_inference.append(label[0])


        # compare ONNX Runtime and PyTorch results
        np.testing.assert_allclose(to_numpy(torch_out), onnx_logits, rtol=1e-03, atol=1e-05)

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    return onnx_inference, pytorch_inference


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':

    text = "tick tock tick"
    convert_bert_to_onnx(text)

    examples, labels = load_data("/usr/local/lib/datasets/GLUE/SST-2/dev.tsv")
    # start_time = time.time()
    # print("labels ", labels)

    # returns results from pytorch pretrained model and onnx
    onnx_labels, pytorch_labels = inference("bert.onnx", examples)
    print("\n ************ \n")

    # print("total time ", time.time() - start_time)
    print("accuracy score of onnx model", accuracy_score(labels, onnx_labels))
    print("accuracy score of pytorch model", accuracy_score(labels, pytorch_labels))
