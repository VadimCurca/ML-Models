import torch
import numpy as np
import openvino as ov
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from typing import List, Tuple
from pathlib import Path
import time

model_name = "GPT-2"


class Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    eos_token_id = tokenizer.eos_token_id
    eos_token = tokenizer.decode(eos_token_id)

    def tokenize(self, text: str):
        """
        tokenize input text using GPT2 tokenizer

        Parameters:
          text, str - input text
        Returns:
          input_ids - np.array with input token ids
          attention_mask - np.array with 0 in place, where should be padding and 1 for places where original tokens are located, represents attention mask for model
        """

        inputs = self.tokenizer(text, return_tensors="np")
        return inputs["input_ids"], inputs["attention_mask"]

    def decode(self, ids: List[int]):
        """
        decode ids using GPT2 tokenizer

        Parameters:
          ids - np.array with token ids
        Returns:
          output_text - text decoded from token ids
        """
        output_text = ""
        for i in ids:
            output_text += self.tokenizer.batch_decode([i])[0]
        return output_text


def softmax(x: np.array) -> np.array:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    summation = e_x.sum(axis=-1, keepdims=True)
    return e_x / summation


def process_logits(
    cur_length: int, scores: np.array, eos_token_id: int, min_length: int = 0
) -> np.array:
    """
    Reduce probability for padded indices.

    Parameters:
      cur_length: Current length of input sequence.
      scores: Model output logits.
      eos_token_id: Index of end of string token in model vocab.
      min_length: Minimum length for applying postprocessing.

    Returns:
      Processed logits with reduced probability for padded indices.
    """
    if cur_length < min_length:
        scores[:, eos_token_id] = -float("inf")
    return scores


def get_top_k_logits(scores: np.array, top_k: int) -> np.array:
    """
    Perform top-k sampling on the logits scores.

    Parameters:
      scores: np.array, model output logits.
      top_k: int, number of elements with the highest probability to select.

    Returns:
      np.array, shape (batch_size, sequence_length, vocab_size),
        filtered logits scores where only the top-k elements with the highest
        probability are kept and the rest are replaced with -inf
    """
    filter_value = -float("inf")
    top_k = min(max(top_k, 1), scores.shape[-1])
    top_k_scores = -np.sort(-scores)[:, :top_k]
    indices_to_remove = scores < np.min(top_k_scores)
    filtred_scores = np.ma.array(
        scores, mask=indices_to_remove, fill_value=filter_value
    ).filled()
    return filtred_scores


def generate_sequence(
    compiled_model: ov.CompiledModel,
    input_ids: List[int],
    attention_mask: List[int],
    eos_token_id: int,
    max_sequence_length: int = 128,
    dynamic_shapes: bool = True,
) -> List[int]:
    """
    Generates a sequence of tokens using a pre-trained language model.

    Parameters:
      input_ids: np.array, tokenized input ids for model
      attention_mask: np.array, attention mask for model
      max_sequence_length: int, maximum sequence length for stopping iteration
      eos_token_id: int, index of the end-of-sequence token in the model's vocabulary
      dynamic_shapes: bool, whether to use dynamic shapes for inference or pad model input to max_sequence_length

    Returns:
      np.array, the predicted sequence of token ids
    """
    while True:
        cur_input_len = len(input_ids[0])
        if not dynamic_shapes:
            pad_len = max_sequence_length - cur_input_len
            model_input_ids = np.concatenate(
                (input_ids, [[eos_token_id] * pad_len]), axis=-1
            )
            model_input_attention_mask = np.concatenate(
                (attention_mask, [[0] * pad_len]), axis=-1
            )
        else:
            model_input_ids = input_ids
            model_input_attention_mask = attention_mask
        output_key = compiled_model.output(0)
        outputs = compiled_model(
            {"input_ids": model_input_ids, "attention_mask": model_input_attention_mask}
        )[output_key]
        next_token_logits = outputs[:, cur_input_len - 1, :]
        # pre-process distribution
        next_token_scores = process_logits(
            cur_input_len, next_token_logits, eos_token_id
        )
        top_k = 20
        next_token_scores = get_top_k_logits(next_token_scores, top_k)
        # get next token id
        probs = softmax(next_token_scores)
        next_tokens = np.random.choice(probs.shape[-1], 1, p=probs[0], replace=True)
        # break the loop if max length or end of text token is reached
        if cur_input_len == max_sequence_length or next_tokens[0] == eos_token_id:
            break
        else:
            input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
            attention_mask = np.concatenate(
                (attention_mask, [[1] * len(next_tokens)]), axis=-1
            )
    return input_ids


def main():
    core = ov.Core()
    tokenizer = Tokenizer()
    pt_model = GPT2LMHeadModel.from_pretrained("gpt2")

    example_input = {
        "input_ids": torch.ones((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    pt_model.config.torchscript = True

    # convert model to openvino
    ov_model = ov.convert_model(
        pt_model,
        example_input=example_input,
        input=[
            ("input_ids", [1, ov.Dimension(1, 128)], ov.Type.i64),
            ("attention_mask", [1, ov.Dimension(1, 128)], ov.Type.i64),
        ],
    )
    compiled_model = core.compile_model(model=ov_model, device_name="CPU")

    text = "A python walks into a bar"

    """
    GPT-2:  A python walks into a bar and begins drinking. 
    Then, the bartender yells, "Get out, you drunkard" and proceeds to make a "stupid" remark.
    After that, he tells someone what a drunkard is.
    You can read more about this video on The Drunkard's Blog.
    The video was uploaded by a drunkard on April 12.
    The video's original copyright date is August 11, 2015 at 9am in the bar in Portland, Oregon.
    It has since been verified with the police.
    """

    input_ids, attention_mask = tokenizer.tokenize(text)

    output_ids = generate_sequence(
        compiled_model, input_ids, attention_mask, tokenizer.eos_token_id
    )

    # Convert IDs to words and make the sentence from it
    output_text = tokenizer.decode(output_ids[0])

    print(f"Input Text:  {text}")
    print()
    print(f"{model_name}: {output_text}")


if __name__ == "__main__":
    main()
