from typing import List

from torch import nn
from transformers import CLIPTokenizer, ClapTextModel


class CLAPTextEmbedder(nn.Module):
    """
    ## CLAP Text Embedder
    """

    def __init__(self, version: str = "laion/clap-htsat-unfused", device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = ClapTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        # Get CLAP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state

clap = CLAPTextEmbedder()
clap.to('cuda')

text_emb = clap.forward(prompts = List('str'))