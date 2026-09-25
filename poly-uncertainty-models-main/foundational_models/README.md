# Derm Foundation Usage

Derm foundation pretrained models can be found here: https://huggingface.co/google/derm-foundation 

Addition Information:
- This model is gated, requiring a hugging face account, and to be logged in.
- To log in via terminal consult, use the hugging face CLI: https://huggingface.co/docs/huggingface_hub/en/guides/cl
  - An access token needs to also be generated: https://huggingface.co/docs/hub/en/security-tokens
  - Once logged in, you should stay logged in.

Known Issues:
- When bulk generated embeddings, the model will kill my shell session with no error.
- Could only run on CPU
- The saved UMAP model in approx. 500mb and I will need to figure out another way to share it.
