# TRIBE v2 no Google Colab 🧠

Guia passo a passo, com código pronto, para rodar o **[TRIBE v2](https://github.com/facebookresearch/tribev2)** (Meta/FAIR) no Google Colab. O TRIBE v2 prevê a resposta cerebral de fMRI (~20 mil vértices corticais, 1 amostra por segundo) a partir de **vídeo, áudio ou texto**.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adrianolopessousa/Tribev2_colab/blob/claude/tribe-v2-google-colab-8cfko9/TribeV2_Colab.ipynb)

## Arquivos

| Arquivo | Para quê |
|---|---|
| `TribeV2_Colab.ipynb` | Notebook pronto: abra no Colab e rode as células em ordem. |
| `tribev2_colab.py` | O mesmo conteúdo em formato de script com células `# %%`. As linhas que começam com `!` são comandos de shell do Colab, então copie cada bloco para uma célula do Colab. |

## Pré-requisitos (uma vez só)

1. **GPU no Colab:** `Ambiente de execução` → `Alterar o tipo de ambiente de execução` → T4/L4/A100.
2. **Acesso ao LLaMA 3.2:** aceite os termos em <https://huggingface.co/meta-llama/Llama-3.2-3B>.
3. **Token do HuggingFace** (*Read*) em <https://huggingface.co/settings/tokens>.
4. No Colab, ícone 🔑 **Secrets** → adicione `HF_TOKEN` com o token e ative o acesso ao notebook.

## Etapas do notebook

0. Verifica a GPU
1. Instala o `tribev2[plotting]` e o `xvfb`, depois **reinicia o runtime automaticamente**
2. Faz login no HuggingFace e confirma o acesso ao LLaMA
3. Carrega `facebook/tribev2`
4. Previsão para um vídeo (trailer do *Sintel*) e visualização 3D (com fallback 2D no nilearn)
5. Previsão para texto (Hamlet, via text-to-speech)
6. Previsão para **seu arquivo** (upload ou Google Drive)
7. Extras: regiões mais ativas (atlas Destrieux), séries temporais, salvar `.npy`/`.zip`, vídeo MP4

## Observações

- As previsões representam um **sujeito médio** e são deslocadas 5 s para compensar o atraso hemodinâmico.
- A transcrição do pipeline (WhisperX) está configurada para **inglês**.
- Licença do modelo: **CC BY-NC 4.0**, apenas para uso não comercial.
