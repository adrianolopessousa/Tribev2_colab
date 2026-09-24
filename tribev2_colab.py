# -*- coding: utf-8 -*-
# %% [markdown]
# # 🧠 TRIBE v2 no Google Colab — guia passo a passo
#
# **TRIBE v2** (Meta / FAIR) é um modelo de *brain encoding* multimodal: dado um
# estímulo naturalista (**vídeo**, **áudio** ou **texto**), ele **prevê a resposta
# de fMRI** de um "sujeito médio" em ~20 mil vértices do córtex (malha
# `fsaverage5`), a cada 1 segundo (1 TR).
#
# Por dentro, ele combina extratores de features estado-da-arte:
#
# | Modalidade | Extrator                          |
# |------------|-----------------------------------|
# | Texto      | LLaMA 3.2 3B (`meta-llama/Llama-3.2-3B`, **acesso restrito**) |
# | Vídeo      | V-JEPA 2 + DINOv2                 |
# | Áudio      | Wav2Vec-BERT                      |
# | Transcrição| WhisperX (roda via `uvx`)         |
#
# Links: [código](https://github.com/facebookresearch/tribev2) ·
# [pesos no HuggingFace](https://huggingface.co/facebook/tribev2) ·
# [demo interativa](https://aidemos.atmeta.com/tribev2/) ·
# licença **CC BY-NC 4.0** (uso não comercial).
#
# ## O que este notebook faz
# 0. Checa a GPU
# 1. Instala o TRIBE v2 (e reinicia o runtime automaticamente)
# 2. Autentica no HuggingFace (necessário para o LLaMA 3.2)
# 3. Carrega o modelo pré-treinado
# 4. Prevê a atividade cerebral para um **vídeo**
# 5. Visualiza no cérebro 3D (com *fallback* 2D caso o 3D falhe)
# 6. Prevê a atividade para **texto** (via text-to-speech)
# 7. Prevê a atividade para **seu próprio arquivo** (upload)
# 8. Análises extras: regiões mais ativas, série temporal, salvar resultados, vídeo MP4
#
# ## ⚠️ Antes de começar (faça uma vez só)
# 1. **Ative a GPU:** menu `Ambiente de execução` → `Alterar o tipo de ambiente de
#    execução` → **T4 GPU** (ou melhor: L4/A100). Sem GPU, a transcrição do WhisperX
#    falha (usa `float16`) e a extração de features fica lentíssima.
# 2. **Peça acesso ao LLaMA 3.2:** entre em
#    <https://huggingface.co/meta-llama/Llama-3.2-3B>, logado na sua conta, e aceite
#    os termos. A aprovação costuma levar de minutos a algumas horas.
# 3. **Crie um token do HuggingFace** (tipo *Read*) em
#    <https://huggingface.co/settings/tokens>.
# 4. **Guarde o token no Colab:** ícone da 🔑 (*Secrets*) na barra lateral esquerda →
#    `Adicionar novo secret` → nome **`HF_TOKEN`**, valor = seu token → ative
#    *Acesso ao notebook*.

# %% [markdown]
# ## Passo 0 — Verificar a GPU

# %%
import subprocess

try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print(
        "❌ Nenhuma GPU encontrada!\n"
        "Vá em 'Ambiente de execução' > 'Alterar o tipo de ambiente de execução' "
        "e selecione uma GPU (T4, L4 ou A100). Depois rode esta célula de novo."
    )

# %% [markdown]
# ## Passo 1 — Instalar o TRIBE v2
#
# Instalamos o pacote direto do GitHub com os extras de visualização
# (`[plotting]`: nilearn, pyvista, matplotlib…). Também instalamos o `xvfb`, um
# "monitor virtual" que permite ao PyVista renderizar o cérebro em 3D num
# servidor sem tela como o Colab.
#
# O TRIBE v2 fixa versões específicas de `torch` e `numpy`, diferentes das que vêm
# no Colab. Por isso **o runtime precisa ser reiniciado** depois da instalação. A
# célula abaixo faz isso **sozinha**: quando aparecer o aviso *"Sua sessão
# falhou/reiniciou"*, é normal. Siga para o **Passo 2** (não rode o Passo 1 de
# novo).
#
# ⏱️ Leva de 3 a 6 minutos.

# %%
import os

!apt-get -qq update > /dev/null
!apt-get -qq install -y xvfb libgl1 > /dev/null
!pip install -q uv
!uv pip install --system -q "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git"
# O WhisperX roda como ferramenta isolada via `uvx`. Pré-baixamos aqui para que a
# primeira transcrição não pareça "travada".
!uvx whisperx --help > /dev/null 2>&1 && echo "✅ WhisperX pronto"

print("✅ Instalação concluída. Reiniciando o runtime...")
os.kill(os.getpid(), 9)  # reinicia o kernel para carregar as novas versões

# %% [markdown]
# ## Passo 2 — Configuração, autenticação no HuggingFace e imports
#
# A partir daqui, **rode as células em ordem**. Se o runtime reiniciar/desconectar
# no futuro, recomece por esta célula (a instalação continua valendo enquanto a
# máquina do Colab não for reciclada).

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- Autenticação no HuggingFace --------------------------------------------
from huggingface_hub import login

hf_token = None
try:
    from google.colab import userdata

    hf_token = userdata.get("HF_TOKEN")
except Exception:
    pass  # secret não configurado ou fora do Colab

if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = hf_token  # visível também para subprocessos
    print("✅ Logado no HuggingFace usando o secret HF_TOKEN")
else:
    # Alternativa: abre uma caixa para colar o token manualmente
    from huggingface_hub import notebook_login

    print("Secret HF_TOKEN não encontrado. Cole seu token abaixo:")
    notebook_login()

# ---- Verifica se você já tem acesso ao LLaMA 3.2 (modelo restrito) ----------
from huggingface_hub import hf_hub_download

try:
    hf_hub_download("meta-llama/Llama-3.2-3B", "config.json")
    print("✅ Acesso ao meta-llama/Llama-3.2-3B confirmado")
except Exception as e:
    print(
        "⚠️ Sem acesso ao meta-llama/Llama-3.2-3B.\n"
        "   Peça acesso em https://huggingface.co/meta-llama/Llama-3.2-3B e aguarde "
        "a aprovação.\n   Sem isso, a etapa de features de TEXTO do modelo vai falhar.\n"
        f"   Detalhe: {type(e).__name__}: {e}"
    )

# ---- Pastas de trabalho ------------------------------------------------------
CACHE_FOLDER = Path("/content/cache")     # features extraídas + modelo
OUTPUT_FOLDER = Path("/content/outputs")  # figuras, .npy, vídeos
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | numpy {np.__version__} | device = {DEVICE}")
if DEVICE == "cpu":
    print("⚠️ Rodando sem GPU: vai ser MUITO lento e a transcrição pode falhar.")

# %% [markdown]
# ## Passo 3 — Carregar o modelo pré-treinado
#
# `TribeModel.from_pretrained("facebook/tribev2")` baixa o checkpoint e a
# configuração (~1 GB) do HuggingFace. Os extratores de features (LLaMA, V-JEPA2,
# Wav2Vec-BERT…) são baixados sob demanda, na primeira previsão.

# %%
from tribev2.demo_utils import TribeModel, download_file

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
    device="auto",  # usa CUDA se disponível
)
print("✅ Modelo carregado")

# %% [markdown]
# ### Funções auxiliares de visualização
#
# O TRIBE v2 traz o `PlotBrain` (PyVista, 3D bonito). Em alguns ambientes a
# renderização 3D *offscreen* falha, então definimos também um **fallback 2D
# com nilearn**, que sempre funciona.
#
# As previsões têm formato `(n_timesteps, 20484)`: os **10242 primeiros vértices
# são o hemisfério esquerdo** e os 10242 seguintes, o direito (malha fsaverage5).

# %%
import matplotlib.pyplot as plt
from nilearn import datasets, plotting

# Liga o monitor virtual para o PyVista (se o xvfb estiver instalado)
try:
    import pyvista as pv

    pv.start_xvfb()
except Exception as e:
    print(f"(xvfb não iniciado: {e})")

plotter = None
try:
    from tribev2.plotting import PlotBrain

    plotter = PlotBrain(mesh="fsaverage5")
    print("✅ PlotBrain (PyVista 3D) disponível")
except Exception as e:
    print(f"⚠️ PlotBrain indisponível ({e}). Vou usar o fallback 2D do nilearn.")

FSAVERAGE5 = datasets.fetch_surf_fsaverage("fsaverage5")
N_VERT_HEMI = 10242


def plot_brain_timesteps(preds, segments, n_timesteps=15, title=None, save_as=None):
    """Plota os primeiros `n_timesteps` segundos de atividade prevista.

    Tenta o PlotBrain 3D oficial; se falhar, cai para o nilearn 2D.
    """
    n = min(n_timesteps, len(preds))
    fig = None
    if plotter is not None:
        try:
            fig = plotter.plot_timesteps(
                preds[:n],
                segments=segments[:n],
                cmap="fire",
                norm_percentile=99,
                vmin=0.6,
                alpha_cmap=(0, 0.2),
                show_stimuli=True,
            )
        except Exception as e:
            print(f"⚠️ Renderização 3D falhou ({type(e).__name__}: {e}). Usando 2D.")
            fig = None
    if fig is None:
        fig = plot_brain_grid_nilearn(preds[:n])
    if fig is not None:
        if title:
            fig.suptitle(title)
        if save_as:
            fig.savefig(OUTPUT_FOLDER / save_as, dpi=120, bbox_inches="tight")
            print(f"💾 Figura salva em {OUTPUT_FOLDER / save_as}")
        plt.show()
    return fig


def plot_brain_grid_nilearn(preds, ncols=5):
    """Fallback: hemisfério esquerdo, vista lateral, um painel por segundo."""
    n = len(preds)
    nrows = int(np.ceil(n / ncols))
    vmax = np.percentile(np.abs(preds), 99)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
        subplot_kw={"projection": "3d"},
    )
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        plotting.plot_surf_stat_map(
            FSAVERAGE5.infl_left,
            preds[i, :N_VERT_HEMI],
            hemi="left",
            view="lateral",
            bg_map=FSAVERAGE5.sulc_left,
            cmap="cold_hot",
            vmax=vmax,
            colorbar=False,
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"t = {i}s", fontsize=9)
    return fig


def plot_brain_mean(preds, title="Atividade média prevista", save_as=None):
    """Mapa médio no tempo, 4 vistas (lateral/medial × esquerdo/direito)."""
    mean_map = preds.mean(axis=0)
    vmax = np.percentile(np.abs(mean_map), 99)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})
    views = [
        ("left", "lateral", FSAVERAGE5.infl_left, FSAVERAGE5.sulc_left, mean_map[:N_VERT_HEMI]),
        ("left", "medial", FSAVERAGE5.infl_left, FSAVERAGE5.sulc_left, mean_map[:N_VERT_HEMI]),
        ("right", "lateral", FSAVERAGE5.infl_right, FSAVERAGE5.sulc_right, mean_map[N_VERT_HEMI:]),
        ("right", "medial", FSAVERAGE5.infl_right, FSAVERAGE5.sulc_right, mean_map[N_VERT_HEMI:]),
    ]
    for ax, (hemi, view, mesh, bg, data) in zip(axes, views):
        plotting.plot_surf_stat_map(
            mesh, data, hemi=hemi, view=view, bg_map=bg, cmap="cold_hot",
            vmax=vmax, colorbar=(ax is axes[-1]), axes=ax, figure=fig,
        )
        ax.set_title(f"{hemi} / {view}")
    fig.suptitle(title)
    if save_as:
        fig.savefig(OUTPUT_FOLDER / save_as, dpi=120, bbox_inches="tight")
    plt.show()
    return fig


# %% [markdown]
# ## Passo 4 — Prever a resposta cerebral a um VÍDEO
#
# A partir de um arquivo de vídeo, o TRIBE v2 automaticamente:
# 1. **Extrai o áudio** da trilha do vídeo;
# 2. **Transcreve a fala** em palavras com timestamps (WhisperX);
# 3. **Extrai features** visuais (V-JEPA2/DINOv2), de áudio (Wav2Vec-BERT) e de
#    texto (LLaMA 3.2);
# 4. **Prevê a atividade fMRI** a cada 1 s em todo o córtex.
#
# Usamos o trailer do curta *Sintel* (Blender Foundation, ~52 s).
#
# ⏱️ Na primeira vez leva ~5–15 min numa T4 (download dos extratores +
# extração de features). Rodadas seguintes usam o cache.

# %%
video_path = CACHE_FOLDER / "sample_video.mp4"
download_file(
    "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4", video_path
)

# 4.1 — Construir o "events dataframe": uma tabela de eventos (vídeo, áudio,
#       palavras) com início, duração e contexto.
df_video = model.get_events_dataframe(video_path=video_path)
print(f"{len(df_video)} eventos | tipos: {df_video['type'].value_counts().to_dict()}")
display(df_video.head(10)[["type", "start", "duration", "filepath", "text", "context"]])

# %%
# 4.2 — Rodar o modelo
preds_video, segments_video = model.predict(events=df_video)
print(f"Formato das previsões: {preds_video.shape}  (n_timesteps, n_vertices)")
print(f"-> {preds_video.shape[0]} segundos de atividade prevista em "
      f"{preds_video.shape[1]} vértices corticais")

# %% [markdown]
# ### 4.3 — Visualizar no cérebro
#
# Cada painel mostra 1 segundo de atividade prevista, com o quadro do vídeo /
# áudio / palavras correspondentes em cima. As previsões são **deslocadas 5 s
# para trás** para compensar o atraso hemodinâmico (a resposta BOLD real
# acontece alguns segundos depois do estímulo).
#
# O que esperar: o **córtex visual** (parte de trás do cérebro) acende quando a
# imagem aparece (~t=4 s) e a **rede de linguagem** (temporal/frontal) quando a
# personagem começa a falar (~t=12 s).

# %%
plot_brain_timesteps(preds_video, segments_video, n_timesteps=15,
                     save_as="video_timesteps.png")
plot_brain_mean(preds_video, title="Vídeo — atividade média prevista",
                save_as="video_mean.png")

# %% [markdown]
# ## Passo 5 — Prever a resposta cerebral a um TEXTO
#
# O modelo foi treinado com áudio/vídeo naturalistas, então o texto é primeiro
# convertido em **fala** (Google Text-to-Speech) e depois transcrito de volta para
# obter o tempo exato de cada palavra.
#
# ⚠️ **Idioma:** o pipeline de transcrição do TRIBE v2 está configurado para
# **inglês**. Textos em português são sintetizados, mas a transcrição/alinhamento
# em inglês fica ruim — use textos em inglês para resultados confiáveis.

# %%
text = """
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
"""

text_path = CACHE_FOLDER / "shakespeare.txt"
text_path.write_text(text, encoding="utf-8")

df_text = model.get_events_dataframe(text_path=text_path)
display(df_text.head(10)[["type", "start", "duration", "filepath", "text", "context"]])

preds_text, segments_text = model.predict(events=df_text)
print(f"Formato das previsões: {preds_text.shape}")

plot_brain_timesteps(preds_text, segments_text, n_timesteps=15,
                     save_as="text_timesteps.png")
plot_brain_mean(preds_text, title="Texto (Hamlet) — atividade média prevista",
                save_as="text_mean.png")

# %% [markdown]
# ## Passo 6 — Usar o SEU próprio arquivo
#
# Rode a célula e escolha um arquivo do seu computador:
# - **Vídeo:** `.mp4 .avi .mkv .mov .webm`
# - **Áudio:** `.wav .mp3 .flac .ogg`
# - **Texto:** `.txt`
#
# Dica: comece com arquivos curtos (30 s – 2 min). Vídeos longos demoram bastante
# para extrair features. Se preferir, troque `USE_UPLOAD = False` e informe um
# caminho em `MY_FILE` (por exemplo, um arquivo no Google Drive montado).

# %%
USE_UPLOAD = True
MY_FILE = "/content/drive/MyDrive/meu_video.mp4"  # usado se USE_UPLOAD = False

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}
TEXT_EXT = {".txt"}

if USE_UPLOAD:
    from google.colab import files

    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("Nenhum arquivo enviado.")
    name = next(iter(uploaded))
    my_path = CACHE_FOLDER / name
    my_path.write_bytes(uploaded[name])
else:
    # from google.colab import drive; drive.mount("/content/drive")
    my_path = Path(MY_FILE)

ext = my_path.suffix.lower()
if ext in VIDEO_EXT:
    df_my = model.get_events_dataframe(video_path=my_path)
elif ext in AUDIO_EXT:
    df_my = model.get_events_dataframe(audio_path=my_path)
elif ext in TEXT_EXT:
    df_my = model.get_events_dataframe(text_path=my_path)
else:
    raise ValueError(f"Extensão não suportada: {ext}")

preds_my, segments_my = model.predict(events=df_my)
print(f"✅ {my_path.name}: previsões {preds_my.shape}")

plot_brain_timesteps(preds_my, segments_my, n_timesteps=15,
                     save_as=f"{my_path.stem}_timesteps.png")
plot_brain_mean(preds_my, title=f"{my_path.name} — atividade média",
                save_as=f"{my_path.stem}_mean.png")

# %% [markdown]
# ## Passo 7 — Análises extras
#
# ### 7.1 — Quais regiões corticais ficam mais ativas?
# Projetamos a atividade média no atlas **Destrieux** (disponível para
# fsaverage5 no nilearn) e listamos as regiões com maior resposta prevista.

# %%
def top_regions(preds, k=15):
    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in destrieux["labels"]]
    mean_map = preds.mean(axis=0)
    rows = []
    for hemi, sl, lab_map in [
        ("E", slice(0, N_VERT_HEMI), destrieux["map_left"]),
        ("D", slice(N_VERT_HEMI, 2 * N_VERT_HEMI), destrieux["map_right"]),
    ]:
        data = mean_map[sl]
        for idx in np.unique(lab_map):
            if labels[idx] in ("Unknown", "Medial_wall"):
                continue
            rows.append({
                "hemisfério": hemi,
                "região": labels[idx],
                "ativação_média": float(data[lab_map == idx].mean()),
            })
    return (pd.DataFrame(rows)
            .sort_values("ativação_média", ascending=False)
            .head(k)
            .reset_index(drop=True))


print("Vídeo (Sintel):")
display(top_regions(preds_video))
print("Texto (Hamlet):")
display(top_regions(preds_text))

# %% [markdown]
# ### 7.2 — Série temporal: visual vs. linguagem
# Comparamos a atividade média ao longo do tempo em uma região visual
# (polo occipital) e em uma de linguagem (giro temporal superior lateral).

# %%
def region_timeseries(preds, region_name, hemi="left"):
    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else l for l in destrieux["labels"]]
    idx = labels.index(region_name)
    if hemi == "left":
        mask = destrieux["map_left"] == idx
        return preds[:, :N_VERT_HEMI][:, mask].mean(axis=1)
    mask = destrieux["map_right"] == idx
    return preds[:, N_VERT_HEMI:][:, mask].mean(axis=1)


fig, ax = plt.subplots(figsize=(12, 4))
for region, label in [
    ("Pole_occipital", "Polo occipital (visual)"),
    ("G_temp_sup-Lateral", "Temporal superior (linguagem/áudio)"),
]:
    ax.plot(region_timeseries(preds_video, region), label=label)
ax.set_xlabel("tempo (s)")
ax.set_ylabel("atividade prevista")
ax.set_title("Vídeo Sintel — série temporal por região (hemisfério esquerdo)")
ax.legend()
plt.show()

# %% [markdown]
# ### 7.3 — Salvar os resultados
# As previsões ficam em `/content/outputs` como `.npy` (numpy). No final,
# compactamos tudo num `.zip` e disparamos o download.

# %%
np.save(OUTPUT_FOLDER / "preds_video.npy", preds_video)
np.save(OUTPUT_FOLDER / "preds_text.npy", preds_text)
if "preds_my" in globals():
    np.save(OUTPUT_FOLDER / f"preds_{my_path.stem}.npy", preds_my)

!cd /content && zip -qr tribev2_outputs.zip outputs && ls -lh tribev2_outputs.zip

try:
    from google.colab import files

    files.download("/content/tribev2_outputs.zip")
except Exception:
    print("Baixe manualmente em /content/tribev2_outputs.zip")

# Opcional: copiar para o Google Drive
# from google.colab import drive; drive.mount("/content/drive")
# !cp -r /content/outputs /content/drive/MyDrive/tribev2_outputs

# %% [markdown]
# ### 7.4 — (Opcional) Gerar um vídeo MP4 da atividade cerebral
# Usa o `plot_timesteps_mp4` do próprio TRIBE v2 (requer a renderização 3D
# funcionando). Pode demorar alguns minutos.

# %%
MAKE_MP4 = False  # troque para True para gerar

if MAKE_MP4 and plotter is not None:
    from IPython.display import Video

    mp4_path = OUTPUT_FOLDER / "brain_video.mp4"
    n = min(30, len(preds_video))
    plotter.plot_timesteps_mp4(
        preds_video[:n],
        mp4_path,
        segments=segments_video[:n],
        norm_percentile=99,
        cmap="fire",
        vmin=0.6,
        alpha_cmap=(0, 0.2),
    )
    display(Video(str(mp4_path), embed=True, width=800))

# %% [markdown]
# ## 🛠️ Solução de problemas
#
# | Problema | Solução |
# |---|---|
# | `ModuleNotFoundError: tribev2` | Rode o **Passo 1** e deixe o runtime reiniciar. |
# | Erro `numpy`/`torch` de versão incompatível | O runtime não reiniciou: `Ambiente de execução` → `Reiniciar sessão`, depois Passo 2. |
# | `401` / `403` / `GatedRepoError` com `meta-llama` | Seu acesso ao LLaMA 3.2 ainda não foi aprovado ou o token não está no secret `HF_TOKEN`. |
# | `whisperx failed` | Verifique se a GPU está ativa (o WhisperX usa `float16`, que não roda em CPU). |
# | `CUDA out of memory` | Use vídeos mais curtos, ou `Ambiente de execução` → `Reiniciar sessão` e rode só o necessário; se possível, use L4/A100. |
# | Figura 3D em branco / erro de OpenGL | O notebook cai automaticamente para o gráfico 2D do nilearn. |
# | Texto em português dá resultados estranhos | A transcrição do pipeline é em inglês: use textos em inglês. |
# | Sessão caiu no meio | O cache em `/content/cache` evita reextrair features enquanto a máquina não for reciclada. |
#
# **Lembretes:** as previsões representam um **sujeito médio** (não uma pessoa
# específica), com resolução de 1 s, deslocadas 5 s para compensar o atraso
# hemodinâmico. Licença do modelo: **CC BY-NC 4.0** (apenas uso não comercial).
